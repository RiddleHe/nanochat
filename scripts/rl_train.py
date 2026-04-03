"""
Minimal RL training loop for research on HuggingFace models.

Uses vLLM for fast rollout generation, HF model for gradient updates.
Loss function is pluggable via --algorithm flag.

Usage:
    # Single GPU
    python -m scripts.rl_train --model Qwen/Qwen3-0.6B --algorithm grpo

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m scripts.rl_train --model Qwen/Qwen3-0.6B --algorithm grpo

    # With distillation (teacher model)
    torchrun --nproc_per_node=4 -m scripts.rl_train --model Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-1.7B --algorithm grpo
"""

import os
import gc
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type

# =============================================================================
# Loss functions (pluggable via --algorithm)
# =============================================================================

def grpo_loss(logprobs, old_logprobs, rewards, clip=0.2, kl_coeff=0.0, **kwargs):
    """GRPO: group-relative policy optimization."""
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    ratio = (logprobs - old_logprobs).exp()
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    if kl_coeff > 0:
        kl = (old_logprobs - logprobs).mean()
        pg_loss = pg_loss + kl_coeff * kl
    return pg_loss

def dapo_loss(logprobs, old_logprobs, rewards, clip_low=0.8, clip_high=1.28, **kwargs):
    """DAPO: decoupled asymmetric policy optimization."""
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    ratio = (logprobs - old_logprobs).exp()
    clipped = torch.clamp(ratio, clip_low, clip_high)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

def reinforce_loss(logprobs, old_logprobs, rewards, **kwargs):
    """Simple REINFORCE with mean-subtracted advantages."""
    advantages = rewards - rewards.mean()
    return -(logprobs * advantages).mean()

def distill_grpo_loss(logprobs, old_logprobs, rewards, teacher_logprobs=None,
                      clip=0.2, distill_coeff=0.1, **kwargs):
    """GRPO + on-policy distillation from teacher logprobs."""
    # Standard GRPO
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    ratio = (logprobs - old_logprobs).exp()
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    # Distillation: KL(teacher || student)
    if teacher_logprobs is not None:
        distill_loss = F.kl_div(logprobs, teacher_logprobs.exp(), log_target=False, reduction='batchmean')
        return pg_loss + distill_coeff * distill_loss
    return pg_loss

ALGORITHMS = {
    "grpo": grpo_loss,
    "dapo": dapo_loss,
    "reinforce": reinforce_loss,
    "distill_grpo": distill_grpo_loss,
}

# =============================================================================
# Utilities
# =============================================================================

def get_logprobs(model, input_ids, attention_mask, response_mask):
    """Compute per-token log-probs for response tokens only."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    # Masked mean over response tokens
    masked_logprobs = (token_logprobs * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
    return masked_logprobs

def generate_rollouts(vllm_engine, tokenizer, prompts, num_samples, max_new_tokens,
                      temperature, top_k):
    """Generate multiple completions per prompt using vLLM."""
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    # Duplicate prompts are handled by vLLM's n parameter
    outputs = vllm_engine.generate(prompts, sampling_params)
    # Flatten: list of (prompt, [completions])
    results = []
    for output in outputs:
        prompt_text = output.prompt
        for completion in output.outputs:
            results.append({
                "prompt": prompt_text,
                "response": completion.text,
                "prompt_ids": output.prompt_token_ids,
                "response_ids": list(completion.token_ids),
            })
    return results

def prepare_batch(rollouts, rewards, tokenizer, max_seq_len, device):
    """Prepare padded tensors for a training batch."""
    input_ids_list = []
    response_mask_list = []
    for rollout in rollouts:
        prompt_ids = rollout["prompt_ids"]
        response_ids = rollout["response_ids"]
        full_ids = prompt_ids + response_ids
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            response_ids = full_ids[len(prompt_ids):]
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        input_ids_list.append(full_ids)
        response_mask_list.append(mask)

    # Pad to max length in batch
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
    padded_masks = [m + [0] * (max_len - len(m)) for m in response_mask_list]
    attn_masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=device),
        "response_mask": torch.tensor(padded_masks, dtype=torch.float, device=device),
        "rewards": torch.tensor(rewards, dtype=torch.float, device=device),
    }

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description="RL training for HF models")
# Model
parser.add_argument("--model", type=str, required=True, help="HF model path (e.g. Qwen/Qwen3-0.6B)")
parser.add_argument("--teacher", type=str, default=None, help="HF teacher model path for distillation")
# Algorithm
parser.add_argument("--algorithm", type=str, default="grpo", choices=list(ALGORITHMS.keys()))
parser.add_argument("--clip", type=float, default=0.2, help="PPO/GRPO clip range")
parser.add_argument("--kl-coeff", type=float, default=0.0, help="KL penalty coefficient")
parser.add_argument("--distill-coeff", type=float, default=0.1, help="Distillation loss coefficient")
# Generation
parser.add_argument("--num-samples", type=int, default=16, help="Completions per prompt")
parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
# Training
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
parser.add_argument("--num-steps", type=int, default=200, help="Number of RL steps")
parser.add_argument("--prompts-per-step", type=int, default=8, help="Prompts per RL step")
parser.add_argument("--train-batch-size", type=int, default=16, help="Training micro-batch size")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--grad-accum", type=int, default=1)
# Evaluation
parser.add_argument("--eval-every", type=int, default=20)
# Task / Reward
parser.add_argument("--task", type=str, default="gsm8k", help="Task name for reward function")
# Runtime
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--save-dir", type=str, default="rl_checkpoints")
args = parser.parse_args()

# =============================================================================
# Setup
# =============================================================================

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

print0(f"Loading model: {args.model}")
print0(f"Algorithm: {args.algorithm}")
print0(f"Device: {device}, World size: {ddp_world_size}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load HF model for training
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model.to(device)
model.train()
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Load vLLM engine for fast generation (rank 0 only, or each rank independently)
vllm_engine = None
if master_process:
    try:
        from vllm import LLM
        vllm_engine = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.3)
        print0("Using vLLM for generation")
    except ImportError:
        print0("vLLM not available, using HF generate (slower)")

# Load teacher model if distillation
teacher_model = None
if args.teacher:
    print0(f"Loading teacher: {args.teacher}")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16)
    teacher_model.to(device)
    teacher_model.eval()

# Load task / reward function
if args.task == "gsm8k":
    from tasks.gsm8k import GSM8K
    task = GSM8K(split="train")
    eval_task = GSM8K(split="test")
else:
    raise ValueError(f"Unknown task: {args.task}")

# Optimizer
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
loss_fn = ALGORITHMS[args.algorithm]

# =============================================================================
# Training loop
# =============================================================================

print0(f"Starting RL training: {args.num_steps} steps, {args.prompts_per_step} prompts/step, {args.num_samples} samples/prompt")

for step in range(args.num_steps):
    t0 = time.time()

    # --- 1. Sample prompts ---
    prompts_per_rank = args.prompts_per_step // max(ddp_world_size, 1)
    prompt_examples = [task.sample() for _ in range(prompts_per_rank)]
    prompt_texts = [ex["prompt"] for ex in prompt_examples]

    # --- 2. Generate rollouts ---
    if vllm_engine is not None and master_process:
        rollouts = generate_rollouts(
            vllm_engine, tokenizer, prompt_texts,
            args.num_samples, args.max_new_tokens,
            args.temperature, args.top_k,
        )
    else:
        # Fallback: HF generate
        rollouts = []
        raw_model.eval()
        with torch.no_grad():
            for prompt_text in prompt_texts:
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
                for _ in range(args.num_samples):
                    output = raw_model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k,
                        do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    )
                    response_ids = output[0, input_ids.shape[1]:].tolist()
                    rollouts.append({
                        "prompt": prompt_text,
                        "response": tokenizer.decode(response_ids, skip_special_tokens=True),
                        "prompt_ids": input_ids[0].tolist(),
                        "response_ids": response_ids,
                    })
        raw_model.train()

    # --- 3. Compute rewards ---
    rewards = []
    for i, rollout in enumerate(rollouts):
        example_idx = i // args.num_samples
        reward = prompt_examples[example_idx].get("reward_fn", task.reward)(
            rollout["prompt"], rollout["response"]
        )
        rewards.append(reward)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # --- 4. Prepare training batch ---
    batch = prepare_batch(rollouts, rewards, tokenizer, args.max_seq_len, device)

    # --- 5. Compute old logprobs (no grad) ---
    with torch.no_grad():
        old_logprobs = get_logprobs(raw_model, batch["input_ids"], batch["attention_mask"], batch["response_mask"])

    # --- 6. Compute teacher logprobs if distillation ---
    teacher_logprobs = None
    if teacher_model is not None:
        with torch.no_grad():
            teacher_logprobs = get_logprobs(teacher_model, batch["input_ids"], batch["attention_mask"], batch["response_mask"])

    # --- 7. Training step (with grad accumulation over micro-batches) ---
    total_samples = batch["input_ids"].shape[0]
    micro_bs = args.train_batch_size
    n_microbatches = math.ceil(total_samples / micro_bs)

    optimizer.zero_grad()
    total_loss = 0.0
    for mb in range(n_microbatches):
        start = mb * micro_bs
        end = min(start + micro_bs, total_samples)
        mb_ids = batch["input_ids"][start:end]
        mb_attn = batch["attention_mask"][start:end]
        mb_resp = batch["response_mask"][start:end]
        mb_rewards = batch["rewards"][start:end]
        mb_old_lp = old_logprobs[start:end]

        logprobs = get_logprobs(model, mb_ids, mb_attn, mb_resp)

        loss_kwargs = dict(
            logprobs=logprobs,
            old_logprobs=mb_old_lp,
            rewards=mb_rewards,
            clip=args.clip,
            kl_coeff=args.kl_coeff,
        )
        if teacher_logprobs is not None:
            loss_kwargs["teacher_logprobs"] = teacher_logprobs[start:end]
            loss_kwargs["distill_coeff"] = args.distill_coeff

        loss = loss_fn(**loss_kwargs) / n_microbatches
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    optimizer.step()

    # --- 8. Sync vLLM weights ---
    if vllm_engine is not None and master_process:
        # Update vLLM model weights from the trained HF model
        from vllm.worker.model_runner import ModelRunner
        # Simple approach: reload weights (slow but correct)
        # For production, use vLLM's weight update API if available
        pass  # TODO: implement efficient weight sync

    dt = time.time() - t0
    print0(f"step {step:04d}/{args.num_steps:04d} | loss: {total_loss:.4f} | reward: {mean_reward:.4f} | dt: {dt:.1f}s")

    # --- 9. Evaluation ---
    if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
        print0("Evaluating...")
        # TODO: pass@k evaluation on eval_task

# =============================================================================
# Save
# =============================================================================

if master_process:
    save_path = os.path.join(args.save_dir, f"{args.model.replace('/', '_')}_{args.algorithm}")
    os.makedirs(save_path, exist_ok=True)
    raw_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print0(f"Saved to {save_path}")

compute_cleanup()
