"""
Minimal RL training loop for HuggingFace causal-LM models.

This script is intentionally thin: it owns model loading (HF for training,
vLLM for rollout generation), the optimization loop, and checkpoint saving.
Everything tweakable lives under nanochat/:

  - nanochat/rl_loss.py    : pluggable loss functions (GRPO, DAPO, REINFORCE)
  - nanochat/rl_rollout.py : vLLM rollout, log-prob scoring, batch packing
  - nanochat/rl_data.py    : RL dataset + reward interface (placeholder)

Usage:
    # Single GPU
    python -m scripts.rl_train --model Qwen/Qwen3-0.6B --algorithm grpo

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m scripts.rl_train --model Qwen/Qwen3-0.6B --algorithm grpo
"""

import os
import math
import time
import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.rl_loss import ALGORITHMS
from nanochat.rl_rollout import get_logprobs, generate_rollouts, prepare_batch, vllm_weight_sync
from nanochat.rl_data import build_rl_dataset, distributed_rl_loader, RewardWorkerPool

# -----------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="RL training for HF models")
# Model
parser.add_argument("--model", type=str, required=True, help="HF model path (e.g. Qwen/Qwen3-0.6B)")
# Algorithm
parser.add_argument("--algorithm", type=str, default="grpo", choices=list(ALGORITHMS.keys()))
parser.add_argument("--clip", type=float, default=0.2, help="PPO/GRPO clip range")
parser.add_argument("--kl-coeff", type=float, default=0.0, help="KL penalty coefficient")
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
# Evaluation
parser.add_argument("--eval-every", type=int, default=20)
# Task / Reward
parser.add_argument("--task", type=str, default="rstar_seed", help="RL dataset name")
parser.add_argument("--reward-workers", type=int, default=0, help="Reward worker pool size")
parser.add_argument("--k-tests", type=int, default=10, help="Tests subsampled per rollout")
# Runtime
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--save-dir", type=str, default="rl_checkpoints")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

print0(f"Loading model: {args.model}")
print0(f"Algorithm: {args.algorithm}")
print0(f"Device: {device}, World size: {ddp_world_size}")

# -----------------------------------------------------------------------------
# Tokenizer + HF training model
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model.to(device)
model.train()
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# vLLM inference engine (rank 0 only)
vllm_engine = None
if master_process:
    from vllm import LLM
    vllm_engine = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.3)
    print0("vLLM inference engine ready")

# -----------------------------------------------------------------------------
# RL dataset + reward worker pool + loss fn
dataset = build_rl_dataset(args.task, split="train")
loader = distributed_rl_loader(
    dataset,
    prompts_per_step=args.prompts_per_step,
    world_size=ddp_world_size,
    rank=ddp_rank,
    seed=args.seed,
)
rewarder = RewardWorkerPool(num_workers=args.reward_workers, k_tests=args.k_tests)
loss_fn = ALGORITHMS[args.algorithm]

# -----------------------------------------------------------------------------
# Optimizer
optimizer = torch.optim.AdamW(
    raw_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
)

# -----------------------------------------------------------------------------
# Training loop
print0(f"Starting RL training: {args.num_steps} steps, {args.prompts_per_step} prompts/step, {args.num_samples} samples/prompt")

for step in range(args.num_steps):
    t0 = time.time()

    # 1. Sample prompts (disjoint slice per rank)
    examples, _loader_state = next(loader)
    prompt_texts = [ex.prompt for ex in examples]

    # 2. Generate rollouts via vLLM (rank 0)
    rollouts = generate_rollouts(
        vllm_engine, tokenizer, prompt_texts,
        args.num_samples, args.max_new_tokens,
        args.temperature, args.top_k,
    )

    # 3. Compute rewards
    # Re-expand examples to align 1:1 with the flat rollout list
    expanded_examples = [examples[i // args.num_samples] for i in range(len(rollouts))]
    responses = [r["response"] for r in rollouts]
    rewards, _infos = rewarder.score(expanded_examples, responses, step=step)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # 4. Pack training batch
    batch = prepare_batch(rollouts, rewards, tokenizer, args.max_seq_len, device)

    # 5. Old log-probs (no grad, frozen behavior model = current policy)
    with torch.no_grad():
        old_logprobs = get_logprobs(
            raw_model, batch["input_ids"], batch["attention_mask"], batch["response_mask"],
        )

    # 6. Policy update with grad accumulation over micro-batches
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
        loss = loss_fn(
            logprobs=logprobs,
            old_logprobs=mb_old_lp,
            rewards=mb_rewards,
            clip=args.clip,
            kl_coeff=args.kl_coeff,
        ) / n_microbatches
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    optimizer.step()

    # 7. Push updated weights into vLLM so next-step rollouts stay on-policy
    if vllm_engine is not None:
        vllm_weight_sync(vllm_engine, raw_model)

    dt = time.time() - t0
    print0(f"step {step:04d}/{args.num_steps:04d} | loss: {total_loss:.4f} | reward: {mean_reward:.4f} | dt: {dt:.1f}s")

    # 8. Evaluation
    if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
        print0("Evaluating...")
        # TODO: pass@k evaluation hook

# -----------------------------------------------------------------------------
# Save
if master_process:
    save_path = os.path.join(args.save_dir, f"{args.model.replace('/', '_')}_{args.algorithm}")
    os.makedirs(save_path, exist_ok=True)
    raw_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print0(f"Saved to {save_path}")

rewarder.close()
compute_cleanup()
