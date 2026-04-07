"""
Rollout / batch utilities for RL training.

This is the engine-side of RL: generate completions with vLLM, score per-token
log-probs with the HF training model, pack rollouts into padded tensors, and
push updated weights back into the vLLM engine between steps.
"""

import torch
import torch.nn.functional as F


def get_logprobs(model, input_ids, attention_mask, response_mask):
    """Compute per-sample masked-mean log-probs over response tokens.

    Returns a tensor of shape [B], where each entry is the mean log-prob of
    the response tokens under `model` (prompt tokens are masked out).
    """
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
    """Generate `num_samples` completions per prompt using vLLM.

    Returns a flat list of dicts, one per (prompt, sample) pair, ordered such
    that the N samples for prompt i occupy positions [i*N, (i+1)*N).
    """
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    outputs = vllm_engine.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        prompt_text = output.prompt
        for completion in output.outputs:
            results.append({
                "prompt": prompt_text,
                "response": completion.text,
                "prompt_ids": list(output.prompt_token_ids),
                "response_ids": list(completion.token_ids),
            })
    return results


def prepare_batch(rollouts, rewards, tokenizer, max_seq_len, device):
    """Pack a list of rollouts into padded training tensors."""
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


def _worker_load_weights(worker, weights):
    """Top-level helper so it's picklable for collective_rpc on TP>1 workers."""
    worker.model_runner.model.load_weights(weights)


def vllm_weight_sync(vllm_engine, hf_model):
    """Push HF-model weights into the vLLM engine in-place.

    We hand the HF state_dict straight to vLLM's `model.load_weights`, which
    is the same path vLLM uses when loading from a HF checkpoint at startup.
    That means name-mapping quirks (fused QKV, fused gate_up_proj, MoE expert
    packing, etc.) are handled by vLLM itself — we don't maintain a table.

    Works for both TP=1 (direct worker access) and TP>1 (collective_rpc).
    Assumes the trainer and the vLLM engine use the same dtype (bf16 here).
    """
    # Unwrap DDP defensively in case the caller forgets
    if hasattr(hf_model, "module"):
        hf_model = hf_model.module

    # Materialize the (name, tensor) pairs once. For small models (≤7B in bf16)
    # this is a few GB and lives on GPU; load_weights will copy/shard as needed.
    weights = list(hf_model.state_dict().items())

    if hasattr(vllm_engine, "collective_rpc"):
        # Modern vLLM (≥0.6.3): one call dispatches to every TP worker.
        vllm_engine.collective_rpc(_worker_load_weights, args=(weights,))
    else:
        # Fallback for older vLLM: reach into the driver worker directly.
        # Only correct for TP=1, which is the only case older vLLM colocates anyway.
        worker = vllm_engine.llm_engine.model_executor.driver_worker
        worker.model_runner.model.load_weights(weights)

    # load_weights leaves transient buffers around; reclaim them so the next
    # rollout doesn't OOM when training and inference share a GPU.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
