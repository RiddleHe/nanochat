"""
Rollout / batch utilities for RL training.

This is the engine-side of RL: generate completions with vLLM, score per-token
log-probs with the HF training model, pack rollouts into padded tensors, and
push updated weights back into the vLLM engine between steps.
"""

import gc
import json
import os
import shutil
import time
import urllib.error
import urllib.request

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


def _remote_json_request(base_url, method, path, payload=None, timeout=600):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"remote rollout request failed: {e.code} {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"remote rollout request failed: {e}") from e

    if not body:
        return None
    return json.loads(body)


def wait_for_rollout_worker(base_url, timeout_s=300):
    """Poll the rollout worker until it reports healthy."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            payload = _remote_json_request(base_url, "GET", "/health", timeout=10)
            if payload and payload.get("ok"):
                return payload
        except Exception as e:  # pragma: no cover - best-effort polling
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(
        f"rollout worker at {base_url} did not become healthy within {timeout_s}s"
        + (f"; last error: {last_err}" if last_err else "")
    )


def generate_rollouts_remote(base_url, prompts, num_samples, max_new_tokens,
                             temperature, top_k):
    """Generate rollouts via a separate rollout worker process."""
    payload = {
        "prompts": prompts,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    resp = _remote_json_request(base_url, "POST", "/generate", payload=payload, timeout=1800)
    return resp["rollouts"]


def remote_vllm_reload(base_url, model_path):
    """Ask the rollout worker to reload weights from a checkpoint path."""
    resp = _remote_json_request(
        base_url, "POST", "/reload", payload={"model_path": model_path}, timeout=1800,
    )
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"rollout worker reload failed for {model_path}: {resp}")
    return resp


def materialize_rollout_checkpoint(hf_model, sync_root, slot_idx, tokenizer_source=None):
    """Write a HF checkpoint into one of two alternating sync slots.

    The worker reloads from the returned slot path. Alternating slots preserves
    strict per-step semantics without accumulating one checkpoint directory per
    RL step.
    """
    if hasattr(hf_model, "module"):
        hf_model = hf_model.module

    os.makedirs(sync_root, exist_ok=True)
    slot_path = os.path.join(sync_root, f"slot{slot_idx}")
    tmp_path = f"{slot_path}.tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    shutil.rmtree(slot_path, ignore_errors=True)
    hf_model.save_pretrained(tmp_path, safe_serialization=True)

    if tokenizer_source is not None:
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
        ):
            src = os.path.join(tokenizer_source, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp_path, name))

    os.replace(tmp_path, slot_path)
    return slot_path


def generate_rollouts_hf(model, tokenizer, prompts, num_samples, max_new_tokens,
                         temperature, top_k, device):
    """Generate rollouts with the HF training model directly.

    This is slower than vLLM but avoids the extra rollout engine dependency
    during debugging or in setups where colocated vLLM is not stable.
    """
    do_sample = temperature > 0
    was_training = model.training
    model.eval()
    results = []
    try:
        with torch.no_grad():
            for prompt_text in prompts:
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_k=top_k if do_sample else None,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                prompt_ids = inputs["input_ids"][0].tolist()
                prompt_len = len(prompt_ids)
                for seq in outputs:
                    response_ids = seq[prompt_len:].tolist()
                    results.append({
                        "prompt": prompt_text,
                        "response": tokenizer.decode(response_ids, skip_special_tokens=True),
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                    })
    finally:
        if was_training:
            model.train()
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


def vllm_reload_model(model_path, tokenizer_path, dtype="bfloat16", gpu_memory_utilization=0.3):
    """Create a fresh vLLM engine from a local checkpoint path."""
    from vllm import LLM

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=True,
    )
