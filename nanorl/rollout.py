"""
Rollout / batch utilities for RL training.

Generation runs in a separate vLLM worker process. The trainer calls the
`*_remote` helpers (HTTP to the worker); the worker itself reuses
`generate_rollouts` and `vllm_reload_weights_inplace` from this module.
"""

import json
import os
import shutil
import time
import urllib.error
import urllib.request

import torch


def get_logprobs(model, input_ids, attention_mask, response_mask):
    """Compute per-response-token log-probs under `model`.

    Returns a tuple ``(token_logprobs, shift_mask)`` each of shape ``[B, T-1]``:
      - ``token_logprobs[b, t] = log π_θ(input_ids[b, t+1] | input_ids[b, :t+1])``
      - ``shift_mask[b, t] = 1.0`` iff ``input_ids[b, t+1]`` is a response token.

    Per-token (not per-sequence) log-probs are required so that PPO/DAPO/GRPO
    can apply per-token importance ratios and per-token clipping — the
    sample-level masked-mean form makes the clip bounds essentially non-
    functional (the geometric mean of many per-token ratios is always ≈ 1).
    """
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]
    # log_softmax(x)[k] = x[k] - logsumexp(x). logsumexp saves only [B,T] for
    # backward vs log_softmax's [B,T,V] — avoids a ~V× persistent allocation.
    gathered = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = gathered - torch.logsumexp(shift_logits, dim=-1)
    return token_logprobs, shift_mask


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


def vllm_reload_weights_inplace(vllm_engine, model_path):
    """Reload checkpoint-format weights into an existing vLLM engine.

    Refreshes model parameters and invalidates generation-time caches without
    tearing down the engine.
    """
    vllm_engine.collective_rpc(
        "reload_weights",
        kwargs={"weights_path": model_path, "is_checkpoint_format": True},
    )
    vllm_engine.reset_prefix_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
