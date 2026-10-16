"""
Rollout / batch utilities for RL training.

Generation runs in a separate vLLM worker process. The trainer calls the
`*_remote` helpers (HTTP to the worker) and pushes weights into the worker
in-place via NCCL using `sync_weights_to_vllm_inplace`. The worker itself
uses `generate_rollouts` from this module.
"""

import json
import time
import logging
from typing import Any
import urllib.error
import urllib.request
from vllm import SamplingParams
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine, NCCLTrainerSendWeightsArgs

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nanorl.rollout_worker")

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
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_new_tokens,
        logprobs=1,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    outputs = vllm_engine.generate(prompts, sampling_params)
    results: list[dict[str,Any]] = []
    for output in outputs:
        prompt_text = output.prompt
        for completion in output.outputs:
            inference_logprobs = [
                lp_dict[token_id].logprob
                for token_id, lp_dict in zip(completion.token_ids, completion.logprobs)
            ]
            results.append({
                "prompt": prompt_text,
                "response": completion.text,
                "prompt_ids": list(output.prompt_token_ids),
                "response_ids": list(completion.token_ids),
                "inference_logprobs": inference_logprobs,
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


def prepare_batch(rollouts, rewards, tokenizer, max_seq_len, device):
    """Pack a list of rollouts into padded training tensors."""
    input_ids_list = []
    response_mask_list = []
    prompt_lens = []
    response_lens = []
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
        prompt_lens.append(len(prompt_ids))
        response_lens.append(len(response_ids))

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
    padded_masks = [m + [0] * (max_len - len(m)) for m in response_mask_list]
    attn_masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]

    # Build inference_logprobs aligned with shift_mask shape [B, max_len-1].
    # token_logprobs[b, t] = logprob of input_ids[b, t+1], so response token i
    # (0-indexed within the response) lands at position P-1+i in the shifted tensor.
    inf_lp_list = []
    for rollout, P, R in zip(rollouts, prompt_lens, response_lens):
        row = [0.0] * (max_len - 1)
        for i, lp in enumerate(rollout["inference_logprobs"][:R]):
            row[P - 1 + i] = lp
        inf_lp_list.append(row)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=device),
        "response_mask": torch.tensor(padded_masks, dtype=torch.float, device=device),
        "rewards": torch.tensor(rewards, dtype=torch.float, device=device),
        "inference_logprobs": torch.tensor(inf_lp_list, dtype=torch.float, device=device),
    }

def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _iter_fsdp_full_params(model):
    for name, param in model.named_parameters():
        if hasattr(param, "full_tensor"):
            yield name, param.full_tensor()
            continue
        yield name, param


def _iter_model_parameters(model, fsdp: bool):
    if fsdp:
        yield from _iter_fsdp_full_params(model)
        return
    yield from model.named_parameters()

def collect_weight_metadata(model, fsdp: bool = False) -> dict[str, Any]:
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    for name, param in _iter_model_parameters(model, fsdp=fsdp):
        names.append(name)
        dtype_names.append(_dtype_name(param.dtype))
        shapes.append(list(param.shape))
    return {
        "names": names,
        "dtype_names": dtype_names,
        "shapes": shapes,
    }

def remote_vllm_start_update_weights(base_url, metadata: dict[str, Any], packed: bool):
    payload = {
        "names": metadata["names"],
        "dtype_names": metadata["dtype_names"],
        "shapes": metadata["shapes"],
        "packed": packed,
        "is_checkpoint_format": True,
    }
    logger.info(f"Starting in-place vLLM weight update with {packed=}, {len(metadata['names'])=}")
    resp = _remote_json_request(
        base_url, "POST", "/update_weights_start", payload=payload, timeout=1800,
    )
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"rollout worker update-start failed: {resp}")
    return resp

def remote_vllm_finish_update_weights(base_url):
    logger.info("Waiting for vLLM weight update to complete.")
    resp = _remote_json_request(
        base_url, "POST", "/update_weights_finish", payload={}, timeout=1800,
    )
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"rollout worker update-finish failed: {resp}")
    return resp

def sync_weights_to_vllm_inplace(
    train_model,
    base_url,
    model_update_group,
    *,
    packed: bool = True,
    fsdp: bool = False,
):
    """Sync trainer weights into the running vLLM worker without checkpoints."""

    if hasattr(train_model, "module"):
        train_model = train_model.module

    metadata = collect_weight_metadata(train_model, fsdp=fsdp)
    remote_vllm_start_update_weights(base_url, metadata, packed=packed)

    param_iterator = _iter_model_parameters(train_model, fsdp=fsdp)
    logger.info(f"Sending trainer weights via NCCL with {packed=}, {fsdp=}.")
    NCCLWeightTransferEngine.trainer_send_weights(
        param_iterator,
        NCCLTrainerSendWeightsArgs(group=model_update_group, packed=packed),
    )

    remote_vllm_finish_update_weights(base_url)
    logger.info("Completed in-place vLLM weight update.")

def remote_vllm_init_weight_transfer(
    base_url,
    *,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
):
    payload = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": rank_offset,
        "world_size": world_size,
    }
    logger.info(
        "Initializing vLLM weight transfer engine with "
        f"{master_address=}, {master_port=}, {rank_offset=}, {world_size=}."
    )
    resp = _remote_json_request(
        base_url, "POST", "/init_weight_transfer", payload=payload, timeout=1800,
    )
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"rollout worker weight-transfer init failed: {resp}")
    return resp


