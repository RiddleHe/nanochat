"""Bin response tokens by softmax entropy and measure mean grad norm per bin,
restricted to 9 target matrices: (k_proj, v_proj, mlp.down_proj) at layers
L/4, L/2, 3L/4.

Loads the same model and dataset that `nanorl/runs/train.sh` uses, generates
the first RL batch (prompts_per_step prompts * num_samples rollouts), then for
every response token:
  1. computes the softmax entropy at that position (no-grad pass),
  2. bins the token into 20% entropy percentiles,
  3. computes the per-token logprob-loss grad norm for each of the 9 target
     matrices (one backward per token, `retain_graph=True` on a shared
     forward),
  4. accumulates a running (sum, count) per (bin, target).

Every non-target parameter has requires_grad=False, so autograd allocates
.grad tensors only for the 9 targets and skips their weight-grad matmuls.
Activation gradients still propagate through frozen weights. Layers before
the earliest target (L/4) have no backward graph at all — those activations
aren't retained.

Memory: ~60 MB of gradient storage (vs ~3 GB if all params were trainable),
plus the shared forward graph for `--batch-size` sequences and the model
weights. Gradient checkpointing is DISABLED on purpose — per-token backward
with `retain_graph=True` would otherwise re-run the full forward on every
call.

Compute: ~= (total response tokens) backward passes. On an H100 with batch=2
and ~4k seq len, budget ~10s-30s per sequence depending on length. For a
128-rollout batch this runs on the order of an hour; the task is sequential
(per-token backwards cannot be fused in stock autograd).

Assumes a vLLM rollout worker is already serving at --rollout-worker-url. Use
`scripts/mech_interp/run.sh` to launch a worker and this script together.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorl.rollout import (
    generate_rollouts_remote,
    prepare_batch,
    wait_for_rollout_worker,
)
from nanorl.data import build_rl_dataset, distributed_rl_loader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--rollout-worker-url", default="http://127.0.0.1:8047")
    # Must match train.sh defaults to get the "first training batch".
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-seq-len", type=int, default=12288)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    # Analysis knobs
    p.add_argument("--num-bins", type=int, default=5, help="# entropy percentile bins")
    p.add_argument("--batch-size", type=int, default=2,
                   help="# rollouts per shared forward/backward graph")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get(
                "NANOCHAT_BASE_DIR",
                os.path.join(os.path.dirname(__file__), "..", "..", ".nanochat"),
            ),
            "mech_interp",
            "entropy_grad_norm.json",
        ),
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.train()
    return model, tokenizer


def fetch_first_batch(args, tokenizer):
    wait_for_rollout_worker(args.rollout_worker_url, timeout_s=600)
    dataset = build_rl_dataset()
    loader = distributed_rl_loader(
        dataset, prompts_per_step=args.prompts_per_step,
        world_size=1, rank=0, seed=args.seed,
    )
    examples, _ = next(loader)
    prompts = [ex.prompt for ex in examples]
    rollouts = generate_rollouts_remote(
        args.rollout_worker_url, prompts,
        args.num_samples, args.max_new_tokens,
        args.temperature, args.top_k,
    )
    rewards = [0.0] * len(rollouts)  # rewards are irrelevant for this probe
    return prepare_batch(rollouts, rewards, tokenizer, args.max_seq_len, "cpu")


def trim_to_actual_len(ids, attn, resp=None):
    actual_len = int(attn.sum(dim=1).max().item())
    actual_len = max(actual_len, 1)
    if resp is None:
        return ids[:, :actual_len], attn[:, :actual_len]
    return ids[:, :actual_len], attn[:, :actual_len], resp[:, :actual_len]


def collect_entropies(model, batch, device, mb: int):
    """Pass 1: per-token softmax entropy (no grad). Returns list[list[(pos, ent)]]."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    B = input_ids.shape[0]
    entropies: list[list[tuple[int, float]]] = [[] for _ in range(B)]
    with torch.no_grad():
        for s in range(0, B, mb):
            e = min(s + mb, B)
            ids = input_ids[s:e].to(device)
            attn = attention_mask[s:e].to(device)
            resp = response_mask[s:e].to(device)
            ids, attn, resp = trim_to_actual_len(ids, attn, resp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=attn).logits
            # logits[:, t] predicts ids[:, t+1]; response at t+1 uses logit t.
            shift_logits = logits[:, :-1, :]
            shift_resp = resp[:, 1:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            ent = -(log_probs.exp() * log_probs).sum(dim=-1)  # [b, T-1]
            for bi in range(e - s):
                seq_idx = s + bi
                pos_idx = torch.nonzero(shift_resp[bi] > 0.5, as_tuple=True)[0].tolist()
                ent_row = ent[bi].detach().float().cpu()
                for pos in pos_idx:
                    entropies[seq_idx].append((pos, float(ent_row[pos])))
            del logits, shift_logits, log_probs, ent
    torch.cuda.empty_cache()
    return entropies


def compute_bin_edges(entropies, num_bins: int):
    flat = np.array([e for seq in entropies for (_, e) in seq], dtype=np.float64)
    if flat.size == 0:
        raise RuntimeError("no response tokens in batch — cannot bin")
    edges = np.quantile(flat, np.linspace(0.0, 1.0, num_bins + 1))
    # Guard against duplicate edges (e.g., many ties) — nudge to strictly increasing.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges, flat.size


def bin_index(ent: float, edges: np.ndarray) -> int:
    # edges has num_bins + 1 entries. Use interior edges for searchsorted.
    idx = int(np.searchsorted(edges[1:-1], ent, side="right"))
    return max(0, min(idx, len(edges) - 2))


def select_target_params(model) -> dict:
    """Return {name: Parameter} for k_proj, v_proj, mlp.down_proj at L/4, L/2, 3L/4.

    Works with HF Qwen2-style models (model.model.layers[i].self_attn.{k,v}_proj,
    model.model.layers[i].mlp.down_proj).
    """
    layers = model.model.layers
    L = len(layers)
    depth_fracs = [(1, 4), (2, 4), (3, 4)]
    target_layer_idx = [max(0, min(L - 1, (num * L) // den)) for num, den in depth_fracs]
    targets: dict = {}
    for li in target_layer_idx:
        layer = layers[li]
        targets[f"L{li:02d}.self_attn.k_proj"] = layer.self_attn.k_proj.weight
        targets[f"L{li:02d}.self_attn.v_proj"] = layer.self_attn.v_proj.weight
        targets[f"L{li:02d}.mlp.down_proj"] = layer.mlp.down_proj.weight
    return targets


def freeze_all_except(model, target_params: dict):
    target_ids = {id(p) for p in target_params.values()}
    for p in model.parameters():
        p.requires_grad_(id(p) in target_ids)


def per_target_grad_norms(target_params: dict) -> dict:
    out = {}
    for name, p in target_params.items():
        g = p.grad
        if g is None:
            out[name] = 0.0
        else:
            # grads are bf16 under autocast backward; upcast for stable accum.
            out[name] = float(g.detach().float().norm().item())
    return out


def per_target_row_cosine(target_params: dict) -> dict:
    """For each target, mean cos(W[i], G[i]) over output rows i.

    `nn.Linear.weight` is laid out as [out_features, in_features]; W[i, :] is
    the vector that dots with the input activation to produce output i. This
    is the "direction" being updated by G[i, :]. A mean cosine close to 0
    means updates rotate rows; close to ±1 means updates scale them.
    """
    out = {}
    for name, p in target_params.items():
        g = p.grad
        if g is None:
            out[name] = 0.0
            continue
        w = p.detach().float()
        gf = g.detach().float()
        cos = torch.nn.functional.cosine_similarity(w, gf, dim=1, eps=1e-12)
        out[name] = float(cos.mean().item())
    return out


def run(args):
    device = args.device
    torch.manual_seed(args.seed)

    print(f"[mech_interp] loading model: {args.model}")
    model, tokenizer = build_model(args.model, device)
    target_params = select_target_params(model)
    freeze_all_except(model, target_params)
    print(f"[mech_interp] {len(target_params)} target params:")
    for name, p in target_params.items():
        print(f"  {name}: shape={tuple(p.shape)}")

    print("[mech_interp] fetching first RL batch (rollouts via worker)")
    batch = fetch_first_batch(args, tokenizer)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    B, T = input_ids.shape
    print(f"[mech_interp] batch: {B} rollouts, padded seq_len={T}")

    print("[mech_interp] pass 1: entropies")
    t0 = time.time()
    entropies = collect_entropies(model, batch, device, args.batch_size)
    print(f"[mech_interp] pass 1 done in {time.time() - t0:.1f}s")

    edges, total_tokens = compute_bin_edges(entropies, args.num_bins)
    print(f"[mech_interp] total response tokens: {total_tokens}")
    print(f"[mech_interp] entropy bin edges: {edges.tolist()}")

    # Running accumulators per bin. Mean + std are derived from sum and
    # sum-of-squares via Var = E[x^2] - E[x]^2 at the end.
    #   bin_sum[b][name]        Σ grad_norm  (Frobenius)
    #   bin_sq_sum[b][name]     Σ grad_norm²
    #   bin_cos_sum[b][name]    Σ mean-row cos(W, G)
    #   bin_cos_sq_sum[b][name] Σ (mean-row cos)²
    #   bin_count[b]            token count
    target_names = list(target_params.keys())
    bin_sum = [{n: 0.0 for n in target_names} for _ in range(args.num_bins)]
    bin_sq_sum = [{n: 0.0 for n in target_names} for _ in range(args.num_bins)]
    bin_cos_sum = [{n: 0.0 for n in target_names} for _ in range(args.num_bins)]
    bin_cos_sq_sum = [{n: 0.0 for n in target_names} for _ in range(args.num_bins)]
    bin_count = [0] * args.num_bins

    print("[mech_interp] pass 2: per-token backward + running mean")
    t0 = time.time()
    processed = 0

    for s in range(0, B, args.batch_size):
        e = min(s + args.batch_size, B)
        ids = input_ids[s:e].to(device)
        attn = attention_mask[s:e].to(device)
        ids, attn = trim_to_actual_len(ids, attn)

        # Pre-compute bin indices for every (local_batch, pos) we'll visit.
        positions: list[tuple[int, int, int]] = []  # (bi, shifted_pos, bin_idx)
        for bi in range(e - s):
            seq_idx = s + bi
            for (pos, ent) in entropies[seq_idx]:
                positions.append((bi, pos, bin_index(ent, edges)))
        if not positions:
            continue

        for p in target_params.values():
            p.grad = None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :]
        labels = ids[:, 1:]
        # Per-token logprob, memory-light: avoid materializing log_softmax [b,T,V].
        gathered = shift_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = gathered - torch.logsumexp(shift_logits, dim=-1)  # [b, T-1]

        for i, (bi, pos, b_idx) in enumerate(positions):
            is_last = (i == len(positions) - 1)
            loss = -token_logprobs[bi, pos]
            loss.backward(retain_graph=not is_last)
            gn_dict = per_target_grad_norms(target_params)
            cos_dict = per_target_row_cosine(target_params)
            for n in target_names:
                gn = gn_dict[n]
                cs = cos_dict[n]
                bin_sum[b_idx][n] += gn
                bin_sq_sum[b_idx][n] += gn * gn
                bin_cos_sum[b_idx][n] += cs
                bin_cos_sq_sum[b_idx][n] += cs * cs
            bin_count[b_idx] += 1
            for p in target_params.values():
                p.grad = None
            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta_min = (total_tokens - processed) / rate / 60.0 if rate > 0 else float("inf")
                print(
                    f"  [{processed}/{total_tokens}] rate={rate:.1f} tok/s "
                    f"eta={eta_min:.1f}min bin_counts={bin_count}"
                )

        del logits, shift_logits, labels, gathered, token_logprobs
        torch.cuda.empty_cache()

    def _mean_std(sum_d, sq_sum_d, count):
        if not count:
            return {n: float("nan") for n in target_names}, {n: float("nan") for n in target_names}
        mean = {n: sum_d[n] / count for n in target_names}
        var = {n: max(0.0, sq_sum_d[n] / count - mean[n] ** 2) for n in target_names}
        std = {n: var[n] ** 0.5 for n in target_names}
        return mean, std

    means, stds = [], []
    cos_means, cos_stds = [], []
    for i in range(args.num_bins):
        m, s = _mean_std(bin_sum[i], bin_sq_sum[i], bin_count[i])
        cm, cs = _mean_std(bin_cos_sum[i], bin_cos_sq_sum[i], bin_count[i])
        means.append(m); stds.append(s)
        cos_means.append(cm); cos_stds.append(cs)

    result = {
        "model": args.model,
        "total_tokens": int(total_tokens),
        "num_bins": args.num_bins,
        "bin_edges": edges.tolist(),
        "bin_counts": bin_count,
        "target_names": target_names,
        "target_shapes": {n: list(target_params[n].shape) for n in target_names},
        "bin_grad_norm_sum": bin_sum,
        "bin_grad_norm_sq_sum": bin_sq_sum,
        "mean_grad_norm_per_bin": means,
        "std_grad_norm_per_bin": stds,
        "bin_row_cosine_sum": bin_cos_sum,
        "bin_row_cosine_sq_sum": bin_cos_sq_sum,
        "mean_row_cosine_per_bin": cos_means,
        "std_row_cosine_per_bin": cos_stds,
        "batch_size": args.batch_size,
        "prompts_per_step": args.prompts_per_step,
        "num_samples": args.num_samples,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[mech_interp] wrote {args.output}")

    def _print_table(label, data, fmt):
        print(f"[mech_interp] entropy bin -> {label}:")
        header = "  bin  range                   n     " + "  ".join(f"{n:>28s}" for n in target_names)
        print(header)
        for i in range(args.num_bins):
            lo, hi = edges[i], edges[i + 1]
            row = f"  {i:3d}  [{lo:.4f}, {hi:.4f}]  {bin_count[i]:6d}  "
            row += "  ".join(fmt.format(data[i][n]) for n in target_names)
            print(row)

    _print_table("mean per-target grad norm", means, "{:28.6f}")
    _print_table("std  per-target grad norm", stds, "{:28.6f}")
    _print_table("mean per-target row cosine(W, G)", cos_means, "{:28.6f}")
    _print_table("std  per-target row cosine(W, G)", cos_stds, "{:28.6f}")


if __name__ == "__main__":
    run(parse_args())
