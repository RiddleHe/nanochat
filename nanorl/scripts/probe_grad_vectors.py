"""Dump per-token ∇log π vectors on one target row, for Exp 1 (compression)
and Exp 2 (correct-vs-incorrect direction) analyses.

For the supplied model:
  1. Sample N prompts × K rollouts (temp=1).
  2. Compute per-response-token entropy; bucket response tokens by this model's
     own entropy into 5 quantile bins (edges fit globally over all sampled
     response tokens).
  3. Per rollout, pick up to --pos-per-bin-per-rollout positions from each bin
     (so every bucket is represented per rollout for Exp 1; bucket 4 ∈ these
     covers Exp 2's top-20% tokens).
  4. For each selected position, compute ∇log π_t with respect to ONE target
     row (default: model.layers.L.mlp.down_proj.weight row 0, dim = intermediate).
  5. Save float32 numpy array (N_positions, row_dim) + per-position meta jsonl.

Run twice (once per model), then analyze with the two analyze_exp*.py scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanorl.data import build_rl_dataset, verify_math


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--param-suffix", type=str, default="mlp.down_proj.weight",
                   help="e.g. mlp.down_proj.weight (row_dim=intermediate) or "
                        "self_attn.k_proj.weight (row_dim=hidden)")
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--pos-per-bin-per-rollout", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def bucket_of(edges: list[float], value: float) -> int:
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if lo <= value <= hi:
                return i
        elif lo <= value < hi:
            return i
    return len(edges) - 2


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[probe] loading tokenizer + model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    target_name = f"model.layers.{args.layer}.{args.param_suffix}"
    named = dict(model.named_parameters())
    if target_name not in named:
        raise KeyError(f"param not found: {target_name}")
    target_param = named[target_name]
    # For weight tensors, each row is one output direction; shape[1] is the row dim.
    row_dim = int(target_param.shape[1])
    print(f"[probe] target = {target_name}  shape={tuple(target_param.shape)}  row_idx={args.row_idx}  row_dim={row_dim}")

    dataset = build_rl_dataset()
    rng = random.Random(args.seed)
    prompt_idxs = rng.sample(range(len(dataset)), min(args.num_prompts, len(dataset)))
    examples = [(i, dataset[i]) for i in prompt_idxs]

    # =========================================================================
    # Pass 1: sample rollouts + per-token entropies + correctness
    # =========================================================================
    rollouts = []
    for ex_i, (prompt_idx, example) in enumerate(examples):
        enc = tokenizer(example.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
                use_cache=True,
            )
        prompt_ids = input_ids[0].tolist()
        plen = len(prompt_ids)
        n_correct = 0
        for s_idx, seq in enumerate(out):
            resp = seq.tolist()[plen:]
            if not resp:
                continue
            if eos_id in resp:
                first = resp.index(eos_id)
                resp = resp[:first + 1]
            full = prompt_ids + resp
            ids = torch.tensor(full, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_ids=ids).logits[:, :-1, :].float()
                lp = logits.log_softmax(dim=-1)
                ents = -(lp.exp() * lp).sum(dim=-1)[0]
            start = plen - 1
            resp_ents = ents[start:].tolist()
            resp_text = tokenizer.decode(resp, skip_special_tokens=False)
            reward, _ = verify_math(example, resp_text, step=0)
            correct = bool(reward == 1.0)
            n_correct += int(correct)
            rollouts.append({
                "prompt_idx": prompt_idx,
                "sample_idx": s_idx,
                "prompt_ids": prompt_ids,
                "response_ids": resp,
                "entropies": resp_ents,
                "correct": correct,
            })
            del logits, lp, ents
        torch.cuda.empty_cache()
        print(f"[probe] prompt {ex_i+1}/{len(examples)} idx={prompt_idx} "
              f"correct={n_correct}/{args.num_samples}  total_rollouts={len(rollouts)}")

    # =========================================================================
    # Global entropy-quantile edges over the sampled response tokens.
    # =========================================================================
    all_ents = [e for r in rollouts for e in r["entropies"]]
    q = torch.linspace(0.0, 1.0, args.bins + 1)
    edges = torch.quantile(torch.tensor(all_ents, dtype=torch.float32), q).tolist()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    print(f"[probe] entropy edges (quantile): {[f'{e:.3f}' for e in edges]}")

    # =========================================================================
    # Select positions: per rollout, up to --pos-per-bin-per-rollout per bucket.
    # =========================================================================
    selected = []
    for r_idx, r in enumerate(rollouts):
        by_bin: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for pos, e in enumerate(r["entropies"]):
            by_bin[bucket_of(edges, e)].append((pos, e))
        pr = random.Random(args.seed + r_idx)
        for b, items in by_bin.items():
            pr.shuffle(items)
            for (pos, e) in items[:args.pos_per_bin_per_rollout]:
                selected.append((r_idx, pos, b, float(e)))
    print(f"[probe] selected {len(selected)} token positions total")

    # =========================================================================
    # Pass 2: compute ∇log π_t w.r.t. ONE target row, per selected position.
    # =========================================================================
    grads = np.zeros((len(selected), row_dim), dtype=np.float32)

    by_rollout: dict[int, list] = defaultdict(list)
    for out_idx, (r_idx, pos, b, e) in enumerate(selected):
        by_rollout[r_idx].append((out_idx, pos, b, e))

    meta_path = Path(args.output_dir) / "meta.jsonl"
    meta_f = open(meta_path, "w", encoding="utf-8")

    for r_step, (r_idx, positions) in enumerate(by_rollout.items()):
        r = rollouts[r_idx]
        full_ids = r["prompt_ids"] + r["response_ids"]
        ids = torch.tensor(full_ids, device=device).unsqueeze(0)
        plen = len(r["prompt_ids"])
        start = plen - 1

        logits = model(input_ids=ids).logits  # (1, L, V)
        positions = sorted(positions, key=lambda x: x[1])
        for j, (out_idx, pos, b, e) in enumerate(positions):
            t_pos = start + pos
            logit = logits[0, t_pos]
            target_token = ids[0, t_pos + 1]
            lp = logit.float().log_softmax(dim=-1)[target_token]
            g = torch.autograd.grad(
                lp, target_param,
                retain_graph=(j + 1) < len(positions),
            )[0]
            grads[out_idx] = g[args.row_idx].detach().float().cpu().numpy()
            meta_f.write(json.dumps({
                "row": out_idx,
                "prompt_idx": r["prompt_idx"],
                "sample_idx": r["sample_idx"],
                "pos": pos,
                "bin": b,
                "entropy": e,
                "correct": r["correct"],
            }) + "\n")
        del logits
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        if (r_step + 1) % 16 == 0 or (r_step + 1) == len(by_rollout):
            print(f"[probe] probed {r_step+1}/{len(by_rollout)} rollouts")

    meta_f.close()
    np.save(Path(args.output_dir) / "grads.npy", grads)

    with open(Path(args.output_dir) / "manifest.json", "w") as f:
        json.dump({
            "model_path": args.model_path,
            "target": target_name,
            "row_idx": args.row_idx,
            "row_dim": row_dim,
            "num_prompts": len(examples),
            "num_samples_per_prompt": args.num_samples,
            "num_rollouts": len(rollouts),
            "num_positions": len(selected),
            "entropy_edges": edges,
            "bins": args.bins,
            "pos_per_bin_per_rollout": args.pos_per_bin_per_rollout,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "num_correct_rollouts": sum(1 for r in rollouts if r["correct"]),
            "prompt_idxs": prompt_idxs,
        }, f, indent=2)
    print(f"[probe] wrote grads.npy shape={grads.shape}  meta.jsonl  manifest.json -> {args.output_dir}")


if __name__ == "__main__":
    main()
