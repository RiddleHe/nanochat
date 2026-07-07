"""Step B matrix: source x target heatmaps of patchscope accuracy,
RAW vs TRANSLATED, on Qwen3-8B.

For every grid cell (source layer Ls, target layer Lt):
  - raw       : norm-matched h_Ls patched into target layer Lt, candidate-logit
                readout over the 15 entity names.
  - translated: train a linear translator A: h_Ls -> h_Lt (MSE, no entity
                labels, held-out-template folds), patch A(h_Ls) instead.
Accuracies are computed on held-out templates only (3 folds, averaged), for
both raw and translated, so the two panels are exactly comparable.

Outputs three heatmaps (raw / translated / delta) + json.

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.step_b_matrix \
      --hf-model Qwen/Qwen3-8B-Base --stride 2 --out results/step_bc
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.probe_step1_robust import ENTITIES
from scripts.inspect.step2_probe_vs_patchscope import (
    load_hf, build_source, target_positions, ref_norm_at, patch_readout)
from scripts.inspect.step_root_cause_bc import train_translator, kfold_template_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--translator-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/step_bc")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers, grid stride {args.stride}", flush=True)
    X, y = build_source(model, tok, layers, n_layer, device)
    tmpl = torch.tensor([ti for _ in ENTITIES for ti in range(12)])
    target_ids, patch_pos, read_pos = target_positions(tok)
    cand = torch.tensor([tok(" " + n, add_special_tokens=False)["input_ids"][0]
                         for n, _, _ in ENTITIES])

    grid = list(range(0, n_layer, args.stride))
    G = len(grid)
    raw = torch.zeros(G, G)
    trans = torch.zeros(G, G)
    ref = {T: ref_norm_at(model, layers, target_ids, T, patch_pos, device) for T in grid}

    def ps(H, T):
        Hn = H / H.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref[T]
        pred = patch_readout(model, layers, target_ids, T, patch_pos, read_pos,
                             Hn, cand, device)
        return pred

    folds = list(kfold_template_mask(tmpl))
    for ti, T in enumerate(grid):
        for si, Ls in enumerate(grid):
            r_acc, t_acc = [], []
            for trm, tem in folds:
                yte = y[tem]
                r_acc.append(float((ps(X[tem, Ls, :], T) == yte).float().mean()))
                A = train_translator(X[trm, Ls, :], X[trm, T, :], device,
                                     steps=args.translator_steps)
                with torch.no_grad():
                    Ht = A(X[tem, Ls, :].to(device)).cpu()
                t_acc.append(float((ps(Ht, T) == yte).float().mean()))
            raw[si, ti] = sum(r_acc) / len(r_acc)
            trans[si, ti] = sum(t_acc) / len(t_acc)
        print(f"  target L{T:02d} done: raw col-max {raw[:, ti].max():.2f}  "
              f"translated col-max {trans[:, ti].max():.2f}", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer, "grid": grid,
           "chance": 1.0 / len(ENTITIES),
           "raw": raw.tolist(), "translated": trans.tolist()}
    with open(os.path.join(args.out, "step_b_matrix.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    ticks = list(range(0, G, max(1, 6 // args.stride)))
    ticklabels = [str(grid[i]) for i in ticks]
    vmax = float(max(raw.max(), trans.max()))
    for ax, M, title, cmap, vmin_, vmax_ in [
            (axes[0], raw, "RAW patchscope", "viridis", res["chance"], vmax),
            (axes[1], trans, "TRANSLATED (linear A) patchscope", "viridis", res["chance"], vmax),
            (axes[2], trans - raw, "improvement (translated - raw)", "RdBu_r", -vmax, vmax)]:
        im = ax.imshow(M.numpy(), origin="lower", aspect="auto", cmap=cmap,
                       vmin=vmin_, vmax=vmax_)
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks); ax.set_yticklabels(ticklabels)
        ax.set_xlabel("target layer (patch INTO)")
        ax.set_ylabel("source layer (hidden state FROM)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{args.hf_model}: patchscope accuracy, raw vs linearly-translated "
                 f"(held-out templates, chance {res['chance']:.2f})")
    fig.tight_layout()
    p = os.path.join(args.out, "step_b_matrix.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
