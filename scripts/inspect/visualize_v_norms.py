"""Per-layer norm diagnostics for v-skip variants of gpt_base.

Produces two plots, one subplot per matched checkpoint:

1. c_v Frobenius norm per layer (`--output-cv`).
   For variants where late-layer c_v is excluded from the optimizer (currently
   only `v_from_v1`), those layers stay at pristine uniform-init magnitude and
   carry no signal — we drop those points rather than plot meaningless flat-init
   dots. Detection is config-driven (`model.config.v_from_v1`), not name-based.

2. Per-layer attention-output and MLP-output L2 norm (`--output-l2`).
   Forward hooks on each block's `attn` and `mlp` capture the output tensors;
   we report the per-token L2 norm (sqrt(sum_d x_d^2)) averaged over batch·seq.
   Input is a fixed random-token batch (`--seed`, `--batch-size`, `--seq-len`)
   so the comparison across variants is honest.

Usage (run as a module so the nanochat package resolves):
    # both _learn variants from the v_from_x0 / v_from_v1 family
    uv run python -m scripts.inspect.visualize_v_norms \\
        --pattern "arch_d12_gpt_base_v_from_*_learn"

    # arbitrary pair
    uv run python -m scripts.inspect.visualize_v_norms \\
        --pattern "arch_d12_gpt_base_v_from_v1*"

    # custom output paths
    uv run python -m scripts.inspect.visualize_v_norms \\
        --pattern "arch_d12_gpt_base_v_from_*_learn" \\
        --output-cv results/cv_frob.png --output-l2 results/out_l2.png

CUDA: launch with CUDA_VISIBLE_DEVICES=<idx> to pick a GPU; the script uses
cuda:0 within that visibility window.
"""
import argparse
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from nanochat.checkpoint_manager import build_model


def find_latest_step(ckpt_dir: Path) -> int:
    steps = [int(re.search(r"_(\d+)\.pt$", str(p)).group(1)) for p in ckpt_dir.glob("model_*.pt")]
    if not steps:
        raise FileNotFoundError(f"No model_*.pt in {ckpt_dir}")
    return max(steps)


def collect(ckpt_dir: Path, device, ids):
    """Returns dict with per-layer cv_frob, attn_l2, mlp_l2, n_layer, late_start, and a
    'dead_late_cv' flag (True when late-layer c_v is excluded from the optimizer)."""
    step = find_latest_step(ckpt_dir)
    m, _, _ = build_model(str(ckpt_dir), step=step, device=device, phase="eval")
    n_layer = len(m.transformer.h)
    late_start = (2 * n_layer) // 3
    dead_late_cv = bool(getattr(m.config, "v_from_v1", False))

    cv_frob = [float(blk.attn.c_v.weight.detach().float().norm().cpu())
               for blk in m.transformer.h]

    attn_l2 = [None] * n_layer
    mlp_l2 = [None] * n_layer
    handles = []
    for i, blk in enumerate(m.transformer.h):
        def attn_hook(mod, inp, out, idx=i):
            y = out[0] if isinstance(out, tuple) else out
            attn_l2[idx] = y.float().norm(dim=-1).mean().item()
        def mlp_hook(mod, inp, out, idx=i):
            mlp_l2[idx] = out.float().norm(dim=-1).mean().item()
        handles.append(blk.attn.register_forward_hook(attn_hook))
        handles.append(blk.mlp.register_forward_hook(mlp_hook))
    with torch.no_grad():
        m(ids)
    for h in handles:
        h.remove()

    return dict(
        cv_frob=cv_frob, attn_l2=attn_l2, mlp_l2=mlp_l2,
        n_layer=n_layer, late_start=late_start, dead_late_cv=dead_late_cv,
    )


def plot_cv_frobenius(results: dict, output: Path) -> None:
    items = list(results.items())
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (label, r) in zip(axes, items):
        xs = list(range(r["n_layer"]))
        ys = r["cv_frob"]
        # Drop pristine-init late layers when they're unoptimized.
        if r["dead_late_cv"]:
            xs = xs[: r["late_start"]]
            ys = ys[: r["late_start"]]
        ax.plot(xs, ys, marker='o', linewidth=2, markersize=7, color="tab:cyan")
        ax.axvspan(r["late_start"] - 0.5, r["n_layer"] - 0.5, alpha=0.08, color='blue')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Layer index", fontsize=11)
        ax.set_xticks(range(r["n_layer"]))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("||c_v.weight||_F", fontsize=11)
    fig.suptitle("c_v Frobenius norm per layer  (shaded = late layers)", fontsize=14)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches='tight')
    print(f"Saved {output}")
    plt.close(fig)


def plot_output_l2(results: dict, output: Path, batch_info: str) -> None:
    items = list(results.items())
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (label, r) in zip(axes, items):
        layers = list(range(r["n_layer"]))
        ax.plot(layers, r["attn_l2"], marker='o', linewidth=2, markersize=7,
                color="tab:blue",   label="attn_out L2")
        ax.plot(layers, r["mlp_l2"],  marker='s', linewidth=2, markersize=7,
                color="tab:orange", label="mlp_out L2")
        ax.axvspan(r["late_start"] - 0.5, r["n_layer"] - 0.5, alpha=0.08, color='blue')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Layer index", fontsize=11)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
    axes[0].set_ylabel("Per-token L2 norm (avg over batch·seq)", fontsize=11)
    fig.suptitle(f"Per-layer output L2 norm  ({batch_info})", fontsize=14)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches='tight')
    print(f"Saved {output}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local-ssd/mh3897/base_checkpoints")
    ap.add_argument("--pattern", default="arch_d12_gpt_base_v_from_*_learn",
                    help="glob pattern under --root to match checkpoint directories")
    ap.add_argument("--label-strip", default="arch_d12_gpt_base_",
                    help="prefix stripped from each checkpoint dir name to form the subplot label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--output-cv", default="results/v_compare_cv_frobenius.png")
    ap.add_argument("--output-l2", default="results/v_compare_output_l2.png")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    batch_info = f"fixed random-token batch, seed={args.seed}, {args.batch_size}×{args.seq_len}"

    root = Path(args.root)
    ckpts = sorted(d for d in root.glob(args.pattern) if d.is_dir())
    if not ckpts:
        raise SystemExit(f"No checkpoints matched {root}/{args.pattern}")
    print(f"Found {len(ckpts)} checkpoints")

    results = {}
    for d in ckpts:
        label = d.name.removeprefix(args.label_strip) or d.name
        print(f"  loading {d.name} ...")
        results[label] = collect(d, device, ids)

    plot_cv_frobenius(results, Path(args.output_cv))
    plot_output_l2(results, Path(args.output_l2), batch_info)

    # Text summary
    layers = list(range(next(iter(results.values()))["n_layer"]))
    print("\n=== c_v ||·||_F per layer ===")
    print("layer", *[f"{i:>7}" for i in layers])
    for label, r in results.items():
        vals = "  ".join(f"{v:7.2f}" for v in r["cv_frob"])
        print(f"{label:30s}  {vals}")
    print("\n=== attn_out per-token L2 (avg over batch·seq) ===")
    for label, r in results.items():
        vals = "  ".join(f"{v:7.2f}" for v in r["attn_l2"])
        print(f"{label:30s}  {vals}")
    print("\n=== mlp_out per-token L2 (avg over batch·seq) ===")
    for label, r in results.items():
        vals = "  ".join(f"{v:7.2f}" for v in r["mlp_l2"])
        print(f"{label:30s}  {vals}")


if __name__ == "__main__":
    main()
