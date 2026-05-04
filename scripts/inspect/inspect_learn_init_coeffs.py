"""
Inspect the trained alpha/beta blend coefficients for each `_learn` variant.

For each checkpoint matching the directory pattern, this loads the model_*.pt
state dict, filters out the per-late-layer alpha_* / beta_* scalars, and
reports their values as text and/or a 2x2 plot.

Naming in the plot: alpha (main path, init 1.0) is rendered as 'current_*';
beta (x0 skip path, init 0.0) is rendered as 'init_*'.

Usage:
    # text report (mean / std / min / max per coeff name)
    uv run python scripts/inspect/inspect_learn_init_coeffs.py
    # text + per-layer values
    uv run python scripts/inspect/inspect_learn_init_coeffs.py --per-layer
    # 2x2 plot of per-layer values
    uv run python scripts/inspect/inspect_learn_init_coeffs.py --plot
    # both, custom roots / pattern / output
    uv run python scripts/inspect/inspect_learn_init_coeffs.py --plot --per-layer \
        --root /local-ssd/mh3897/base_checkpoints \
        --pattern "arch_d12_gpt_base_add_init_*_learn" \
        --output results/learn_init_coeffs_per_layer.png
"""
import argparse
import re
import statistics
from pathlib import Path

import torch


COEFF_NAMES = ("alpha_res", "beta_res",
               "alpha_q",   "beta_q",
               "alpha_k",   "beta_k",
               "alpha_v",   "beta_v")

# Color scheme for the plot (consistent across subplots).
PLOT_COLORS = {
    "alpha_res": "tab:blue",
    "alpha_v":   "tab:blue",
    "beta_res":  "tab:orange",
    "beta_v":    "tab:orange",
    "alpha_q":   "tab:green",
    "beta_q":    "tab:red",
    "alpha_k":   "tab:purple",
    "beta_k":    "tab:brown",
}

# Display names for the plot legend: alpha = "current" (main path), beta = "init" (x0 skip).
PLOT_DISPLAY_NAMES = {
    "alpha_res": "current_res", "beta_res": "init_res",
    "alpha_q":   "current_q",   "beta_q":   "init_q",
    "alpha_k":   "current_k",   "beta_k":   "init_k",
    "alpha_v":   "current_v",   "beta_v":   "init_v",
}

KEY_RE = re.compile(r"^(?:_orig_mod\.)?(?:module\.)?transformer\.h\.(\d+)\.(?:attn\.)?(\w+)$")


def find_latest_state_dict(ckpt_dir: Path) -> Path:
    paths = sorted(ckpt_dir.glob("model_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No model_*.pt in {ckpt_dir}")
    return paths[-1]


def collect_coeffs(ckpt_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Returns name -> sorted [(layer_idx, value), ...] for each coeff name found."""
    sd = torch.load(find_latest_state_dict(ckpt_dir), map_location="cpu", weights_only=True)
    out: dict[str, list[tuple[int, float]]] = {}
    for k, v in sd.items():
        m = KEY_RE.match(k)
        if not m:
            continue
        layer_idx, name = int(m.group(1)), m.group(2)
        if name not in COEFF_NAMES:
            continue
        out.setdefault(name, []).append((layer_idx, float(v.detach().to(torch.float32).item())))
    for name in out:
        out[name].sort()
    return out


def fmt_stats(values: list[float]) -> str:
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else 0.0
    return f"n={n}  mean={mean:+.4f}  std={std:.4f}  min={min(values):+.4f}  max={max(values):+.4f}"


def report_text(variant_to_coeffs: dict[str, dict], per_layer: bool) -> None:
    for variant, coeffs in variant_to_coeffs.items():
        print(f"\n=== {variant} ===")
        if not coeffs:
            print("  (no alpha_* / beta_* found)")
            continue
        for name in COEFF_NAMES:
            if name not in coeffs:
                continue
            entries = coeffs[name]
            vals = [v for _, v in entries]
            print(f"  {name:<10} {fmt_stats(vals)}")
            if per_layer:
                per = "  ".join(f"L{li}={v:+.4f}" for li, v in entries)
                print(f"             {per}")


def render_plot(variant_to_coeffs: dict[str, dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    items = list(variant_to_coeffs.items())
    if not items:
        print("nothing to plot")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    for ax, (variant, coeffs) in zip(axes, items):
        title = variant.replace("arch_d12_gpt_base_", "")
        if not coeffs:
            ax.set_title(f"{title}\n(no coeffs)")
            continue
        first_late = min(li for entries in coeffs.values() for li, _ in entries)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        for name in COEFF_NAMES:
            if name not in coeffs:
                continue
            xs = [li - first_late for li, _ in coeffs[name]]
            ys = [v for _, v in coeffs[name]]
            ax.plot(xs, ys, marker="o", markersize=8, color=PLOT_COLORS[name],
                    label=PLOT_DISPLAY_NAMES[name], linewidth=1.5,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("late-layer offset", fontsize=11)
        ax.set_ylabel("coefficient value", fontsize=11)
        ax.grid(True, alpha=0.3)
        n_late = max(li for entries in coeffs.values() for li, _ in entries) - first_late + 1
        ax.set_xticks(range(n_late))
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("Trained learn_init_coeffs (current = main path init 1.0, init = x0 skip path init 0.0)",
                 fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local-ssd/mh3897/base_checkpoints")
    ap.add_argument("--pattern", default="arch_d12_gpt_base_add_init_*_learn")
    ap.add_argument("--per-layer", action="store_true",
                    help="text report: include per-layer values")
    ap.add_argument("--plot", action="store_true",
                    help="also save a 2x2 plot of per-layer values")
    ap.add_argument("--output", default="results/learn_init_coeffs_per_layer.png",
                    help="path to write the plot (only with --plot)")
    args = ap.parse_args()

    root = Path(args.root)
    ckpts = sorted(d for d in root.glob(args.pattern) if d.is_dir())
    if not ckpts:
        print(f"No checkpoints matched {root}/{args.pattern}")
        return

    print(f"Found {len(ckpts)} checkpoints")
    variant_to_coeffs = {d.name: collect_coeffs(d) for d in ckpts}

    report_text(variant_to_coeffs, per_layer=args.per_layer)

    if args.plot:
        render_plot(variant_to_coeffs, Path(args.output))


if __name__ == "__main__":
    main()
