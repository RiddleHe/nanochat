"""Plot val_bpb vs FLOPs for the attn-only / mlp-only pre_norm ablation.

Reads /local-ssd/mh3897/arch_d12/val_loss_curves.csv and overlays the relevant
cohort:
  - gpt_base                                 (no x0 injection)
  - gpt_base_add_init_pre_norm_learn         (attn ✓ + mlp ✓, both at pre-norm input)
  - gpt_base_add_init_pre_norm_attn_only     (attn ✓ + mlp ✗)        ← new
  - gpt_base_add_init_pre_norm_mlp_only      (attn ✗ + mlp ✓)        ← new
  - gpt_base_add_init_qkv_learn              (best so far)
  - gpt_base_add_init_res_learn              (residual-stream injection — control)

Each curve is val_bpb vs FLOPs (log-x). A horizontal annotation of each
model's minimum bpb is placed next to its final point.
"""
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/local-ssd/mh3897/arch_d12/val_loss_curves.csv"
OUT_DIR = "/local-ssd/mh3897/arch_d12/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# (model_type, label, color, linestyle)
COHORT = [
    ("gpt_base",                                  "baseline (no x0)",          "#888888", "-"),
    ("gpt_base_add_init_pre_norm_mlp_only_learn", "pre_norm mlp_only (NEW)",   "#dd8452", "-"),
    ("gpt_base_add_init_pre_norm_attn_only_learn","pre_norm attn_only (NEW)",  "#55a868", "-"),
    ("gpt_base_add_init_pre_norm_learn",          "pre_norm full (attn+mlp)",  "#4c72b0", "-"),
    ("gpt_base_add_init_res_learn",               "add_init_res_learn",        "#937860", "--"),
    ("gpt_base_add_init_qkv_learn",               "add_init_qkv_learn (best)", "#c44e52", "-"),
]

# Load curves
data = defaultdict(list)
with open(CSV) as f:
    next(f)
    for line in f:
        m, step, flops, bpb = line.strip().split(",")
        data[m].append((int(step), float(flops), float(bpb)))
for m in data:
    data[m].sort(key=lambda r: r[0])

fig, ax = plt.subplots(figsize=(11, 6.5))
for model_type, label, color, ls in COHORT:
    rows = data.get(model_type, [])
    if not rows:
        print(f"WARN: no rows for {model_type}")
        continue
    flops = [r[1] for r in rows]
    bpb   = [r[2] for r in rows]
    ax.plot(flops, bpb, ls, color=color, lw=2.0, label=label, marker="o", markersize=3)
    final = bpb[-1]
    minv  = min(bpb)
    ax.annotate(f"min={minv:.4f}", xy=(flops[-1], final),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=color, va="center", fontweight="bold")

ax.set_xscale("log")
ax.set_xlabel("training FLOPs (log scale)")
ax.set_ylabel("validation bpb")
ax.set_title("x0 injection ablation (depth=12, 1.5e18 FLOPs):\n"
             "where in the block does x0 want to be added?",
             fontsize=12, fontweight="bold")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

# Zoomed inset — only show the late-training portion where curves separate
inset = fig.add_axes([0.45, 0.45, 0.32, 0.30])
for model_type, label, color, ls in COHORT:
    rows = data.get(model_type, [])
    if not rows:
        continue
    flops = [r[1] for r in rows]
    bpb   = [r[2] for r in rows]
    cutoff = flops[len(flops) // 2]
    fl = [f for f in flops if f >= cutoff]
    bp = [b for f, b in zip(flops, bpb) if f >= cutoff]
    inset.plot(fl, bp, ls, color=color, lw=1.6, marker="o", markersize=2.5)
inset.set_xscale("log")
inset.set_title("late-training zoom", fontsize=9)
inset.tick_params(labelsize=8)
inset.grid(True, alpha=0.3)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "pre_norm_attn_vs_mlp.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_path}")

# Print final-step val_bpb table for the cohort
print("\nFinal val_bpb per model:")
for model_type, label, _, _ in COHORT:
    rows = data.get(model_type, [])
    if rows:
        minv = min(r[2] for r in rows)
        finv = rows[-1][2]
        print(f"  {label:36s}  min={minv:.4f}  final={finv:.4f}")
