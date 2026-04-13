"""Plot LM loss and balance loss (dual y-axis).

Usage:
    python scripts/visualizations/visualize_balanced_losses.py --log a.log --label "Model A"
    python scripts/visualizations/visualize_balanced_losses.py --log a.log --label "Model A" --output out.png
"""
import re
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, required=True, help="path to log file")
parser.add_argument("--label", type=str, required=True, help="label for the run")
parser.add_argument("--output", type=str, default="balanced_losses.png", help="output file path")
args = parser.parse_args()

steps, lm_losses, bal_losses = [], [], []
with open(args.log) as f:
    for line in f:
        if line.startswith("step "):
            step_m = re.search(r'step (\d+)/', line)
            lm_m = re.search(r'\| lm: ([\d.]+)', line)
            bal_m = re.search(r'\| bal: ([-\d.]+)', line)
            if step_m and lm_m and bal_m:
                steps.append(int(step_m.group(1)))
                lm_losses.append(float(lm_m.group(1)))
                bal_losses.append(float(bal_m.group(1)))

fig, ax1 = plt.subplots(figsize=(12, 6))

color_lm = "#2196F3"
color_bal = "#E91E63"

ax1.plot(steps, lm_losses, color=color_lm, linewidth=1.5, alpha=0.8, label="LM loss")
ax1.set_xlabel("Training Step", fontsize=13)
ax1.set_ylabel("LM Loss", fontsize=13, color=color_lm)
ax1.tick_params(axis='y', labelcolor=color_lm)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(steps, [-b for b in bal_losses], color=color_bal, linewidth=1.5, alpha=0.8, label="Layer attn entropy")
ax2.set_ylabel("Layer Attention Entropy", fontsize=13, color=color_bal)
ax2.tick_params(axis='y', labelcolor=color_bal)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')

plt.title(f"{args.label}: LM Loss vs Layer Attention Entropy", fontsize=14)
plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
