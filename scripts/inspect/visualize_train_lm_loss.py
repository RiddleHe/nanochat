"""Plot training LM loss comparison: two panels — full and last 10% zoomed.

For standard logs (gpt, attn_res), `loss:` IS the smoothed LM loss.
For balanced logs (those with `lm:` field), the raw `lm:` value is extracted
and smoothed with a debiased EMA (beta=0.9) to match base_train.py behavior.

Usage:
    python scripts/inspect/visualize_train_lm_loss.py --logs a.log b.log --labels "Model A" "Model B"
    python scripts/inspect/visualize_train_lm_loss.py --logs a.log b.log --labels "A" "B" --output out.png
"""
import re
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--logs", nargs="+", required=True, help="absolute paths to log files")
parser.add_argument("--labels", nargs="+", required=True, help="labels for each log file")
parser.add_argument("--output", type=str, default="train_lm_loss.png", help="output file path")
args = parser.parse_args()

assert len(args.logs) == len(args.labels), f"Got {len(args.logs)} logs but {len(args.labels)} labels"

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#795548"]


def has_lm_field(log_path):
    """Check if log has separate lm: field (balanced model)."""
    with open(log_path) as f:
        for line in f:
            if line.startswith("step ") and "| lm: " in line:
                return True
    return False


def extract_smoothed_loss(log_path):
    """Extract already-smoothed loss: field (standard models)."""
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            if line.startswith("step "):
                m = re.search(r'step (\d+)/\d+.*\| loss: ([\d.]+)', line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    return steps, losses


def extract_raw_lm_and_smooth(log_path, beta=0.9):
    """Extract raw lm: field and apply debiased EMA (balanced models)."""
    steps, smoothed = [], []
    ema = 0.0
    with open(log_path) as f:
        for line in f:
            if line.startswith("step "):
                step_m = re.search(r'step (\d+)/', line)
                lm_m = re.search(r'\| lm: ([\d.]+)', line)
                if step_m and lm_m:
                    step = int(step_m.group(1))
                    raw = float(lm_m.group(1))
                    ema = beta * ema + (1 - beta) * raw
                    debiased = ema / (1 - beta ** (step + 1))
                    steps.append(step)
                    smoothed.append(debiased)
    return steps, smoothed


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

all_data = []
for i, (log_path, label) in enumerate(zip(args.logs, args.labels)):
    color = COLORS[i % len(COLORS)]
    if has_lm_field(log_path):
        steps, losses = extract_raw_lm_and_smooth(log_path)
    else:
        steps, losses = extract_smoothed_loss(log_path)
    if steps:
        all_data.append((steps, losses, label, color))

# Left: full training
for steps, losses, label, color in all_data:
    ax1.plot(steps, losses, label=f'{label} ({losses[-1]:.3f})', color=color, linewidth=1.5, alpha=0.85)
ax1.set_xlabel("Training Step", fontsize=13)
ax1.set_ylabel("LM Loss (smoothed)", fontsize=13)
ax1.set_title("Full Training", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: last 10%
max_step = max(s[-1] for s, _, _, _ in all_data)
zoom_start = int(max_step * 0.9)
for steps, losses, label, color in all_data:
    s_zoom = [s for s in steps if s >= zoom_start]
    l_zoom = [l for s, l in zip(steps, losses) if s >= zoom_start]
    if s_zoom:
        ax2.plot(s_zoom, l_zoom, label=f'{label} ({losses[-1]:.3f})', color=color, linewidth=2, marker='o', markersize=3)
ax2.set_xlabel("Training Step", fontsize=13)
ax2.set_ylabel("LM Loss (smoothed)", fontsize=13)
ax2.set_title("Last 10% (zoomed)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
