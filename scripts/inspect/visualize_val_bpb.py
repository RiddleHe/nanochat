"""Plot val BPB comparison: two subplots — full training and last 10% zoomed.

Usage:
    python scripts/inspect/visualize_val_bpb.py --logs a.log b.log --labels "Model A" "Model B"
    python scripts/inspect/visualize_val_bpb.py --logs a.log b.log --labels "A" "B" --output out.png
"""
import re
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--logs", nargs="+", required=True, help="absolute paths to log files")
parser.add_argument("--labels", nargs="+", required=True, help="labels for each log file")
parser.add_argument("--output", type=str, default="comparison.png", help="output file path")
args = parser.parse_args()

assert len(args.logs) == len(args.labels), f"Got {len(args.logs)} logs but {len(args.labels)} labels"

# Color palette
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#795548"]

def extract_from_log(log_path):
    flops_per_token = None
    batch_size = None
    flops, bpbs = [], []
    with open(log_path) as f:
        for line in f:
            if "Estimated FLOPs per token:" in line:
                m = re.search(r'([\d.]+e\+?\d+)', line)
                if m:
                    flops_per_token = float(m.group(1))
            if "Total batch size" in line and batch_size is None:
                m = re.search(r'Total batch size ([\d,]+)', line)
                if m:
                    batch_size = int(m.group(1).replace(',', ''))
            if "Validation bpb:" in line:
                m = re.search(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
                if m:
                    step = int(m.group(1))
                    if step == 0:
                        continue
                    flops.append(step * flops_per_token * batch_size)
                    bpbs.append(float(m.group(2)))
    return flops, bpbs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

all_data = []
for i, (log_path, label) in enumerate(zip(args.logs, args.labels)):
    color = COLORS[i % len(COLORS)]
    flops, bpbs = extract_from_log(log_path)
    if flops:
        all_data.append((flops, bpbs, label, color))

# Left: full training
for flops, bpbs, label, color in all_data:
    ax1.plot(flops, bpbs, label=f'{label} ({bpbs[-1]:.3f})', color=color, linewidth=2)
ax1.set_xlabel("Training FLOPs", fontsize=13)
ax1.set_ylabel("Validation BPB", fontsize=13)
ax1.set_title("Full Training", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

# Right: last 10% of FLOPs
max_flops = max(f[-1] for f, _, _, _ in all_data)
zoom_start = max_flops * 0.9
for flops, bpbs, label, color in all_data:
    f_zoom = [f for f in flops if f >= zoom_start]
    b_zoom = [b for f, b in zip(flops, bpbs) if f >= zoom_start]
    if f_zoom:
        ax2.plot(f_zoom, b_zoom, label=f'{label} ({bpbs[-1]:.3f})', color=color, linewidth=2, marker='o', markersize=4)
ax2.set_xlabel("Training FLOPs", fontsize=13)
ax2.set_ylabel("Validation BPB", fontsize=13)
ax2.set_title("Last 10% (zoomed)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
