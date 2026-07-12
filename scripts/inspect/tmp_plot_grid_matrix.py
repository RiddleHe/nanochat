"""Plot the source-layer x target-layer decode matrix for the Diana bare-source
patchscope grid, using the subagent (LLM-judge) yes/no verdicts."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

N = 36
verdict = {}
for vf in sorted(glob.glob("results/patchscopes/verify_chunks/chunk_*_verdicts.jsonl")):
    for line in open(vf):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        verdict[(r["sl"], r["tl"])] = 1 if str(r["verdict"]).lower().startswith("y") else 0

M = np.zeros((N, N))  # rows = source layer, cols = target layer
for (sl, tl), v in verdict.items():
    M[sl, tl] = v
total_yes = int(M.sum())

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, ax = plt.subplots(figsize=(6.6, 5.8))

cmap = ListedColormap(["#f0f0f0", "#2c8a8a"])  # no = light grey, yes = teal
ax.imshow(M, cmap=cmap, origin="upper", vmin=0, vmax=1, aspect="equal")

# faint gridlines between cells
ax.set_xticks(np.arange(-.5, N, 1), minor=True)
ax.set_yticks(np.arange(-.5, N, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.5)
ax.tick_params(which="minor", length=0)

ticks = [0, 6, 12, 18, 24, 30, 35]
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.tick_params(labelsize=8, length=2)
ax.set_xlabel("target layer  (where the residual is injected)", fontsize=10)
ax.set_ylabel("source layer  (where the residual is extracted)", fontsize=10)

# diagonal (source == target) and the standard-completion corner
ax.plot([0, N-1], [0, N-1], color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)
ax.scatter([N-1], [N-1], s=70, facecolors="none", edgecolors="#d1495b", linewidths=1.6)
ax.annotate("S35→T35\n(=standard\ncompletion)", xy=(N-1, N-1), xytext=(N-9.5, N-6.5),
            fontsize=7.5, color="#d1495b",
            arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1))

# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#2c8a8a", label="decoded Diana (yes)"),
                   Patch(facecolor="#f0f0f0", label="no")],
          loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
          frameon=False, fontsize=8)

ax.set_title(f"Patchscopes decode matrix — Qwen3-8B, bare \"Diana, princess of Wales\"\n"
             f"source set, few-shot target.  {total_yes}/1296 cells decoded "
             f"(LLM-judged).", fontsize=9.5, pad=26)

plt.tight_layout()
out = "results/patchscopes/diana_decode_matrix"
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print("wrote", out + ".png", " total_yes=", total_yes)
