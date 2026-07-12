"""Hand-graded plain vs whitespace-scratch decode, per phrase (I read every full
completion in ws_scratch_output.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.inspect.patchscope_few_shot import ENTITIES, ENTITY_TITLES, _style_axes

N = 36
PLAIN = {
    "diana":     [8,10,13,17,18,19,20,21,22,24],
    "alexander": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    "jurassic":  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "nyc":       [5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29],
}
WS = {
    "diana":     [16,17,18,19,20,24,25,26,27],
    "alexander": [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    "jurassic":  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    "nyc":       [7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
}

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.4), sharey=True)
for ax, ent in zip(axes, ENTITIES):
    p = [1 if L in PLAIN[ent] else 0 for L in range(N)]
    w = [1 if L in WS[ent] else 0 for L in range(N)]
    ax.step(range(N), p, where="mid", color="#1f77b4", linewidth=1.5)
    ax.step(range(N), w, where="mid", color="#e8a87c", linewidth=1.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
    ax.set_xticks([0, 12, 24, 35]); ax.tick_params(labelsize=8)
    ax.set_xlabel("source layer", fontsize=9)
    ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
    _style_axes(ax)
handles = [Line2D([0], [0], color="#1f77b4", lw=1.5, label="plain patchscope"),
           Line2D([0], [0], color="#e8a87c", lw=1.5, label="+ whitespace scratch")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = "results/patchscopes/ws_scratch_decode"
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".png")
