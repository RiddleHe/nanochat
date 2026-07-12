"""Hand-graded (by me, reading every completion) raw vs projection decode,
replacing the unreliable regex. 28-token completions, target L6."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.inspect.patchscope_few_shot import ENTITIES, ENTITY_TITLES, _style_axes

N = 36
# decode layers judged by hand from projection_patch_v2.json completions
PROJ = {
    "diana":     [15,16,17,18,19,20,21,22,23,24,25,26,35],
    "alexander": [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,35],
    "ali":       [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    "jurassic":  [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    "nyc":       [5,6,7,8,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,35],
}
RAW = {
    "diana":     [8,10,13,17,18,19,20,21,22,24],
    "alexander": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    "jurassic":  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "nyc":       [5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29],
}

# all-five intersections
proj_common = sorted(set.intersection(*[set(PROJ[e]) for e in ENTITIES]))
raw_common = sorted(set.intersection(*[set(RAW[e]) for e in ENTITIES]))
print("proj all-5 decode:", proj_common)
print("raw  all-5 decode:", raw_common)
print("L19 in both:", 19 in proj_common and 19 in raw_common)

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.4), sharey=True)
for ax, ent in zip(axes, ENTITIES):
    raw = [1 if L in RAW[ent] else 0 for L in range(N)]
    prj = [1 if L in PROJ[ent] else 0 for L in range(N)]
    ax.step(range(N), raw, where="mid", color="#1f77b4", linewidth=1.5)
    ax.step(range(N), prj, where="mid", color="#e8a87c", linewidth=1.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
    ax.set_xticks([0, 12, 24, 35]); ax.tick_params(labelsize=8)
    ax.set_xlabel("source layer", fontsize=9)
    ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
    _style_axes(ax)
handles = [Line2D([0], [0], color="#1f77b4", lw=1.5, label="raw $h_L$  (full activation)"),
           Line2D([0], [0], color="#e8a87c", lw=1.5, label="projection onto L19  (component along L19)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = "results/patchscopes/projection_patch_handgraded"
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".png")
