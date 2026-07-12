"""Vanilla patchscope, per source layer, two lines:
  - has ':'  (first token is the colon; format followed) — computed from completions
  - correct decode (did it identify the entity) — hand-inspected from full text
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.inspect.patchscope_few_shot import ENTITIES, ENTITY_TITLES, _style_axes

N = 36
d = json.load(open("results/patchscopes/ws_scratch.json"))

# hand-inspected correct-decode layers (read every plain completion)
DECODE = {
    "diana":     [8,10,13,17,18,19,20,21,22,24],
    "alexander": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    "jurassic":  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "nyc":       [5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29],
}

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.5), sharey=True)
for ax, ent in zip(axes, ENTITIES):
    rows = {r["L"]: r for r in d[ent]}
    colon = [1 if rows[L]["plain"].startswith(":") else 0 for L in range(N)]
    dec = [1 if L in DECODE[ent] else 0 for L in range(N)]
    ax.step(range(N), colon, where="mid", color="#c0c0c0", linewidth=3.2)
    ax.step(range(N), dec, where="mid", color="#1f77b4", linewidth=1.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
    ax.set_xticks([0, 12, 24, 35]); ax.tick_params(labelsize=8)
    ax.set_xlabel("source layer", fontsize=9)
    ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
    _style_axes(ax)
handles = [Line2D([0], [0], color="#c0c0c0", lw=3.2, label='has ":"  (format followed)'),
           Line2D([0], [0], color="#1f77b4", lw=1.5, label="correct decode")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 1.02))
plt.tight_layout(rect=[0, 0, 1, 0.9])
out = "results/patchscopes/colon_vs_decode"
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".png")
