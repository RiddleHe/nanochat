"""Plot B: correct-decode yes/no per source layer, THREE modes as rows.
Row1 raw (plain patchscope). Row2 + whitespace scratch. Row3 + ': a historical
figure who' suffix. All hand-graded from full completions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.inspect.patchscope_few_shot import ENTITIES, ENTITY_TITLES, _style_axes

N = 36
RAW = {
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
HIST = {
    "diana":     [9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    "alexander": [6,8,10,11,22,23,24,25,26,27,28,29],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32],
    "jurassic":  [8,9,10,11],
    "nyc":       [],
}
ROWS = [("raw", "#1f77b4", RAW),
        ("+ whitespace", "#e8a87c", WS),
        ('+ \": a historical\n   figure who\"', "#5aa75a", HIST)]

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, axes = plt.subplots(3, 5, figsize=(11.5, 4.8), sharey=True, sharex=True)
for r, (label, color, D) in enumerate(ROWS):
    for c, ent in enumerate(ENTITIES):
        ax = axes[r, c]
        ys = [1 if L in D[ent] else 0 for L in range(N)]
        ax.step(range(N), ys, where="mid", color=color, linewidth=1.6)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
        ax.set_xticks([0, 12, 24, 35]); ax.tick_params(labelsize=8)
        _style_axes(ax)
        if r == 0:
            ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
        if r == 2:
            ax.set_xlabel("source layer", fontsize=9)
        if c == 0:
            ax.set_ylabel(label, fontsize=8.5)
fig.suptitle("Correct decode by source layer — three prompt modes (target L6)",
             fontsize=10, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/patchscopes/three_modes_decode"
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".png")
