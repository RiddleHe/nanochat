"""Plot A: vanilla patchscope, TWO ROWS x 5 cols.
Row 1 = has ':' (format followed, blue). Row 2 = correct decode (orange)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.inspect.patchscope_few_shot import ENTITIES, ENTITY_TITLES, _style_axes

N = 36
d = json.load(open("results/patchscopes/ws_scratch.json"))
DECODE = {
    "diana":     [8,10,13,17,18,19,20,21,22,24],
    "alexander": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    "jurassic":  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "nyc":       [5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29],
}
ROWS = [("has \":\"  (format)", "#1f77b4",
         lambda ent, L: {r["L"]: r for r in d[ent]}[L]["plain"].startswith(":")),
        ("correct decode", "#e8a87c",
         lambda ent, L: L in DECODE[ent])]

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, axes = plt.subplots(2, 5, figsize=(11.5, 3.6), sharey=True, sharex=True)
for r, (label, color, fn) in enumerate(ROWS):
    for c, ent in enumerate(ENTITIES):
        ax = axes[r, c]
        ys = [1 if fn(ent, L) else 0 for L in range(N)]
        ax.step(range(N), ys, where="mid", color=color, linewidth=1.6)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
        ax.set_xticks([0, 12, 24, 35]); ax.tick_params(labelsize=8)
        _style_axes(ax)
        if r == 0:
            ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
        if r == 1:
            ax.set_xlabel("source layer", fontsize=9)
        if c == 0:
            ax.set_ylabel(label, fontsize=9)
fig.suptitle("Vanilla patchscope", fontsize=10, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/patchscopes/colon_vs_decode_rows"
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".png")
