"""Per source layer: did the vanilla (plain) patchscope completion start with
the expected ':' token? yes/no, 5 panels."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.inspect.patchscope_few_shot import ENTITIES, ENTITY_TITLES, _style_axes

N = 36
d = json.load(open("results/patchscopes/ws_scratch.json"))

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.4), sharey=True)
for ax, ent in zip(axes, ENTITIES):
    rows = {r["L"]: r for r in d[ent]}
    ys = [1 if rows[L]["plain"].startswith(":") else 0 for L in range(N)]
    ax.step(range(N), ys, where="mid", color="#1f77b4", linewidth=1.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
    ax.set_xticks([0, 12, 24, 35]); ax.tick_params(labelsize=8)
    ax.set_xlabel("source layer", fontsize=9)
    ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
    _style_axes(ax)
fig.suptitle("Vanilla patchscope: first generated token = \":\"  (format followed)",
             fontsize=10, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.88])
out = "results/patchscopes/colon_followed"
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".png")
for ent in ENTITIES:
    rows = {r["L"]: r for r in d[ent]}
    yes = [L for L in range(N) if rows[L]["plain"].startswith(":")]
    print(f"{ent}: colon at {yes}")
