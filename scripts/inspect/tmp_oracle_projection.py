"""Simple oracle-projection plot: for each of the 5 canonical phrases, capture
the last-token residual at every layer, then plot cosine(h_L, h_L*) across
source layers L, where L* is a single fixed projection layer (the layer we
found where all five decode). One vertical line marks L*."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    _load_hf, SOURCE_SETS, ENTITIES, ENTITY_TITLES, capture_source_hiddens,
    _style_axes,
)

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
ORACLE_L = 19          # single layer to project onto (all-five-decode layer)
CANON = SOURCE_SETS["canonical"]
COLORS = {"diana": "#1f77b4", "alexander": "#e8a87c", "ali": "#2c8a8a",
          "jurassic": "#5aa75a", "nyc": "#9467bd"}
OUT = "results/patchscopes/oracle_projection"

adapter = _load_hf(HF, torch.device(DEV))
N = adapter["n_layer"]

resid = {}
for ent in ENTITIES:
    ids = adapter["encode"](CANON[ent])
    if len(ids) < 2:
        ids = [ids[0]] + ids
    H = capture_source_hiddens(adapter, ids, len(ids) - 1)
    resid[ent] = torch.stack([H[L] for L in range(N)]).float()

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, ax = plt.subplots(figsize=(7.0, 4.0))
xs = np.arange(N)
for ent in ENTITIES:
    H = resid[ent]
    v = H[ORACLE_L]
    cos = ((H / (H.norm(dim=-1, keepdim=True) + 1e-8))
           @ (v / (v.norm() + 1e-8))).cpu().numpy()
    ax.plot(xs, cos, color=COLORS[ent], linewidth=1.3, marker="o",
            markersize=3, label=ENTITY_TITLES[ent])

ax.axvline(ORACLE_L, color="#888888", linestyle="--", linewidth=1.0)
ax.set_xlabel("source layer", fontsize=10)
ax.set_ylabel(rf"cosine( $h_L$ , $h_{{{ORACLE_L}}}$ )", fontsize=10)
ax.set_xlim(-1, N)
ax.set_xticks([0, 6, 12, ORACLE_L, 24, 30, N - 1])
ax.tick_params(labelsize=8)
_style_axes(ax)
ax.legend(fontsize=8, loc="lower center", ncol=2, frameon=False)
plt.tight_layout()
fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
print(f"wrote {OUT}.png  (projected onto layer {ORACLE_L})")
