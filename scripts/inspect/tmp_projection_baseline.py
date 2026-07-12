"""Layer-matched baseline for the L19 projection curves.

For each phrase p and layer L:
  own[p,L]   = cos(h_L^p, h_19^p)          # alignment to its OWN L19 oracle
  cross[p,L] = mean_{q!=p} cos(h_L^p, h_19^q)   # alignment to OTHER entities' L19
The gap own-cross at each layer is the entity-specific signal. This replaces the
earlier (wrong) single-number L19-vs-L19 baseline: the baseline must be compared
LAYER-BY-LAYER, since a residual's direction changes a lot across depth.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    _load_hf, SOURCE_SETS, ENTITIES, capture_source_hiddens, _style_axes,
)

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
ORACLE_L = 19
CANON = SOURCE_SETS["canonical"]
OUT = "results/patchscopes/projection_baseline"

adapter = _load_hf(HF, torch.device(DEV))
N = adapter["n_layer"]

resid = {}
for ent in ENTITIES:
    ids = adapter["encode"](CANON[ent])
    if len(ids) < 2:
        ids = [ids[0]] + ids
    H = capture_source_hiddens(adapter, ids, len(ids) - 1)
    resid[ent] = torch.stack([H[L].float() for L in range(N)])  # (N, d)

oracle = {e: resid[e][ORACLE_L] for e in ENTITIES}


def cosv(a, b):
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


own = np.zeros((len(ENTITIES), N))
cross = np.zeros((len(ENTITIES), N))
for i, p in enumerate(ENTITIES):
    for L in range(N):
        h = resid[p][L]
        own[i, L] = cosv(h, oracle[p])
        cross[i, L] = np.mean([cosv(h, oracle[q]) for q in ENTITIES if q != p])

own_m, cross_m = own.mean(0), cross.mean(0)
print("layer :  own   cross   gap")
for L in [0, 6, 10, 12, 16, 19, 24, 30, 34, 35]:
    print(f"  L{L:02d} : {own_m[L]:.3f}  {cross_m[L]:.3f}   {own_m[L]-cross_m[L]:+.3f}")

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
fig, ax = plt.subplots(figsize=(7.0, 4.0))
xs = np.arange(N)
ax.plot(xs, own_m, color="#2c8a8a", linewidth=1.3, marker="o", markersize=3,
        label="self (own L19)")
ax.plot(xs, cross_m, color="#d1495b", linewidth=1.3, marker="o", markersize=3,
        label="other (mean of other 4)")
ax.axvline(ORACLE_L, color="#bbbbbb", linestyle="--", linewidth=0.8)
ax.set_xlabel("source layer", fontsize=10)
ax.set_ylabel(r"cosine( $h_L$ , L19 )", fontsize=10)
ax.set_xlim(-1, N)
ax.set_xticks([0, 6, 12, ORACLE_L, 24, 30, N - 1])
ax.tick_params(labelsize=8)
_style_axes(ax)
ax.legend(fontsize=9, loc="upper right", frameon=False)
plt.tight_layout()
fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
print("wrote", OUT + ".png")
