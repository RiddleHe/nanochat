"""Does patchscope decoding depend on DIRECTION or MAGNITUDE?

For each phrase, patch two things into the target x-slot at target L6 and check
whether the completion decodes the entity:
  raw : h_L               (source-layer residual as-is; direction+magnitude vary)
  proj: (h_L . v19hat) v19hat   (unnormalized projection onto the L19 oracle
        direction; DIRECTION is identical for every L, only MAGNITUDE s_L varies)
If proj-decode still varies across layers, decoding is magnitude-driven (since
the direction is pinned to the oracle). Also reports a backbone-cosine baseline.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    _load_hf, TARGET_DEFAULT, SOURCE_SETS, ENTITIES, ENTITY_TITLES, CRITERIA,
    _grade, capture_source_hiddens, patched_generate, _style_axes,
)

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
TGT = 6
ORACLE_L = 19
MAXTOK = 28
CANON = SOURCE_SETS["canonical"]
OUT = "results/patchscopes/projection_patch_v2"


def clean(t):
    return t.lstrip().split("\n")[0]


def main():
    adapter = _load_hf(HF, torch.device(DEV))
    N = adapter["n_layer"]
    tgt_ids = adapter["encode"](TARGET_DEFAULT)
    tgt_pos = len(tgt_ids) - 1

    resid = {}
    data = {}
    for ent in ENTITIES:
        ids = adapter["encode"](CANON[ent])
        if len(ids) < 2:
            ids = [ids[0]] + ids
        H = capture_source_hiddens(adapter, ids, len(ids) - 1)
        Hf = {L: H[L].float() for L in range(N)}
        resid[ent] = Hf
        v = Hf[ORACLE_L]
        vhat = v / (v.norm() + 1e-8)
        vnorm = v.norm().item()
        rows = []
        for L in range(N):
            h = Hf[L]
            s = (h @ vhat).item()               # scalar projection (signed)
            proj = s * vhat                       # unnormalized projection vec
            raw_txt = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos, h,
                                             MAXTOK, op="replace"))
            proj_txt = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos,
                                              proj, MAXTOK, op="replace"))
            rows.append(dict(
                L=L, s=s, hnorm=h.norm().item(), vnorm=vnorm,
                cos=s / (h.norm().item() + 1e-8),
                raw_hit=_grade(raw_txt, CRITERIA[ent]),
                proj_hit=_grade(proj_txt, CRITERIA[ent]),
                raw_txt=raw_txt, proj_txt=proj_txt))
        data[ent] = dict(vnorm=vnorm, rows=rows)

    # ---- backbone-cosine baseline: cross-entity L19 alignment ----
    print("\n=== backbone baseline: cos(h_19^i, h_19^j) ===")
    ovec = {e: resid[e][ORACLE_L] for e in ENTITIES}
    offdiag = []
    for i in ENTITIES:
        for j in ENTITIES:
            if i < j:
                c = (ovec[i] @ ovec[j] / (ovec[i].norm() * ovec[j].norm())).item()
                offdiag.append(c)
    print(f"  mean cross-entity L19 cosine = {np.mean(offdiag):.3f} "
          f"(range {min(offdiag):.3f}..{max(offdiag):.3f})")
    print("  => absolute cosine above this baseline is the entity-specific part.\n")

    # ---- per-phrase summary + flips ----
    for ent in ENTITIES:
        rows = data[ent]["rows"]
        raw_layers = [r["L"] for r in rows if r["raw_hit"]]
        proj_layers = [r["L"] for r in rows if r["proj_hit"]]
        flips_on = [r["L"] for r in rows if r["proj_hit"] and not r["raw_hit"]]
        flips_off = [r["L"] for r in rows if r["raw_hit"] and not r["proj_hit"]]
        print(f"[{ent:9s}] |v19|={data[ent]['vnorm']:.1f}")
        print(f"   raw  decode layers: {raw_layers}")
        print(f"   proj decode layers: {proj_layers}")
        print(f"   proj-only (raw failed, proj works): {flips_on}")
        print(f"   raw-only  (raw works, proj failed): {flips_off}")
        # magnitude at the deep end vs oracle
        deep = [r for r in rows if r["L"] >= 28]
        print("   deep-layer proj magnitude s_L (should be >> |v19| if norm grows):")
        print("     " + ", ".join(f"L{r['L']}:{r['s']:.0f}" for r in deep))
        print()

    with open(OUT + ".json", "w") as f:
        json.dump({e: data[e]["rows"] for e in ENTITIES}, f, indent=1)

    # ---- plot: raw vs proj decode per phrase (5 panels) ----
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.4), sharey=True)
    for ax, ent in zip(axes, ENTITIES):
        rows = data[ent]["rows"]
        xs = [r["L"] for r in rows]
        ax.step(xs, [r["raw_hit"] for r in rows], where="mid",
                color="#1f77b4", linewidth=1.3, label="raw $h_L$")
        ax.step(xs, [r["proj_hit"] for r in rows], where="mid",
                color="#e8a87c", linewidth=1.3, label="proj onto L19")
        ax.axvline(ORACLE_L, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["no", "yes"], fontsize=8)
        ax.set_xticks([0, 12, 19, 35])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("source layer", fontsize=9)
        ax.set_title(ENTITY_TITLES[ent], fontsize=8.5)
        _style_axes(ax)
    axes[0].legend(fontsize=7.5, loc="center left", frameon=False)
    plt.tight_layout()
    fig.savefig(OUT + ".png", dpi=200)
    fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    main()
