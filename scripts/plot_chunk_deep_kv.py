"""
Render the chunk deep-KV results figure (research_chunk_deep_kv.md).

Three diverging-bar panels, each answering one question against its own baseline:
  1. equal FLOPs   -- does any variant beat the baseline?          (no)
  2. equal tokens  -- is the branch itself useful?                 (yes, and it is
                      the PROCESSING, not the visibility)
  3. seq length    -- does the benefit grow with more context?     (no -- the kill)

Numbers are read from the run metadata by default so the figure cannot silently
drift from the checkpoints; pass --hardcoded to render from the values recorded
in research_chunk_deep_kv.md instead (e.g. on a machine without the checkpoints).

Usage:
    python -m scripts.plot_chunk_deep_kv [--hardcoded] [-o figure.png]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# val_bpb as recorded in research_chunk_deep_kv.md (single seed each).
RECORDED = {
    "arch_d12_gpt_base_1.5e18": 0.85401,
    "arch_d12_gpt_base_chunk_deep_kv_1.5e18": 0.86757,
    "arch_d12_gpt_base_chunk_same_kv_1.5e18": 0.86928,
    "arch_d12_gpt_base_equaltoken_1.06e18": 0.86863,
    "arch_d12_gpt_base_chunk_deep_kv_v2_1.5e18": 0.85684,
    "arch_d12_gpt_base_chunk_deep_kv_v2_slim_1.5e18": 0.85524,
    "arch_d12_gpt_base_seq4096_1.5e18": 0.86086,
    "arch_d12_gpt_base_chunk_v2slim_seq4096_1.5e18": 0.86295,
}

# Validated diverging pair (blue <-> red) with a neutral gray midpoint.
WORSE, BETTER, NEUTRAL = "#e34948", "#2a78d6", "#b9b8b3"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#6f6e6a", "#e4e3df"
SURFACE = "#fcfcfb"


def load_bpb(use_recorded):
    if use_recorded:
        return dict(RECORDED)
    ckpt = os.path.join(os.environ.get("NANOCHAT_BASE_DIR", ""), "base_checkpoints")
    out = {}
    for tag, fallback in RECORDED.items():
        metas = sorted(glob.glob(os.path.join(ckpt, tag, "meta_*.json")))
        if not metas:
            print(f"  ! no checkpoint for {tag}, using recorded value {fallback}")
            out[tag] = fallback
            continue
        out[tag] = json.load(open(metas[-1]))["val_bpb"]
    return out


def draw_panel(ax, rows, zero_label, title, question):
    """rows: list of (label, sublabel, delta_bpb). Positive delta = worse."""
    deltas = [r[2] * 1000 for r in rows]  # per-mille
    lo, hi = min(0, *deltas), max(0, *deltas)
    pad = (hi - lo) * 0.30 or 0.1
    ax.set_xlim(lo - (pad if lo < 0 else pad * 0.12), hi + (pad if hi > 0 else 0))

    ys = list(range(len(rows)))[::-1]
    for y, (label, sub, d) in zip(ys, rows):
        v = d * 1000
        ax.barh(y, v, height=0.52, color=WORSE if v > 0 else BETTER, zorder=3)
        off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.014
        ax.text(v + (off if v > 0 else -off), y, f"{v:+.2f}‰",
                va="center", ha="left" if v > 0 else "right",
                fontsize=8.5, color=INK2, zorder=4)

    ax.axvline(0, color=NEUTRAL, lw=1.1, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab}\n{sub}" for lab, sub, _ in rows], fontsize=8.5, color=INK)
    for t, (_, _, d) in zip(ax.get_yticklabels(), rows):
        t.set_linespacing(1.5)

    # Offsets in POINTS, not axes fractions: the panels have different heights, so
    # an axes-fraction offset would collide on the tall panel.
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", fontsize=10.5, color=INK,
                ha="left", va="bottom", fontweight="semibold", annotation_clip=False)
    ax.annotate(question, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", fontsize=8, color=INK3,
                ha="left", va="bottom", annotation_clip=False)
    ax.set_xlabel(f"val_bpb minus {zero_label}  (‰;  right = worse)",
                  fontsize=8, color=INK3, labelpad=7)

    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK3, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardcoded", action="store_true",
                    help="use the values recorded in the writeup instead of reading checkpoints")
    ap.add_argument("-o", "--out", default="chunk_deep_kv_results.png")
    args = ap.parse_args()

    b = load_bpb(args.hardcoded)
    base = b["arch_d12_gpt_base_1.5e18"]
    base_tok = b["arch_d12_gpt_base_equaltoken_1.06e18"]
    base_4k = b["arch_d12_gpt_base_seq4096_1.5e18"]

    panels = [
        (
            [("chunk_same_kv", "v1 control · same-layer sources",
              b["arch_d12_gpt_base_chunk_same_kv_1.5e18"] - base),
             ("chunk_deep_kv", "v1 · +41% FLOPs tax",
              b["arch_d12_gpt_base_chunk_deep_kv_1.5e18"] - base),
             ("chunk_deep_kv_v2", "v2 · +8% tax",
              b["arch_d12_gpt_base_chunk_deep_kv_v2_1.5e18"] - base),
             ("chunk_deep_kv_v2_slim", "v2-slim · +4% tax",
              b["arch_d12_gpt_base_chunk_deep_kv_v2_slim_1.5e18"] - base)],
            f"baseline {base:.5f}",
            "1 · Equal FLOPs — no variant beats the baseline",
            "Costlier architectures get fewer steps. The deficit tracks the FLOPs tax, not harm.",
        ),
        (
            [("chunk_deep_kv", "reads processed content",
              b["arch_d12_gpt_base_chunk_deep_kv_1.5e18"] - base_tok),
             ("chunk_same_kv", "reads unprocessed content",
              b["arch_d12_gpt_base_chunk_same_kv_1.5e18"] - base_tok)],
            f"equal-token baseline {base_tok:.5f}",
            "2 · Equal tokens — the branch itself IS useful",
            "Predicted ordering holds: deep < baseline < same. The processing is the ingredient.",
        ),
        (
            [("seq 2048", "v2-slim vs baseline",
              b["arch_d12_gpt_base_chunk_deep_kv_v2_slim_1.5e18"] - base),
             ("seq 4096", "v2-slim vs baseline",
              b["arch_d12_gpt_base_chunk_v2slim_seq4096_1.5e18"] - base_4k)],
            "the baseline at that length",
            "3 · Longer context — the gap WIDENED (the kill)",
            "The bet was that the gap narrows with more range. It nearly doubled instead.",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2),
                             gridspec_kw={"height_ratios": [4, 2.1, 2.1], "hspace": 0.62})
    fig.patch.set_facecolor(SURFACE)
    for ax, (rows, zl, title, q) in zip(axes, panels):
        draw_panel(ax, rows, zl, title, q)

    fig.suptitle("Chunk deep-KV: can early layers use already-processed distant context?",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985, fontweight="semibold")
    fig.text(0.012, 0.955,
             "d12 · ClimbMix-400B · SSSL window · single seed per run. "
             "Horizontal scale differs per panel.",
             fontsize=8.5, color=INK3, ha="left")
    fig.text(0.012, 0.012,
             "Panel 2's effects (1.1‰ / 0.7‰) are the size at which seed noise matters and seed "
             "variance was never measured.\nThe ordering was predicted in advance, but a 3-seed repeat "
             "is needed before building on those two numbers.",
             fontsize=7.5, color=INK3, ha="left", va="bottom")

    # top leaves room for the figure title + subtitle AND panel 1's own title,
    # which sits 24pt above the axes.
    fig.subplots_adjust(left=0.27, right=0.965, top=0.865, bottom=0.09)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
