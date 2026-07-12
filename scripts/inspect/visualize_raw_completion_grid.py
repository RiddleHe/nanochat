"""Grid view of the canonical raw-completion (logit lens) experiment.

Two stacked subplots, one per model (Qwen3-8B on top, Pythia-12B on bottom).
Within each subplot:
  - rows = the 5 canonical entity prompts
  - cols = source layers in the second half of the model (L18..L35)
  - each cell text = the top-1 next-token predicted by that layer's residual,
                     with whitespace visualised (` ` → `·`, `\n` → `\\n`)
  - cell background colour = red if the few-shot patchscope (tgt L6) FAILED
                              to decode the entity at that same source layer,
                              white otherwise.

So you can see at a glance:
  - which mid/late layers' raw residuals already commit to a sensible next token
  - which of those same residuals also drive a correct few-shot entity decode
    (white cells) vs which decode the residual to something else when given the
    target prompt's few-shot context (red cells).
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts.inspect.patchscope_few_shot import (
    CRITERIA, ENTITIES, ENTITY_TITLES, MODEL_DISPLAY,
)

RESULTS = "/hdd/mh3897/nanochat/results/patchscopes"
OUT_BASE = "/hdd/mh3897/nanochat/results/patchscopes/raw_completion_canonical_grid"

# Rows top→bottom
MODELS = ["Qwen_Qwen3-8B-Base", "EleutherAI_pythia-12b"]

# Layers shown (second half of 36)
START_L, END_L = 18, 36

FAIL_COLOR = "#f4c4c4"   # soft red for "few-shot patchscope failed at this layer"
PASS_COLOR = "white"


def vis(s, max_chars=14):
    """Visible whitespace, truncate long tokens."""
    s = s.replace(" ", "·").replace("\n", "\\n").replace("\t", "\\t")
    if len(s) > max_chars:
        s = s[:max_chars - 1] + "…"
    return s


def load_top1(model_slug, ent_key):
    """logit_lens canonical: dict layer → top1 string."""
    path = os.path.join(RESULTS,
                        f"{model_slug}__logit_lens__canonical__{ent_key}.txt")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("L") or "\t" not in line:
                continue
            head, tail = line.split("\t", 1)
            parts = tail.split("\t", 2)
            top1 = parts[1] if len(parts) > 1 else ""
            out[int(head[1:])] = top1
    return out


def load_fewshot(model_slug, ent_key):
    """few-shot tgt6 canonical: dict src_layer → 20-token completion."""
    path = os.path.join(RESULTS,
                        f"{model_slug}__tgt6__canonical__{ent_key}.txt")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("S") or "\t" not in line:
                continue
            head, text = line.split("\t", 1)
            src = int(head.split("_")[0][1:])
            out[src] = text
    return out


def grade(text, crit):
    t = text.lower()
    if any(n in t for n in crit["neg"]):
        return 0
    return 1 if any(p in t for p in crit["pos"]) else 0


def main():
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"

    n_layers = END_L - START_L
    n_ents = len(ENTITIES)

    # Width: ~0.7" per column + label margin ≈ 14"
    # Height: per subplot ~ n_ents * 0.32 + title + xlabel ≈ 2.3", times 2 = ~5"
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 5.4))

    for r, model_slug in enumerate(MODELS):
        ax = axes[r]
        for row, ent in enumerate(ENTITIES):
            top1 = load_top1(model_slug, ent)
            fs = load_fewshot(model_slug, ent)
            for col, L in enumerate(range(START_L, END_L)):
                tok = top1.get(L, "")
                completion = fs.get(L, "")
                passed = grade(completion, CRITERIA[ent])
                color = PASS_COLOR if passed else FAIL_COLOR
                ax.add_patch(Rectangle((col, row), 1, 1,
                                       facecolor=color,
                                       edgecolor="#888", linewidth=0.4))
                ax.text(col + 0.5, row + 0.5, vis(tok),
                        ha="center", va="center", fontsize=6.5)

        ax.set_xlim(0, n_layers)
        ax.set_ylim(0, n_ents)
        ax.invert_yaxis()
        ax.set_xticks([i + 0.5 for i in range(n_layers)])
        ax.set_xticklabels([f"L{L}" for L in range(START_L, END_L)],
                           fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(n_ents)])
        ax.set_yticklabels([ENTITY_TITLES[e] for e in ENTITIES],
                           fontsize=8)
        ax.set_title(MODEL_DISPLAY.get(model_slug, model_slug),
                     fontsize=10, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        if r == 1:
            ax.set_xlabel("source layer", fontsize=9)

    fig.suptitle(
        "Raw-completion top-1 (canonical sources, L18–L35).  "
        "Red = few-shot patchscope at tgt L6 failed entity decode at this layer.",
        fontsize=10, y=0.99)
    plt.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.10,
                        hspace=0.45)

    for ext in ("pdf", "png"):
        path = f"{OUT_BASE}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=200)
        else:
            fig.savefig(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
