"""TEMP experiment: does extracting the entity's last token from WITHIN the
few-shot prompt (so its residual already attended to the exemplars) make the
patchscope decode, unlike the bare-entity source?

Two conditions, same model, same target prompt, same fixed target layer L6:
  bare : source = "Diana, princess of Wales"                      (standard)
  fs   : source = PREFIX + "Diana, princess of Wales"            (few-shot ctx)
For each, sweep source layer 0..N-1, capture residual at the entity's LAST
token (" Wales"/" Great"/" Ali"/" Park"/" City"), REPLACE the target x-slot
residual at L6, generate. Grade with the shared CRITERIA and plot a 2-row grid.
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    _load_hf, TARGET_DEFAULT, ENTITIES, ENTITY_TITLES, SOURCE_SETS, CRITERIA,
    PALETTE, _style_axes, _grade, run_one_source,
)

PREFIX = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer "
    "electronics corporation, "
)
CANON = SOURCE_SETS["canonical"]
TARGET_LAYER = 6
MAX_TOKENS = 20
HF_MODEL = "Qwen/Qwen3-8B-Base"
OUT_DIR = "./results/patchscopes"


def run_condition(adapter, source_fn):
    """source_fn(entity_phrase) -> source string. Returns {ent: [(layer, text)]}"""
    out = {}
    for ent in ENTITIES:
        source = source_fn(CANON[ent])
        out[ent] = run_one_source(adapter, source, TARGET_DEFAULT,
                                  TARGET_LAYER, MAX_TOKENS, inject_mode="residual")
    return out


def main():
    device = torch.device("cuda")
    adapter = _load_hf(HF_MODEL, device)
    n_layer = adapter["n_layer"]
    print(f"Model {adapter['name']}  n_layer={n_layer}  target_layer={TARGET_LAYER}")

    conditions = [
        ("bare entity source", lambda e: e),
        ("few-shot-prefixed source", lambda e: PREFIX + e),
    ]
    res = {}
    for label, fn in conditions:
        print(f"\n######## condition: {label} ########")
        res[label] = run_condition(adapter, fn)
        # console: show fs condition completions at a few layers + hit totals
        for ent in ENTITIES:
            rows = res[label][ent]
            hits = sum(_grade(t, CRITERIA[ent]) for _, t in rows)
            print(f"  {ent:9s} hits={hits:2d}/{n_layer}")

    # Print the few-shot-source completions at representative layers for eyeballing
    fs = res["few-shot-prefixed source"]
    print("\n==== few-shot-source completions (selected layers) ====")
    for ent in ENTITIES:
        print(f"\n[{ent}]  last-token source = {CANON[ent]!r}")
        rows = dict(fs[ent])
        for L in sorted(set([n_layer//4, n_layer//2, 3*n_layer//4, n_layer-1])):
            print(f"  L{L:02d}: {rows[L]}")

    # ---- plot 2-row grid ----
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    n_rows, n_cols = len(conditions), len(ENTITIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.0, 1.6 + 1.4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for r, (label, _) in enumerate(conditions):
        color = PALETTE[r % len(PALETTE)]
        for c, ent in enumerate(ENTITIES):
            ax = axes[r, c]
            rows = res[label][ent]
            xs = [L for L, _ in rows]
            ys = [_grade(t, CRITERIA[ent]) for _, t in rows]
            ax.plot(xs, ys, color=color, linewidth=1.2, drawstyle="steps-mid", zorder=3)
            ax.scatter([x for x, y in zip(xs, ys) if y], [1]*sum(ys), color=color, s=8, zorder=4)
            ax.set_ylim(-0.15, 1.15); ax.set_yticks([0, 1])
            ax.set_yticklabels(["no", "yes"], fontsize=8)
            ax.set_xlim(-1, max(xs)+1)
            ax.set_xticks([0, 12, 24, 35])
            ax.tick_params(axis="x", labelsize=8)
            _style_axes(ax)
            if r == 0:
                ax.set_title(f"{ENTITY_TITLES[ent]}\n(last tok {CANON[ent].split()[-1]!r})",
                             fontsize=9, pad=4)
            if r == n_rows-1:
                ax.set_xlabel("source layer", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"Patchscopes on {adapter['name']} — bare vs few-shot-extracted "
                 f"source, fixed target L{TARGET_LAYER}", fontsize=10, y=0.99)
    plt.subplots_adjust(left=0.13, right=0.99, top=0.80, bottom=0.16, wspace=0.45, hspace=0.55)
    out_base = os.path.join(OUT_DIR, "patchscopes_qwen8b_fewshot_source_grid")
    fig.savefig(out_base + ".pdf"); fig.savefig(out_base + ".png", dpi=200)
    plt.close(fig)
    print(f"\nwrote {out_base}.png")


if __name__ == "__main__":
    main()
