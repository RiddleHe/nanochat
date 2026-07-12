"""Multi-model patchscopes decode grid.

Plots an N-model x 5-entity binary decode grid from saved patchscope run files
(as produced by patchscope_few_shot.py). Models are passed on the CLI, so this
is model-agnostic; it just reads the `{slug}__tgt{T}__{set}__{entity}.txt`
outputs and grades them with the shared CRITERIA.

Usage:
  python -m scripts.inspect.plot_d24_100b_patchscope \
      --models "arch_d24_gpt_base_100B:GPT-base d24 (100B)" \
               "EleutherAI_pythia-12b:Pythia-12B" \
      --target 6 --source-set canonical
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    ENTITIES, ENTITY_TITLES, SOURCE_SETS, CRITERIA, PALETTE,
    _grade, _load_run_file, _style_axes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more 'slug:label' pairs (the run-file model slug "
                         "and its display label).")
    ap.add_argument("--target", type=int, default=6)
    ap.add_argument("--source-set", default="canonical", choices=list(SOURCE_SETS))
    ap.add_argument("--out-dir", default="./results/patchscopes")
    ap.add_argument("--out-name", default="patchscopes_grid")
    args = ap.parse_args()
    models = [(m.split(":", 1)[0], m.split(":", 1)[1] if ":" in m else m)
              for m in args.models]
    TGT, SET = args.target, args.source_set

    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    n_rows, n_cols = len(models), len(ENTITIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.0, 1.6 + 1.4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    totals = {}
    for r, (slug, label) in enumerate(models):
        color = PALETTE[r % len(PALETTE)]
        hits = 0
        for c, ent in enumerate(ENTITIES):
            ax = axes[r, c]
            fname = f"{slug}__tgt{TGT}__{SET}__{ent}.txt"
            rows = _load_run_file(os.path.join(args.out_dir, fname))
            xs = [x for x, _ in rows]
            ys = [_grade(t, CRITERIA[ent]) for _, t in rows]
            hits += sum(ys)
            ax.plot(xs, ys, color=color, linewidth=1.2, drawstyle="steps-mid", zorder=3)
            hit_xs = [x for x, y in zip(xs, ys) if y == 1]
            ax.scatter(hit_xs, [1] * len(hit_xs), color=color, s=8, zorder=4)
            ax.set_ylim(-0.15, 1.15)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"], fontsize=8)
            ax.set_xlim(-1, max(xs) + 1)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.tick_params(axis="x", labelsize=8)
            _style_axes(ax)
            if r == 0:
                src = SOURCE_SETS[SET][ent]
                ax.set_title(f"{ENTITY_TITLES[ent]}\n({src!r})", fontsize=9, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("source layer", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
        totals[label] = hits

    fig.suptitle(f"Patchscopes few-shot — fixed target L{TGT}, {SET} source set",
                 fontsize=10, y=0.98)
    plt.subplots_adjust(left=0.12, right=0.99, top=0.78, bottom=0.16,
                        wspace=0.45, hspace=0.55)
    out_base = os.path.join(args.out_dir, args.out_name)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=200)
    plt.close(fig)
    for label, h in totals.items():
        print(f"  {label:32s} layers-decoded (sum over 5 entities): {h}")
    print(f"wrote {out_base}.png / .pdf")


if __name__ == "__main__":
    main()
