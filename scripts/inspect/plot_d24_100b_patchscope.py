"""Focused 2-model patchscopes grid: gpt_base d24 vs v_from_value_emb d24,
both trained on 100B tokens. Reuses grading/styling from patchscope_few_shot
so it matches the existing figures, but restricts to just these two models
(the full regenerate_plot would also pull in pythia/qwen rows)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    ENTITIES, ENTITY_TITLES, SOURCE_SETS, CRITERIA, PALETTE,
    _grade, _load_run_file, _style_axes,
)

OUT_DIR = "./results/patchscopes"
TGT, SET = 6, "canonical"

MODELS = [
    ("arch_d24_gpt_base_100B",                       "GPT-base d24 (100B)"),
    ("arch_d24_gpt_base_v_from_value_emb_learn_100B", "V-from-value-emb d24 (100B)"),
]

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

n_rows, n_cols = len(MODELS), len(ENTITIES)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.0, 1.6 + 1.4 * n_rows),
                         sharex=True, sharey=True, squeeze=False)

totals = {}
for r, (slug, label) in enumerate(MODELS):
    color = PALETTE[r % len(PALETTE)]
    hits = 0
    for c, ent in enumerate(ENTITIES):
        ax = axes[r, c]
        fname = f"{slug}__tgt{TGT}__{SET}__{ent}.txt"
        rows = _load_run_file(os.path.join(OUT_DIR, fname))
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

fig.suptitle(
    "Patchscopes few-shot — d24 @100B tokens, fixed target L6, canonical source set",
    fontsize=10, y=0.98)
plt.subplots_adjust(left=0.12, right=0.99, top=0.78, bottom=0.16,
                    wspace=0.45, hspace=0.55)

out_base = os.path.join(OUT_DIR, "patchscopes_d24_100B_canonical_grid")
fig.savefig(out_base + ".pdf")
fig.savefig(out_base + ".png", dpi=200)
plt.close(fig)
for label, h in totals.items():
    print(f"  {label:32s} layers-decoded (sum over 5 entities, /120): {h}")
print(f"wrote {out_base}.png / .pdf")
