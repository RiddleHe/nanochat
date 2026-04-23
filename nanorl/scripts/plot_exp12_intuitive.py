"""Intuitive headline plots for Exp 1 + Exp 2.

Reads merged exp12_base / exp12_trained and produces:
  - exp1_headline.png       (2x2: PR bars, rank95 bars, PCA base, PCA trained)
  - exp1_bin4_energy.png    (cumulative energy for bin 4 only, base vs trained)
  - exp2_headline.png       (2 panels: histogram + per-prompt sorted bars)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/hdd/mh3897/cc/nanochat/.nanochat/probe/exp12_figures"
BASE_D = "/hdd/mh3897/cc/nanochat/.nanochat/probe/exp12_base"
TRAINED_D = "/hdd/mh3897/cc/nanochat/.nanochat/probe/exp12_trained"
os.makedirs(OUT, exist_ok=True)


def load_run(d):
    g = np.load(os.path.join(d, "grads.npy"))
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    bins = np.array([m["bin"] for m in meta])
    corr = np.array([m["correct"] for m in meta])
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    return g, bins, corr, manifest


def pr_of(s):
    s2 = s.astype(np.float64) ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def rank_at(s, frac=0.95):
    s2 = s.astype(np.float64) ** 2
    return int(np.searchsorted(s2.cumsum() / s2.sum(), frac) + 1)


def bucket_svd(g, bins, n_bins):
    out = {}
    for b in range(n_bins):
        mat = g[bins == b].astype(np.float64)
        if len(mat) < 2:
            continue
        s = np.linalg.svd(mat, compute_uv=False)
        out[b] = {"n": int(len(mat)), "PR": pr_of(s), "rank95": rank_at(s), "S": s}
    return out


def pca_2d(g, seed=0, subsample_basis=2500):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(g), min(subsample_basis, len(g)), replace=False)
    sub = g[idx].astype(np.float64)
    mean = sub.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(sub - mean, full_matrices=False)
    coords = (g.astype(np.float64) - mean) @ Vt[:2].T
    return coords


def main():
    base_g, base_bins, base_corr, base_manifest = load_run(BASE_D)
    trn_g, trn_bins, trn_corr, trn_manifest = load_run(TRAINED_D)
    n_bins = base_manifest["bins"]

    base_stats = bucket_svd(base_g, base_bins, n_bins)
    trn_stats = bucket_svd(trn_g, trn_bins, n_bins)

    cmap = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, n_bins))

    # ------------------------------------------------------------------------
    # FIGURE 1: Exp 1 headline (2x2)
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(n_bins)
    w = 0.4

    # --- (a) PR bars ---
    ax = axes[0, 0]
    base_pr = [base_stats[b]["PR"] for b in range(n_bins)]
    trn_pr = [trn_stats[b]["PR"] for b in range(n_bins)]
    b1 = ax.bar(x - w / 2, base_pr, w, label="base", color="#888888", edgecolor="black")
    b2 = ax.bar(x + w / 2, trn_pr, w, label="500-step RL", color="#c44e52", edgecolor="black")
    for bars, vals in [(b1, base_pr), (b2, trn_pr)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {b}" for b in range(n_bins)])
    ax.set_xlabel("entropy percentile bucket (0=low, 4=high)")
    ax.set_ylabel("participation ratio  (effective # of directions)")
    ax.set_title("(a) Effective dimension of gradient set shrinks after RL\n"
                 "higher PR = more spread-out; lower PR = collapsed to fewer directions",
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper left")

    # --- (b) Rank95 bars ---
    ax = axes[0, 1]
    base_r = [base_stats[b]["rank95"] for b in range(n_bins)]
    trn_r = [trn_stats[b]["rank95"] for b in range(n_bins)]
    b1 = ax.bar(x - w / 2, base_r, w, label="base", color="#888888", edgecolor="black")
    b2 = ax.bar(x + w / 2, trn_r, w, label="500-step RL", color="#c44e52", edgecolor="black")
    for bars, vals in [(b1, base_r), (b2, trn_r)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 10, f"{v}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {b}" for b in range(n_bins)])
    ax.set_xlabel("entropy percentile bucket")
    ax.set_ylabel("# singular values needed for 95% energy")
    ax.set_title("(b) Rank required to explain 95% of gradient variance ↓\n"
                 "(target row dim = 8960; base uses ~560, trained uses ~270)",
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper left")

    # --- (c) PCA base ---
    base_xy = pca_2d(base_g, seed=0)
    ax = axes[1, 0]
    for b in range(n_bins):
        mask = base_bins == b
        ax.scatter(base_xy[mask, 0], base_xy[mask, 1],
                   s=5, alpha=0.35, color=cmap[b], label=f"bin {b}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(c) Base model: gradients by entropy bucket in top-2 PCs\n"
                 "(all buckets overlap — one shared low-dim structure)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, markerscale=2)

    # --- (d) PCA trained ---
    trn_xy = pca_2d(trn_g, seed=0)
    ax = axes[1, 1]
    for b in range(n_bins):
        mask = trn_bins == b
        ax.scatter(trn_xy[mask, 0], trn_xy[mask, 1],
                   s=5, alpha=0.35, color=cmap[b], label=f"bin {b}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(d) 500-step RL model: gradients by entropy bucket in top-2 PCs\n"
                 "(bin 4 = top-entropy separates from the rest)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, markerscale=2)

    fig.suptitle(
        "Exp 1: per-entropy-bucket gradient structure, base vs 500-step RL\n"
        f"target = model.layers.14.mlp.down_proj.weight row 0 (dim=8960), "
        f"{len(base_g)} token gradients per model",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "exp1_headline.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # FIGURE 2: Exp 1 bin-4 cumulative energy (base vs trained on one axes)
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, st, color in [
        ("base  bin 4 (top-20% entropy)", base_stats[4], "#888888"),
        ("500-step RL  bin 4", trn_stats[4], "#c44e52"),
    ]:
        S = st["S"]
        cum = (S ** 2).cumsum() / (S ** 2).sum()
        ax.plot(np.arange(1, len(S) + 1), cum, "-", lw=2.5, color=color,
                label=f"{lbl}   PR={st['PR']:.0f}, rank95={st['rank95']}")
    ax.axhline(0.95, ls=":", color="#555", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("# singular values included (log)")
    ax.set_ylabel("cumulative singular-value energy")
    ax.set_title(
        "Exp 1 detail: top-20%-entropy gradients need ~2× fewer directions after RL\n"
        f"({base_stats[4]['n']} base tokens vs {trn_stats[4]['n']} trained tokens)",
        fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp1_bin4_energy.png"), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------------
    # FIGURE 3: Exp 2 headline
    # ------------------------------------------------------------------------
    from collections import defaultdict
    trn_meta = [json.loads(l) for l in open(os.path.join(TRAINED_D, "meta.jsonl"))]
    top_bin = n_bins - 1
    idxs_by_pc = defaultdict(list)
    for i, m in enumerate(trn_meta):
        if m["bin"] != top_bin:
            continue
        pidx = m["prompt_idx"]
        if isinstance(pidx, list):
            pidx = tuple(pidx)
        idxs_by_pc[(pidx, m["correct"])].append(i)

    prompts = sorted({k[0] for k in idxs_by_pc})
    per_prompt = []
    for p in prompts:
        ci = idxs_by_pc.get((p, True), [])
        ii = idxs_by_pc.get((p, False), [])
        if len(ci) >= 2 and len(ii) >= 2:
            gc = trn_g[ci].mean(axis=0)
            gi = trn_g[ii].mean(axis=0)
            c = float(np.dot(gc, gi) / (np.linalg.norm(gc) * np.linalg.norm(gi) + 1e-12))
            per_prompt.append(c)
    per_prompt = np.array(per_prompt)

    control = []
    rng = np.random.RandomState(0)
    for (p, c), idxs in idxs_by_pc.items():
        if len(idxs) >= 4:
            arr = list(idxs)
            rng.shuffle(arr)
            half = len(arr) // 2
            g1 = trn_g[arr[:half]].mean(axis=0)
            g2 = trn_g[arr[half:]].mean(axis=0)
            control.append(float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-12)))
    control = np.array(control)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))
    bin_edges = np.linspace(-1, 1, 31)
    axL.hist(per_prompt, bins=bin_edges, alpha=0.7, color="#c44e52",
             edgecolor="white",
             label=f"correct rollouts vs incorrect rollouts\nsame prompt  (n={len(per_prompt)} prompts)\nmean={per_prompt.mean():+.3f}")
    axL.hist(control, bins=bin_edges, alpha=0.55, color="#4c72b0",
             edgecolor="white",
             label=f"control: random half-split within one group\n(n={len(control)})\nmean={control.mean():+.3f}")
    axL.axvline(0, color="black", ls=":", lw=1)
    axL.axvline(1, color="#2b7a2b", ls="--", lw=1.2,
                label="if correct/incorrect were identical → +1")
    axL.axvline(-1, color="#7a2b2b", ls="--", lw=1.2,
                label="if exactly opposite → −1")
    axL.set_xlim(-1.05, 1.05)
    axL.set_xlabel("cosine similarity between mean ∇log π")
    axL.set_ylabel("# prompts")
    axL.set_title(
        "(a) Direction of ∇log π at top-20%-entropy tokens:\n"
        "correct vs incorrect rollouts — essentially orthogonal",
        fontsize=11)
    axL.legend(fontsize=9, loc="upper left")
    axL.grid(True, alpha=0.3)

    order = np.argsort(per_prompt)
    sorted_cos = per_prompt[order]
    colors = ["#c44e52" if c >= 0 else "#4c72b0" for c in sorted_cos]
    axR.bar(range(len(sorted_cos)), sorted_cos, color=colors, edgecolor="black", linewidth=0.5)
    axR.axhline(0, color="black", lw=1)
    ctl_lo, ctl_hi = np.percentile(control, [2.5, 97.5])
    axR.axhspan(ctl_lo, ctl_hi, color="#4c72b0", alpha=0.18,
                label=f"95% band of control  [{ctl_lo:+.2f}, {ctl_hi:+.2f}]")
    axR.set_xlabel("prompt (sorted by cosine)")
    axR.set_ylabel("cos(g_correct, g_incorrect)")
    axR.set_ylim(-1.05, 1.05)
    axR.set_title(
        "(b) Per-prompt view: every prompt inside the control noise band\n"
        "→ no detectable signed alignment",
        fontsize=11)
    axR.legend(fontsize=9, loc="upper left")
    axR.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Exp 2: do correct and incorrect rollouts push the same direction at high-entropy tokens?\n"
        f"500-step RL model, {len(per_prompt)} prompts with both correct+incorrect rollouts",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp2_headline.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("wrote figures:")
    for fn in ["exp1_headline.png", "exp1_bin4_energy.png", "exp2_headline.png"]:
        p = os.path.join(OUT, fn)
        print(f"  {p}  ({os.path.getsize(p)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
