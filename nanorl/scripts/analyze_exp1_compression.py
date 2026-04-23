"""Exp 1: gradient compression per entropy bucket.

For each model run (base vs trained), compute:
  - per-bucket SVD on the matrix of per-token row-gradients (N × row_dim)
  - participation ratio PR = (Σσ²)² / Σσ⁴
  - cumulative singular-value energy curve
  - cross-bucket cosine: top-1 right singular vector of bucket i vs bucket j

Outputs:
  - exp1_cumulative_energy.png  (one panel per run)
  - exp1_cross_bucket_cosine.png  (one panel per run)
  - exp1_summary.json

Usage:
  python analyze_exp1_compression.py \
    base=/path/to/base_probe_out \
    trained=/path/to/trained_probe_out
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/hdd/mh3897/cc/nanochat/.nanochat/probe/exp12_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_run(run_dir):
    grads = np.load(os.path.join(run_dir, "grads.npy"))
    meta = [json.loads(l) for l in open(os.path.join(run_dir, "meta.jsonl"))]
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    bins = np.array([m["bin"] for m in meta])
    correct = np.array([m["correct"] for m in meta])
    return grads, bins, correct, manifest


def participation_ratio(s: np.ndarray) -> float:
    s2 = s.astype(np.float64) ** 2
    denom = (s2 ** 2).sum()
    return float((s2.sum() ** 2) / denom) if denom > 0 else 0.0


def per_bucket_svd(grads, bins, n_bins, max_per_bucket=None):
    out = {}
    for b in range(n_bins):
        mask = bins == b
        mat = grads[mask]
        if len(mat) < 2:
            continue
        if max_per_bucket and len(mat) > max_per_bucket:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(mat), max_per_bucket, replace=False)
            mat = mat[idx]
        mat64 = mat.astype(np.float64)
        U, S, Vt = np.linalg.svd(mat64, full_matrices=False)
        out[b] = {
            "n": int(len(mat)),
            "singular_values": S,
            "participation_ratio": participation_ratio(S),
            "top1_right_vec": Vt[0],
            "top1_energy_frac": float((S[0] ** 2) / (S ** 2).sum()) if len(S) > 0 else 0.0,
            "rank95": int(np.searchsorted((S ** 2).cumsum() / (S ** 2).sum(), 0.95) + 1),
        }
    return out


def cross_bucket_cosine(bucket_info):
    bkeys = sorted(bucket_info.keys())
    mat = np.zeros((len(bkeys), len(bkeys)))
    for i, bi in enumerate(bkeys):
        vi = bucket_info[bi]["top1_right_vec"]
        ni = np.linalg.norm(vi) + 1e-12
        for j, bj in enumerate(bkeys):
            vj = bucket_info[bj]["top1_right_vec"]
            nj = np.linalg.norm(vj) + 1e-12
            mat[i, j] = float(np.dot(vi, vj) / (ni * nj))
    return bkeys, mat


def parse_runs(argv):
    runs = []
    for arg in argv[1:]:
        if "=" not in arg:
            continue
        tag, path = arg.split("=", 1)
        runs.append((tag, path))
    return runs


def main():
    runs = parse_runs(sys.argv)
    if not runs:
        print("usage: python analyze_exp1_compression.py tag1=run_dir1 tag2=run_dir2 ...")
        sys.exit(1)

    summaries = {}
    cmap = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 5))

    # =========================================================================
    # Plot 1: cumulative singular-value energy per bucket (panel per run)
    # =========================================================================
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.8), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, (tag, path) in zip(axes, runs):
        grads, bins, correct, manifest = load_run(path)
        n_bins = manifest["bins"]
        info = per_bucket_svd(grads, bins, n_bins)
        summaries[tag] = {
            "n_positions": int(len(grads)),
            "row_dim": manifest["row_dim"],
            "target": manifest["target"],
            "per_bucket": {
                b: {
                    "n": v["n"],
                    "participation_ratio": v["participation_ratio"],
                    "top1_energy_frac": v["top1_energy_frac"],
                    "rank95": v["rank95"],
                }
                for b, v in info.items()
            },
        }
        for b in sorted(info.keys()):
            S = info[b]["singular_values"]
            cum = (S ** 2).cumsum() / (S ** 2).sum()
            ax.plot(
                np.arange(1, len(S) + 1), cum, "-",
                color=cmap[b], lw=2,
                label=f"bin {b}  PR={info[b]['participation_ratio']:.1f}  rank95={info[b]['rank95']}  n={info[b]['n']}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("rank (log)")
        ax.set_ylabel("cumulative singular-value energy")
        ax.set_title(tag)
        ax.set_ylim(0, 1.02)
        ax.axhline(0.95, ls=":", color="#888", alpha=0.7)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")

    example_manifest = json.load(open(os.path.join(runs[0][1], "manifest.json")))
    fig.suptitle(
        f"Exp 1: per-bucket gradient compression  "
        f"(target {example_manifest['target']} row {example_manifest['row_idx']}, dim={example_manifest['row_dim']})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp1_cumulative_energy.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Plot 2: cross-bucket top-1 right-singular-vector cosine heatmap (panel per run)
    # =========================================================================
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.5))
    if len(runs) == 1:
        axes = [axes]
    for ax, (tag, path) in zip(axes, runs):
        grads, bins, correct, manifest = load_run(path)
        info = per_bucket_svd(grads, bins, manifest["bins"])
        bkeys, cos_mat = cross_bucket_cosine(info)
        summaries[tag]["cross_bucket_top1_cosine"] = cos_mat.tolist()
        im = ax.imshow(cos_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(bkeys)))
        ax.set_yticks(range(len(bkeys)))
        ax.set_xticklabels([f"bin {b}" for b in bkeys])
        ax.set_yticklabels([f"bin {b}" for b in bkeys])
        for i in range(len(bkeys)):
            for j in range(len(bkeys)):
                ax.text(j, i, f"{cos_mat[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if abs(cos_mat[i, j]) > 0.5 else "black",
                        fontsize=10)
        ax.set_title(tag)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Exp 1: cosine between top-1 gradient directions across entropy buckets")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp1_cross_bucket_cosine.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "exp1_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print("Exp 1 summary:")
    for tag, s in summaries.items():
        print(f"  {tag}  (target={s['target']}, dim={s['row_dim']}, n={s['n_positions']})")
        for b, info in sorted(s["per_bucket"].items()):
            print(f"    bin {b}: n={info['n']:>4}  PR={info['participation_ratio']:7.2f}  "
                  f"top1_energy={info['top1_energy_frac']:.3f}  rank95={info['rank95']}")


if __name__ == "__main__":
    main()
