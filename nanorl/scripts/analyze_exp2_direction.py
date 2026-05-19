"""Exp 2: correct vs incorrect rollout gradient direction at top-20% entropy tokens.

For each prompt that has BOTH correct and incorrect rollouts in the sampled set:
  - collect ∇log π at bin-4 (top-20% entropy) positions from its correct rollouts  -> mean  g_c
  - same from its incorrect rollouts                                                -> mean  g_i
  - cos(g_c, g_i)

Control: same-correctness half-split cosine (split one group's tokens in half;
cosine of the two halves' means). This sets a noise floor — anything above is
meaningful alignment.

Interpretation: cos ≈ 1  ⇒ correct and incorrect push the SAME direction; ±1
advantage makes them cancel (entropy-masking / filtering 'bad' rollouts helps).
cos ≈ 0       ⇒ they are orthogonal; signed-advantage update doesn't fight
itself (entropy masking buys less).

Usage:
  python analyze_exp2_direction.py /path/to/trained_probe_out
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/hdd/mh3897/cc/nanochat/.nanochat/probe/exp12_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def cosine(a, b):
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def main():
    if len(sys.argv) < 2:
        print("usage: python analyze_exp2_direction.py <probe_output_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    grads = np.load(os.path.join(run_dir, "grads.npy"))
    meta = [json.loads(l) for l in open(os.path.join(run_dir, "meta.jsonl"))]
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))

    n_bins = manifest["bins"]
    top_bin = n_bins - 1

    idxs_by_pc: dict[tuple, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if m["bin"] != top_bin:
            continue
        pidx = m["prompt_idx"]
        # After merge_shards, prompt_idx is [shard_tag, int]; JSON round-trips a
        # tuple into a list. Re-tuplize so it's hashable.
        if isinstance(pidx, list):
            pidx = tuple(pidx)
        idxs_by_pc[(pidx, m["correct"])].append(i)

    prompts = sorted({k[0] for k in idxs_by_pc.keys()})
    rows = []
    for p in prompts:
        ci = idxs_by_pc.get((p, True), [])
        ii = idxs_by_pc.get((p, False), [])
        if len(ci) >= 2 and len(ii) >= 2:
            gc = grads[ci].mean(axis=0)
            gi = grads[ii].mean(axis=0)
            rows.append({
                "prompt_idx": list(p) if isinstance(p, tuple) else int(p),
                "n_correct_tokens": len(ci),
                "n_incorrect_tokens": len(ii),
                "cos_correct_vs_incorrect": cosine(gc, gi),
            })

    coses = np.array([r["cos_correct_vs_incorrect"] for r in rows]) if rows else np.array([])

    # Control: within-group half-split (noise floor).
    rng = np.random.RandomState(0)
    control = []
    for (p, c), idxs in idxs_by_pc.items():
        if len(idxs) >= 4:
            shuffled = list(idxs)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            g1 = grads[shuffled[:half]].mean(axis=0)
            g2 = grads[shuffled[half:]].mean(axis=0)
            control.append(cosine(g1, g2))
    control = np.array(control) if control else np.array([])

    # ------------------------------------------------------------------------
    print(f"[exp2] n prompts usable (both correct+incorrect w/ >=2 top-bin tokens): {len(rows)}")
    if len(coses):
        print(f"[exp2] correct-vs-incorrect cos:  mean={coses.mean():.3f}  "
              f"median={np.median(coses):.3f}  std={coses.std():.3f}  "
              f"frac>0={(coses>0).mean():.3f}  frac>0.5={(coses>0.5).mean():.3f}")
    if len(control):
        print(f"[exp2] control (within-group split):  mean={control.mean():.3f}  "
              f"median={np.median(control):.3f}")

    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bins_edges = np.linspace(-1, 1, 21)
    if len(coses):
        ax.hist(coses, bins=bins_edges, alpha=0.7, color="#c44e52",
                label=f"correct vs incorrect  (n={len(coses)} prompts)  mean={coses.mean():.2f}")
    if len(control):
        ax.hist(control, bins=bins_edges, alpha=0.5, color="#4c72b0",
                label=f"control: within-group split  (n={len(control)})  mean={control.mean():.2f}")
    ax.axvline(0, color="black", linestyle=":", lw=1)
    ax.set_xlabel("cosine between mean ∇log π at top-20% entropy tokens")
    ax.set_ylabel("# prompts")
    ax.set_title(
        f"Exp 2: do correct and incorrect rollouts push the same direction at high-entropy tokens?\n"
        f"(target {manifest['target']} row {manifest['row_idx']}, model={os.path.basename(manifest['model_path'])})",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp2_direction_cosine_hist.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "exp2_summary.json"), "w") as f:
        json.dump({
            "run_dir": run_dir,
            "model": manifest["model_path"],
            "target": manifest["target"],
            "n_prompts_usable": len(rows),
            "correct_vs_incorrect": {
                "mean": float(coses.mean()) if len(coses) else None,
                "median": float(np.median(coses)) if len(coses) else None,
                "std": float(coses.std()) if len(coses) else None,
                "frac_positive": float((coses > 0).mean()) if len(coses) else None,
                "frac_gt_0p5": float((coses > 0.5).mean()) if len(coses) else None,
            },
            "control_within_group": {
                "mean": float(control.mean()) if len(control) else None,
                "median": float(np.median(control)) if len(control) else None,
                "n": int(len(control)),
            },
            "per_prompt": rows,
        }, f, indent=2)
    print(f"[exp2] wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
