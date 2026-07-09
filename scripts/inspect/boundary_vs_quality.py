"""Boundary-vs-quality summary plot.

X = the layer where the BoV-style value swap STARTS (from that layer on,
values are context-free). Y = final val_bpb minus the matched attention
baseline's (positive = worse). Vertical line = the causally measured
'stop-reading' boundary (step_d_nanochat). Prediction under test: degradation
kinks in when the swap start crosses to the left of the boundary.

Reads val_bpb directly from each checkpoint's last meta json (stored during
training); no GPU needed.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "/local-ssd/mh3897/base_checkpoints"
PALETTE = ["#1f77b4", "#e8a87c", "#2c8a8a", "#5aa75a"]

# depth -> (baseline_tag, {swap_start_layer: tag}, every2_tag)
SETS = {
    12: ("arch_d12_gpt_base_1.5e18",
         {8: "arch_d12_gpt_base_v_from_value_emb_learn",
          6: "arch_d12_gpt_base_v_from_value_emb_learn_second_half",
          4: "arch_d12_gpt_base_v_from_value_emb_learn_last_two_thirds",
          0: "arch_d12_gpt_base_v_from_value_emb_learn_every_layer"},
         "arch_d12_gpt_base_v_from_value_emb_learn_every2"),
    24: ("arch_d24_gpt_base",
         {16: "arch_d24_gpt_base_v_from_value_emb_learn",
          12: "arch_d24_gpt_base_v_from_value_emb_learn_second_half",
          8: "arch_d24_gpt_base_v_from_value_emb_learn_last_two_thirds",
          0: "arch_d24_gpt_base_v_from_value_emb_learn_every_layer"},
         "arch_d24_gpt_base_v_from_value_emb_learn_every2"),
}


def last_meta(tag):
    ms = sorted(glob.glob(os.path.join(CKPT, tag, "meta_*.json")))
    if not ms:
        return None
    with open(ms[-1]) as f:
        return json.load(f)


def main():
    out = "results/boundary"
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    summary = {}
    for ax, (depth, (base_tag, points, every2_tag)) in zip(axes, sorted(SETS.items())):
        bmeta = last_meta(base_tag)
        if bmeta is None:
            ax.set_title(f"d{depth}: baseline missing"); continue
        base_bpb = bmeta["val_bpb"]
        xs, ys, missing = [], [], []
        for start, tag in sorted(points.items()):
            m = last_meta(tag)
            if m is None:
                missing.append((start, tag)); continue
            xs.append(start); ys.append(m["val_bpb"] - base_bpb)
        e2 = last_meta(every2_tag)
        bfile = os.path.join(out, f"d{depth}__boundary.json")
        boundary = json.load(open(bfile))["boundary"] if os.path.exists(bfile) else None
        summary[depth] = {"base_bpb": base_bpb, "base_step": bmeta["step"],
                          "points": dict(zip(xs, ys)), "boundary": boundary,
                          "every2_delta": (e2["val_bpb"] - base_bpb) if e2 else None,
                          "missing": [t for _, t in missing]}
        ax.plot(xs, ys, "-o", color=PALETTE[0], label="swap-start sweep")
        if e2:
            ax.axhline(e2["val_bpb"] - base_bpb, color=PALETTE[1], ls=":",
                       label="every-2 (interleaved)")
        ax.axhline(0, color="0.6", lw=0.8)
        if boundary is not None:
            ax.axvline(boundary, color="r", ls="--", lw=1,
                       label=f"measured stop-reading boundary L{boundary}")
        ax.set_xlabel("swap start layer (context-free values from here on)")
        ax.set_ylabel("val_bpb - baseline (higher = worse)")
        ax.set_title(f"d{depth} (baseline {base_bpb:.4f} @ step {bmeta['step']})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.invert_xaxis()  # left = swap starts earlier (cuts deeper into the net)
    fig.suptitle("Does quality degrade exactly when the swap crosses the measured boundary?")
    fig.tight_layout()
    p = os.path.join(out, "boundary_vs_quality.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    with open(os.path.join(out, "boundary_vs_quality.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"saved {p}")


if __name__ == "__main__":
    main()
