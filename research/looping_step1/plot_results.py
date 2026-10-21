"""Publication-style static figures from measured results only (no placeholders)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, SOURCE, GROUPS, VARIANTS, write_json, now

OUT = ROOT / "figures"
RESULTS = ROOT / "results"
COLORS = {"shared_small": "#0072B2", "independent": "#D55E00", "shared_large": "#009E73"}
LABELS = {"shared_small": "Shared, small rank", "independent": "Independent per loop",
          "shared_large": "Shared, parameter-matched"}
MODULES = {"q": "Query (Q)", "k": "Key (K)", "v": "Value (V)", "o": "Attention output (O)",
           "up": "MLP expansion", "down": "MLP projection"}


def read(path):
    return json.loads(path.read_text())


def save(fig, name, footer):
    fig.text(0.015, 0.015, footer, fontsize=9, color="#596579", va="bottom")
    fig.savefig(OUT / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gradients():
    path = RESULTS / "gradient/results.json"
    if not path.exists():
        return
    summary = read(path)["summary"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 11), sharey=True)
    fig.subplots_adjust(left=.09, right=.88, bottom=.13, top=.89, wspace=.24, hspace=.28)
    fig.suptitle("Do different loops request the same parameter update?", x=.09, ha="left", fontsize=18, weight="bold")
    fig.text(.09, .925, "Cosine within each mini-batch, then average across 32 batches.", fontsize=11, color="#596579")
    for ax, (short, title) in zip(axes.flat, MODULES.items()):
        rows = [r for r in summary if r["name"].endswith("." + short)]
        values = np.array([[r["mean_cosine"][a][b] for a, b in pairs] for r in rows], dtype=float)
        im = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_title(title, loc="left", pad=10, weight="bold")
        ax.set_xticks(range(6), [f"{a+1} / {b+1}" for a, b in pairs])
        ax.set_yticks([0, 3, 7, 11, 15, 19, 23], [1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Loop pair")
        if ax in axes[:, 0]:
            ax.set_ylabel("Physical layer")
        ax.tick_params(length=0)
    cb = fig.colorbar(im, cax=fig.add_axes([.91, .24, .018, .5]))
    cb.set_label("Mean of per-batch gradient cosines")
    save(fig, "01_gradient_directions",
         "Red = opposing; blue = aligned. Not the cosine of the mean gradients.\nFinal checkpoint; 32 batches. Disagreement alone does not establish sharing harm.")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.84, hspace=.34, wspace=.2)
    fig.suptitle("Which loop contributes the largest gradient norm?", x=.08, ha="left", fontsize=18, weight="bold")
    loop_colors = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
    for ax, (short, title) in zip(axes.flat, MODULES.items()):
        rows = [r for r in summary if r["name"].endswith("." + short)]
        values = np.array([r["norm_shares"] for r in rows])
        for loop in range(4):
            ax.plot(range(1, 25), values[:, loop], color=loop_colors[loop],
                    lw=1.8, label=f"Loop {loop+1}")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Physical layer")
        if ax in axes[:, 0]:
            ax.set_ylabel("Fraction of sum of loop norms")
        ax.grid(axis="y", alpha=.22)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.52, .93), ncol=4, frameon=False)
    save(fig, "02_gradient_magnitudes",
         "Average of per-batch norm shares · Four shares sum to one at each matrix\nNorm magnitude is not a measure of causal importance.")


def sensitivity():
    path = RESULTS / "sensitivity/results.json"
    if not path.exists():
        return
    record = read(path)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    fig.subplots_adjust(left=.22, right=.96, bottom=.16, top=.8)
    fig.suptitle("What changes when a loop's module output is scaled?", x=.06, ha="left", fontsize=17, weight="bold")
    fig.text(.06, .89, "Left of zero: lower loss. Right of zero: higher loss.", fontsize=11, color="#596579")
    labels = [f"Loop {loop} · {kind}" for loop in range(1, 5) for kind in ("Attention", "MLP")]
    for scale, offset, color, label in [
        (.9, -.13, "#0072B2", "Output × 0.9"),
        (1.1, .13, "#D55E00", "Output × 1.1"),
    ]:
        rows = [r for r in record["conditions"] if r["scale"] == scale]
        vals = np.array([r["delta_ce"] for r in rows])
        low = np.array([r["ci95"][0] for r in rows])
        high = np.array([r["ci95"][1] for r in rows])
        ax.errorbar(vals, np.arange(8) + offset, xerr=np.stack([vals-low, high-vals]),
                    fmt="o", color=color, capsize=3, ms=7, lw=1.5, label=label)
    ax.axvline(0, color="#596579", lw=1)
    ax.set_yticks(range(8), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in final cross-entropy (nats / token)")
    ax.grid(axis="x", alpha=.22)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "03_output_sensitivity",
         "128 identical evaluation sequences · Each intervention affects one module class in all 24 layers of one loop\nWhiskers: exploratory paired-sequence 95% bootstrap intervals; not adjusted for multiple comparisons.")


def adaptation():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.75, wspace=.18)
    fig.suptitle("Does independent adaptation outperform shared adaptation?", x=.07, ha="left",
                 fontsize=17, weight="bold")
    any_curve = False
    for ax, group in zip(axes, GROUPS):
        for variant in VARIANTS:
            folder = RESULTS / f"{group}_{variant}"
            records = [(int(p.stem.split("_")[-1]), read(p)["summary"])
                       for p in sorted(folder.glob("development_*.json"))]
            if not records:
                continue
            any_curve = True
            ax.plot([s * 524288 / 1e6 for s, _ in records], [v["bpb"] for _, v in records],
                    marker="o", ms=5, lw=2, color=COLORS[variant], label=LABELS[variant])
        ax.set_title({"qk": "Q / K", "vo": "V / O", "mlp": "MLP"}[group], loc="left", weight="bold")
        ax.set_xlabel("Adaptation tokens (millions)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Development BPB · lower is better")
    if any_curve:
        handles = {}
        for ax in axes:
            for h, label in zip(*ax.get_legend_handles_labels()):
                handles[label] = h
        fig.legend(handles.values(), handles.keys(), loc="upper center",
                   bbox_to_anchor=(.53, .9), ncol=3, frameon=False)
        save(fig, "04_lora_learning_curves",
             "Same frozen baseline and token order · One adaptation seed · 4,194,304 development tokens per evaluation\nParameter-matched shared adapters use more adapter FLOPs; these are not matched-total-compute comparisons.")
    else:
        plt.close(fig)
    baseline_path = RESULTS / "baseline_eval/confirmation.json"
    if not baseline_path.exists():
        return
    baseline = read(baseline_path)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(left=.1, right=.97, bottom=.24, top=.76)
    fig.suptitle("Final held-out improvement over the original model", x=.1, ha="left", fontsize=17, weight="bold")
    rows_summary = []
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(baseline["rows"]), size=(2000, len(baseline["rows"])))
    base_nats = np.array([r["byte_nats"] for r in baseline["rows"]])
    base_bytes = np.array([r["bytes"] for r in baseline["rows"]])
    for j, variant in enumerate(VARIANTS):
        xvalues, values, lows, highs = [], [], [], []
        for i, group in enumerate(GROUPS):
            path = RESULTS / f"{group}_{variant}/confirmation_200.json"
            if not path.exists():
                continue
            result = read(path)
            nats = np.array([r["byte_nats"] for r in result["rows"]])
            assert [r["bytes"] for r in result["rows"]] == base_bytes.tolist()
            boots = (base_nats - nats)[idx].sum(1) / (np.log(2) * base_bytes[idx].sum(1))
            value = baseline["summary"]["bpb"] - result["summary"]["bpb"]
            ci = np.quantile(boots, [.025, .975])
            xvalues.append(i + (j - 1) * .18)
            values.append(value)
            lows.append(value - ci[0])
            highs.append(ci[1] - value)
            rows_summary.append(dict(group=group, variant=variant, improvement_bpb=value,
                                     ci95=ci.tolist(), confirmation_bpb=result["summary"]["bpb"]))
        if values:
            ax.errorbar(xvalues, values, yerr=[lows, highs], fmt="o", ms=8,
                        capsize=4, color=COLORS[variant], label=LABELS[variant])
    ax.axhline(0, color="#596579", lw=1)
    ax.set_xticks(range(3), ["Q / K", "V / O", "MLP"])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel("BPB reduction · higher is better")
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.25), ncol=3, frameon=False)
    save(fig, "05_lora_confirmation",
         "Final step 200 · Separate 4,194,304-token confirmation set · Exploratory paired-sequence bootstrap intervals\nShared-small: 2.95M added parameters; independent and shared-large: 11.80M. One seed is not a robustness result.")
    write_json(OUT / "confirmation_summary.json", dict(
        generated_utc=now(), baseline=baseline["summary"], comparisons=rows_summary))


def main():
    global OUT, RESULTS
    parser = argparse.ArgumentParser(description="Plot recorded results without a model or GPU")
    parser.add_argument("--recorded", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.recorded:
        RESULTS = SOURCE / "recorded/results"
    OUT = args.output or (ROOT / "figures")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": "#24324A",
        "text.color": "#24324A", "xtick.color": "#596579", "ytick.color": "#596579",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CAD2DC", "grid.color": "#AEB9C7", "pdf.fonttype": 42,
    })
    gradients()
    sensitivity()
    adaptation()
    write_json(OUT / "inventory.json", dict(
        generated_utc=now(), files=[p.name for p in sorted(OUT.glob("*.png"))]))


if __name__ == "__main__":
    main()
