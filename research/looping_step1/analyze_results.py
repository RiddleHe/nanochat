"""Post-run analysis only. Reads collected results; never touches frozen run code."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Validate recorded results and reproduce paired comparisons")
parser.add_argument("--input", type=Path, default=ROOT / "recorded")
parser.add_argument("--output", type=Path, default=ROOT / "runs/analysis")
args = parser.parse_args()
COLLECTED = args.input
RESULTS = COLLECTED / "results"
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ("qk", "vo", "mlp")
VARIANTS = ("shared_small", "independent", "shared_large")
SEED = 42


def read(path):
    return json.loads(path.read_text())


def check_eval(record):
    rows = record["rows"]
    summary = record["summary"]
    assert [r["sequence"] for r in rows] == list(range(2048))
    assert sum(r["tokens"] for r in rows) == summary["tokens"] == 4194304
    assert sum(r["bytes"] for r in rows) == summary["bytes"]
    bpb = sum(r["byte_nats"] for r in rows) / (math.log(2) * summary["bytes"])
    assert abs(bpb - summary["bpb"]) < 1e-12
    assert all(math.isfinite(r["nats"]) and math.isfinite(r["byte_nats"]) for r in rows)


def paired_difference(first, second):
    a, b = first["rows"], second["rows"]
    assert [(r["sequence"], r["tokens"], r["bytes"]) for r in a] == [
        (r["sequence"], r["tokens"], r["bytes"]) for r in b]
    diff = np.array([x["byte_nats"] - y["byte_nats"] for x, y in zip(a, b)])
    sizes = np.array([r["bytes"] for r in a])
    idx = np.random.default_rng(SEED).integers(0, len(a), size=(2000, len(a)))
    samples = diff[idx].sum(1) / (math.log(2) * sizes[idx].sum(1))
    value = float(diff.sum() / (math.log(2) * sizes.sum()))
    assert abs(value - (first["summary"]["bpb"] - second["summary"]["bpb"])) < 1e-12
    return {"delta_bpb": value, "ci95": np.quantile(samples, [.025, .975]).tolist()}


baseline = {split: read(RESULTS / f"baseline_eval/{split}.json")
            for split in ("development", "confirmation")}
for value in baseline.values():
    check_eval(value)
manifest = read(COLLECTED / "provenance.json")
report = {
    "scope": "Final-checkpoint exploratory screen; one pretraining/adaptation seed.",
    "baseline": {s: o["summary"] for s, o in baseline.items()},
    "confirmation": {}, "paired": {}, "training": {},
    "interval_method": "2000 paired packed-sequence bootstrap draws, seed 42; "
                       "unadjusted descriptive 95% intervals, not seed uncertainty.",
}
for group in GROUPS:
    evals = {}
    report["confirmation"][group] = {}
    for variant in VARIANTS:
        name = f"{group}_{variant}"
        folder = RESULTS / name
        config = read(folder / "config.json")
        result = read(folder / "results.json")
        assert result["updates"] == 200 and result["train_tokens"] == 104857600
        assert result["frozen_weights_unchanged"]
        assert config["initialization_matches_baseline"]
        for key, value in manifest["settings"].items():
            assert config[key] == value, (name, key)
        assert config["trainable_parameters"] == (
            2949120 if variant == "shared_small" else 11796480)
        for step in (0, 50, 100, 200):
            evaluation = read(folder / f"development_{step:03d}.json")
            check_eval(evaluation)
            if step == 0:
                assert evaluation == baseline["development"]
        evaluation = read(folder / "confirmation_200.json")
        check_eval(evaluation)
        assert evaluation["summary"] == result["confirmation"]
        evals[variant] = evaluation
        report["confirmation"][group][variant] = evaluation["summary"]["bpb"]
        rows = [json.loads(x) for x in (folder / "training.jsonl").read_text().splitlines()]
        assert [x["step"] for x in rows] == list(range(1, 201))
        assert all(x["tokens"] == x["step"] * 524288 for x in rows)
        assert all(math.isfinite(x[k]) for x in rows
                   for k in ("lr", "train_ce", "gradient_norm", "seconds"))
        spike = max(rows, key=lambda x: x["gradient_norm"])
        report["training"][name] = {
            "median_step_seconds": float(np.median([x["seconds"] for x in rows])),
            "largest_gradient_norm": spike["gradient_norm"],
            "largest_gradient_step": spike["step"],
            "final_development": result["development"]["bpb"],
        }
    report["paired"][group] = {
        v: paired_difference(evals["independent"], evals[v])
        for v in ("shared_small", "shared_large")
    }

gradient = read(RESULTS / "gradient/results.json")
report["gradient"] = {
    "aggregation": "Mean of within-mini-batch cosines; NOT cosine of mean gradients.",
    "max_sum_relative_error": gradient["max_sum_relative_error"],
    "loop1_average_norm_share_by_module": {},
    "strongest_negative_pairs": [],
}
entries = []
for short in ("q", "k", "v", "o", "up", "down"):
    rows = [r for r in gradient["summary"] if r["name"].endswith("." + short)]
    report["gradient"]["loop1_average_norm_share_by_module"][short] = float(
        np.mean([r["norm_shares"][0] for r in rows]))
    for row in rows:
        for a in range(4):
            for b in range(a + 1, 4):
                entries.append({"matrix": row["name"], "loops": [a + 1, b + 1],
                                "cosine": row["mean_cosine"][a][b]})
report["gradient"]["strongest_negative_pairs"] = sorted(entries, key=lambda e: e["cosine"])[:8]
batch_rows = [json.loads(line) for line in (RESULTS / "gradient/batches.jsonl").read_text().splitlines()]
assert [row["batch"] for row in batch_rows] == list(range(32))
report["gradient"]["layer14_expansion"] = {}
for a, b in ((1, 2), (1, 3)):
    values = np.array([next(m for m in row["matrices"] if m["name"] == "L14.up")["cosine"][a][b]
                       for row in batch_rows])
    report["gradient"]["layer14_expansion"][f"{a+1}/{b+1}"] = {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "negative_batches": int((values < 0).sum()), "total_batches": len(values),
        "range": [float(values.min()), float(values.max())],
    }
report["sensitivity"] = read(RESULTS / "sensitivity/results.json")["conditions"]
report["audit"] = {
    "nine_runs_200_steps_and_all_evaluations_valid": True,
    "all_zero_adapter_evaluations_equal_baseline": True,
    "all_nine_frozen_weights_checks_passed": True,
    "historical_remote_checkpoint_audit": manifest["historical_validation"]["checkpoint_audit"],
    "audit_scope": "This script validates numeric records only. Checkpoints and GPU kernels are not bundled or rechecked here.",
    "original_checkpoint_file_sha256": "e1f2a8988470cf346a9dfb23e8934879a64c70f6b8044997afa486bc93751e05",
}
(OUT / "final_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

# Complement the baseline-improvement plot with the actual sharing contrasts.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "text.color": "#24324A", "axes.labelcolor": "#24324A",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#CAD2DC", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
fig.subplots_adjust(left=.08, right=.98, top=.73, bottom=.26, wspace=.17)
fig.suptitle("Does giving each loop its own adapter improve held-out quality?",
             x=.08, ha="left", fontsize=17, weight="bold")
fig.text(.08, .86, "Independent BPB minus shared BPB: left of zero favors independent.",
         fontsize=11, color="#596579")
for ax, comparator, title, color in zip(
        axes, ("shared_small", "shared_large"),
        ("vs small shared adapter (fewer parameters)",
         "vs large shared adapter (same parameters)"),
        ("#0072B2", "#009E73")):
    for i, group in enumerate(GROUPS):
        item = report["paired"][group][comparator]
        value = item["delta_bpb"] * 1000
        lo, hi = np.array(item["ci95"]) * 1000
        ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt="o",
                    color=color, capsize=4, ms=8)
        ax.annotate(f"{value:+.3f}", (value, i), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.axvline(0, color="#596579", lw=1)
    ax.set_title(title, fontsize=11, loc="left", pad=13)
    ax.set_yticks(range(3), ["Q / K", "V / O", "MLP"])
    ax.set_xlim(-9, 5.5)
    ax.set_ylim(2.5, -.5)
    ax.set_xlabel("BPB difference × 1,000")
    ax.grid(axis="x", alpha=.18)
fig.text(.08, .06,
         "Whiskers: paired-sequence 95% intervals; unadjusted, one seed. They do not measure run-to-run variability.\n"
         "Same-parameter shared adapters use more adapter compute. Neither panel establishes a matched-total-FLOPs result.",
         color="#596579", fontsize=9)
for extension in ("png", "pdf"):
    fig.savefig(OUT / f"06_direct_sharing_comparisons.{extension}", dpi=190,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print(json.dumps({k: report[k] for k in ("baseline", "confirmation", "paired", "gradient")},
                 indent=2))
