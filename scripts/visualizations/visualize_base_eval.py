"""Bar-chart comparison of base_eval CORE results across N models.

Reads the CSVs written by `scripts/base_eval.py` (one row per task,
plus a final CORE row with the centered-average metric) and produces
a 2-row bar chart with consistent per-model colors. The best-scoring
bar in each task is drawn at full alpha; the rest are faded to make
winners easy to spot at a glance. The CORE centered metric for each
model is printed in a box at the bottom.

Usage:
    python -m scripts.visualizations.visualize_base_eval \\
      --csvs path/to/model_a.csv path/to/model_b.csv path/to/model_c.csv \\
      --labels "GPT" "AttnRes" "AttnRes+LB" \\
      --title "Benchmark Comparison: AttnRes Variants" \\
      --subtitle "depth = 12, step = 2,511, FLOPs = 1e18" \\
      --output comparison.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Pretty multi-line names for long task labels (keeps the x-axis readable).
PRETTY = {
    "hellaswag_zeroshot": "HellaSwag\n(0-shot)",
    "hellaswag": "HellaSwag\n(10-shot)",
    "jeopardy": "Jeopardy",
    "bigbench_qa_wikidata": "BB: QA\nWikidata",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-\nChallenge",
    "copa": "COPA",
    "commonsense_qa": "Common-\nsenseQA",
    "piqa": "PIQA",
    "openbook_qa": "Openbook\nQA",
    "lambada_openai": "LAMBADA",
    "winograd": "Winograd",
    "winogrande": "Wino-\nGrande",
    "bigbench_dyck_languages": "BB: Dyck\nLangs",
    "agi_eval_lsat_ar": "AGI Eval\nLSAT AR",
    "bigbench_cs_algorithms": "BB: CS\nAlgos",
    "bigbench_operators": "BB:\nOperators",
    "bigbench_repeat_copy_logic": "BB: Repeat\nCopy Logic",
    "squad": "SQuAD",
    "coqa": "CoQA",
    "boolq": "BoolQ",
    "bigbench_language_identification": "BB:\nLang ID",
}

# Consistent colors across runs; extend if you ever compare > 6 models.
COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#76b7b2", "#b07aa1"]


def parse_eval_csv(path):
    """Parse one base_eval CSV. Returns (tasks, accs, centereds, core_metric)."""
    tasks, accs, centereds = [], [], []
    core = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Task"):
                continue
            parts = [p.strip() for p in line.split(",")]
            task = parts[0]
            if task == "CORE":
                core = float(parts[2])
            else:
                tasks.append(task)
                accs.append(float(parts[1]))
                centereds.append(float(parts[2]))
    if core is None:
        raise ValueError(f"No CORE row found in {path}")
    return tasks, accs, centereds, core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Paths to base_eval CSVs (one per model)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Model labels, one per csv")
    parser.add_argument("--title", type=str, required=True,
                        help="Main plot title")
    parser.add_argument("--subtitle", type=str, default="",
                        help="Optional subtitle line below the title")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path (e.g. comparison.png)")
    parser.add_argument("--alpha-lose", type=float, default=0.35,
                        help="Alpha for non-winning bars (default: 0.35)")
    parser.add_argument("--rows", type=int, default=2,
                        help="Number of subplot rows to split tasks across (default: 2)")
    args = parser.parse_args()

    assert len(args.csvs) == len(args.labels), (
        f"Got {len(args.csvs)} csvs but {len(args.labels)} labels")

    all_data = []
    for label, path in zip(args.labels, args.csvs):
        tasks, accs, centereds, core = parse_eval_csv(path)
        all_data.append((label, tasks, accs, centereds, core))

    # Sanity check: all models evaluated on the same task set.
    task_names = all_data[0][1]
    for label, tasks, *_ in all_data[1:]:
        if tasks != task_names:
            raise ValueError(
                f"Task set mismatch: {label} has {tasks}, expected {task_names}")

    n = len(task_names)
    nm = len(all_data)
    n_per_row = (n + args.rows - 1) // args.rows  # ceil divide

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
    })

    fig, axes = plt.subplots(args.rows, 1, figsize=(16, 4 * args.rows),
                             gridspec_kw={"hspace": 0.55})
    if args.rows == 1:
        axes = [axes]
    fig.suptitle(args.title, fontsize=15, fontweight="bold", style="italic", y=0.98)
    if args.subtitle:
        fig.text(0.5, 0.945, args.subtitle, ha="center", fontsize=11, color="#555")

    bar_width = 0.8 / nm

    for row_idx in range(args.rows):
        ax = axes[row_idx]
        start = row_idx * n_per_row
        end = min(start + n_per_row, n)
        row_tasks = task_names[start:end]
        row_n = len(row_tasks)
        if row_n == 0:
            ax.axis("off")
            continue

        x = np.arange(row_n)

        for m_idx in range(nm):
            accs = [all_data[m_idx][2][start + i] for i in range(row_n)]
            positions = x + (m_idx - (nm - 1) / 2) * bar_width

            for i in range(row_n):
                task_idx = start + i
                all_accs = [all_data[j][2][task_idx] for j in range(nm)]
                best_val = max(all_accs)
                is_winner = best_val > 1e-8 and abs(accs[i] - best_val) < 1e-8
                alpha = 1.0 if is_winner else args.alpha_lose

                ax.bar(positions[i], accs[i], bar_width * 0.9,
                       color=COLORS[m_idx % len(COLORS)], alpha=alpha,
                       label=args.labels[m_idx] if (i == 0 and row_idx == 0) else None)

        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(t, t) for t in row_tasks], fontsize=7.5)
        ax.set_ylabel("Accuracy", fontsize=10)
        row_max = max(max(all_data[j][2][start:end]) for j in range(nm))
        ax.set_ylim(0, min(1.0, row_max * 1.25) if row_max > 0 else 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # CORE metric banner at the bottom.
    core_strs = [f"{all_data[j][0]}: {all_data[j][4]:.4f}" for j in range(nm)]
    core_text = "CORE (centered avg.)   —   " + "   |   ".join(core_strs)
    fig.text(0.5, 0.02, core_text, ha="center", fontsize=10,
             fontfamily="serif", style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f0", edgecolor="#ccc"))

    # Legend in the upper-right corner.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9,
               bbox_to_anchor=(0.98, 0.96), framealpha=0.9)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
