#!/usr/bin/env python3
"""Plot causal recovery by layer for each exact prompt token position."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITION_LABELS = {
    0: '"Everyone" (before entity)',
    1: '"knows" (before entity)',
    2: 'entity name (e.g., Newton)',
    3: '"was" immediately after entity',
    4: '"a"',
    5: '"celebrated"',
    6: 'first role token (e.g., scientist)',
    7: 'period "."',
    8: '"The"',
    9: 'second role token (e.g., scientist)',
    10: 'final "was" (next-token readout)',
}

POSITION_STEMS = {
    0: 'Everyone',
    1: 'knows',
    2: 'subject_entity',
    3: 'was_(after_subject)',
    4: 'a',
    5: 'celebrated',
    6: 'first_role_mention',
    7: 'period',
    8: 'The',
    9: 'second_role_mention',
    10: 'final_was',
}

POSITION_COLORS = {
    0: "#6B7280",
    1: "#6B7280",
    2: "#3F8F4E",
    3: "#6B7280",
    4: "#6B7280",
    5: "#6B7280",
    6: "#D97706",
    7: "#6B7280",
    8: "#6B7280",
    9: "#B45309",
    10: "#2563A6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def position_number(position_group: str) -> int:
    match = re.fullmatch(r"token_(\d+)", position_group)
    if not match:
        raise ValueError(f"Unexpected position_group: {position_group}")
    return int(match.group(1))


def summarize(df: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260713)
    rows: list[dict[str, float | int | str]] = []
    for (position, layer), group in df.groupby(["position", "layer"], sort=True):
        values = group["normalized_recovery"].to_numpy(dtype=float)
        samples = rng.choice(
            values,
            size=(bootstrap_samples, len(values)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "position": int(position),
                "position_label": POSITION_LABELS[int(position)],
                "layer": int(layer),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "n_pairs": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
        }
    )


def style_axis(ax: plt.Axes, y_limits: tuple[float, float]) -> None:
    ax.axvspan(23.5, 35.5, color="#D1D5DB", alpha=0.32, linewidth=0)
    ax.axhline(0.0, color="#4B5563", linewidth=1.0, alpha=0.75)
    ax.axhline(1.0, color="#9CA3AF", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.grid(axis="both", color="#D1D5DB", linewidth=0.8, alpha=0.55)
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(*y_limits)
    ax.set_xticks(np.arange(0, 36, 5))
    ax.set_axisbelow(True)


def draw_curve(
    ax: plt.Axes,
    frame: pd.DataFrame,
    color: str,
    y_limits: tuple[float, float],
    show_ci: bool,
    label: str | None = None,
) -> None:
    style_axis(ax, y_limits)
    x = frame["layer"].to_numpy()
    mean = frame["mean"].to_numpy()
    if show_ci:
        ax.fill_between(
            x,
            frame["ci_low"].to_numpy(),
            frame["ci_high"].to_numpy(),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=3.4,
        label=label,
    )


def plot_individual(
    summary: pd.DataFrame,
    output_dir: Path,
    y_limits: tuple[float, float],
) -> None:
    for position in sorted(summary["position"].unique()):
        frame = summary[summary["position"] == position].sort_values("layer")
        label = POSITION_LABELS[position]
        fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
        draw_curve(
            ax,
            frame,
            POSITION_COLORS[position],
            y_limits,
            show_ci=True,
        )
        ax.set_title(
            f"Patch position {position}: {label}\n"
            "Qwen3-8B-Base, mean over 16 ordered entity pairs"
        )
        ax.set_xlabel("patched layer")
        ax.set_ylabel("normalized logit-difference recovery")
        ax.text(
            0.99,
            0.03,
            "shading: layers 24-35; band: 95% pair-bootstrap CI",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#4B5563",
            fontsize=9,
        )
        stem = f"patch_position_{position:02d}_{POSITION_STEMS[position]}"
        fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_small_multiples(
    summary: pd.DataFrame,
    output_dir: Path,
    y_limits: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(17.0, 10.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for position, ax in zip(sorted(summary["position"].unique()), flat_axes):
        frame = summary[summary["position"] == position].sort_values("layer")
        draw_curve(
            ax,
            frame,
            POSITION_COLORS[position],
            y_limits,
            show_ci=False,
        )
        ax.set_title(f"P{position:02d}  {POSITION_LABELS[position]}", loc="left")
        if position >= 8:
            ax.set_xlabel("patched layer")
        if position in (0, 4, 8):
            ax.set_ylabel("mean recovery")
    for ax in flat_axes[len(POSITION_LABELS) :]:
        ax.axis("off")
    fig.suptitle(
        "Causal recovery by exact patch position and layer\n"
        "Prompt: Everyone knows {entity} was a celebrated {role}. The {role} was\n"
        "P02 = entity name; P06/P09 = first/second role tokens; "
        "P10 = final readout token",
        fontsize=18,
    )
    fig.savefig(
        output_dir / "all_patch_positions_by_layer_small_multiples.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "all_patch_positions_by_layer_small_multiples.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_key_positions(
    summary: pd.DataFrame,
    output_dir: Path,
    y_limits: tuple[float, float],
) -> None:
    selected = [2, 6, 9, 10]
    fig, ax = plt.subplots(figsize=(11.0, 6.5), constrained_layout=True)
    style_axis(ax, y_limits)
    for position in selected:
        frame = summary[summary["position"] == position].sort_values("layer")
        ax.plot(
            frame["layer"],
            frame["mean"],
            color=POSITION_COLORS[position],
            linewidth=2.4,
            marker="o",
            markersize=3.6,
            label=f"P{position:02d} {POSITION_LABELS[position]}",
        )
    ax.set_title(
        "Entity influence moves from P02 (entity name) to P10 "
        "(final readout token)\n"
        "Qwen3-8B-Base, mean over 16 ordered entity pairs"
    )
    ax.set_xlabel("patched layer")
    ax.set_ylabel("normalized logit-difference recovery")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
    )
    fig.savefig(
        output_dir / "key_patch_positions_by_layer.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "key_patch_positions_by_layer.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    token_rows = df[df["position_group"].str.fullmatch(r"token_\d+")].copy()
    token_rows["position"] = token_rows["position_group"].map(position_number)

    expected_positions = set(POSITION_LABELS)
    actual_positions = set(token_rows["position"].unique())
    if actual_positions != expected_positions:
        raise ValueError(
            f"Expected positions {sorted(expected_positions)}, got {sorted(actual_positions)}"
        )

    summary = summarize(token_rows, args.bootstrap_samples)
    summary.to_csv(args.output_dir / "patch_position_curve_summary.csv", index=False)

    lower = min(-0.08, float(summary["ci_low"].min()) - 0.03)
    upper = max(1.08, float(summary["ci_high"].max()) + 0.03)
    y_limits = (lower, upper)

    configure_style()
    plot_individual(summary, args.output_dir, y_limits)
    plot_small_multiples(summary, args.output_dir, y_limits)
    plot_key_positions(summary, args.output_dir, y_limits)

    print(f"rows={len(token_rows)}")
    print(f"pairs={token_rows['pair_id'].nunique()}")
    print(f"layers={token_rows['layer'].min()}..{token_rows['layer'].max()}")
    print(f"positions={sorted(actual_positions)}")
    print(f"mean_range={summary['mean'].min():.4f}..{summary['mean'].max():.4f}")
    print(f"ci_range={summary['ci_low'].min():.4f}..{summary['ci_high'].max():.4f}")
    print(f"y_limits={y_limits[0]:.4f}..{y_limits[1]:.4f}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
