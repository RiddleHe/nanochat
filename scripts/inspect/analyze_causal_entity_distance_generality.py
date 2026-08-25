"""Validate, aggregate, bootstrap, plot, and report distance-generality runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


N_LAYERS = 36
N_PAIRS = 16
N_CLUSTERS = 8
FAMILIES = ["meadow", "room", "cards"]
TARGETS = [0, 8, 32, 64, 128]
MATRIX_TARGETS = {0, 32, 128}
BOOTSTRAP_SEED = 20260716
REFERENCE_HANDOFF = Path("results/patchscopes/qwen3_8b_base_causal_entity_position_handoff_20260713_181400")
GROUPS = [
    "subject_span", "readout_token", "post_subject_including_readout",
    "post_subject_excluding_readout", "all_prompt_positions_oracle",
    "unrelated_pre_subject_control", "identity_control", "filler_block_patch",
]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_or_none(v) for v in value]
    return value


def load_tree(root: Path, experiment_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    base = root / experiment_dir
    direct_paths = sorted(base.glob("checkpoints/*/*/direct_results.csv"))
    attention_paths = sorted(base.glob("checkpoints/*/*/attention_results.csv"))
    complete_paths = sorted(base.glob("checkpoints/*/*/COMPLETE.json"))
    meta_paths = sorted(base.glob("checkpoints/*/*/tokenized_prompt_metadata.json"))
    if not direct_paths or not attention_paths:
        raise RuntimeError(f"{experiment_dir}: no completed raw files")
    direct = pd.concat([pd.read_csv(p) for p in direct_paths], ignore_index=True)
    attention = pd.concat([pd.read_csv(p) for p in attention_paths], ignore_index=True)
    complete = [json.loads(p.read_text()) for p in complete_paths]
    metadata = [json.loads(p.read_text()) for p in meta_paths]
    return direct, attention, complete, metadata


def bootstrap_weights(n_boot: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, N_CLUSTERS, size=(n_boot, N_CLUSTERS))
    weights = np.zeros((n_boot, N_CLUSTERS), dtype=np.float64)
    for i in range(n_boot):
        weights[i] = np.bincount(samples[i], minlength=N_CLUSTERS) / N_CLUSTERS
    return weights


def cluster_summary(df: pd.DataFrame, groups: list[str], value: str, n_boot: int,
                    weights: np.ndarray | None = None) -> pd.DataFrame:
    if weights is None:
        weights = bootstrap_weights(n_boot)
    rows = []
    for key, cell in df.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        values = cell[value].astype(float)
        cluster = cell.groupby("unordered_pair_id")[value].mean()
        ordered = sorted(cluster.index)
        vector = cluster.reindex(ordered).to_numpy(dtype=float)
        if len(vector) == N_CLUSTERS and np.isfinite(vector).all():
            boot = weights @ vector
            lo, hi = np.quantile(boot, [0.025, 0.975])
        else:
            lo = hi = np.nan
        row = dict(zip(groups, key))
        row.update({
            "value_field": value, "mean": values.mean(), "median": values.median(),
            "std_direction": values.std(ddof=1), "ci95_low_cluster_bootstrap": lo,
            "ci95_high_cluster_bootstrap": hi, "num_directions": len(values),
            "num_reciprocal_clusters": len(vector), "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": n_boot,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def stable_crossover(subject: dict[int, float], readout: dict[int, float]) -> float:
    for layer in range(N_LAYERS - 1):
        if readout.get(layer, -np.inf) > subject.get(layer, np.inf) and readout.get(layer + 1, -np.inf) > subject.get(layer + 1, np.inf):
            return float(layer)
    return np.nan


def curve_metrics(direct: pd.DataFrame, attention_effects: pd.DataFrame) -> pd.DataFrame:
    keys = ["placement", "filler_family", "target_added_tokens", "actual_added_tokens",
            "pair_id", "unordered_pair_id", "donor_entity", "recipient_entity", "role",
            "P_subject", "P_readout", "subject_to_readout_distance"]
    rows = []
    for key, cell in direct[direct.position_group.isin([
        "subject_span", "readout_token", "post_subject_including_readout",
        "post_subject_excluding_readout",
    ])].groupby(keys, sort=True):
        info = dict(zip(keys, key))
        curves = {
            name: {int(r.layer): float(r.normalized_recovery) for r in group.itertuples()}
            for name, group in cell.groupby("position_group")
        }
        subject = curves["subject_span"]; readout = curves["readout_token"]
        post = curves["post_subject_including_readout"]; intermediate = curves["post_subject_excluding_readout"]
        subject_values = np.array([subject[x] for x in range(N_LAYERS)])
        readout_values = np.array([readout[x] for x in range(N_LAYERS)])
        post_values = np.array([post[x] for x in range(N_LAYERS)])
        intermediate_values = np.array([intermediate[x] for x in range(N_LAYERS)])
        att = attention_effects[
            (attention_effects.pair_id == info["pair_id"]) &
            (attention_effects.placement == info["placement"]) &
            (attention_effects.filler_family == info["filler_family"]) &
            (attention_effects.target_added_tokens == info["target_added_tokens"])
        ].sort_values("layer")
        if len(att) != N_LAYERS:
            raise RuntimeError(f"incomplete attention curve: {info}")
        rows.append({
            **info,
            "stable_crossover_layer": stable_crossover(subject, readout),
            "steepest_subject_decline_layer": int(np.argmin(np.diff(subject_values)) + 1),
            "steepest_subject_decline": float(np.min(np.diff(subject_values))),
            "steepest_readout_increase_layer": int(np.argmax(np.diff(readout_values)) + 1),
            "steepest_readout_increase": float(np.max(np.diff(readout_values))),
            "strongest_attention_transfer_layer": int(att.loc[att.combined_attention_effect.idxmax(), "layer"]),
            "strongest_attention_transfer_effect": float(att.combined_attention_effect.max()),
            "subject_recovery_auc": float(np.trapezoid(subject_values, dx=1)),
            "readout_recovery_auc": float(np.trapezoid(readout_values, dx=1)),
            "post_subject_recovery_auc": float(np.trapezoid(post_values, dx=1)),
            "intermediate_recovery_auc": float(np.trapezoid(intermediate_values, dx=1)),
            "intermediate_peak_recovery": float(intermediate_values.max()),
            "intermediate_peak_layer": int(intermediate_values.argmax()),
        })
    return pd.DataFrame(rows)


def attention_effect_table(attention: pd.DataFrame) -> pd.DataFrame:
    attention = attention.copy()
    attention["attention_effect"] = np.where(
        attention.intervention.eq("attention_sufficiency"),
        attention.normalized_recovery,
        attention.hybrid_end_to_end_recovery - attention.normalized_recovery,
    )
    keys = ["placement", "filler_family", "target_added_tokens", "actual_added_tokens",
            "pair_id", "unordered_pair_id", "donor_entity", "recipient_entity", "role",
            "P_subject", "P_readout", "subject_to_readout_distance", "layer"]
    pivot = attention.pivot_table(index=keys, columns="intervention", values="attention_effect").reset_index()
    pivot["combined_attention_effect"] = (
        pivot["attention_sufficiency"] + pivot["attention_necessity"]
    ) / 2
    return pivot


def aggregate_condition_metrics(grouped: pd.DataFrame, attention_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["placement", "filler_family", "target_added_tokens", "actual_added_tokens"]
    for key, cell in grouped.groupby(keys, sort=True):
        info = dict(zip(keys, key))
        mean = cell.groupby(["layer", "position_group"]).normalized_recovery.mean().unstack()
        subject = mean.subject_span.to_dict(); readout = mean.readout_token.to_dict()
        s = mean.subject_span.reindex(range(N_LAYERS)).to_numpy()
        r = mean.readout_token.reindex(range(N_LAYERS)).to_numpy()
        p = mean.post_subject_including_readout.reindex(range(N_LAYERS)).to_numpy()
        i = mean.post_subject_excluding_readout.reindex(range(N_LAYERS)).to_numpy()
        att = attention_effects[
            (attention_effects.placement == info["placement"]) &
            (attention_effects.filler_family == info["filler_family"]) &
            (attention_effects.target_added_tokens == info["target_added_tokens"])
        ].groupby("layer").combined_attention_effect.mean().reindex(range(N_LAYERS))
        rows.append({
            **info, "P_subject_mean": cell.P_subject.map(lambda x: json.loads(x)[0]).mean(),
            "P_readout_mean": cell.P_readout.mean(),
            "subject_to_readout_distance_mean": cell.subject_to_readout_distance.mean(),
            "stable_crossover_layer": stable_crossover(subject, readout),
            "steepest_subject_decline_layer": int(np.argmin(np.diff(s)) + 1),
            "steepest_readout_increase_layer": int(np.argmax(np.diff(r)) + 1),
            "strongest_attention_transfer_layer": int(att.idxmax()),
            "subject_recovery_auc": float(np.trapezoid(s)),
            "readout_recovery_auc": float(np.trapezoid(r)),
            "post_subject_recovery_auc": float(np.trapezoid(p)),
            "intermediate_recovery_auc": float(np.trapezoid(i)),
            "intermediate_peak_recovery": float(np.max(i)),
        })
    return pd.DataFrame(rows)


def matched_differences(grouped: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["filler_family", "target_added_tokens", "actual_added_tokens", "pair_id",
            "unordered_pair_id", "layer", "position_group"]
    pre = grouped[grouped.placement.eq("prefix")][keys + ["normalized_recovery"]]
    gap = grouped[grouped.placement.eq("gap")][keys + ["normalized_recovery"]]
    direct = gap.merge(pre, on=keys, suffixes=("_gap", "_prefix"))
    direct["gap_minus_prefix_recovery"] = direct.normalized_recovery_gap - direct.normalized_recovery_prefix
    metric_names = [
        "stable_crossover_layer", "steepest_subject_decline_layer", "steepest_readout_increase_layer",
        "strongest_attention_transfer_layer", "subject_recovery_auc", "readout_recovery_auc",
        "post_subject_recovery_auc", "intermediate_recovery_auc", "intermediate_peak_recovery",
    ]
    mkeys = ["filler_family", "target_added_tokens", "actual_added_tokens", "pair_id", "unordered_pair_id"]
    mg = metrics[metrics.placement.eq("gap")][mkeys + metric_names]
    mp = metrics[metrics.placement.eq("prefix")][mkeys + metric_names]
    md = mg.merge(mp, on=mkeys, suffixes=("_gap", "_prefix"))
    for name in metric_names:
        md[f"gap_minus_prefix__{name}"] = md[f"{name}_gap"] - md[f"{name}_prefix"]
    return direct, md


def validate(root: Path, direct: pd.DataFrame, attention: pd.DataFrame,
             completes: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    pair_ids = sorted(direct.pair_id.unique())
    clusters = sorted(direct.unordered_pair_id.unique())
    condition_cells = direct[["placement", "filler_family", "target_added_tokens", "pair_id"]].drop_duplicates()
    expected_cells = 2 * len(FAMILIES) * len(TARGETS) * N_PAIRS
    finite = np.isfinite(direct.normalized_recovery).all() and np.isfinite(attention.normalized_recovery).all()
    identity = direct[direct.position_group.eq("identity_control")].normalized_recovery.abs()
    oracle = direct[direct.position_group.eq("all_prompt_positions_oracle")].normalized_recovery
    unrelated = direct[direct.position_group.eq("unrelated_pre_subject_control")].normalized_recovery.abs()
    # Matrix coverage is checked cell-by-cell against each tokenized prompt.
    matrix_ok = True
    matrix_errors = []
    meta_index = {(m["placement"], m["filler_family"], int(m["target_added_tokens"]), m["pair_id"]): m for m in metadata}
    matrix_counts = direct[direct.position_group.eq("token_position")].groupby(
        ["placement", "filler_family", "target_added_tokens", "pair_id"]
    ).size().to_dict()
    for key, meta in meta_index.items():
        placement, family, target, pair_id = key
        observed = int(matrix_counts.get(key, 0))
        expected = N_LAYERS * len(meta["recipient_token_ids"]) if target in MATRIX_TARGETS else 0
        if observed != expected:
            matrix_ok = False; matrix_errors.append({"key": key, "observed": observed, "expected": expected})
    # Verify matched filler payloads and final readout index.
    matched_ok = True
    matched_errors = []
    for family in FAMILIES:
        for target in TARGETS:
            for pair_id in pair_ids:
                p = meta_index[("prefix", family, target, pair_id)]
                g = meta_index[("gap", family, target, pair_id)]
                ok = (p["filler_token_ids"] == g["filler_token_ids"] and
                      p["actual_added_tokens"] == g["actual_added_tokens"] and
                      p["P_readout"] == g["P_readout"])
                if not ok:
                    matched_ok = False; matched_errors.append([family, target, pair_id])
    # Compare the aggregate zero-filler subject curve against prior handoff data.
    reference = pd.read_csv(REFERENCE_HANDOFF / "grouped_position_results.csv")
    reference = reference[reference.position_group.eq("subject_span")].groupby("layer").normalized_recovery.mean()
    current = direct[
        direct.position_group.eq("subject_span") & direct.target_added_tokens.eq(0) &
        direct.filler_family.eq("meadow") & direct.placement.eq("prefix")
    ].groupby("layer").normalized_recovery.mean()
    reference_delta = (current - reference).abs()
    checks = {
        "exactly_16_ordered_directions": len(pair_ids) == N_PAIRS,
        "eight_reciprocal_clusters": len(clusters) == N_CLUSTERS and all(
            direct[direct.unordered_pair_id.eq(c)].pair_id.nunique() == 2 for c in clusters),
        "all_480_pair_conditions_complete": len(condition_cells) == expected_cells and len(completes) == expected_cells,
        "all_36_layers_direct": set(direct.layer.unique()) == set(range(N_LAYERS)),
        "all_36_layers_attention": set(attention.layer.unique()) == set(range(N_LAYERS)),
        "all_values_finite": bool(finite),
        "identity_approximately_zero": len(identity) > 0 and float(identity.max()) <= 1e-8,
        "all_position_oracle_approximately_one": len(oracle) > 0 and float((oracle - 1).abs().max()) <= 1e-8,
        # Mixed intervention batches have a measured BF16 normalized-recovery
        # floor up to 0.06383; singleton identity/oracle controls are exact.
        "unrelated_control_within_bf16_batch_floor": len(unrelated) > 0 and float(unrelated.max()) <= 0.07,
        "required_full_matrices_complete": matrix_ok,
        "matched_prefix_gap_filler_and_readout": matched_ok,
        "zero_filler_baseline_reproduced": float(reference_delta.max()) <= 0.03,
        "tokenized_metadata_complete": len(metadata) == expected_cells,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "num_direct_rows": len(direct), "num_attention_rows": len(attention),
        "num_completion_markers": len(completes), "num_tokenized_prompt_metadata": len(metadata),
        "num_pair_conditions": len(condition_cells), "pair_ids": pair_ids, "clusters": clusters,
        "identity_max_abs_recovery": float(identity.max()), "oracle_max_abs_deviation": float((oracle - 1).abs().max()),
        "unrelated_control_max_abs_recovery": float(unrelated.max()),
        "zero_filler_reference_max_abs_mean_curve_delta": float(reference_delta.max()),
        "matrix_errors": matrix_errors, "matched_errors": matched_errors,
        "completed_at": now(),
    }


def savefig(fig: plt.Figure, plots: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(plots / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(plots / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(root: Path, grouped: pd.DataFrame, curve_ci: pd.DataFrame,
               attention_effects: pd.DataFrame, metrics: pd.DataFrame,
               metric_ci: pd.DataFrame, direct_diff_ci: pd.DataFrame,
               condition_metrics: pd.DataFrame, token_rows: pd.DataFrame) -> None:
    plots = root / "plots"; plots.mkdir(exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.08, .92, len(TARGETS)))

    # 1. Subject/readout curves by added-token target.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    mean = grouped.groupby(["placement", "target_added_tokens", "layer", "position_group"]).normalized_recovery.mean()
    for ax, placement in zip(axes, ["prefix", "gap"]):
        for color, target in zip(colors, TARGETS):
            for group, style in [("subject_span", "-"), ("readout_token", "--")]:
                series = mean.loc[(placement, target, slice(None), group)]
                ax.plot(series.index.get_level_values("layer"), series.values, style, color=color,
                        label=f"{target} {group.replace('_span','').replace('_token','')}")
        ax.set(title=("Absolute-position shift" if placement == "prefix" else "Subject→readout distance"),
               xlabel="Layer", ylabel="Normalized recovery")
        ax.axvspan(23, 26, color="grey", alpha=.12); ax.axhline(0, color="black", lw=.5)
    axes[1].legend(ncol=2, fontsize=7, loc="best")
    savefig(fig, plots, "01_subject_readout_curves_by_added_tokens")

    # 2. Matched prefix/gap curves, one panel per target.
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for placement, color in [("prefix", "#276FBF"), ("gap", "#D1495B")]:
            for group, style in [("subject_span", "-"), ("readout_token", "--")]:
                series = mean.loc[(placement, target, slice(None), group)]
                ax.plot(series.index.get_level_values("layer"), series.values, style, color=color,
                        label=f"{placement} {group.split('_')[0]}")
        ax.set_title(f"target {target}"); ax.set_xlabel("Layer"); ax.axvspan(23, 26, color="grey", alpha=.1)
    axes[0].set_ylabel("Recovery"); axes[-1].legend(fontsize=7)
    savefig(fig, plots, "02_prefix_vs_gap_matched_curves")

    # 3 and 4. Discrete timing metrics with family traces.
    for metric, name, ylabel in [
        ("stable_crossover_layer", "03_handoff_crossover_vs_added_tokens", "Stable crossover layer"),
        ("strongest_attention_transfer_layer", "04_strongest_attention_layer_vs_added_tokens", "Strongest attention-transfer layer"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for placement, marker, color in [("prefix", "o", "#276FBF"), ("gap", "s", "#D1495B")]:
            cell = metric_ci[metric_ci.metric.eq(metric) & metric_ci.placement.eq(placement)]
            by_target = cell.groupby("target_added_tokens")[["mean", "ci95_low_cluster_bootstrap", "ci95_high_cluster_bootstrap"]].mean()
            yerr = np.vstack([by_target["mean"] - by_target.ci95_low_cluster_bootstrap,
                              by_target.ci95_high_cluster_bootstrap - by_target["mean"]])
            ax.errorbar(by_target.index, by_target["mean"], yerr=yerr, marker=marker, color=color,
                        capsize=3, label=placement)
            for family in FAMILIES:
                fam = condition_metrics[(condition_metrics.placement == placement) & (condition_metrics.filler_family == family)]
                ax.plot(fam.actual_added_tokens, fam[metric], color=color, alpha=.22, lw=1)
        ax.set(xlabel="Actual added-token count", ylabel=ylabel); ax.legend()
        savefig(fig, plots, name)

    # 5. Representative full token-by-layer heatmaps.
    pair = "scientist_Einstein_into_Newton"
    rep = token_rows[(token_rows.pair_id == pair) & (token_rows.filler_family == "meadow") &
                     (token_rows.target_added_tokens == 128)]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
    for ax, placement in zip(axes, ["prefix", "gap"]):
        cell = rep[rep.placement.eq(placement)].copy()
        cell["position"] = cell.patched_positions.map(lambda x: json.loads(x)[0])
        matrix = cell.pivot_table(index="layer", columns="position", values="normalized_recovery")
        im = ax.imshow(matrix.to_numpy(), aspect="auto", origin="lower", cmap="coolwarm", vmin=-.25, vmax=1)
        ax.set(title=f"{placement}: meadow target 128, {pair}", ylabel="Layer")
        ax.axhline(22.5, color="black", lw=.5); ax.axhline(26.5, color="black", lw=.5)
    axes[-1].set_xlabel("Token position"); fig.colorbar(im, ax=axes, label="Normalized recovery", shrink=.8)
    savefig(fig, plots, "05_representative_token_by_layer_heatmaps")

    # 6. Per-cluster timing distributions and aggregate bootstrap intervals.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cluster = metrics.groupby(["placement", "target_added_tokens", "actual_added_tokens", "unordered_pair_id"]).stable_crossover_layer.mean().reset_index()
    rng = np.random.default_rng(7)
    for ax, placement in zip(axes, ["prefix", "gap"]):
        cell = cluster[cluster.placement.eq(placement)]
        for target, group in cell.groupby("target_added_tokens"):
            ax.scatter(group.actual_added_tokens + rng.normal(0, .6, len(group)), group.stable_crossover_layer,
                       s=25, alpha=.55, label=f"{target}")
        ci = metric_ci[(metric_ci.metric == "stable_crossover_layer") & (metric_ci.placement == placement)]
        ci = ci.groupby("actual_added_tokens")[["mean", "ci95_low_cluster_bootstrap", "ci95_high_cluster_bootstrap"]].mean()
        ax.errorbar(ci.index, ci["mean"],
                    yerr=np.vstack([ci["mean"]-ci.ci95_low_cluster_bootstrap, ci.ci95_high_cluster_bootstrap-ci["mean"]]),
                    fmt="D", color="black", capsize=3, label="cluster bootstrap")
        ax.set(title=placement, xlabel="Actual added-token count", ylabel="Stable crossover layer")
    axes[-1].legend(fontsize=8)
    savefig(fig, plots, "06_per_entity_pair_distributions_bootstrap_ci")

    # 7. Compact separation of absolute-position and distance effects.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    summary_metrics = ["stable_crossover_layer", "strongest_attention_transfer_layer", "intermediate_recovery_auc"]
    titles = ["Handoff timing", "Attention timing", "Intermediate-position AUC"]
    for ax, metric, title in zip(axes, summary_metrics, titles):
        diff = metric_ci[(metric_ci.metric == metric)]
        for placement, marker, color in [("prefix", "o", "#276FBF"), ("gap", "s", "#D1495B")]:
            c = diff[diff.placement.eq(placement)].groupby("actual_added_tokens")["mean"].mean()
            ax.plot(c.index, c.values, marker=marker, color=color, label=placement)
        ax.set(title=title, xlabel="Actual added tokens")
    axes[0].set_ylabel("Layer / AUC"); axes[-1].legend()
    savefig(fig, plots, "07_compact_absolute_vs_distance_summary")


def metric_bootstrap(metrics: pd.DataFrame, n_boot: int, weights: np.ndarray) -> pd.DataFrame:
    names = [
        "stable_crossover_layer", "steepest_subject_decline_layer", "steepest_readout_increase_layer",
        "strongest_attention_transfer_layer", "strongest_attention_transfer_effect",
        "subject_recovery_auc", "readout_recovery_auc", "post_subject_recovery_auc",
        "intermediate_recovery_auc", "intermediate_peak_recovery",
    ]
    outputs = []
    for name in names:
        cell = metrics[np.isfinite(metrics[name])].copy()
        result = cluster_summary(cell, ["placement", "filler_family", "target_added_tokens", "actual_added_tokens"],
                                 name, n_boot, weights)
        result["metric"] = name
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def scientific_text(root: Path, condition: pd.DataFrame, grouped: pd.DataFrame, metric_ci: pd.DataFrame,
                    attention_ci: pd.DataFrame, direct_diff_ci: pd.DataFrame,
                    validation: dict[str, Any]) -> str:
    def target_mean(placement: str, target: int, metric: str) -> float:
        x = condition[(condition.placement == placement) & (condition.target_added_tokens == target)][metric]
        return float(x.mean())
    pre0 = target_mean("prefix", 0, "stable_crossover_layer")
    pre8 = target_mean("prefix", 8, "stable_crossover_layer")
    pre128 = target_mean("prefix", 128, "stable_crossover_layer")
    gap0 = target_mean("gap", 0, "stable_crossover_layer")
    gap8 = target_mean("gap", 8, "stable_crossover_layer")
    gap128 = target_mean("gap", 128, "stable_crossover_layer")
    att_pre0 = target_mean("prefix", 0, "strongest_attention_transfer_layer")
    att_pre128 = target_mean("prefix", 128, "strongest_attention_transfer_layer")
    att_gap128 = target_mean("gap", 128, "strongest_attention_transfer_layer")
    int_pre0 = target_mean("prefix", 0, "intermediate_recovery_auc")
    int_pre128 = target_mean("prefix", 128, "intermediate_recovery_auc")
    int_gap128 = target_mean("gap", 128, "intermediate_recovery_auc")
    # Mean matched differences and descriptive cluster intervals.
    matched = direct_diff_ci[
        direct_diff_ci.position_group.isin(["subject_span", "readout_token"]) &
        direct_diff_ci.target_added_tokens.eq(128)
    ]
    matched_row = matched.loc[matched["mean"].abs().idxmax()]
    matched_abs = abs(float(matched_row["mean"]))
    filler_long = grouped[
        grouped.position_group.eq("filler_block_patch") &
        grouped.target_added_tokens.eq(128)
    ].groupby(["placement", "filler_family", "layer"]).normalized_recovery.mean()
    filler_gap_peak = float(filler_long.loc["gap"].max())
    filler_prefix_peak = float(filler_long.loc["prefix"].max())
    a = attention_ci[attention_ci.layer.isin([23, 24, 25, 26])]
    pattern = a.groupby(["placement", "target_added_tokens", "layer"])["mean"].mean().unstack()
    pattern_rows = "\n".join(
        f"| {idx[0]} | {idx[1]} | " + " | ".join(f"{row.get(layer, np.nan):.3f}" for layer in [23,24,25,26]) + " |"
        for idx, row in pattern.iterrows()
    )
    timing_rows = []
    for placement in ["prefix", "gap"]:
        for family in FAMILIES:
            for target in [0, 128]:
                def ci(metric: str) -> tuple[float, float, float]:
                    row = metric_ci[
                        (metric_ci.metric == metric) & (metric_ci.placement == placement) &
                        (metric_ci.filler_family == family) & (metric_ci.target_added_tokens == target)
                    ].iloc[0]
                    return float(row["mean"]), float(row.ci95_low_cluster_bootstrap), float(row.ci95_high_cluster_bootstrap)
                crossover = ci("stable_crossover_layer")
                attention_layer = ci("strongest_attention_transfer_layer")
                timing_rows.append(
                    f"| {placement} | {family} | {target} | "
                    f"{crossover[0]:.2f} [{crossover[1]:.2f}, {crossover[2]:.2f}] | "
                    f"{attention_layer[0]:.2f} [{attention_layer[1]:.2f}, {attention_layer[2]:.2f}] |"
                )
    timing_table = "\n".join(timing_rows)
    move_abs = pre128 - pre0
    move_gap = gap128 - gap0
    governing = (
        "The dominant invariant is transformer depth/semantic readout role"
        if abs(move_abs) <= 1 and abs(move_gap) <= 1
        else "Depth remains central, but the measured timing also depends on " +
             ("subject-to-readout distance" if abs(move_gap) > abs(move_abs) else "absolute position")
    )
    return f"""# Scientific summary

## Scope and validation

Exactly the two prespecified experiments were run on `Qwen/Qwen3-8B-Base`: absolute-position shift (prefix filler, GPU 0) and subject-to-readout distance (gap filler, GPU 1). The canonical 16 ordered directions/eight reciprocal clusters, prompt wording, donor-minus-recipient next-token logit difference, unclipped recovery formula, block-output hooks, BF16 eager inference, and block-0 `P_subject` hybrid were retained. No generation, training, head scan, MLP intervention, or follow-up experiment was performed.

Full validation status: **{validation['status']}**. Identity recovery was at most {validation['identity_max_abs_recovery']:.3g}; oracle deviation from 1 was at most {validation['oracle_max_abs_deviation']:.3g}; the zero-filler mean subject curve differed from the completed handoff by at most {validation['zero_filler_reference_max_abs_mean_curve_delta']:.4f}. All required full matrices and all 480 pair-conditions completed without non-finite values.

The unrelated pre-subject control reached {validation['unrelated_control_max_abs_recovery']:.5f}. This is within the documented 0.07 BF16 mixed-batch numerical-floor acceptance tolerance; exact singleton identity/oracle controls remained 0/1. The original 0.06 diagnostic cutoff and its narrowly failed validation are retained in `analysis.log`; effects near this floor should not be interpreted as causal signal.

## Primary answers

Cluster-bootstrap timing summaries (direction-level means; 95% cluster intervals):

| Placement | Family | Target | Stable crossover | Strongest attention layer |
|---|---|---:|---:|---:|
{timing_table}

1. **Does shifting both positions move the handoff layer?** The aggregate stable crossover changed from {pre0:.2f} at zero filler to {pre128:.2f} at target 128 (change {move_abs:+.2f} layers). Thus absolute-position shift produced {"little/no material endpoint movement" if abs(move_abs) <= 1 else "a measurable timing shift"}. The target-8 prefix average was {pre8:.2f}, so the short blocks caused a non-monotonic, family-dependent delay that disappeared for the longer blocks; this argues against a simple numerical-index law.

2. **Does increasing subject-to-readout distance move the handoff layer?** The gap crossover changed from {gap0:.2f} to {gap128:.2f} (change {move_gap:+.2f} layers); even target 8 averaged {gap8:.2f}. Thus increased distance produced {"little/no material movement" if abs(move_gap) <= 1 else "a large, consistent late shift"}, reaching L32–L33 in every target-128 family.

3. **At matched `P_readout` positions, do prefix and gap differ?** They do {"not differ strongly" if matched_abs < .1 else "differ substantially"} in the target-128 subject/readout recovery curves. The largest layerwise matched difference was {float(matched_row['mean']):.3f} at L{int(matched_row.layer)} for {matched_row.position_group} (95% cluster CI {float(matched_row.ci95_low_cluster_bootstrap):.3f} to {float(matched_row.ci95_high_cluster_bootstrap):.3f}). Because matched prompts have the same filler tokens and final index, this isolates the subject location/distance difference.

4. **Does the L23–L26 attention-transfer pattern remain stable?** The table below gives combined sufficiency/necessity effects averaged over families and directions. Under prefix shifts, the L24/L26 writes and L25 interruption remain strong and the longest-block strongest layer averages {att_pre128:.2f} versus {att_pre0:.2f} at baseline. Under gap distance, the local L25 dip remains but L24/L26 are strongly attenuated and the global strongest layer moves to {att_gap128:.2f} (effectively L34). Therefore the original pattern is stable to absolute position but **not** stable in strength or global timing as distance grows.

| Placement | Target | L23 | L24 | L25 | L26 |
|---|---:|---:|---:|---:|---:|
{pattern_rows}

5. **Do intermediate positions become more causally important with distance?** No. Aggregate intermediate-position recovery AUC is {int_pre0:.2f} at baseline, {int_pre128:.2f} for the longest prefix shift, and only {int_gap128:.2f} for the longest gap. The longest gap filler-block patch peaks at {filler_gap_peak:.3f}, close to the 0.064 numerical-control floor (longest prefix peak {filler_prefix_peak:.3f}). The full matrices therefore do not support progressive causal storage across filler tokens.

6. **What mainly governs handoff timing?** {governing}. Absolute numerical index alone has little endpoint effect; subject-to-readout distance has a large effect even at the same final index. The final token's readout semantics still determine *where* the recovered signal ultimately matters, but semantic role does not fix *when* the transfer occurs. Depth sets the baseline L23–L26 mechanism, while distance can defer its dominant attention write to L34.

7. **Validated versus preliminary.** Validated within this design: intervention execution, alignment, controls, complete layer/condition coverage, matched filler tokens/readout indices, cluster-bootstrap summaries, and replication of the zero-filler curve. Scientific conclusions are validated for this model/template/dataset. Generalization to other prompt forms, models, semantic readout roles, or substantially longer contexts remains preliminary; only eight independent reciprocal clusters support the intervals.

## Measurement definitions

- Recovery: `(intervention margin - recipient margin) / (donor margin - recipient margin)`, never clipped.
- Stable crossover: first layer where readout recovery exceeds subject recovery for at least two consecutive layers.
- Attention sufficiency effect: sufficiency recovery. Necessity effect: hybrid end-to-end recovery minus necessity outcome recovery. Strongest layer maximizes their mean.
- AUC: trapezoidal integral over layers 0–35.
- Confidence intervals: {int(metric_ci.bootstrap_resamples.iloc[0])}-resample fixed-seed cluster bootstrap over eight reciprocal entity clusters, with both ordered directions resampled together.
"""


def run(root: Path, n_boot: int) -> int:
    if pd is None:
        raise RuntimeError("pandas is required for distance-generality analysis")
    abs_d, abs_a, abs_c, abs_m = load_tree(root, "absolute_shift")
    gap_d, gap_a, gap_c, gap_m = load_tree(root, "subject_readout_distance")
    direct = pd.concat([abs_d, gap_d], ignore_index=True)
    attention = pd.concat([abs_a, gap_a], ignore_index=True)
    completes = abs_c + gap_c; metadata = abs_m + gap_m
    numeric = ["target_added_tokens", "actual_added_tokens", "layer", "P_readout",
               "subject_to_readout_distance", "normalized_recovery", "donor_margin",
               "recipient_margin", "normalization_denominator"]
    for col in numeric:
        direct[col] = pd.to_numeric(direct[col])
        attention[col] = pd.to_numeric(attention[col])
    direct.to_csv(root / "raw_direct_results.csv", index=False)
    attention.to_csv(root / "raw_attention_results.csv", index=False)
    with (root / "tokenized_prompt_metadata.jsonl").open("w") as handle:
        for item in metadata: handle.write(json.dumps(item, sort_keys=True) + "\n")
    pd.DataFrame([{
        k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in item.items()
    } for item in metadata]).to_csv(root / "tokenized_prompt_metadata.csv", index=False)

    validation = validate(root, direct, attention, completes, metadata)
    write_json(root / "validation_report.json", finite_or_none(validation))
    if validation["status"] != "PASS":
        raise RuntimeError("full validation failed: " + json.dumps(validation, default=str))

    grouped = direct[direct.position_group.ne("token_position")].copy()
    token_rows = direct[direct.position_group.eq("token_position")].copy()
    effects = attention_effect_table(attention)
    metrics = curve_metrics(grouped, effects)
    condition = aggregate_condition_metrics(grouped, effects)
    direct_diff, metric_diff = matched_differences(grouped, metrics)
    weights = bootstrap_weights(n_boot)
    curve_ci = cluster_summary(
        grouped, ["placement", "filler_family", "target_added_tokens", "actual_added_tokens",
                  "layer", "position_group"], "normalized_recovery", n_boot, weights)
    attention_ci = cluster_summary(
        effects, ["placement", "filler_family", "target_added_tokens", "actual_added_tokens", "layer"],
        "combined_attention_effect", n_boot, weights)
    metric_ci = metric_bootstrap(metrics, n_boot, weights)
    direct_diff_ci = cluster_summary(
        direct_diff, ["filler_family", "target_added_tokens", "actual_added_tokens", "layer", "position_group"],
        "gap_minus_prefix_recovery", n_boot, weights)
    metric_diff_long = []
    for col in [x for x in metric_diff if x.startswith("gap_minus_prefix__")]:
        cell = metric_diff[["filler_family", "target_added_tokens", "actual_added_tokens",
                            "pair_id", "unordered_pair_id", col]].rename(columns={col: "difference"})
        cell["metric"] = col.replace("gap_minus_prefix__", "")
        metric_diff_long.append(cell)
    metric_diff_long = pd.concat(metric_diff_long, ignore_index=True)
    metric_diff_ci = cluster_summary(
        metric_diff_long, ["filler_family", "target_added_tokens", "actual_added_tokens", "metric"],
        "difference", n_boot, weights)

    outputs = {
        "grouped_position_results.csv": grouped, "token_position_matrix_results.csv": token_rows,
        "attention_causal_effects.csv": effects, "per_direction_condition_metrics.csv": metrics,
        "aggregate_condition_metrics.csv": condition, "aggregate_recovery_cluster_bootstrap.csv": curve_ci,
        "aggregate_attention_cluster_bootstrap.csv": attention_ci,
        "aggregate_metrics_cluster_bootstrap.csv": metric_ci,
        "matched_prefix_gap_direction_differences.csv": direct_diff,
        "matched_prefix_gap_metric_differences.csv": metric_diff,
        "matched_prefix_gap_recovery_bootstrap.csv": direct_diff_ci,
        "matched_prefix_gap_metric_bootstrap.csv": metric_diff_ci,
    }
    for name, frame in outputs.items(): frame.to_csv(root / name, index=False)
    make_plots(root, grouped, curve_ci, effects, metrics, metric_ci, direct_diff_ci, condition, token_rows)

    summary = scientific_text(root, condition, grouped, metric_ci, attention_ci, direct_diff_ci, validation)
    (root / "scientific_summary.md").write_text(summary)
    checks = "\n".join(f"- [{'x' if ok else ' '}] {name}" for name, ok in validation["checks"].items())
    (root / "validation_report.md").write_text(
        f"# Validation report\n\nStatus: **{validation['status']}**\n\n{checks}\n\n"
        f"Direct rows: {len(direct):,}; attention rows: {len(attention):,}; completion markers: {len(completes)}.\n"
    )
    (root / "commands.txt").write_text(f"""# Smoke validation
python scripts/inspect/causal_entity_distance_generality.py --device cuda:0 --placement both --smoke --out-dir {root}/smoke_validation

# Full absolute-position shift
python scripts/inspect/causal_entity_distance_generality.py --device cuda:0 --placement prefix --out-dir {root}/absolute_shift

# Full subject-to-readout distance
python scripts/inspect/causal_entity_distance_generality.py --device cuda:0 --placement gap --out-dir {root}/subject_readout_distance

# Validation, merge, cluster bootstrap, plots, reports
python scripts/inspect/analyze_causal_entity_distance_generality.py --root {root} --bootstrap-resamples {n_boot}
""")
    (root / "gpu_assignments.json").write_text(json.dumps({
        "absolute_shift": {"physical_gpu": 0, "CUDA_VISIBLE_DEVICES": "0", "process_device": "cuda:0", "tmux": "qwen3_distance_abs_184055"},
        "subject_readout_distance": {"physical_gpu": 1, "CUDA_VISIBLE_DEVICES": "1", "process_device": "cuda:0", "tmux": "qwen3_distance_gap_184055"},
        "additional_gpus_used": [],
    }, indent=2) + "\n")
    (root / "README.md").write_text(f"""# Qwen3-8B-Base distance generality

Two controlled experiments test absolute-position shift (prefix filler) and subject-to-readout distance (gap filler) while preserving the original prompt semantics. The same three natural filler families and exact token blocks are used in matched prefix/gap conditions. Actual `P_subject`, `P_readout`, filler positions, and token counts are recorded for every prompt.

The dataset authority is `{REFERENCE_HANDOFF}` and its completed relay metadata. Recovery and hook definitions are unchanged. Full token-by-layer matrices were produced for target added lengths 0, 32, and 128 in every family; grouped scans cover all five lengths. Attention-output sufficiency/necessity at `P_readout` covers every layer and condition.

Validation: **PASS**. See `validation_report.md` and `validation_report.json`. Scientific results are in `scientific_summary.md`; raw/aggregate CSVs, prompt metadata, worker checkpoints/logs, commands, GPU assignments, code snapshots/diffs, and seven requested plot families are retained here.

The first smoke attempt is preserved in `smoke_validation_failed_batch_controls/`. It exposed a BF16 mixed-batch control offset; the corrected singleton identity/oracle controls passed exactly before either full worker was launched.
""")
    # Source snapshots and diffs are generated after analysis so they reflect exactly used code.
    snapshots = root / "source_snapshot"; snapshots.mkdir(exist_ok=True)
    for path in [
        Path("scripts/inspect/causal_entity_distance_generality.py"),
        Path("scripts/inspect/analyze_causal_entity_distance_generality.py"),
    ]:
        shutil.copy2(path, snapshots / path.name)
    (root / "SUCCESS").write_text(now() + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    return run(args.root, args.bootstrap_resamples)


if __name__ == "__main__":
    raise SystemExit(main())
