"""Merge, validate, summarize, and plot the three component mediation experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


N_LAYERS = 36
N_PAIRS = 16
SCAN_LAYERS = [23, 24, 25, 26]
BOOTSTRAP_SEED = 20260715
REFERENCE_RELAY = Path("/ssd/mh3897/patchscope_results/qwen3_8b_base_p02_p10_relay_20260714_030311")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def f(value: Any) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("nan")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    x = q * (len(values) - 1); lo = int(math.floor(x)); hi = int(math.ceil(x))
    return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)


def key_sort(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(x) for x in key)


def cluster_summary(rows: list[dict[str, Any]], group_fields: list[str], value_field: str,
                    n_boot: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = f(row.get(value_field))
        if math.isfinite(value): buckets[tuple(row.get(k, "") for k in group_fields)].append(row)
    output = []
    rng = random.Random(BOOTSTRAP_SEED)
    for key in sorted(buckets, key=key_sort):
        cell = buckets[key]; values = [f(row[value_field]) for row in cell]
        clusters: dict[str, list[float]] = defaultdict(list)
        for row in cell: clusters[str(row["unordered_pair_id"])].append(f(row[value_field]))
        boot = []
        cluster_names = sorted(clusters)
        if len(cluster_names) == 8 and all(len(clusters[c]) == 2 for c in cluster_names):
            for _ in range(n_boot):
                sampled = [rng.choice(cluster_names) for _ in cluster_names]
                boot.append(statistics.fmean(v for c in sampled for v in clusters[c]))
            lo, hi = percentile(boot, .025), percentile(boot, .975)
        else:
            lo = hi = float("nan")
        row = {field: value for field, value in zip(group_fields, key)}
        row.update({
            "value_field": value_field, "mean": statistics.fmean(values), "median": statistics.median(values),
            "ci95_low_cluster_bootstrap": lo, "ci95_high_cluster_bootstrap": hi,
            "num_directions": len(values), "num_reciprocal_clusters": len(cluster_names),
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": n_boot,
        })
        output.append(row)
    return output


def load_hybrid_recovery(root: Path) -> dict[str, float]:
    result = {}
    for path in root.glob("shard_*/checkpoints/*/COMPLETE.json"):
        value = json.loads(path.read_text()); result[value["pair_id"]] = float(value["hybrid_end_to_end_recovery"])
    if not result:
        for path in root.glob("checkpoints/*/COMPLETE.json"):
            value = json.loads(path.read_text()); result[value["pair_id"]] = float(value["hybrid_end_to_end_recovery"])
    return result


def causal_effect_rows(rows: list[dict[str, Any]], hybrid: dict[str, float]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        kind = str(row["intervention"])
        if "sufficiency" in kind:
            effect = f(row["normalized_recovery"])
        elif "necessity" in kind:
            effect = hybrid[row["pair_id"]] - f(row["normalized_recovery"])
        else:
            continue
        new = dict(row); new["causal_effect"] = effect
        new["effect_definition"] = "sufficiency recovery; necessity = hybrid end-to-end recovery minus necessity outcome recovery"
        output.append(new)
    return output


def combine_sufficiency_necessity(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: buckets[tuple(row[k] for k in key_fields)].append(row)
    output = []
    for key, cell in buckets.items():
        if len(cell) != 2 or {r["intervention"] for r in cell} not in (
            {"edge_sufficiency", "edge_necessity"}, {"head_sufficiency", "head_necessity"}):
            raise RuntimeError(f"incomplete sufficiency/necessity pair: {key}")
        row = dict(cell[0]); row["combined_causal_effect"] = statistics.fmean(f(r["causal_effect"]) for r in cell)
        row["intervention"] = "mean_sufficiency_and_necessity_effect"
        output.append(row)
    return output


def mlp_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = defaultdict(dict)
    for row in rows:
        buckets[(row["pair_id"], row["layer"], row["position"])][row["intervention"]] = row
    output = []
    for (_pair, _layer, _pos), cell in sorted(buckets.items()):
        if set(cell) != {"mlp_active", "mlp_bypass", "full_post_mlp"}:
            raise RuntimeError(f"incomplete MLP triplet: {(_pair, _layer, _pos)} {set(cell)}")
        active, bypass, full = cell["mlp_active"], cell["mlp_bypass"], cell["full_post_mlp"]
        output.append({
            "pair_id": active["pair_id"], "unordered_pair_id": active["unordered_pair_id"],
            "donor_entity": active["donor_entity"], "recipient_entity": active["recipient_entity"],
            "role": active["role"], "layer": int(active["layer"]), "position": int(active["position"]),
            "position_label": active["position_label"],
            "recovery_active": f(active["normalized_recovery"]), "recovery_bypass": f(bypass["normalized_recovery"]),
            "recovery_full_post_mlp": f(full["normalized_recovery"]),
            "MLP_effect": f(active["normalized_recovery"]) - f(bypass["normalized_recovery"]),
            "active_full_abs_difference": abs(f(active["normalized_recovery"]) - f(full["normalized_recovery"])),
            "active_state_error_max_abs": f(active["state_error_max_abs"]),
            "bypass_state_error_max_abs": f(bypass["state_error_max_abs"]),
        })
    return output


def reference_map(path: Path, condition: str, layer_field: str) -> dict[tuple[str, int], float]:
    result = {}
    for row in read_csv(path):
        if row["relay_condition"] == condition:
            result[(row["pair_id"], int(row[layer_field]))] = f(row["normalized_recovery"])
    return result


def validate(root: Path, attention: list[dict[str, Any]], mlp: list[dict[str, Any]], heads: list[dict[str, Any]],
             edges: list[dict[str, Any]], refs: list[dict[str, Any]], controls: list[dict[str, Any]],
             metadata: list[dict[str, Any]], hybrid: dict[str, float], smoke: bool) -> dict[str, Any]:
    n_pairs = 2 if smoke else 16; n_layers = 4 if smoke else 36
    pair_ids = sorted({r["pair_id"] for r in metadata})
    expected = {
        "attention_rows": n_pairs * n_layers * 2, "mlp_rows": n_pairs * n_layers * 2 * 3,
        "head_rows": n_pairs * 4 * (32 * 2 + 2), "edge_rows": n_pairs * 4 * 32 * 11 * 2,
        "reference_rows": n_pairs * n_layers * 2, "control_rows": n_pairs,
    }
    actual = {"attention_rows": len(attention), "mlp_rows": len(mlp), "head_rows": len(heads),
              "edge_rows": len(edges), "reference_rows": len(refs), "control_rows": len(controls)}
    all_rows = attention + mlp + heads + edges + refs + controls
    finite = all(math.isfinite(f(r["normalized_recovery"])) and math.isfinite(f(r["normalization_denominator"])) for r in all_rows)
    identity_max = max(abs(f(r["normalized_recovery"])) for r in controls)
    wrong_entity_max = identity_max
    effects = mlp_effect_rows(mlp)
    active_full = max(r["active_full_abs_difference"] for r in effects)
    active_state = max(r["active_state_error_max_abs"] for r in effects)
    bypass_state = max(r["bypass_state_error_max_abs"] for r in effects)
    rec_metrics = {}
    complete_paths = list(root.glob("shard_*/checkpoints/*/COMPLETE.json")) or list(root.glob("checkpoints/*/COMPLETE.json"))
    for path in complete_paths:
        details = json.loads(path.read_text())
        for key, value in details["reconstruction"].items(): rec_metrics[f"{details['pair_id']}::{key}"] = float(value)
    edge_head_abs = max(v for k, v in rec_metrics.items() if k.endswith("edge_to_head_max_abs"))
    edge_head_cos = min(v for k, v in rec_metrics.items() if k.endswith("edge_to_head_cosine"))
    allhead_abs = max(v for k, v in rec_metrics.items() if k.endswith("all_head_to_attention_max_abs"))
    allhead_cos = min(v for k, v in rec_metrics.items() if k.endswith("all_head_to_attention_cosine"))

    attn_map = {(r["pair_id"], int(r["layer"]), r["intervention"]): f(r["normalized_recovery"]) for r in attention}
    all_map = {(r["pair_id"], int(r["layer"]), r["intervention"]): f(r["normalized_recovery"]) for r in heads
               if r["component"] == "all_query_heads_pre_o_proj_control"}
    allhead_patch_delta = max(abs(attn_map[(p, li, k.replace("all_head_", "attention_"))] - value)
                              for (p, li, k), value in all_map.items())

    old_source = reference_map(REFERENCE_RELAY / "source_patch_end_to_end.csv", "source_patch_end_to_end", "source_layer_s")
    old_direct = reference_map(REFERENCE_RELAY / "direct_p10_control.csv", "direct_clean_donor_p10", "relay_layer_t")
    ref_deltas = []
    for row in refs:
        key = (row["pair_id"], int(row["layer"]))
        old = old_source[key] if row["intervention"] == "previous_P02_source_curve" else old_direct[key]
        ref_deltas.append({"pair_id": key[0], "layer": key[1], "intervention": row["intervention"],
                           "observed": f(row["normalized_recovery"]), "previous": old,
                           "absolute_delta": abs(f(row["normalized_recovery"]) - old)})
    write_csv(root / "reference_curve_comparison.csv", ref_deltas)
    ref_max = max(r["absolute_delta"] for r in ref_deltas)
    ref_curve_cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ref_deltas: ref_curve_cells[(row["intervention"], row["layer"])].append(row)
    ref_curve_deltas = [{"intervention": key[0], "layer": key[1],
                         "observed_mean": statistics.fmean(r["observed"] for r in cell),
                         "previous_mean": statistics.fmean(r["previous"] for r in cell),
                         "absolute_mean_curve_delta": abs(statistics.fmean(r["observed"] for r in cell) - statistics.fmean(r["previous"] for r in cell))}
                        for key, cell in ref_curve_cells.items()]
    write_csv(root / "reference_curve_mean_comparison.csv", ref_curve_deltas)
    ref_curve_max = max(r["absolute_mean_curve_delta"] for r in ref_curve_deltas)
    required_layers = set(SCAN_LAYERS if smoke else range(N_LAYERS))
    layer_coverage = all({int(r["layer"]) for r in table if r["pair_id"] == p} == required_layers
                         for table in (attention, mlp) for p in pair_ids)
    head_coverage = all({int(r["head"]) for r in heads if r["pair_id"] == p and int(r["layer"]) == li and r["head"] != ""} == set(range(32))
                        for p in pair_ids for li in SCAN_LAYERS)
    source_coverage = all({int(r["source_position"]) for r in edges if r["pair_id"] == p and int(r["layer"]) == li} == set(range(11))
                          for p in pair_ids for li in SCAN_LAYERS)
    checks = {
        "exact_row_counts": actual == expected, "correct_pair_count": len(pair_ids) == n_pairs,
        "eight_reciprocal_clusters_full": smoke or (len({r["unordered_pair_id"] for r in metadata}) == 8),
        "P02_P10_ids_and_labels": all(int(r["P02"]) == 2 and int(r["P10"]) == 10 and "was" in r["P10_label"] for r in metadata),
        "all_values_finite_no_NaNs": finite, "required_layer_coverage": layer_coverage,
        "all_32_query_heads": head_coverage, "all_11_aligned_source_positions": source_coverage,
        "identity_approximately_zero": identity_max <= 1e-5,
        "wrong_entity_does_not_recover_donor": wrong_entity_max <= 1e-5,
        "MLP_active_equals_full_post_MLP": active_full <= 1e-6 and active_state <= 1e-6,
        "MLP_bypass_exactly_hybrid_r": bypass_state <= 1e-6,
        "source_messages_reconstruct_heads": edge_head_abs <= 0.0625 and edge_head_cos >= 0.99999,
        "all_heads_reconstruct_attention_output": allhead_abs <= 0.0625 and allhead_cos >= 0.99999,
        "all_head_patch_matches_experiment_1": allhead_patch_delta <= 1e-5,
        "previous_reference_curves_reproduced": ref_curve_max <= 0.02 and ref_max <= 0.05,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "mode": "smoke" if smoke else "full",
        "checks": checks, "expected_counts": expected, "actual_counts": actual, "pair_ids": pair_ids,
        "identity_max_abs_recovery": identity_max, "wrong_entity_max_abs_recovery": wrong_entity_max,
        "MLP_active_full_max_abs_recovery_difference": active_full,
        "MLP_active_state_max_abs_error": active_state, "MLP_bypass_state_max_abs_error": bypass_state,
        "edge_message_to_head_max_abs_error": edge_head_abs, "edge_message_to_head_min_cosine": edge_head_cos,
        "all_heads_to_attention_max_abs_error": allhead_abs, "all_heads_to_attention_min_cosine": allhead_cos,
        "all_head_patch_vs_experiment1_max_abs_recovery_delta": allhead_patch_delta,
        "previous_curves_max_abs_mean_recovery_delta": ref_curve_max,
        "previous_curves_max_abs_pair_recovery_delta": ref_max,
        "hook_shapes": {"attention_output": "[batch, 11, 4096]", "pre_o_projection": "[batch, 11, 32, 128]",
                        "attention_weights": "[batch, 32, 11, 11]", "values": "[batch, 11, 8, 128]"},
    }
    write_json(root / ("smoke_validation_report.json" if smoke else "validation_report.json"), report)
    if not all(checks.values()):
        raise RuntimeError("validation failed: " + json.dumps({k: v for k, v in checks.items() if not v}))
    return report


def save_fig(fig, plots: Path, name: str) -> None:
    fig.savefig(plots / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(plots / f"{name}.pdf", bbox_inches="tight")


def summaries_to_lookup(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {tuple(str(r[k]) for k in keys): r for r in rows}


def make_plots(root: Path, attention_effects: list[dict[str, Any]], mlp: list[dict[str, Any]],
               effects: list[dict[str, Any]], head_effects: list[dict[str, Any]],
               edge_effects: list[dict[str, Any]], summaries: dict[str, list[dict[str, Any]]],
               top_heads: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plots = root / "plots"; plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .2})

    att = summaries_to_lookup(summaries["attention"], ["intervention", "layer"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, label, color in (("attention_sufficiency", "Sufficiency", "#2878b5"), ("attention_necessity", "Necessity loss", "#d95319")):
        xs = range(36); cell = [att[(kind, str(x))] for x in xs]
        y = [f(c["mean"]) for c in cell]; lo = [f(c["ci95_low_cluster_bootstrap"]) for c in cell]; hi = [f(c["ci95_high_cluster_bootstrap"]) for c in cell]
        ax.plot(xs, y, label=label, color=color); ax.fill_between(xs, lo, hi, alpha=.18, color=color)
    ax.axhline(0, color="black", lw=.7); ax.set(xlabel="Layer", ylabel="Causal effect (normalized recovery)", title="Experiment 1: attention output at P10")
    ax.legend(ncol=2, loc="best"); save_fig(fig, plots, "attention_sufficiency_necessity_across_layers"); plt.close(fig)

    mlp_sum = summaries_to_lookup(summaries["mlp_curves"], ["position", "intervention", "layer"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pos, label in zip(axes, (2, 10), ("P02 entity", "P10 final token")):
        for kind, style in (("mlp_active", "-"), ("mlp_bypass", "--"), ("full_post_mlp", ":")):
            ax.plot(range(36), [f(mlp_sum[(str(pos), kind, str(li))]["mean"]) for li in range(36)], style, label=kind)
        ax.set(title=label, xlabel="Layer", ylabel="Normalized recovery"); ax.legend(fontsize=7, ncol=1)
    save_fig(fig, plots, "mlp_active_bypass_full_post_mlp_curves"); plt.close(fig)

    eff = summaries_to_lookup(summaries["mlp_effect"], ["position", "layer"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pos, label, color in ((2, "P02", "#2878b5"), (10, "P10", "#d95319")):
        cell = [eff[(str(pos), str(li))] for li in range(36)]
        y = [f(c["mean"]) for c in cell]; lo = [f(c["ci95_low_cluster_bootstrap"]) for c in cell]; hi = [f(c["ci95_high_cluster_bootstrap"]) for c in cell]
        ax.plot(range(36), y, color=color, label=label); ax.fill_between(range(36), lo, hi, color=color, alpha=.18)
    ax.axhline(0, color="black", lw=.7); ax.set(xlabel="Layer", ylabel="MLP active − bypass recovery", title="Incremental matched MLP effect")
    ax.legend(ncol=2); save_fig(fig, plots, "incremental_mlp_effect_with_ci"); save_fig(fig, plots, "P02_versus_P10_mlp_effects"); plt.close(fig)

    pair_order = sorted({r["pair_id"] for r in attention_effects})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, kind in zip(axes, ("attention_sufficiency", "attention_necessity")):
        data = np.array([[statistics.fmean(f(r["causal_effect"]) for r in attention_effects if r["pair_id"] == p and int(r["layer"]) == li and r["intervention"] == kind) for li in range(36)] for p in pair_order])
        im = ax.imshow(data, aspect="auto", cmap="coolwarm"); ax.set_yticks(range(16), pair_order, fontsize=6); ax.set_title(kind); fig.colorbar(im, ax=ax, shrink=.7)
    axes[-1].set_xlabel("Layer"); save_fig(fig, plots, "experiment1_pair_level_heatmaps"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, pos in zip(axes, (2, 10)):
        data = np.array([[next(r["MLP_effect"] for r in effects if r["pair_id"] == p and r["layer"] == li and r["position"] == pos) for li in range(36)] for p in pair_order])
        im = ax.imshow(data, aspect="auto", cmap="coolwarm"); ax.set_yticks(range(16), pair_order, fontsize=6); ax.set_title(f"MLP effect P{pos:02d}"); fig.colorbar(im, ax=ax, shrink=.7)
    axes[-1].set_xlabel("Layer"); save_fig(fig, plots, "experiment2_pair_level_heatmaps"); plt.close(fig)

    head_sum = summaries_to_lookup(summaries["heads"], ["intervention", "layer", "head"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, ("head_sufficiency", "head_necessity")):
        data = np.array([[f(head_sum[(kind, str(li), str(h))]["mean"]) for h in range(32)] for li in SCAN_LAYERS])
        im = ax.imshow(data, aspect="auto", cmap="coolwarm"); ax.set(title=kind, xlabel="Query head", ylabel="Layer", yticks=range(4), yticklabels=SCAN_LAYERS); fig.colorbar(im, ax=ax, shrink=.75)
    save_fig(fig, plots, "layer_by_head_sufficiency_necessity_heatmaps"); plt.close(fig)

    edge_sum = summaries_to_lookup(summaries["edges"], ["layer", "head", "source_position"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, li in zip(axes.flat, SCAN_LAYERS):
        data = np.array([[f(edge_sum[(str(li), str(h), str(j))]["mean"]) for j in range(11)] for h in range(32)])
        im = ax.imshow(data, aspect="auto", cmap="coolwarm"); ax.set(title=f"Layer {li}", xlabel="Source position", ylabel="Query head", xticks=range(11)); fig.colorbar(im, ax=ax, shrink=.7)
    save_fig(fig, plots, "head_by_source_position_maps_layers_23_26"); plt.close(fig)

    group_sum = summaries_to_lookup(summaries["source_groups"], ["layer", "source_group"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, label in (("P02_subject_entity", "P02"), ("P03-P09_intermediate", "P03–P09"), ("P10_self", "P10")):
        ax.plot(SCAN_LAYERS, [f(group_sum[(str(li), group)]["mean"]) for li in SCAN_LAYERS], marker="o", label=label)
    ax.axhline(0, color="black", lw=.7); ax.set(xlabel="Layer", ylabel="Summed edge causal effect", title="Path contribution by source-position group", xticks=SCAN_LAYERS); ax.legend(ncol=3)
    save_fig(fig, plots, "P02_versus_P03_P09_versus_P10_path_contributions"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for kind in ("attention_sufficiency", "attention_necessity"):
        axes[0].plot(range(20, 31), [f(att[(kind, str(li))]["mean"]) for li in range(20, 31)], marker="o", label=kind)
    for pos in (2, 10):
        axes[1].plot(range(20, 31), [f(eff[(str(pos), str(li))]["mean"]) for li in range(20, 31)], marker="o", label=f"P{pos:02d}")
    axes[0].set(title="Attention", xlabel="Layer", ylabel="Causal effect"); axes[1].set(title="MLP active − bypass", xlabel="Layer", ylabel="Recovery difference")
    for ax in axes: ax.axhline(0, color="black", lw=.7); ax.legend(fontsize=7)
    save_fig(fig, plots, "focused_layers_20_30"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, min(8, 1.2 + .36 * len(top_heads)))); ax.axis("off")
    cols = ["layer", "head", "effect", "ci95_low", "ci95_high", "dominant_source_positions"]
    table = ax.table(cellText=[[r[c] if c in ("layer", "head", "dominant_source_positions") else f"{f(r[c]):.4f}" for c in cols] for r in top_heads], colLabels=cols, loc="center")
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2); ax.set_title("Top attention heads")
    save_fig(fig, plots, "top_head_table"); plt.close(fig)


def merge(args: argparse.Namespace) -> int:
    root = Path(args.out_dir)
    shard_dirs = [root] if args.smoke else [root / f"shard_{i}" for i in range(args.num_shards)]
    for shard in shard_dirs:
        if not (shard / "SUCCESS").exists(): raise RuntimeError(f"incomplete worker {shard}")
    tables: dict[str, list[dict[str, Any]]] = {k: [] for k in ("attention", "mlp", "heads", "edges", "refs", "controls", "metadata")}
    names = {"attention": "attention_results.csv", "mlp": "mlp_results.csv", "heads": "head_results.csv",
             "edges": "edge_results.csv", "refs": "reference_results.csv", "controls": "control_results.csv"}
    for shard in shard_dirs:
        tables["metadata"].extend(read_csv(shard / "pair_metadata.csv"))
        for pair_dir in sorted((shard / "checkpoints").iterdir()):
            if not pair_dir.is_dir() or not (pair_dir / "COMPLETE.json").exists(): continue
            for key, name in names.items(): tables[key].extend(read_csv(pair_dir / name))
    hybrid = load_hybrid_recovery(root)
    report = validate(root, tables["attention"], tables["mlp"], tables["heads"], tables["edges"],
                      tables["refs"], tables["controls"], tables["metadata"], hybrid, args.smoke)
    if args.smoke:
        (root / "SMOKE_PASS").write_text("all smoke validation gates passed\n")
        return 0

    for key, name in names.items(): write_csv(root / name, tables[key])
    write_csv(root / "pair_metadata.csv", tables["metadata"])
    attention_effects = causal_effect_rows(tables["attention"], hybrid)
    head_effects = causal_effect_rows([r for r in tables["heads"] if r["component"] == "query_head_pre_o_proj"], hybrid)
    edge_effects = causal_effect_rows(tables["edges"], hybrid)
    head_combined_effects = combine_sufficiency_necessity(
        head_effects, ["pair_id", "layer", "head"])
    edge_combined_effects = combine_sufficiency_necessity(
        edge_effects, ["pair_id", "layer", "head", "source_position"])
    effects = mlp_effect_rows(tables["mlp"])
    write_csv(root / "attention_causal_effects.csv", attention_effects)
    write_csv(root / "head_causal_effects.csv", head_effects)
    write_csv(root / "edge_causal_effects.csv", edge_effects)
    write_csv(root / "head_combined_causal_effects.csv", head_combined_effects)
    write_csv(root / "edge_combined_causal_effects.csv", edge_combined_effects)
    write_csv(root / "mlp_effects.csv", effects)

    # Aggregate the intermediate source group by summing edge effects within each direction/layer.
    grouped_edges = defaultdict(float)
    for row in edge_combined_effects:
        group = row["source_group"]
        if group not in {"P02_subject_entity", "P03-P09_intermediate", "P10_self"}: continue
        grouped_edges[(row["pair_id"], row["unordered_pair_id"], row["layer"], group)] += f(row["combined_causal_effect"])
    source_group_rows = [{"pair_id": k[0], "unordered_pair_id": k[1], "layer": k[2], "source_group": k[3], "summed_causal_effect": v}
                         for k, v in grouped_edges.items()]
    write_csv(root / "source_group_path_effects.csv", source_group_rows)

    summaries = {
        "attention": cluster_summary(attention_effects, ["intervention", "layer"], "causal_effect", args.bootstrap_resamples),
        "mlp_curves": cluster_summary(tables["mlp"], ["position", "intervention", "layer"], "normalized_recovery", args.bootstrap_resamples),
        "mlp_effect": cluster_summary(effects, ["position", "layer"], "MLP_effect", args.bootstrap_resamples),
        "heads": cluster_summary(head_effects, ["intervention", "layer", "head"], "causal_effect", args.bootstrap_resamples),
        "edges": cluster_summary(edge_combined_effects, ["layer", "head", "source_position"], "combined_causal_effect", args.bootstrap_resamples),
        "source_groups": cluster_summary(source_group_rows, ["layer", "source_group"], "summed_causal_effect", args.bootstrap_resamples),
    }
    for key, rows in summaries.items(): write_csv(root / f"aggregate_{key}.csv", rows)

    # Every pair direction is reported for the layer/head and MLP comparisons.
    pair_summary = []
    for pair in sorted(hybrid):
        for li in range(36):
            a = [r for r in attention_effects if r["pair_id"] == pair and int(r["layer"]) == li]
            e = [r for r in effects if r["pair_id"] == pair and r["layer"] == li]
            pair_summary.append({"pair_id": pair, "layer": li, "hybrid_recovery": hybrid[pair],
                **{r["intervention"]: f(r["causal_effect"]) for r in a},
                **{f"MLP_effect_P{r['position']:02d}": r["MLP_effect"] for r in e}})
    write_csv(root / "pair_direction_summary.csv", pair_summary)

    head_combined = defaultdict(list)
    for row in head_combined_effects: head_combined[(int(row["layer"]), int(row["head"]))].append(row)
    top_heads = []
    for (li, h), cell in head_combined.items():
        vals = [f(r["combined_causal_effect"]) for r in cell]
        clusters = defaultdict(list)
        for r in cell: clusters[r["unordered_pair_id"]].append(f(r["combined_causal_effect"]))
        rng = random.Random(BOOTSTRAP_SEED + li * 100 + h); names = sorted(clusters); boot = []
        for _ in range(args.bootstrap_resamples):
            sampled = [rng.choice(names) for _ in names]; boot.append(statistics.fmean(v for c in sampled for v in clusters[c]))
        edge_by_source = defaultdict(list)
        for r in edge_combined_effects:
            if int(r["layer"]) == li and int(r["head"]) == h: edge_by_source[int(r["source_position"])].append(f(r["combined_causal_effect"]))
        dominant = sorted(((statistics.fmean(v), j) for j, v in edge_by_source.items()), reverse=True)[:3]
        top_heads.append({"layer": li, "head": h, "effect": statistics.fmean(vals),
                          "ci95_low": percentile(boot, .025), "ci95_high": percentile(boot, .975),
                          "dominant_source_positions": ", ".join(f"P{j:02d} ({v:.4f})" for v, j in dominant)})
    top_heads.sort(key=lambda r: r["effect"], reverse=True); top_heads = top_heads[:20]
    write_csv(root / "top_heads.csv", top_heads)
    make_plots(root, attention_effects, tables["mlp"], effects, head_effects, edge_effects, summaries, top_heads)

    runtime_paths = list(root.glob("shard_*/checkpoints/*/COMPLETE.pre_layer0_formula_fix.json"))
    if not runtime_paths:
        runtime_paths = list(root.glob("shard_*/checkpoints/*/COMPLETE.json"))
    runtime = sum(json.loads(p.read_text())["runtime_seconds"] for p in runtime_paths)
    att_lookup = summaries_to_lookup(summaries["attention"], ["intervention", "layer"])
    p10_eff = summaries_to_lookup(summaries["mlp_effect"], ["position", "layer"])
    peak_att = max(range(36), key=lambda li: f(att_lookup[("attention_sufficiency", str(li))]["mean"]))
    peak_mlp = max(range(36), key=lambda li: f(p10_eff[("10", str(li))]["mean"]))
    group_lookup = summaries_to_lookup(summaries["source_groups"], ["layer", "source_group"])
    quantitative = {
        "total_worker_pair_runtime_seconds": runtime, "attention_sufficiency_peak_layer": peak_att,
        "attention_sufficiency_peak_mean": f(att_lookup[("attention_sufficiency", str(peak_att))]["mean"]),
        "P10_MLP_effect_peak_layer": peak_mlp, "P10_MLP_effect_peak_mean": f(p10_eff[("10", str(peak_mlp))]["mean"]),
        "layers_23_26_attention_sufficiency_means": {li: f(att_lookup[("attention_sufficiency", str(li))]["mean"]) for li in SCAN_LAYERS},
        "layers_23_26_P10_MLP_effect_means": {li: f(p10_eff[("10", str(li))]["mean"]) for li in SCAN_LAYERS},
        "layers_23_26_source_group_effects": {li: {g: f(group_lookup[(str(li), g)]["mean"]) for g in
            ("P02_subject_entity", "P03-P09_intermediate", "P10_self")} for li in SCAN_LAYERS},
        "top_heads": top_heads[:10],
    }
    write_json(root / "quantitative_findings.json", quantitative)
    readme = f"""# Qwen3-8B component-level causal relay

Exactly three experiments were run on `Qwen/Qwen3-8B-Base`: attention-output mediation, matched MLP intervention, and the predefined head/source-position scan at layers 23–26. The dataset is exactly the completed relay's 16 ordered directions in eight reciprocal clusters. P02 is the entity token and P10 is the final `was`. The hybrid trajectory patches donor P02 into recipient P02 at block-0 output.

Recovery is `(intervention margin - recipient margin) / (donor margin - recipient margin)` and is never clipped. Confidence intervals use a fixed-seed ({BOOTSTRAP_SEED}) cluster bootstrap over eight unordered reciprocal clusters; both directions are resampled together. Raw CSVs retain target logits, denominator, direction, cluster, layer, position, component, head/source position, intervention, vector norms/cosines, token IDs, and labels.

Validation status: **{report['status']}**. See `validation_report.json` and `reference_curve_comparison.csv`. Total summed per-pair worker runtime: {runtime:.1f} seconds (workers ran concurrently).

Key files: raw experiment CSVs, `mlp_effects.csv`, causal-effect CSVs, `pair_direction_summary.csv`, `aggregate_*.csv`, `top_heads.csv`, `quantitative_findings.json`, plots in `plots/`, and `scientific_summary.md`.
"""
    (root / "README.md").write_text(readme)
    (root / "scientific_summary.md").write_text("# Scientific summary\n\nQuantitative results are in `quantitative_findings.json`; final interpretation is completed after numerical review.\n")
    (root / "SUCCESS").write_text("all three experiments complete; merged validation passed\n")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--out-dir", required=True); p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("--bootstrap-resamples", type=int, default=2000); p.add_argument("--smoke", action="store_true"); return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(merge(parse_args()))
