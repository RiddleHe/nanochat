"""Rebuild all statistics from verified, completed per-token chunks (CPU only)."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare import sha256, write_json


def summarize(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    records = sorted((run / "tokens").glob("chunk_*.json"))
    expected = manifest["config"]["num_sequences"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} completed chunks, found {len(records)}")
    sums, squares, counts = {}, {}, {}
    worst_share_error = 0.0
    for index, record_path in enumerate(records):
        record = json.loads(record_path.read_text())
        if record["sequence_id"] != index:
            raise ValueError("Chunk sequence IDs are not contiguous")
        data_path = run / "tokens" / record["file"]
        if sha256(data_path) != record["sha256"]:
            raise ValueError(f"Chunk hash mismatch: {data_path}")
        with np.load(data_path) as chunk:
            for key in chunk.files:
                values = chunk[key].astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"Undefined/nonfinite measurements: {data_path}:{key}")
                sums.setdefault(key, np.zeros(values.shape[1:], dtype=np.float64))
                squares.setdefault(key, np.zeros(values.shape[1:], dtype=np.float64))
                counts.setdefault(key, 0)
                sums[key] += values.sum(axis=0)
                squares[key] += np.square(values).sum(axis=0)
                counts[key] += len(values)
                if key.startswith("share_"):
                    error = np.max(np.abs(values.sum(axis=-1) - 1))
                    worst_share_error = max(worst_share_error, float(error))
                    absolute = chunk[key.replace("share_", "abs_", 1)]
                    np.testing.assert_allclose(values, absolute / absolute.sum(axis=-1, keepdims=True), rtol=2e-6, atol=2e-7)
        if (index + 1) % 16 == 0 or index == 0:
            print(f"Summarized {index + 1}/{expected} chunks", flush=True)
    arrays, cells = {}, []
    for key in sorted(sums):
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(squares[key] / counts[key] - mean * mean, 0))
        arrays[key + "_token_mean"] = mean
        arrays[key + "_token_std"] = std
        if not key.startswith("abs_"):
            continue
        _, kind, layer = key.split("_")
        for name, value in (("head_mean", mean.mean(0)), ("head_std", mean.std(0)),
                            ("head_min", mean.min(0)), ("head_max", mean.max(0))):
            arrays[key + "_" + name] = value
        for source in range(mean.shape[1]):
            cells.append({"kind": kind, "layer": int(layer), "source": source - 1,
                          "mean": float(mean[:, source].mean()),
                          "std": float(mean[:, source].std()),
                          "min": float(mean[:, source].min()), "max": float(mean[:, source].max()),
                          "heads": mean[:, source].tolist(),
                          "token_std": std[:, source].tolist(), "count": counts[key]})
    np.savez(run / "summaries/statistics.npz", **arrays)
    write_json(run / "summaries/heatmaps.json", {"run_id": manifest["run_id"],
               "tokens": expected * manifest["config"]["sequence_length"],
               "sequences": expected, "measurement": manifest["config"]["projection_location"],
               "cells": cells})
    limit = manifest["config"]["validation"]["share_sum_max_error"]
    if worst_share_error > limit:
        raise ValueError(f"Share sum error {worst_share_error} exceeds {limit}")
    write_json(run / "validation/summary.json", {
        "passed": True, "chunks_verified": len(records), "share_sum_max_error": worst_share_error,
        "relative_shares_recomputed_from_absolute_norms": True,
        "summaries_rebuilt_from_all_saved_token_records": True,
        "token_counts": counts, "std_ddof": 0})
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    summarize(parser.parse_args().run_dir)
