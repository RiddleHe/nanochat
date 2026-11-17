"""Rebuild full-bank and source-family routing statistics from verified chunks."""
import argparse
import json
from pathlib import Path

import numpy as np

from .prepare import sha256, write_json


def summarize(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    sources = json.loads((run / "inputs/source_index.json").read_text())
    assert sources == manifest["source_index"]
    manifest_hash = sha256(run / "manifest.json")
    assert sha256(run / "inputs/tokens.npy") == manifest["tokens_sha256"]
    depth = json.loads((run / "inputs/checkpoint_metadata.json").read_text())["model_config"]["n_layer"]
    assert len(list((run / "tokens").glob("chunk_*.json"))) == config["num_sequences"]
    sums, squares, maxima = {}, {}, {}
    count = 0

    def add(key, values):
        values = values.astype(np.float64)
        sums[key] = sums.get(key, 0) + values.sum(0)
        squares[key] = squares.get(key, 0) + np.square(values).sum(0)

    for index in range(config["num_sequences"]):
        path = run / "tokens" / f"chunk_{index:05d}.npz"
        record = json.loads(path.with_suffix(".json").read_text())
        validation_path = run / "validation" / path.with_suffix(".json").name
        assert record["sequence_id"] == index and record["tokens"] == config["sequence_length"]
        assert record["sha256"] == sha256(path) and record["manifest_sha256"] == manifest_hash
        assert record["tokens_sha256"] == manifest["tokens_sha256"]
        assert record["validation_sha256"] == sha256(validation_path)
        validation = json.loads(validation_path.read_text())
        assert validation["passed"] and validation["exact_forward_replay"]
        assert len(validation["checks"]) == depth
        for check in validation["checks"]:
            assert check["exact_bf16_mixture_replay"] and check["source_identity_passed"]
            for key, value in check.items():
                if key.endswith("error") or key.endswith("relative_rms"):
                    maxima[key] = max(maxima.get(key, 0), value)
        with np.load(path) as chunk:
            assert sorted(chunk.files) == [f"weights_{layer:02d}" for layer in range(depth)]
            for layer in range(depth):
                key = f"weights_{layer:02d}"
                weights = chunk[key]
                assert weights.dtype == np.float32 and weights.shape == (config["sequence_length"], 3, 2*layer+1)
                assert np.isfinite(weights).all() and (weights >= 0).all() and (weights <= 1).all()
                assert ((weights.view(np.uint32) & 0xffff) == 0).all(), "Weights not lossless BF16 values"
                error = float(np.abs(weights.astype(np.float64).sum(-1)-1).max())
                assert error <= config["validation"]["applied_weight_sum_max_error"]
                maxima["saved_weight_sum_error"] = max(maxima.get("saved_weight_sum_error", 0), error)
                add(key, weights)
                family_totals = []
                for family in ("embedding", "attention", "mlp"):
                    indices = [s["index"] for s in sources[:2*layer+1] if s["source_family"] == family]
                    total = weights[:, :, indices].astype(np.float64).sum(-1)
                    add(f"{family}_mass_{layer:02d}", total)
                    family_totals.append(total)
                np.testing.assert_allclose(sum(family_totals), weights.astype(np.float64).sum(-1), atol=1e-12, rtol=0)
        count += config["sequence_length"]
    stats = {}
    for key in sums:
        mean = sums[key] / count
        stats[key+"_token_mean"] = mean
        stats[key+"_token_std"] = np.sqrt(np.maximum(squares[key]/count-mean**2, 0))
    np.savez(run / "summaries/statistics.npz", **stats)
    cells, rows = [], []
    for layer in range(depth):
        for query_index, kind in enumerate(config["query_types"]):
            rows.append(dict(layer=layer, kind=kind,
                attention_mass_mean=float(stats[f"attention_mass_{layer:02d}_token_mean"][query_index]),
                attention_mass_std=float(stats[f"attention_mass_{layer:02d}_token_std"][query_index])))
            for source in sources[:2*layer+1]:
                if source["source_family"] == "attention":
                    cells.append(dict(layer=layer, kind=kind, source_layer=source["source_layer"],
                        mean=float(stats[f"weights_{layer:02d}_token_mean"][query_index,source["index"]]),
                        std=float(stats[f"weights_{layer:02d}_token_std"][query_index,source["index"]]), count=count))
    write_json(run / "summaries/attention_heatmaps.json", dict(run_id=manifest["run_id"], tokens=count,
        cells=cells, rows=rows, source_family="attention", renormalized=False, std_ddof=0,
        measurement=config["measurement"]))
    write_json(run / "summaries/schema.json", dict(source_index=sources, query_types=config["query_types"],
        weight_shape="[3,2*receiving_layer+1]", family_mass_shape="[3]", std_ddof=0,
        family_mass_std="std_of_per_token_family_total_not_sum_of_source_stds"))
    result = dict(passed=True, chunks_verified=config["num_sequences"], tokens=count, maxima=maxima,
                  all_sequences_exact_forward_replay=True, all_layers_exact_bf16_mixture_replay=True,
                  lossless_bf16_weights_verified=True, attention_heatmap_cells=len(cells))
    write_json(run / "validation/summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    summarize(parser.parse_args().run_dir)
