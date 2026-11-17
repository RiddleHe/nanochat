"""Verify weighted separate-source records and rebuild means/std on CPU without plotting."""
import argparse
import json
from pathlib import Path

import numpy as np

from .prepare import sha256, write_json


def summarize(run, manifest=None):
    run = Path(run)
    manifest = manifest or json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    metadata = json.loads((run / "inputs/checkpoint_metadata.json").read_text())["model_config"]
    depth, length = metadata["n_layer"], config["sequence_length"]
    manifest_hash = sha256(run / "manifest.json")
    assert sha256(run / "inputs/tokens.npy") == manifest["tokens_sha256"]
    assert json.loads((run / "inputs/source_index.json").read_text()) == manifest["source_index"]
    assert len(list((run / "tokens").glob("chunk_*.json"))) == config["num_sequences"]
    sums, squares, counts, maxima = {}, {}, {}, {}
    reference = json.loads((run / "inputs/routing_reference.json").read_text())
    assert sha256(run / "inputs/routing_reference.json") == manifest["routing_reference_sha256"]

    def accumulate(key, values):
        values = values.astype(np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid measurement: {key}")
        sums[key] = sums.get(key, 0) + values.sum(0)
        squares[key] = squares.get(key, 0) + np.square(values).sum(0)
        counts[key] = counts.get(key, 0) + len(values)

    for index in range(config["num_sequences"]):
        path = run / "tokens" / f"chunk_{index:05d}.npz"
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["sequence_id"] == index and record["tokens"] == length
        assert record["manifest_sha256"] == manifest_hash and record["tokens_sha256"] == manifest["tokens_sha256"]
        assert sha256(path) == record["sha256"]
        parent_path = Path(reference["path"]) / "tokens" / path.name
        assert sha256(parent_path) == reference["records"][index]["sha256"]
        with np.load(path) as chunk, np.load(parent_path) as parent_weights:
            assert len(chunk.files) == 2 + depth * 9
            raw, denominator = chunk["source_norm"], chunk["input_denominator"]
            assert raw.shape == (length, 2 * depth + 1) and denominator.shape == (length, depth, 3)
            assert (denominator > 0).all()
            accumulate("source_norm", raw)
            accumulate("input_denominator", denominator)
            for layer in range(depth):
                for qi, kind in enumerate("QKV"):
                    input_norm = chunk[f"input_{kind}_{layer:02d}"]
                    assert input_norm.shape == (length, 2*layer+1)
                    expected_input = raw[:, :2*layer+1] * parent_weights[f"weights_{layer:02d}"][:,qi,:] / denominator[:,layer,qi:qi+1]
                    np.testing.assert_allclose(input_norm, expected_input, rtol=3e-6, atol=1e-7)
                    accumulate(f"input_{kind}_{layer:02d}", input_norm)
                    key = f"abs_{kind}_{layer:02d}"
                    heads = metadata["n_head"] if kind == "Q" else metadata["n_kv_head"]
                    absolute, share = chunk[key], chunk[key.replace("abs_", "share_", 1)]
                    assert absolute.shape == share.shape == (length, heads, 2*layer+1)
                    total = absolute.sum(-1, keepdims=True)
                    assert (total > 0).all()
                    np.testing.assert_allclose(share, absolute / total, atol=2e-7, rtol=2e-6)
                    error = float(np.abs(share.astype(np.float64).sum(-1) - 1).max())
                    maxima["saved_share_sum_error"] = max(maxima.get("saved_share_sum_error", 0), error)
                    assert error <= config["validation"]["share_sum_max_error"]
                    accumulate(key, absolute)
                    accumulate(key.replace("abs_", "share_", 1), share)
                    # Full output norm per token, then token statistics.
                    accumulate(f"full_{kind}_{layer:02d}", np.sqrt(np.square(absolute.astype(np.float64)).sum(1)))
        validation = json.loads((run / "validation" / path.with_suffix(".json").name).read_text())
        assert sha256(run / "validation" / path.with_suffix(".json").name) == record["validation_sha256"]
        assert validation["passed"] and validation["exact_forward_replay"] and len(validation["checks"]) == depth * 3
        for check in validation["checks"]:
            assert check["exact_parent_weights"] and check["exact_bf16_mixture_replay"] and check["source_identity_passed"]
            for key, value in check.items():
                if key.endswith("relative_rms") or key.endswith("error"):
                    maxima[key] = max(maxima.get(key, 0), value)
        if (index + 1) % 16 == 0:
            print(f"Verified and summarized {index + 1}/{config['num_sequences']} chunks", flush=True)
    arrays = {}
    for key in sums:
        assert counts[key] == length * config["num_sequences"]
        mean = sums[key] / counts[key]
        arrays[key + "_token_mean"] = mean
        arrays[key + "_token_std"] = np.sqrt(np.maximum(squares[key] / counts[key] - mean**2, 0))
        if key.startswith(("abs_", "share_")):
            for name, value in (("head_mean", mean.mean(0)), ("head_std", mean.std(0)),
                                ("head_min", mean.min(0)), ("head_max", mean.max(0))):
                arrays[key + "_" + name] = value
    path = run / "summaries/statistics.npz"
    with path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez(handle, **arrays)
    path.with_suffix(".npz.tmp").replace(path)
    tokens = length * config["num_sequences"]
    write_json(run / "summaries/schema.json", dict(source_index=manifest["source_index"],
        available_sources="source_index[:2*receiving_layer+1]", input="input_{Q|K|V}_{layer}: [source]; distinct query weights and denominators",
        abs_and_share="[actual_heads,source]", full="[source]", std_ddof=0,
        full_norm="sqrt(sum(head_norm**2)) for each token before mean/std", tokens=tokens))
    result = dict(passed=True, maxima=maxima, tokens=tokens,
        chunks=config["num_sequences"], chunks_checksum_verified=True, relative_shares_recomputed=True,
        counts=counts, std_ddof=0, exact_parent_weights=True, all_sequences_exact_forward_replay=True,
        all_layers_exact_bf16_mixture_replay=True, input_norms_independently_recomputed=True, visualization_status="held_by_user")
    write_json(run / "validation/summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    summarize(parser.parse_args().run_dir)
