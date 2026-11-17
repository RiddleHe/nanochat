"""Replay a frozen QKV run and collect norms of its pre-projection components."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import build_model
from nanochat.flash_attention import USE_FA3
from nanochat.model.gpt_base import norm
from .collect import capture_forward, now, relative_rms
from .prepare import sha256, write_json


@torch.inference_mode()
def measure(model, ids, reference, config):
    captured = capture_forward(model, ids)
    sources = [captured["input"][0].float()]
    sources.extend(captured["attn"][layer].float() + captured["mlp"][layer].float()
                   for layer in range(len(model.transformer.h)))
    source_norm = torch.stack([source.norm(dim=-1).flatten() for source in sources], -1)
    denominators, checks = [], []
    for layer, block in enumerate(model.transformer.h):
        actual = captured["input"][layer]
        denominator = (actual.float().square().mean(-1, keepdim=True) + torch.finfo(actual.dtype).eps).sqrt()
        denominators.append(denominator.flatten())
        source_sum = torch.stack(sources[:layer + 1]).sum(0)
        residual_error = relative_rms(source_sum, actual)
        normalization_error = relative_rms(actual.float() / denominator, norm(actual))
        assert residual_error <= config["validation"]["residual_relative_rms_max"]
        assert normalization_error <= config["validation"]["normalization_relative_rms_max"]
        check = dict(layer=layer, residual_relative_rms=residual_error,
                     normalization_relative_rms=normalization_error)
        if layer in config["replay_layers"]:
            # Verify every token/source/head against saved Q/K/V norms.
            # Source groups match the original analysis's batching convention.
            for kind in "QKV":
                weight = getattr(block.attn, f"c_{kind.lower()}").weight.to(actual.dtype).float()
                pieces, full_pieces = [], []
                for start in range(0, layer + 1, config["source_batch_size"]):
                    group = torch.stack(sources[start:min(start + config["source_batch_size"], layer + 1)]) / denominator
                    projected = F.linear(group, weight)
                    shaped = projected.reshape(len(group), -1, block.attn.n_head, block.attn.head_dim)
                    pieces.append(shaped.norm(dim=-1).permute(1, 2, 0))
                    full_pieces.append(projected.norm(dim=-1).reshape(len(group), -1).T)
                replay = torch.cat(pieces, -1)
                saved = torch.from_numpy(reference[f"abs_{kind}_{layer:02d}"].copy()).to(replay.device)
                error = relative_rms(replay, saved)
                assert error <= config["replay_relative_rms_max"], (layer, kind, error)
                check[f"replay_{kind}_relative_rms"] = error
                reconstructed_full = saved.double().square().sum(1).sqrt()
                full_error = relative_rms(torch.cat(full_pieces, -1), reconstructed_full)
                assert full_error <= config["replay_relative_rms_max"]
                check[f"full_norm_{kind}_relative_rms"] = full_error
        checks.append(check)
    arrays = dict(source_norm=source_norm.cpu().numpy(),
                  input_denominator=torch.stack(denominators, -1).cpu().numpy())
    assert all(np.isfinite(value).all() and (value > 0).all() for value in arrays.values())
    return arrays, checks


def summarize(run, manifest):
    tokens = np.load(run / "inputs/tokens.npy", mmap_mode="r")
    totals, squares, count = {}, {}, 0
    maxima = {}
    for sequence_id in range(len(tokens)):
        path = run / "tokens" / f"chunk_{sequence_id:05d}.npz"
        marker = json.loads(path.with_suffix(".json").read_text())
        assert marker["sha256"] == sha256(path)
        assert marker["tokens_sha256"] == manifest["tokens_sha256"]
        assert marker["manifest_sha256"] == sha256(run / "manifest.json")
        with np.load(path) as chunk:
            raw, denominator = chunk["source_norm"], chunk["input_denominator"]
            assert raw.shape == (tokens.shape[1], 25) and denominator.shape == (tokens.shape[1], 24)
            values = {"source_norm": raw, "input_denominator": denominator}
            for layer in range(24):
                # Per-token division precedes aggregation; save population std too.
                values[f"input_{layer:02d}"] = raw[:, :layer + 1] / denominator[:, layer:layer + 1]
            parent_path = Path(manifest["parent_run"]) / "tokens" / path.name
            assert sha256(parent_path) == marker["parent_chunk_sha256"]
            with np.load(parent_path) as projected:
                for layer in manifest["config"]["replay_layers"]:
                    for kind in "QKV":
                        heads = projected[f"abs_{kind}_{layer:02d}"].astype(np.float64)
                        assert heads.shape == (len(raw), 12, layer + 1)
                        values[f"full_{kind}_{layer:02d}"] = np.sqrt(np.square(heads).sum(1))
            for key, value in values.items():
                value = value.astype(np.float64)
                totals[key] = totals.get(key, 0) + value.sum(0)
                squares[key] = squares.get(key, 0) + np.square(value).sum(0)
            count += len(raw)
        for check in json.loads((run / "validation" / f"chunk_{sequence_id:05d}.json").read_text())["checks"]:
            for key, value in check.items():
                if key.endswith("relative_rms"):
                    maxima[key] = max(maxima.get(key, 0), value)
    stats = {}
    for key in totals:
        mean = totals[key] / count
        stats[f"{key}_token_mean"] = mean
        stats[f"{key}_token_std"] = np.sqrt(np.maximum(squares[key] / count - mean ** 2, 0))
    np.savez(run / "summaries/statistics.npz", **stats)
    write_json(run / "validation/summary.json", dict(passed=True, maxima=maxima, chunks=len(tokens), tokens=count))
    return count, maxima


def run_analysis(run, measure_fn=measure, summarize_fn=summarize):
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    parent = Path(manifest["parent_run"])
    assert sha256(parent / "manifest.json") == manifest["parent_manifest_sha256"]
    assert sha256(parent / "summaries/statistics.npz") == manifest["parent_statistics_sha256"]
    assert sha256(manifest["checkpoint_path"]) == manifest["checkpoint_sha256"]
    assert sha256(run / "inputs/tokens.npy") == manifest["tokens_sha256"]
    for relative, expected in manifest["source_hashes"].items():
        assert sha256(run / "code" / relative) == expected, relative
    for relative, expected in manifest["kernel_cache_sha256"].items():
        assert sha256(Path(manifest["kernel_cache_path"]) / relative) == expected, relative
    assert torch.cuda.device_count() == 1
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                       gpu=torch.cuda.get_device_name(0), visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                       use_fa3=USE_FA3, tf32=False, float32_matmul_precision="highest")
    if (run / "environment.json").exists():
        assert json.loads((run / "environment.json").read_text()) == environment
    write_json(run / "environment.json", environment)
    (run / "environment-packages.txt").write_text("\n".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())) + "\n")
    model, _, _ = build_model(str(Path(manifest["checkpoint_path"]).parent), config["checkpoint_step"], torch.device("cuda:0"), "eval")
    model.requires_grad_(False)
    tokens = np.load(run / "inputs/tokens.npy")
    assert tokens.shape == (config["num_sequences"], config["sequence_length"])
    start = time.monotonic()
    manifest_hash = sha256(run / "manifest.json")
    for sequence_id, sequence in enumerate(tokens):
        path = run / "tokens" / f"chunk_{sequence_id:05d}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            record = json.loads(marker.read_text())
            assert record["sha256"] == sha256(path) and record["manifest_sha256"] == manifest_hash
            continue
        parent_path = parent / "tokens" / path.name
        parent_record = json.loads(parent_path.with_suffix(".json").read_text())
        assert parent_record["sha256"] == sha256(parent_path)
        assert parent_record["tokens_sha256"] == manifest["tokens_sha256"]
        assert parent_record["manifest_sha256"] == manifest["parent_manifest_sha256"]
        ids = torch.from_numpy(sequence.copy()).unsqueeze(0).cuda()
        with np.load(parent_path) as reference:
            arrays, checks = measure_fn(model, ids, reference, config)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(path)
        write_json(run / "validation" / marker.name, dict(passed=True, checks=checks))
        write_json(marker, dict(sha256=sha256(path), sequence_id=sequence_id, tokens=len(sequence),
                               tokens_sha256=manifest["tokens_sha256"], manifest_sha256=manifest_hash,
                               parent_chunk_sha256=parent_record["sha256"]))
        write_json(run / "status.json", dict(state="running", completed_chunks=sequence_id + 1,
                                             total_chunks=len(tokens), updated_utc=now()))
        if (sequence_id + 1) % 8 == 0:
            print(f"Completed {sequence_id + 1}/{len(tokens)} sequences; {time.monotonic()-start:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    write_json(run / "status.json", dict(state="summarizing", completed_chunks=len(tokens), updated_utc=now()))
    count, maxima = summarize_fn(run, manifest)
    write_json(run / "results.json", dict(state="analysis_complete", tokens=count, validation=maxima,
        duration_seconds=time.monotonic()-start, gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
        statistics_sha256=sha256(run / "summaries/statistics.npz"),
        token_records_bytes=sum(p.stat().st_size for p in (run / "tokens").glob("*.npz")),
        visualization_status=manifest.get("visualization_status", "pending")))
    write_json(run / "status.json", dict(state="analysis_complete", completed_chunks=len(tokens), updated_utc=now()))
    print(f"ANALYSIS_COMPLETE {run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_analysis(args.run_dir)
    except BaseException as error:
        write_json(args.run_dir / "status.json", dict(state="failed", error=repr(error), traceback=traceback.format_exc(), updated_utc=now()))
        raise


if __name__ == "__main__":
    main()
