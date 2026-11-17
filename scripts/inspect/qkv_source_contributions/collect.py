"""Collect frozen-forward Q/K/V source norms, validating before each chunk commits."""
import argparse
from datetime import datetime, timezone
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
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.dataset import list_parquet_files
from nanochat.flash_attention import USE_FA3
from nanochat.model.gpt_base import norm
from .prepare import sha256, write_json
from .summarize import summarize


def now():
    return datetime.now(timezone.utc).isoformat()


def relative_rms(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return float((error / scale).item())


def capture_forward(model, ids):
    """Hooks observe the original unmodified forward; no replacement attention."""
    captured = {"input": {}, "attn": {}, "mlp": {}, "projection": {}}
    handles = []
    for layer, block in enumerate(model.transformer.h):
        def before(module, args, layer=layer):
            captured["input"][layer] = args[0].detach()
        def attn(module, args, output, layer=layer):
            captured["attn"][layer] = output[0].detach()
        def mlp(module, args, output, layer=layer):
            captured["mlp"][layer] = output.detach()
        handles.extend([block.register_forward_pre_hook(before),
                        block.attn.register_forward_hook(attn),
                        block.mlp.register_forward_hook(mlp)])
        for kind in ("Q", "K", "V"):
            def projected(module, args, output, layer=layer, kind=kind):
                captured["projection"][(layer, kind)] = output.detach()
            handles.append(getattr(block.attn, f"c_{kind.lower()}").register_forward_hook(projected))
    try:
        logits = model(ids)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite original-forward logits")
        del logits
    finally:
        for handle in handles:
            handle.remove()
    return captured


@torch.inference_mode()
def measure_sequence(model, ids, config):
    captured = capture_forward(model, ids)
    sources = [captured["input"][0].float()]
    arrays, checks = {}, []
    limits = config["validation"]
    for layer, block in enumerate(model.transformer.h):
        actual_input = captured["input"][layer]
        source_sum = torch.stack(sources).sum(0)
        # eps=None in F.rms_norm uses the input dtype's machine epsilon.
        eps = torch.finfo(actual_input.dtype).eps
        denominator = (actual_input.float().square().mean(-1, keepdim=True) + eps).sqrt()
        normalized_actual = actual_input.float() / denominator
        residual_error = relative_rms(source_sum, actual_input)
        normalization_error = relative_rms(normalized_actual, norm(actual_input))
        if residual_error > limits["residual_relative_rms_max"]:
            raise ValueError(f"Residual reconstruction failed at layer {layer}: {residual_error}")
        if normalization_error > limits["normalization_relative_rms_max"]:
            raise ValueError(f"RMSNorm denominator mismatch at layer {layer}: {normalization_error}")
        for kind in ("Q", "K", "V"):
            projection = getattr(block.attn, f"c_{kind.lower()}")
            # These are exactly the weights seen by the BF16 forward linear op,
            # represented in FP32 to avoid rounding each source independently.
            weight = projection.weight.to(actual_input.dtype).float()
            heads = block.attn.n_head if kind == "Q" else block.attn.n_kv_head
            head_dim = block.attn.head_dim
            norm_parts = []
            projected_sum = torch.zeros_like(captured["projection"][(layer, kind)], dtype=torch.float32)
            source_batch = config["source_batch_size"]
            for start in range(0, len(sources), source_batch):
                source_group = torch.stack(sources[start:start + source_batch]) / denominator
                projected = F.linear(source_group, weight)
                projected_sum += projected.sum(0)
                shaped = projected.reshape(projected.shape[0], -1, heads, head_dim)
                norm_parts.append(shaped.norm(dim=-1).permute(1, 2, 0).cpu())
            absolute = torch.cat(norm_parts, dim=-1).numpy()
            total = absolute.sum(-1, keepdims=True)
            if np.any(total == 0) or not np.isfinite(absolute).all():
                raise ValueError(f"Undefined source share at layer {layer}, {kind}")
            share = absolute / total
            expected_linear_sum = F.linear(source_sum / denominator, weight)
            linear_error = relative_rms(projected_sum, expected_linear_sum)
            forward_error = relative_rms(projected_sum, captured["projection"][(layer, kind)])
            # Isolate projection/normalization rounding from residual accumulation.
            forward_rounding_error = relative_rms(F.linear(normalized_actual, weight),
                                                  captured["projection"][(layer, kind)])
            share_error = float(np.abs(share.sum(-1) - 1).max())
            if linear_error > limits["linear_reconstruction_relative_rms_max"]:
                raise ValueError(f"Linear reconstruction failed: {layer} {kind} {linear_error}")
            if forward_error > limits["projection_relative_rms_max"]:
                raise ValueError(f"Forward reconstruction failed: {layer} {kind} {forward_error}")
            if share_error > limits["share_sum_max_error"]:
                raise ValueError(f"Share sum failed: {layer} {kind} {share_error}")
            arrays[f"abs_{kind}_{layer:02d}"] = absolute
            arrays[f"share_{kind}_{layer:02d}"] = share
            checks.append({"layer": layer, "kind": kind, "rmsnorm_epsilon": eps,
                           "residual_relative_rms": residual_error,
                           "normalization_relative_rms": normalization_error,
                           "linear_reconstruction_relative_rms": linear_error,
                           "projection_relative_rms": forward_error,
                           "forward_rounding_relative_rms": forward_rounding_error,
                           "share_sum_error": share_error})
        sources.append(captured["attn"][layer].float() + captured["mlp"][layer].float())
    return arrays, checks


def prepare_inputs(run, manifest, tokenizer):
    path = run / "inputs/tokens.npy"
    provenance_path = run / "inputs/provenance.json"
    config = manifest["config"]
    if path.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if sha256(path) != provenance["tokens_sha256"]:
            raise ValueError("Saved input tokens changed")
        return np.load(path)
    paths = list_parquet_files()
    validation_path = Path(paths[-1])
    if validation_path.name != "shard_06542.parquet":
        raise ValueError(f"Unexpected validation shard: {validation_path}")
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B=1, T=config["sequence_length"], split="val", device="cpu",
        tokenizer_threads=config["tokenizer_threads"],
        tokenizer_batch_size=config["tokenizer_batch_size"], buffer_size=config["document_buffer_size"])
    sequences = []
    for _ in range(config["num_sequences"]):
        inputs, _ = next(loader)
        sequences.append(inputs[0].numpy().copy())
    tokens = np.stack(sequences)
    np.save(path, tokens)
    tokenizer_dir = Path(manifest["base_dir"]) / "tokenizer"
    provenance = {
        "selection": config["sample_selection"], "split": "val", "tokens_sha256": sha256(path),
        "sequence_ids": list(range(len(tokens))), "position_range": [0, tokens.shape[1] - 1],
        "all_positions_including_bos": True, "padding": False,
        "shape": list(tokens.shape), "dtype": str(tokens.dtype),
        "validation_shard": str(validation_path), "validation_shard_sha256": sha256(validation_path),
        "validation_shard_bytes": validation_path.stat().st_size,
        "tokenizer_sha256": {str(p): sha256(p) for p in sorted(tokenizer_dir.iterdir()) if p.is_file()},
        "seed": config["seed"], "selection_is_deterministic_not_random": True,
    }
    write_json(provenance_path, provenance)
    return tokens


def run_analysis(run):
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    for relative, expected in manifest["source_hashes"].items():
        if sha256(run / "code" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    if sha256(manifest["checkpoint_path"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for relative, expected in manifest.get("kernel_cache_sha256", {}).items():
        if sha256(Path(manifest["kernel_cache_path"]) / relative) != expected:
            raise ValueError(f"Shared kernel cache changed: {relative}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU must be visible")
    if config["batch_size"] != 1:
        raise ValueError("This collector uses one sequence per atomic chunk")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pyarrow", "tiktoken")}
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "packages": packages, "cuda_version": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "use_fa3": USE_FA3, "tf32": False, "float32_matmul_precision": "highest",
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "reproducibility": "numerical_tolerance_not_cross_platform_bitwise"}
    environment_path = run / "environment.json"
    if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
        raise ValueError("Execution environment changed; create a new run")
    write_json(environment_path, environment)
    freeze = sorted({f"{dist.metadata['Name']}=={dist.version}"
                     for dist in importlib.metadata.distributions()})
    (run / "environment-packages.txt").write_text("\n".join(freeze) + "\n")
    print(f"GPU: {environment['gpu']}; FA3: {USE_FA3}; loading checkpoint", flush=True)
    checkpoint_dir = str(Path(manifest["checkpoint_path"]).parent)
    model, tokenizer, metadata = build_model(checkpoint_dir, config["checkpoint_step"], device, "eval")
    model.requires_grad_(False)
    tokens = prepare_inputs(run, manifest, tokenizer)
    assert tokens.shape == (config["num_sequences"], config["sequence_length"])
    token_hash = sha256(run / "inputs/tokens.npy")
    manifest_hash = sha256(run / "manifest.json")
    print(f"Inputs frozen: {tokens.shape}; beginning first-sequence validation", flush=True)
    start_time = time.monotonic()
    completed = 0
    for sequence_id, sequence in enumerate(tokens):
        marker = run / "tokens" / f"chunk_{sequence_id:05d}.json"
        path = marker.with_suffix(".npz")
        if marker.exists():
            record = json.loads(marker.read_text())
            if (record["manifest_sha256"] != manifest_hash or record["tokens_sha256"] != token_hash
                    or sha256(path) != record["sha256"]):
                raise ValueError(f"Invalid completed chunk: {marker}")
            completed += 1
            continue
        chunk_start = time.monotonic()
        write_json(run / "status.json", {"state": "running", "current_sequence": sequence_id,
                   "completed_chunks": completed, "total_chunks": len(tokens), "updated_utc": now()})
        ids = torch.from_numpy(sequence.copy()).unsqueeze(0).to(device)
        arrays, checks = measure_sequence(model, ids, config)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(path)
        write_json(run / "validation" / f"chunk_{sequence_id:05d}.json", {"passed": True, "checks": checks})
        write_json(marker, {"file": path.name, "sha256": sha256(path),
                   "sequence_id": sequence_id, "tokens": len(sequence), "tokens_sha256": token_hash,
                   "manifest_sha256": manifest_hash, "bytes": path.stat().st_size})
        completed += 1
        print(f"Chunk {completed}/{len(tokens)} complete in {time.monotonic()-chunk_start:.1f}s; "
              f"projection error max={max(c['projection_relative_rms'] for c in checks):.5f}; "
              f"GPU peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        del arrays, ids
    del model
    torch.cuda.empty_cache()
    write_json(run / "status.json", {"state": "summarizing", "completed_chunks": completed, "updated_utc": now()})
    cells = summarize(run)
    validation_records = [json.loads(p.read_text()) for p in sorted((run / "validation").glob("chunk_*.json"))]
    all_checks = [check for record in validation_records for check in record["checks"]]
    maxima = {key: max(check[key] for check in all_checks) for key in all_checks[0]
              if key.endswith("_rms") or key.endswith("_error")}
    write_json(run / "validation/reconstruction.json", {"passed": True, "maxima": maxima,
               "thresholds": config["validation"], "sequence_count": len(tokens)})
    write_json(run / "results.json", {"state": "analysis_complete", "tokens": int(tokens.size),
               "sequences": len(tokens), "duration_seconds": time.monotonic() - start_time,
               "validation": maxima, "heatmap_cells": len(cells),
               "summary_artifacts": {str(p.relative_to(run)): sha256(p)
                                     for p in sorted((run / "summaries").iterdir()) if p.is_file()},
               "visualization_status": "pending"})
    write_json(run / "status.json", {"state": "analysis_complete", "completed_chunks": completed,
               "total_chunks": len(tokens), "updated_utc": now()})
    print(f"ANALYSIS_COMPLETE {run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_analysis(args.run_dir)
    except BaseException as error:
        write_json(args.run_dir / "status.json", {"state": "failed", "error": repr(error),
                   "traceback": traceback.format_exc(), "updated_utc": now()})
        raise


if __name__ == "__main__":
    main()
