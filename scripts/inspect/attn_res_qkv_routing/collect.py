"""Observe Q/K/V depth-routing weights; validate every sequence; never plot."""
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

from nanochat.checkpoint_manager import build_model
from nanochat.flash_attention import USE_FA3
from nanochat.model.gpt_base import norm
from .prepare import sha256, write_json
from .summarize import summarize


def now():
    return datetime.now(timezone.utc).isoformat()


@torch.inference_mode()
def measure(model, ids, config):
    baseline = model(ids)
    if not torch.isfinite(baseline).all():
        raise ValueError("Nonfinite baseline logits")
    original = model._attn_res
    arrays, checks, seen = {}, [], {}
    handles = []
    layer_count = model.config.n_layer
    for layer, block in enumerate(model.transformer.h):
        for family, module in (("attention", block.attn), ("mlp", block.mlp)):
            def capture(module, args, output, layer=layer, family=family):
                seen[family, layer] = output
            handles.append(module.register_forward_hook(capture))

    def observe(queries, sources):
        actual = original(queries, sources)
        if queries.shape[0] != 3:
            return actual  # MLP and final-output routing are not measured.
        layer = len(arrays)
        assert layer < layer_count and len(sources) == 2*layer+1
        assert torch.equal(queries, model.attn_res_queries[4*layer:4*layer+3])
        for previous in range(layer):
            assert sources[2*previous+1] is seen["attention", previous]
            assert sources[2*previous+2] is seen["mlp", previous]
        values = torch.stack(sources, 0)
        assert values.dtype == torch.bfloat16
        keys = norm(values)
        scores = torch.einsum('rd,nbtd->rnbt', queries.to(keys.dtype), keys)
        probabilities = scores.float().softmax(dim=1)
        weights = probabilities.to(values.dtype)
        precast_error = float((probabilities.sum(1)-1).abs().max())
        applied_error = float((weights.float().sum(1)-1).abs().max())
        limits = config["validation"]
        assert precast_error <= limits["precast_weight_sum_max_error"]
        assert applied_error <= limits["applied_weight_sum_max_error"]
        assert torch.isfinite(weights).all() and (weights >= 0).all() and (weights <= 1).all()
        replay = torch.einsum('rnbt,nbtd->rbtd', weights, values)
        if not torch.equal(replay, actual):
            raise ValueError(f"BF16 routing replay mismatch at layer {layer}")
        linear = torch.einsum('rnbt,nbtd->rbtd', weights.float(), values.float())
        error = float((linear-actual.float()).square().mean().sqrt() /
                      actual.float().square().mean().sqrt().clamp_min(1e-12))
        assert error <= limits["fp32_mixture_relative_rms_max"], (layer, error)
        arrays[f"weights_{layer:02d}"] = weights[:, :, 0, :].permute(2, 0, 1).float().cpu().numpy()
        checks.append(dict(layer=layer, sources=len(sources), source_identity_passed=True,
            precast_weight_sum_error=precast_error, applied_weight_sum_error=applied_error,
            exact_bf16_mixture_replay=True, fp32_mixture_relative_rms=error))
        return actual  # Return the untouched original router result.

    model._attn_res = observe
    try:
        observed = model(ids)
        if not torch.equal(observed, baseline):
            raise ValueError("Instrumentation changed full-forward logits")
    finally:
        del model._attn_res
        for handle in handles:
            handle.remove()
    assert len(arrays) == layer_count
    return arrays, dict(passed=True, exact_forward_replay=True, checks=checks)


def run_analysis(run, measure_fn=None, summarize_fn=None):
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    assert config["batch_size"] == 1 and config["forward_dtype"] == "bfloat16"
    for relative, expected in manifest["source_hashes"].items():
        assert sha256(run / "code" / relative) == expected, relative
    assert sha256(manifest["checkpoint_path"]) == manifest["checkpoint_sha256"]
    assert sha256(run / "inputs/checkpoint_metadata.json") == manifest["metadata_sha256"]
    assert sha256(run / "inputs/tokens.npy") == manifest["tokens_sha256"]
    for path, expected in manifest["tokenizer_sha256"].items():
        assert sha256(path) == expected, path
    for relative, expected in manifest["kernel_cache_sha256"].items():
        assert sha256(Path(manifest["kernel_cache_path"]) / relative) == expected, relative
    assert torch.cuda.device_count() == 1
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        use_fa3=USE_FA3, tf32=False, float32_matmul_precision="highest",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if (run / "environment.json").exists():
        assert json.loads((run / "environment.json").read_text()) == environment
    write_json(run / "environment.json", environment)
    (run / "environment-packages.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    model, _, _ = build_model(str(Path(manifest["checkpoint_path"]).parent), config["checkpoint_step"], torch.device("cuda:0"), "eval")
    model.requires_grad_(False)
    tokens = np.load(run / "inputs/tokens.npy")
    assert tokens.shape == (config["num_sequences"], config["sequence_length"])
    manifest_hash = sha256(run / "manifest.json")
    start = time.monotonic()
    for index, sequence in enumerate(tokens):
        path = run / "tokens" / f"chunk_{index:05d}.npz"
        marker = path.with_suffix(".json")
        validation_path = run / "validation" / marker.name
        if marker.exists():
            record = json.loads(marker.read_text())
            assert record["sha256"] == sha256(path) and record["manifest_sha256"] == manifest_hash
            assert record["validation_sha256"] == sha256(validation_path)
            continue
        write_json(run / "status.json", dict(state="running", completed_chunks=index, total_chunks=len(tokens), updated_utc=now()))
        ids = torch.from_numpy(sequence.copy()).unsqueeze(0).cuda()
        arrays, validation = (measure_fn(model, ids, config, index) if measure_fn else measure(model, ids, config))
        with path.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez(handle, **arrays)
        path.with_suffix(".npz.tmp").replace(path)
        write_json(validation_path, validation)
        write_json(marker, dict(sequence_id=index, tokens=len(sequence), sha256=sha256(path),
            tokens_sha256=manifest["tokens_sha256"], manifest_sha256=manifest_hash,
            validation_sha256=sha256(validation_path)))
        if (index+1) % 8 == 0 or index == 0:
            print(f"Collected {index+1}/{len(tokens)}; exact mixture and forward replay passed; {time.monotonic()-start:.1f}s", flush=True)
    del model, ids
    torch.cuda.empty_cache()
    write_json(run / "status.json", dict(state="summarizing", completed_chunks=len(tokens), updated_utc=now()))
    validation = (summarize_fn or summarize)(run)
    write_json(run / "results.json", dict(state="analysis_complete", tokens=int(tokens.size),
        sequences=len(tokens), duration_seconds=time.monotonic()-start,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
        token_records_bytes=sum(p.stat().st_size for p in (run / "tokens").glob("*.npz")),
        validation=validation, summary_sha256={p.name:sha256(p) for p in (run / "summaries").iterdir() if p.is_file()},
        visualization_status="held_by_user"))
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
