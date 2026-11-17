"""Freeze a fresh analysis run; execute its frozen collector on one explicit GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("/local-ssd/mh3897"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--launch", action="store_true")
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument("--input-norms-of", type=Path,
                        help="Collect input-component norms using this completed QKV run's exact tokens and model")
    parent_group.add_argument("--separate-sources-of", type=Path,
                             help="Separate attention/MLP sources using this completed combined-source run")
    parser.add_argument("--separate-config", type=Path,
                        default=Path(__file__).with_name("separate_config.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    parent_arg = args.separate_sources_of or args.input_norms_of
    parent = parent_arg.resolve() if parent_arg else None
    if parent:
        parent_manifest = json.loads((parent / "manifest.json").read_text())
        config = dict(parent_manifest["config"], experiment="d24_qkv_input_norms",
                      replay_layers=[6, 12, 18, 23], replay_relative_rms_max=2e-5)
        if args.separate_sources_of:
            overrides = json.loads(args.separate_config.read_text())
            config = dict(parent_manifest["config"], **overrides)
            config["validation"] = dict(parent_manifest["config"]["validation"],
                                        **overrides.get("validation", {}))
            if parent_manifest["config"]["experiment"] != "d24_qkv_source_contributions":
                raise ValueError("Separate-source replay requires the original combined-source QKV run")
        args.base_dir = Path(parent_manifest["base_dir"])
        for relative, expected in parent_manifest["source_hashes"].items():
            if sha256(parent / "code" / relative) != expected:
                raise ValueError(f"Parent frozen source changed: {relative}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"gpt_base_step{config['checkpoint_step']}_{stamp}"
    if Path(run_id).name != run_id:
        raise ValueError("run-id must be a single path component")
    gpu_query = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip()
    gpu_fields = [part.strip() for part in gpu_query.split(",")]
    if int(gpu_fields[-2]) > 1000 or int(gpu_fields[-1]) > 5:
        raise RuntimeError(f"Selected GPU is busy: {gpu_query}")
    checkpoint_dir = args.base_dir / "base_checkpoints" / config["checkpoint_run"]
    checkpoint = checkpoint_dir / f"model_{config['checkpoint_step']:06d}.pt"
    metadata = checkpoint_dir / f"meta_{config['checkpoint_step']:06d}.json"
    model_meta = json.loads(metadata.read_text())
    assert model_meta["model_type"] == "gpt_base"
    assert all(not value for value in model_meta["model_config"].values() if isinstance(value, bool))
    assert model_meta["model_config"]["sequence_len"] == config["sequence_length"]
    run = args.base_dir / "analysis_runs" / config["experiment"] / run_id
    run.mkdir(parents=True, exist_ok=False)
    for name in ("code", "inputs", "logs", "tokens", "summaries", "validation", "visualization"):
        (run / name).mkdir()
    shutil.copytree(parent / "code/nanochat" if parent else repo / "nanochat", run / "code/nanochat",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    dest = run / "code/scripts/inspect/qkv_source_contributions"
    shutil.copytree(Path(__file__).parent, dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(metadata, run / "inputs/checkpoint_metadata.json")
    if parent:
        if sha256(checkpoint) != parent_manifest["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint changed")
        provenance = json.loads((parent / "inputs/provenance.json").read_text())
        if sha256(parent / "inputs/tokens.npy") != provenance["tokens_sha256"]:
            raise ValueError("Parent tokens changed")
        shutil.copy2(parent / "inputs/tokens.npy", run / "inputs/tokens.npy")
        shutil.copy2(parent / "inputs/provenance.json", run / "inputs/provenance.json")
    source_hashes = {str(p.relative_to(run / "code")): sha256(p)
                     for p in sorted((run / "code").rglob("*")) if p.is_file()}
    # Reuse the maintained shared kernel cache; record exact file identities.
    runtime_cache = run / "runtime_cache"
    hub_cache = Path(os.environ.get("KERNELS_CACHE", os.environ.get(
        "HF_KERNELS_CACHE", os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))))
    kernel_name = "models--varunneal--flash-attention-3"
    kernel_cache = hub_cache / kernel_name
    if not kernel_cache.is_dir():
        raise FileNotFoundError(f"Required offline kernel cache missing: {kernel_cache}")
    cache_hashes = {str(p.relative_to(kernel_cache)): sha256(p)
                    for p in sorted(kernel_cache.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_fields[1],
               NANOCHAT_BASE_DIR=str(args.base_dir), NANOCHAT_DTYPE=config["forward_dtype"],
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
               XDG_CACHE_HOME=str(runtime_cache), HF_HOME=str(runtime_cache / "huggingface"),
               HF_HUB_CACHE=str(hub_cache), HUGGINGFACE_HUB_CACHE=str(hub_cache),
               KERNELS_CACHE=str(hub_cache), HF_KERNELS_CACHE=str(hub_cache), TORCH_HOME=str(runtime_cache / "torch"),
               CUDA_CACHE_PATH=str(runtime_cache / "cuda"), TRITON_CACHE_DIR=str(runtime_cache / "triton"))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PYTHONPATH"):
        env.pop(key, None)
    module = "collect_input_norms" if parent else "collect"
    if args.separate_sources_of:
        module = "collect_separate_sources"
    command = [sys.executable, "-m", f"scripts.inspect.qkv_source_contributions.{module}",
               "--run-dir", str(run)]
    manifest = {
        "schema_version": 1, "run_id": run_id, "created_utc": stamp,
        "config": config, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "metadata_sha256": sha256(metadata),
        "base_dir": str(args.base_dir), "source_hashes": source_hashes,
        "kernel_cache_path": str(kernel_cache), "kernel_cache_sha256": cache_hashes,
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "parent_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "selected_gpu": gpu_query, "gpu_uuid": gpu_fields[1],
        "command": command, "cwd": str(run / "code"),
        "environment_overrides": {k: env[k] for k in (
            "CUDA_VISIBLE_DEVICES", "NANOCHAT_BASE_DIR", "NANOCHAT_DTYPE", "HF_HUB_OFFLINE", "OMP_NUM_THREADS",
            "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_KERNELS_CACHE", "KERNELS_CACHE",
            "TORCH_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR", "PYTHONDONTWRITEBYTECODE")},
        "token_chunk_schema": "NPZ keys abs_{Q,K,V}_{layer:02d}, share_{Q,K,V}_{layer:02d}; each [tokens, actual_heads, sources]; source 0 embedding, source j+1 layer j; one sequence per chunk",
        "all_positions_included": True,
    }
    if parent:
        manifest.update(parent_run=str(parent), parent_manifest_sha256=sha256(parent / "manifest.json"),
                        tokens_sha256=provenance["tokens_sha256"],
                        parent_statistics_sha256=sha256(parent / "summaries/statistics.npz"),
                        token_chunk_schema="source_norm [tokens,25], input_denominator [tokens,24]; source 0 Emb, source j+1 layer j")
    if args.separate_sources_of:
        n_layers = model_meta["model_config"]["n_layer"]
        sources = [{"index": 0, "source_family": "embedding", "source_layer": -1}]
        for layer in range(n_layers):
            for family in ("attention", "mlp"):
                sources.append(dict(index=len(sources), source_family=family, source_layer=layer))
        manifest.update(schema_version=2, source_index=sources,
                        token_chunk_schema="source_norm [tokens,2*depth+1]; input_denominator [tokens,depth]; abs_{Q,K,V}_{layer:02d} and share_{Q,K,V}_{layer:02d} [tokens,heads,2*layer+1]; available sources are the source_index prefix",
                        receiving_modules=["Q", "K", "V"], visualization_status="held_by_user")
        write_json(run / "inputs/source_index.json", sources)
    write_json(run / "manifest.json", manifest)
    write_json(run / "status.json", {"state": "prepared", "completed_chunks": 0})
    print(f"RUN_DIR={run}", flush=True)
    if args.launch:
        with open(run / "logs/collect.log", "a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=run / "code", env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
