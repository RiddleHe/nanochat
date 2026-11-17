"""Freeze and launch a single-GPU weighted source-contribution probe on exact previously saved tokens."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.inspect.qkv_source_contributions.prepare import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("/local-ssd/mh3897"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    gpu = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [s.strip() for s in gpu.split(",")]
    if int(fields[-2]) > 1000 or int(fields[-1]) > 5:
        raise RuntimeError(f"GPU busy: {gpu}")
    checkpoint_dir = args.base_dir / "base_checkpoints" / config["checkpoint_run"]
    checkpoint = checkpoint_dir / f"model_{config['checkpoint_step']:06d}.pt"
    metadata_path = checkpoint_dir / f"meta_{config['checkpoint_step']:06d}.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["model_type"] == "gpt_base_attn_res_qkv"
    assert metadata["model_config"]["n_layer"] == 24
    assert metadata["model_config"]["sequence_len"] == config["sequence_length"]
    training_run = args.base_dir / "run_queues" / config["checkpoint_run"]
    parent = args.base_dir / "analysis_runs" / config["token_run"]
    provenance = json.loads((parent / "inputs/provenance.json").read_text())
    assert sha256(parent / "inputs/tokens.npy") == provenance["tokens_sha256"]
    tokenizer_hashes = {str(p):sha256(p) for p in sorted((args.base_dir / "tokenizer").iterdir()) if p.is_file()}
    assert tokenizer_hashes == provenance["tokenizer_sha256"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"attn_res_qkv_step{config['checkpoint_step']}_{stamp}"
    if Path(run_id).name != run_id:
        raise ValueError("run-id must be one path component")
    run = args.base_dir / "analysis_runs" / config["experiment"] / run_id
    run.mkdir(parents=True, exist_ok=False)
    for directory in ("code", "inputs", "tokens", "summaries", "validation", "logs", "visualization"):
        (run / directory).mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(training_run / "code/nanochat", run / "code/nanochat", ignore=ignore)
    shutil.copytree(Path(__file__).parent, run / "code/scripts/inspect/attn_res_qkv_source_contributions", ignore=ignore)
    shutil.copytree(repo / "scripts/inspect/attn_res_qkv_routing", run / "code/scripts/inspect/attn_res_qkv_routing", ignore=ignore)
    routing_parent = args.base_dir / "analysis_runs" / config["routing_run"]
    parent_manifest = json.loads((routing_parent / "manifest.json").read_text())
    assert parent_manifest["checkpoint_sha256"] == sha256(checkpoint)
    assert parent_manifest["tokens_sha256"] == provenance["tokens_sha256"]
    records = [json.loads(p.read_text()) for p in sorted((routing_parent / "tokens").glob("chunk_*.json"))]
    assert len(records) == config["num_sequences"]
    write_json(run / "inputs/routing_reference.json", dict(path=str(routing_parent), manifest_sha256=sha256(routing_parent / "manifest.json"), records=records))
    helper = run / "code/scripts/inspect/qkv_source_contributions/prepare.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(repo / "scripts/inspect/qkv_source_contributions/prepare.py", helper)
    for name in ("tokens.npy", "provenance.json"):
        shutil.copy2(parent / "inputs" / name, run / "inputs" / name)
    shutil.copy2(metadata_path, run / "inputs/checkpoint_metadata.json")
    source_index = [dict(index=0, source_family="embedding", source_layer=-1)]
    for layer in range(24):
        for family in ("attention", "mlp"):
            source_index.append(dict(index=len(source_index), source_family=family, source_layer=layer))
    write_json(run / "inputs/source_index.json", source_index)
    hub = Path(os.environ.get("KERNELS_CACHE", os.environ.get("HF_KERNELS_CACHE",
        os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))))
    cache = hub / "models--varunneal--flash-attention-3"
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    runtime = run / "runtime_cache"
    overrides = dict(CUDA_VISIBLE_DEVICES=fields[1], NANOCHAT_BASE_DIR=str(args.base_dir),
        NANOCHAT_DTYPE=config["forward_dtype"], HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
        XDG_CACHE_HOME=str(runtime), HF_HOME=str(runtime / "huggingface"), HF_HUB_CACHE=str(hub),
        HUGGINGFACE_HUB_CACHE=str(hub), KERNELS_CACHE=str(hub), HF_KERNELS_CACHE=str(hub),
        TORCH_HOME=str(runtime / "torch"), CUDA_CACHE_PATH=str(runtime / "cuda"), TRITON_CACHE_DIR=str(runtime / "triton"))
    env = dict(os.environ, **overrides)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PYTHONPATH"):
        env.pop(key, None)
    command = [sys.executable, "-m", "scripts.inspect.attn_res_qkv_source_contributions.collect", "--run-dir", str(run)]
    manifest = dict(schema_version=1, run_id=run_id, created_utc=stamp, config=config,
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint), metadata_sha256=sha256(metadata_path),
        training_run=str(training_run), training_manifest_sha256=sha256(training_run / "manifest.json"),
        token_parent_run=str(parent), token_parent_manifest_sha256=sha256(parent / "manifest.json"),
        tokens_sha256=provenance["tokens_sha256"], tokenizer_sha256=tokenizer_hashes,
        base_dir=str(args.base_dir), source_index=source_index,
        source_hashes={str(p.relative_to(run / "code")):sha256(p) for p in sorted((run / "code").rglob("*")) if p.is_file()},
        kernel_cache_path=str(cache), kernel_cache_sha256={str(p.relative_to(cache)):sha256(p) for p in sorted(cache.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"},
        selected_gpu=gpu, gpu_uuid=fields[1], command=command, cwd=str(run / "code"), environment_overrides=overrides,
        python_executable_resolved=str(Path(sys.executable).resolve()),
        parent_git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        token_chunk_schema="abs/share_{kind}_{layer:02d} [tokens,heads,sources]; input_{kind}_{layer:02d} [tokens,sources]; input_denominator [tokens,layers,3]; source_norm [tokens,49]; QKV before RoPE/QK norm",
        routing_reference_sha256=sha256(run / "inputs/routing_reference.json"),
        visualization_status="held_by_user")
    write_json(run / "manifest.json", manifest)
    write_json(run / "status.json", dict(state="prepared", completed_chunks=0))
    print(f"RUN_DIR={run}", flush=True)
    if args.launch:
        with (run / "logs/collect.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=run / "code", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
