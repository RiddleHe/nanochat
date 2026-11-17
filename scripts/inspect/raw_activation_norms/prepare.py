"""Freeze dependencies, then tokenize the complete validation shard with frozen code."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from .common import read_json, sha256, tree_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("/local-ssd/mh3897"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--token-reference", type=Path, help="Reuse a validated full-shard token stream and its context policy")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    config = read_json(args.config)
    model_type = config.get('model_type', 'gpt_base')
    assert model_type in ('gpt_base', 'gpt_base_no_embed_norm', 'gpt_base_scalar_res_qkv_no_embed_norm')
    reference = args.token_reference.resolve() if args.token_reference else None
    if reference:
        prior = read_json(reference/'manifest.json')
        result = read_json(reference/'results.json')
        assert result['state'] == 'complete' and result['validation_sha256'] == sha256(reference/'validation/summary.json')
        assert read_json(reference/'validation/summary.json')['passed']
        assert prior['config']['sequence_length'] == config['sequence_length']
        assert prior['config']['token_policy'] == config['token_policy']
        for name, digest in prior['inputs_hashes'].items():
            if not name.startswith('checkpoint/') and name != 'config.json':
                assert sha256(reference/'inputs'/name) == digest, name
        assert tree_hashes(reference/'kernel_cache') == prior['kernel_cache_hashes']
    gpu = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    gpu_fields = [v.strip() for v in gpu.split(",")]
    assert int(gpu_fields[-2]) < 1000 and int(gpu_fields[-1]) <= 5, gpu
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = args.base_dir / "analysis_runs/d24_raw_activation_norms" / f"{model_type}_step{config['checkpoint_step']}_fullval_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    for name in ("code", "inputs", "logs", "tokens", "derived", "validation", "visualization"):
        (run / name).mkdir()
    print(f"RUN_DIR={run}", flush=True)
    write_json(run / "status.json", {"state": "preparing"})
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    model_code = Path(config.get('model_code_source', str(repo/'nanochat')))
    if config.get('training_run'):
        training = Path(config['training_run'])
        training_manifest = read_json(training/'manifest.json')
        assert read_json(training/'status.json').get('returncode') == 0
        assert model_code.resolve() == (Path(training_manifest['repo'])/'nanochat').resolve()
        for name, digest in training_manifest['source_sha256'].items():
            if name.startswith('nanochat/'):
                assert sha256(Path(training_manifest['repo'])/name) == digest, name
    shutil.copytree(model_code, run / "code/nanochat", ignore=ignore)
    shutil.copytree(Path(__file__).parent, run / "code/scripts/inspect/raw_activation_norms", ignore=ignore)
    for name in ("pyproject.toml", "uv.lock"):
        if (repo / name).exists():
            shutil.copy2(repo / name, run / "code" / name)
    (run / "code/tracked_changes.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True))
    checkpoint_dir = args.base_dir / "base_checkpoints" / config["checkpoint_run"]
    frozen_checkpoint = run / "inputs/checkpoint"
    frozen_checkpoint.mkdir()
    for prefix, suffix in (("model", "pt"), ("meta", "json")):
        name = f"{prefix}_{config['checkpoint_step']:06d}.{suffix}"
        shutil.copy2(checkpoint_dir / name, frozen_checkpoint / name)
    metadata = read_json(frozen_checkpoint / f"meta_{config['checkpoint_step']:06d}.json")
    assert metadata["model_type"] == model_type
    assert metadata["model_config"]["n_layer"] == 24
    if config.get('training_run'):
        assert metadata['step'] == training_manifest['expected_steps'] == config['checkpoint_step']
        assert metadata['model_config'] == training_manifest['model_config']
    assert all(not v for v in metadata["model_config"].values() if isinstance(v, bool))
    tokenizer = reference/'inputs/runtime_base/tokenizer' if reference else args.base_dir/'tokenizer'
    shutil.copytree(tokenizer, run / "inputs/runtime_base/tokenizer")
    if config.get('training_run'):
        for path, digest in training_manifest['tokenizer_sha256'].items():
            assert sha256(run/'inputs/runtime_base/tokenizer'/Path(path).name) == digest
        write_json(run/'inputs/training_reference.json', dict(run=str(training),
                   manifest_sha256=sha256(training/'manifest.json')))
    data_dir = args.base_dir / "base_data_climbmix"
    if not data_dir.exists():
        data_dir = args.base_dir / "base_data"
    source = reference/'inputs/validation.parquet' if reference else sorted(data_dir.glob("*.parquet"))[-1]
    shutil.copy2(source, run / "inputs/validation.parquet")
    hub = Path(os.environ.get("KERNELS_CACHE", os.environ.get("HF_KERNELS_CACHE",
                os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))))
    kernel = "models--varunneal--flash-attention-3"
    shutil.copytree(reference/'kernel_cache'/kernel if reference else hub/kernel, run / "kernel_cache" / kernel, ignore=ignore)
    env_overrides = dict(NANOCHAT_BASE_DIR=str(run / "inputs/runtime_base"), NANOCHAT_DTYPE="bfloat16",
        CUDA_VISIBLE_DEVICES=gpu_fields[1], PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_CACHE=str(run / "kernel_cache"), HUGGINGFACE_HUB_CACHE=str(run / "kernel_cache"),
        KERNELS_CACHE=str(run / "kernel_cache"), HF_KERNELS_CACHE=str(run / "kernel_cache"),
        XDG_CACHE_HOME=str(run / "runtime_cache"), HF_HOME=str(run / "runtime_cache/huggingface"),
        TORCH_HOME=str(run / "runtime_cache/torch"), CUDA_CACHE_PATH=str(run / "runtime_cache/cuda"),
        TRITON_CACHE_DIR=str(run / "runtime_cache/triton"))
    env = dict(os.environ, **env_overrides)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PYTHONPATH"):
        env.pop(key, None)
    write_json(run / "environment_overrides.json", env_overrides)
    write_json(run / "inputs/config.json", config)
    (run / "environment-packages.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    if reference:
        for name in ('token_ids.u32','document_offsets.npy','token_frequencies.npy','tokenization.json'):
            shutil.copy2(reference/'inputs'/name, run/'inputs'/name)
            assert sha256(run/'inputs'/name) == prior['inputs_hashes'][name]
        write_json(run/'inputs/token_reference.json', dict(run=str(reference), manifest_sha256=sha256(reference/'manifest.json'),
                   results_sha256=sha256(reference/'results.json'), sequence_length=config['sequence_length'], token_policy=config['token_policy']))
    else:
        subprocess.run([sys.executable, "-m", "scripts.inspect.raw_activation_norms.tokenize", "--run-dir", str(run)],
                       cwd=run / "code", env=dict(env, CUDA_VISIBLE_DEVICES=""), check=True)
    manifest = dict(created_utc=stamp, run_id=run.name, config=config, selected_gpu=gpu,
        gpu_uuid=gpu_fields[1], checkpoint_source=str(checkpoint_dir), dataset_source=str(source),
        dataset_source_sha256=sha256(source), tokenization=read_json(run / "inputs/tokenization.json"),
        environment_overrides=env_overrides, python_executable=sys.executable,
        packages_sha256=sha256(run / "environment-packages.txt"),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        inputs_hashes=tree_hashes(run / "inputs"), code_hashes=tree_hashes(run / "code"),
        kernel_cache_hashes=tree_hashes(run / "kernel_cache"))
    write_json(run / "manifest.json", manifest)
    write_json(run / "status.json", {"state": "prepared", "tokens": manifest["tokenization"]["tokens"]})
    print(f"PREPARED={run}", flush=True)


if __name__ == "__main__":
    main()
