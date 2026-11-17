#!/usr/bin/env python3
"""Resolve a repeatable pretraining recipe; no GPU work unless --launch is given."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(args):
    base = Path(args.base_dir).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"Existing data directory required: {base}")
    recipe = json.loads(args.recipe.read_text())
    if recipe["seed"] != 42:
        raise ValueError("This trainer fixes the seed at 42; other seeds need trainer support")
    train = dict(recipe["trainer_args"])
    if args.device_batch_size is not None:
        train["device-batch-size"] = args.device_batch_size
    if args.gpu_count < 1 or train["device-batch-size"] < 1:
        raise ValueError("GPU count and device batch size must be positive")
    if train["num-iterations"] != -1 or train["target-flops"] <= 0:
        raise ValueError("This recipe must resolve its horizon from the FLOP budget")
    batch = train["total-batch-size"]
    micro_batch = train["device-batch-size"] * train["max-seq-len"] * args.gpu_count
    if batch <= 0 or batch % micro_batch:
        raise ValueError("Global token batch must be divisible by the distributed microbatch")
    if train["eval-tokens"] < micro_batch or train["eval-tokens"] % micro_batch:
        raise ValueError("Evaluation token budget must be divisible by the distributed microbatch")

    # Hide GPUs BEFORE importing torch/model code; preflight is always CPU/meta only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NANOCHAT_BASE_DIR"] = str(base)
    os.environ["NANOCHAT_DTYPE"] = recipe["dtype"]
    sys.path.insert(0, str(REPO))
    import torch
    from nanochat.dataset import list_parquet_files
    from nanochat.model_registry import get_model
    from nanochat.tokenizer import get_tokenizer

    tokenizer_paths = [base / "tokenizer" / name for name in ("tokenizer.pkl", "token_bytes.pt")]
    for path in tokenizer_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    vocab_size = get_tokenizer().get_vocab_size()
    paths = list_parquet_files()
    if len(paths) < 2:
        raise ValueError("Need existing training shards plus a validation shard")
    inventory = []
    for path in paths:
        item = Path(path)
        stat = item.stat()
        inventory.append(dict(path=str(item), size=stat.st_size, mtime_ns=stat.st_mtime_ns))

    Config, Model = get_model(args.model_type)
    value_options = {name: True for name in ("norm_value_input", "sum_head_attention")
                     if getattr(args, name, False)}
    width = train["depth"] * train["aspect-ratio"]
    head_dim = train["head-dim"]
    width = ((width + head_dim - 1) // head_dim) * head_dim
    config = Config(sequence_len=train["max-seq-len"], vocab_size=vocab_size,
                    n_layer=train["depth"], n_head=width // head_dim,
                    n_kv_head=width // head_dim, n_embd=width,
                    window_pattern=train["window-pattern"], **value_options)
    flags = {key: value for key, value in asdict(config).items() if type(value) is bool}
    if args.expect_all_flags_false and (not flags or any(flags.values())):
        raise ValueError(f"Baseline requires all architecture flags false: {flags}")
    with torch.device("meta"), redirect_stdout(sys.stderr):
        model = Model(config)
    fpt = model.estimate_flops()
    steps = round(train["target-flops"] / (fpt * batch))
    if steps < 1:
        raise ValueError("FLOP budget resolves to zero steps")
    if args.expect_all_flags_false and any("smear" in name for name in model.state_dict()):
        raise ValueError("This baseline requires the smear-free model")

    tag = args.run_id or f"{args.tag_prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("Run ID must be a simple filename, without slashes")
    train.update({"model-type": args.model_type, "model-tag": tag})
    python = os.path.abspath(sys.executable)  # Keep the virtualenv path, not its symlink target.
    command = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc-per-node={args.gpu_count}", "--module", "--", "scripts.base_train"]
    command.extend(f"--{key}={value}" for key, value in train.items())
    command.extend("--" + name.replace("_", "-") for name in value_options)
    commit = args.source_commit or subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sources = {}
    for directory in ("nanochat", "scripts", "tasks", "configs", "runs"):
        for path in sorted((REPO / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".sh", ".json", ".yaml", ".yml"):
                sources[str(path.relative_to(REPO))] = sha256(path)
    for name in ("pyproject.toml", "uv.lock"):
        sources[name] = sha256(REPO / name)
    return dict(
        schema_version=1, prepared_at=datetime.now(timezone.utc).isoformat(),
        repo=str(REPO), git_head=commit, recipe=str(args.recipe.resolve()),
        recipe_reference=recipe["reference"], model_tag=tag, model_config=asdict(config),
        all_boolean_flags_false=bool(flags) and not any(flags.values()),
        parameter_counts=model.num_scaling_params(), seed=recipe["seed"], command=command,
        env={"NANOCHAT_BASE_DIR": str(base), "NANOCHAT_DTYPE": recipe["dtype"],
             "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": str(REPO)},
        training_log=str(base / "arch_d24" / f"{tag}_train.log"),
        checkpoint_dir=str(base / "base_checkpoints" / tag),
        gpu_count=args.gpu_count, probe_seconds=60, idle_wait_seconds=300,
        max_idle_memory_mib=100, max_idle_utilization=5,
        target_flops=train["target-flops"], estimated_flops_per_token=fpt,
        expected_steps=steps, expected_tokens=steps * batch,
        expected_actual_flops=steps * batch * fpt, global_batch_tokens=batch,
        device_batch_size=train["device-batch-size"], gradient_accumulation_steps=batch // micro_batch,
        source_sha256=sources, tokenizer_sha256={str(p): sha256(p) for p in tokenizer_paths},
        dataset_inventory=inventory, training_shards=len(paths) - 1, validation_shard=paths[-1],
        runtime={"python": sys.version, "torch": torch.__version__},
        resume_policy="Fresh run; complete model/meta/all-rank optimizer files required for any later resume. Exact data/RNG replay is not supported by this trainer.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--expect-all-flags-false", action="store_true")
    parser.add_argument("--norm-value-input", action="store_true")
    parser.add_argument("--sum-head-attention", action="store_true")
    parser.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR"), required=not os.environ.get("NANOCHAT_BASE_DIR"))
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--device-batch-size", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--source-commit", help="Parent Git revision when resolving a source snapshot")
    parser.add_argument("--manifest", type=Path, help="Write a NEW resolved manifest; never starts a queue")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: resolve and print only")
    mode.add_argument("--launch", action="store_true", help="Start training now on explicitly selected GPUs")
    args = parser.parse_args()
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.launch and (len(selected.split(",")) != args.gpu_count or not all(s.strip() for s in selected.split(","))):
        parser.error("--launch requires CUDA_VISIBLE_DEVICES with exactly --gpu-count entries")
    manifest = resolve(args)
    for name in ("training_log", "checkpoint_dir"):
        if Path(manifest[name]).exists():
            raise FileExistsError(f"Refusing to overwrite {manifest[name]}")
    if args.manifest:
        with args.manifest.open("x") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
    print(json.dumps({k: manifest[k] for k in (
        "model_tag", "git_head", "all_boolean_flags_false", "gpu_count", "parameter_counts",
        "expected_steps", "expected_tokens", "expected_actual_flops", "global_batch_tokens",
        "gradient_accumulation_steps", "training_shards", "validation_shard", "training_log", "checkpoint_dir")}, indent=2))
    print(shlex.join(manifest["command"]), flush=True)
    if args.launch:
        env = os.environ.copy()
        env.update(manifest["env"])
        env["CUDA_VISIBLE_DEVICES"] = selected
        log_path = Path(manifest["training_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as output:
            result = subprocess.run(manifest["command"], cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
        raise SystemExit(result.returncode)
    print("Prepared only: no training or queue process started.")


if __name__ == "__main__":
    main()
