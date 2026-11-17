#!/usr/bin/env python3
"""Validate the prepared snapshot by default; only --launch starts training."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(config):
    expected_packages = json.loads((HERE / "environment_packages.json").read_text())
    actual_packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if expected_packages != actual_packages:
        raise RuntimeError("Python package environment changed since validation")
    for base, hashes in (
        (Path(config["repo"]), config["source_sha256"]),
        (Path("/"), config["tokenizer_sha256"]),
        (Path(config["kernel_cache_path"]), config["kernel_cache_sha256"]),
        (HERE, config["artifact_sha256"]),
    ):
        for name, expected in hashes.items():
            if sha(base / name) != expected:
                raise RuntimeError(f"Prepared dependency changed: {base / name}")
    inventory = config["dataset_inventory"]
    directory = Path(inventory[0]["path"]).parent
    if sorted(str(p) for p in directory.glob("*.parquet")) != sorted(p["path"] for p in inventory):
        raise RuntimeError("Dataset shard list changed")
    for item in inventory:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise RuntimeError(f"Dataset shard changed: {item['path']}")
    for name in ("training_log", "checkpoint_dir"):
        if Path(config[name]).exists():
            raise RuntimeError(f"Refusing to overwrite {config[name]}")


def verify_gpus(config):
    def query(option):
        return subprocess.check_output(
            ["nvidia-smi", option, "--format=csv,noheader,nounits"],
            text=True, timeout=20,
        )
    occupied = {row.split(",")[0].strip() for row in
                query("--query-compute-apps=gpu_uuid,pid").splitlines() if row.strip()}
    inventory = {}
    for row in query("--query-gpu=index,uuid,memory.used,utilization.gpu").splitlines():
        index, uuid, memory, utilization = [field.strip() for field in row.split(",")]
        inventory[int(index)] = (uuid, int(memory), int(utilization))
    for gpu in config["selected_gpus"]:
        uuid, memory, utilization = inventory[gpu["index"]]
        if (uuid != gpu["uuid"] or uuid in occupied or
                memory > config["max_idle_memory_mib"] or
                utilization > config["max_idle_utilization"]):
            raise RuntimeError(f"GPU {gpu['index']} is busy or its identity changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    parser.add_argument("--gpus", help="Comma-separated physical GPU indices; required with --launch")
    args = parser.parse_args()
    config = json.loads((HERE / "manifest.json").read_text())
    verify(config)
    if not args.launch:
        print(json.dumps({k: config[k] for k in (
            "model_type", "selected_gpus", "expected_steps", "global_batch_tokens",
            "gradient_accumulation_steps", "validation")}, indent=2))
        print(shlex.join(config["command"]))
        print("Prepared and held: no queue or training started.")
        return
    if config["validation"].get("status") != "passed":
        raise RuntimeError("Validation must pass before launch")
    if not args.gpus:
        parser.error("--launch requires --gpus with explicit physical indices")
    requested = args.gpus.split(",")
    if len(requested) != config["gpu_count"] or not all(g.isdigit() for g in requested) or len(set(map(int, requested))) != len(requested):
        parser.error("Select exactly gpu_count distinct physical GPU indices")
    inventory = {}
    for row in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True).splitlines():
        index, uuid = row.split(",")
        inventory[int(index)] = uuid.strip()
    if any(int(g) not in inventory for g in requested):
        parser.error("Unknown GPU index")
    config["selected_gpus"] = [{"index": int(g), "uuid": inventory[int(g)]} for g in requested]
    config["env"]["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in config["selected_gpus"])
    with (HERE / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(config)
        verify_gpus(config)
        env = dict(os.environ, **config["env"])
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            env.pop(key, None)
        env["PATH"] = str(Path(config["command"][0]).parent) + os.pathsep + env["PATH"]
        # If two GPUs were unavailable at preparation, run the frozen synthetic
        # two-rank check now. It never reads training data or saves a checkpoint.
        # Failure leaves training unstarted, and permits a later explicit retry.
        if config["validation"].get("distributed_status") != "passed":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            preflight_log = HERE / f"distributed_preflight_{stamp}.log"
            command = [config["command"][0], "-m", "torch.distributed.run",
                       "--standalone", "--nproc-per-node=2", str(HERE / "distributed_smoke.py")]
            print(f"Running synthetic two-GPU preflight: {preflight_log}", flush=True)
            with preflight_log.open("x") as output:
                subprocess.run(command, cwd=config["repo"], env=env,
                               stdout=output, stderr=subprocess.STDOUT, check=True)
            result = json.loads((HERE / "distributed_validation.json").read_text())
            if not (result["status"] == "passed" and result["world_size"] == 2
                    and result["parameters_identical_after_update"]
                    and result["parameter_count"] == config["parameter_counts"]["total"]
                    and result["architecture_training_flops_per_token"] == config["estimated_flops_per_token"]):
                raise RuntimeError("Two-GPU preflight result does not match this run")
            verify(config)
            verify_gpus(config)
        with (HERE / "launch_request.json").open("x") as request:
            json.dump({"selected_gpus": config["selected_gpus"], "command": config["command"],
                       "requested_at": datetime.now(timezone.utc).isoformat()}, request, indent=2)
        state = dict(status="running", run_id=config["model_tag"],
                     selected_gpu_indices=[g["index"] for g in config["selected_gpus"]],
                     training_started=True, started_at=datetime.now(timezone.utc).isoformat())
        def publish():
            temporary = HERE / "status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(HERE / "status.json")
        log = Path(config["training_log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("x") as output:
                child = subprocess.Popen(config["command"], cwd=config["repo"], env=env,
                                         stdin=subprocess.DEVNULL, stdout=output,
                                         stderr=subprocess.STDOUT)
                state["training_pid"] = child.pid
                publish()
                result = child.wait()
            state.update(status="exited" if result == 0 else "failed", returncode=result,
                         finished_at=datetime.now(timezone.utc).isoformat())
            publish()
        except Exception as exc:
            state.update(status="failed", error=str(exc))
            publish()
            raise
        raise SystemExit(result)


if __name__ == "__main__":
    main()
