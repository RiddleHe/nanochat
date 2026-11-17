#!/usr/bin/env python3
"""Prepare, check, test, launch, or wait for a saved training run.

Example (preparation only):
  python -m scripts.run prepare --recipe runs/bov/d24_f3p91e19/gpt_base_no_embed_norm.sh \
      --run-dir /absolute/new/run --base-dir /existing/data --gpu-count 2
Then use /absolute/new/run/launch.sh check, test, launch --gpus 0,1,
or wait --gpus 0,1 --after /absolute/previous/run. Launch always tests first.
Nothing installs packages, downloads inputs, or edits an existing run's code.
"""

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
RANK_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Publish status/reports atomically so waiting processes never read half a file."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
    os.replace(out.name, path)


def status(run, state, **extra):
    value = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(),
                 controller_pid=os.getpid(), **extra)
    write(run / "status.json", value)
    with (run / "events.jsonl").open("a") as log:
        log.write(json.dumps(value, allow_nan=False) + "\n")


def clean_env(manifest=None, gpus=""):
    env = os.environ.copy()
    for key in RANK_ENV:
        env.pop(key, None)
    # Do not inherit a caller's compilation/dtype overrides into validation/training.
    env.pop("TORCH_COMPILE_DISABLE", None)
    if manifest:
        env.update(manifest["env"])
    env.update(CUDA_VISIBLE_DEVICES=gpus, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def stop_child(process):
    """Stop this controller's process group, including torchrun workers."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_checked(command, *, cwd, env, output):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        stop_child(process)
        raise


def environment():
    return dict(python=sys.version, executable=os.path.abspath(sys.executable),
                executable_sha256=sha(sys.executable), packages=sorted(
                    [d.metadata["Name"], d.version] for d in importlib.metadata.distributions()
                    if d.metadata["Name"]))


def snapshot(repo, destination):
    """Copy tracked and untracked source; include edits, exclude ignored outputs."""
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).decode().split("\0")
    hashes = {}
    for name in sorted(set(names) - {""}):
        source = repo / name
        # Documentation is deliberately excluded from executable snapshots.
        if not source.is_file() or source.suffix.lower() in (".md", ".pyc"):
            continue
        if source.is_symlink() and not source.resolve().is_relative_to(repo):
            raise ValueError(f"Source symlink leaves checkout: {source}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, target)
        if sha(target) != before or sha(source) != before:
            raise RuntimeError(f"Source changed while copying: {source}")
        hashes[name] = before
    # Detect edits to an earlier file while copying later files.
    for name, digest in hashes.items():
        if sha(repo / name) != digest:
            raise RuntimeError(f"Source changed while copying: {name}")
    return hashes


def model_type(manifest):
    return next(arg.split("=", 1)[1] for arg in manifest["command"] if arg.startswith("--model-type="))


def prepare(args):
    run = args.run_dir.resolve()
    if run.is_relative_to(REPO):
        raise ValueError("Run directory must be outside the source checkout")
    recipe = args.recipe.resolve()
    relative = recipe.relative_to(REPO)
    if recipe.suffix not in (".sh", ".json"):
        raise ValueError("Recipe must be a canonical .sh entry point or a .json recipe")
    if recipe.suffix == ".json" and not args.model_type:
        raise ValueError("A JSON recipe requires --model-type")
    run.mkdir(parents=True, exist_ok=False)
    status(run, "preparing", training_started=False)
    try:
        code = run / "code"
        hashes = snapshot(REPO, code)
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        python = os.path.abspath(sys.executable)
        env = clean_env()
        env.update(PYTHON=python, PYTHONPATH=str(code), NANOCHAT_BASE_DIR=str(args.base_dir.resolve()))
        command = (["bash", str(code / relative)] if recipe.suffix == ".sh" else
                   [python, str(code / "runs/lib/pretrain.py"), "--recipe", str(code / relative),
                    "--tag-prefix", args.run_dir.name])
        command += ["--dry-run", "--manifest", str(run / "manifest.json"), "--source-commit", commit,
                    "--run-id", run.name, "--base-dir", str(args.base_dir.resolve()),
                    "--gpu-count", str(args.gpu_count)]
        if args.model_type:
            command += ["--model-type", args.model_type]
        if args.device_batch_size is not None:
            command += ["--device-batch-size", str(args.device_batch_size)]
        for flag in ("norm_value_input", "sum_head_attention"):
            if getattr(args, flag):
                command += ["--" + flag.replace("_", "-")]
        with (run / "prepare.log").open("x") as log:
            subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = read(run / "manifest.json")
        if manifest["gpu_count"] != args.gpu_count:
            raise ValueError("Recipe overrides requested GPU count; use its supported count")
        manifest.update(run_tools_version=1, model_type=model_type(manifest),
                        snapshot_sha256=hashes, external_sha256={})
        # Freeze environment settings affecting kernel discovery; never acquire packages online.
        manifest["env"].update({k: env[k] for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
        for key in ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
            if key in os.environ:
                manifest["env"][key] = os.environ[key]
        for dependency in args.dependency:
            dependency = dependency.expanduser().resolve()
            if not dependency.exists():
                raise FileNotFoundError(dependency)
            files = [dependency] if dependency.is_file() else sorted(dependency.rglob("*"))
            for path in files:
                if path.is_file():
                    manifest["external_sha256"][str(path)] = sha(path)
        # Cache files are external inputs, just like tokenizer files. Pin existing FA3 cache.
        hub = Path(env.get("HF_HUB_CACHE", str(Path(env.get("HF_HOME", str(
            Path(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))) / "hub")))
        manifest["env"]["HF_HUB_CACHE"] = str(hub.expanduser().resolve())
        kernel = hub / "models--varunneal--flash-attention-3"
        for path in sorted(kernel.rglob("*")):
            if path.is_file() and ".locks" not in path.parts:
                manifest["external_sha256"][str(path.absolute())] = sha(path)
        write(run / "environment.json", environment())
        launcher = run / "launch.sh"
        launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                            'if [ "$#" -eq 0 ]; then set -- check; fi\n'
                            f"exec {shlex.quote(python)} {shlex.quote(str(code / 'scripts/run.py'))} "
                            f'"$@" --run-dir {shlex.quote(str(run))}\n')
        launcher.chmod(0o755)
        manifest["artifact_sha256"] = {name: sha(run / name) for name in ("environment.json", "launch.sh")}
        write(run / "manifest.json", manifest)
        (run / "manifest.sha256").write_text(sha(run / "manifest.json") + "\n")
        status(run, "prepared", training_started=False)
    except BaseException as exc:
        status(run, "preparation_failed", error=str(exc), training_started=False)
        raise
    print(f"Prepared {run}; no tests, queue, or training started.\nLauncher: {launcher}")


def verify(run, fresh=True):
    if sha(run / "manifest.json") != (run / "manifest.sha256").read_text().strip():
        raise ValueError("Saved manifest changed; prepare a new run")
    m = read(run / "manifest.json")
    if m.get("run_tools_version") != 1 or Path(m["repo"]) != run / "code":
        raise ValueError("Unsupported or relocated run")
    for root, key in ((run / "code", "snapshot_sha256"), (run, "artifact_sha256"),
                      (Path("/"), "tokenizer_sha256"), (Path("/"), "external_sha256")):
        for name, digest in m[key].items():
            if sha(root / name) != digest:
                raise ValueError(f"Changed input: {root / name}")
    # New Python modules can change imports even when all old files still match.
    actual = {str(p.relative_to(run / "code")) for p in (run / "code").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(m["snapshot_sha256"]):
        raise ValueError("Saved source file inventory changed")
    for item in m["dataset_inventory"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise ValueError(f"Dataset changed: {item['path']}")
    if dataset_paths(m) != [item["path"] for item in m["dataset_inventory"]]:
        raise ValueError("Dataset shard inventory changed")
    if environment() != read(run / "environment.json"):
        raise ValueError("Python environment changed; prepare a new run")
    if fresh:
        for key in ("training_log", "checkpoint_dir"):
            if Path(m[key]).exists():
                raise FileExistsError(m[key])
        if (run / "started.json").exists():
            raise FileExistsError("This run has already attempted training")
    return m


def dataset_paths(manifest):
    # Use the saved dataset implementation, including its legacy-directory fallback.
    output = subprocess.check_output(
        [manifest["command"][0], "-c", "import json; from nanochat.dataset import list_parquet_files; "
         "print(json.dumps(list_parquet_files()))"], cwd=manifest["repo"], env=clean_env(manifest), text=True)
    return json.loads(output)


@contextmanager
def locked(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another controller holds {path}") from exc
        yield


def gpu_inventory():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if "," in line}
    result = {}
    for row in rows.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in row.split(",")]
        result[index] = dict(index=index, uuid=uuid, memory_mib=int(memory),
                             utilization=int(utilization), compute_process=uuid in busy)
    return result


def select_gpus(selection, count):
    parts = selection.split(",")
    inventory = gpu_inventory()
    by_uuid = {v["uuid"]: v for v in inventory.values()}
    selected = []
    for part in parts:
        item = inventory.get(part.strip(), by_uuid.get(part.strip()))
        if item is None:
            raise ValueError(f"Unknown GPU: {part}")
        selected.append(item)
    if len(selected) != count or len({x["uuid"] for x in selected}) != count:
        raise ValueError(f"Select exactly {count} distinct GPUs")
    return selected


def idle(selected):
    return all(not x["compute_process"] and x["memory_mib"] <= 100 and x["utilization"] <= 5
               for x in selected)


def final_checkpoint(m):
    directory, step = Path(m["checkpoint_dir"]), m["expected_steps"]
    files = [directory / f"model_{step:06d}.pt", directory / f"meta_{step:06d}.json"]
    files += [directory / f"optim_{step:06d}_rank{rank}.pt" for rank in range(m["gpu_count"])]
    if not all(p.is_file() and p.stat().st_size > 0 for p in files):
        raise ValueError("Final checkpoint is incomplete")
    meta = read(files[1])
    bpb = meta.get("val_bpb")
    if (meta.get("step") != step or not isinstance(bpb, (float, int)) or not math.isfinite(bpb)
            or meta.get("model_type") != model_type(m) or meta.get("model_config") != m["model_config"]):
        raise ValueError("Final checkpoint metadata does not match run or has nonfinite validation")


def dependency_ready(run):
    """Accept existing external runs as well as runs prepared by this command."""
    state = read(run / "status.json")
    if state["status"] in ("failed", "interrupted", "preparation_failed", "validation_failed"):
        raise RuntimeError(f"Dependency failed: {run}")
    if state["status"] != "completed":
        if state.get("exit_code") not in (None, 0):
            raise RuntimeError(f"Dependency exited unsuccessfully: {run}")
        return False
    if state.get("exit_code") != 0:
        raise RuntimeError(f"Dependency lacks successful exit: {run}")
    final_checkpoint(read(run / "manifest.json"))
    return True


def validate(run, m, selected=None):
    """Always refresh reports; bind evidence to the exact manifest and chosen GPUs."""
    reports = run / "checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports.mkdir(parents=True)
    modes = ["cpu"] + (["gpu"] + (["distributed"] if m["gpu_count"] > 1 else []) if selected else [])
    result = dict(status="running", manifest_sha256=sha(run / "manifest.json"),
                  selected_gpus=selected or [], reports=str(reports), modes=modes)
    write(run / "validation.json", result)
    try:
        for mode in modes:
            env = clean_env(m)
            if mode == "cpu":
                env.update(NANOCHAT_DTYPE="float32", TORCH_COMPILE_DISABLE="1")
            else:
                # Another user may have occupied an idle GPU during CPU checks or
                # compilation on the first GPU. Check again before every GPU phase.
                current = select_gpus(",".join(x["uuid"] for x in selected), m["gpu_count"])
                if not idle(current):
                    raise RuntimeError(f"Selected GPUs became occupied before {mode} checks")
                env["CUDA_VISIBLE_DEVICES"] = ",".join(x["uuid"] for x in (
                    selected if mode == "distributed" else selected[:1]))
            command = [m["command"][0]]
            if mode == "distributed":
                command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                            f"--nproc-per-node={m['gpu_count']}"]
            command += [str(run / "code/tests/model_checks/check.py"), "--manifest", str(run / "manifest.json"),
                        "--mode", mode, "--report", str(reports / f"{mode}.json")]
            with (reports / f"{mode}.log").open("x") as log:
                run_checked(command, cwd=m["repo"], env=env, output=log)
            report = read(reports / f"{mode}.json")
            if report["status"] != "passed" or report["mode"] != mode:
                raise RuntimeError(f"Invalid check report: {mode}")
        verify(run)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        write(run / "validation.json", result)
    return result


def train(run, m, selected):
    validate(run, m, selected)
    verify(run)
    selected = select_gpus(",".join(x["uuid"] for x in selected), m["gpu_count"])
    if not idle(selected):
        raise RuntimeError("Selected GPUs became occupied during checks")
    env = clean_env(m, ",".join(x["uuid"] for x in selected))
    log_path = Path(m["training_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (run / "started.json").open("x") as marker:
        json.dump(dict(selected_gpus=selected, command=m["command"]), marker)
    process = None
    try:
        with log_path.open("x") as output:
            process = subprocess.Popen(m["command"], cwd=m["repo"], env=env,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            status(run, "running", training_started=True, training_pid=process.pid, selected_gpus=selected)
            code = process.wait()
        if code:
            raise RuntimeError(f"Trainer exited with code {code}")
        final_checkpoint(m)
        status(run, "completed", training_started=True, exit_code=0)
    except BaseException as exc:
        if process is not None:
            stop_child(process)
        status(run, "failed", training_started=process is not None,
               exit_code=process.returncode if process else None, error=str(exc))
        raise


def operate(args):
    run = args.run_dir.resolve()
    verify(run, fresh=args.action != "check")
    if args.action == "check":
        print(f"Inputs verified: {run}; no GPU work or training started.")
        return
    with locked(run / "controller.lock"):
        try:
            operate_locked(args, run)
        except BaseException as exc:
            if not (run / "started.json").exists():
                status(run, "failed", training_started=False, error=str(exc))
            raise


def operate_locked(args, run):
    m = verify(run)
    if args.action == "test" and not args.gpus:
        print(json.dumps(validate(run, m), indent=2))
        return
    if not args.gpus:
        raise ValueError("Explicit --gpus is required")
    if args.action == "wait":
        if (not all(math.isfinite(v) for v in (args.poll_seconds, args.timeout_seconds))
                or args.poll_seconds < 1 or args.idle_samples < 2 or args.timeout_seconds <= 0):
            raise ValueError("Use positive timeout, poll >= 1 second, and at least 2 idle samples")
        after = [p.resolve() for p in args.after]
        if run in after:
            raise ValueError("A run cannot depend on itself")
        deadline, samples = time.monotonic() + args.timeout_seconds, 0
        status(run, "waiting", training_started=False, dependencies=list(map(str, after)))
        # Resolve indices to stable UUIDs before waiting; no CUDA context is created.
        selection = ",".join(x["uuid"] for x in select_gpus(args.gpus, m["gpu_count"]))
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Wait deadline exceeded; training was not launched")
            # Evaluate every dependency so a later failed run is not hidden by an earlier pending one.
            ready = [dependency_ready(p) for p in after]
            if all(ready):
                selected = select_gpus(selection, m["gpu_count"])
                samples = samples + 1 if idle(selected) else 0
                if samples >= args.idle_samples:
                    break
            else:
                samples = 0
            status(run, "waiting", training_started=False, dependencies=list(map(str, after)),
                   dependencies_ready=ready, consecutive_idle_samples=samples)
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    else:
        selected = select_gpus(args.gpus, m["gpu_count"])
    if not idle(selected):
        raise RuntimeError("Selected GPUs are occupied; no GPU checks or training started")
    # Cooperating controllers belonging to this user cannot race on the same GPUs.
    # Other users/tools do not take these locks: this is not a cluster reservation.
    locks = Path(tempfile.gettempdir()) / f"nanochat-gpu-locks-{os.getuid()}"
    locks.mkdir(mode=0o700, exist_ok=True)
    with ExitStack() as stack:
        for item in sorted(selected, key=lambda x: x["uuid"]):
            stack.enter_context(locked(locks / (item["uuid"] + ".lock")))
        selected = select_gpus(",".join(x["uuid"] for x in selected), m["gpu_count"])
        if not idle(selected):
            raise RuntimeError("Selected GPUs became occupied")
        if args.action == "test":
            print(json.dumps(validate(run, m, selected), indent=2))
        else:
            status(run, "validating", training_started=False)
            train(run, m, selected)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare", help="Copy code and resolve settings; no GPU execution")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--gpu-count", type=int, default=2)
    p.add_argument("--device-batch-size", type=int)
    p.add_argument("--model-type", help="Registry name; required for JSON recipes, optional shell recipe override")
    p.add_argument("--norm-value-input", action="store_true")
    p.add_argument("--sum-head-attention", action="store_true")
    p.add_argument("--dependency", type=Path, action="append", default=[], help="Additional file/directory to hash and verify")
    for action in ("check", "test", "launch", "wait"):
        command = commands.add_parser(action)
        if action != "check":
            command.add_argument("--gpus", help="Comma-separated physical indices or full GPU UUIDs; test defaults to CPU")
        if action == "wait":
            command.add_argument("--after", type=Path, action="append", default=[])
            command.add_argument("--poll-seconds", type=float, default=30)
            command.add_argument("--idle-samples", type=int, default=2)
            command.add_argument("--timeout-seconds", type=float, default=86400)
    for command in commands.choices.values():
        command.add_argument("--run-dir", type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    # SIGTERM should stop a controller's own child process, not orphan training.
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    if args.action == "prepare":
        prepare(args)
    else:
        operate(args)


if __name__ == "__main__":
    main()
