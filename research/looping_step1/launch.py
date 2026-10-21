"""Three bounded workers; GPU locks, no duplicate jobs, no automatic failure retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, SOURCE, GROUPS, VARIANTS, SETTINGS, now, write_json, verify_source

PYTHON = sys.executable


def lock(path):
    f = open(path, "a+")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f


def gpu_empty(index):
    uuid = subprocess.check_output([
        "nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"
    ], text=True)
    return not any(uuid in line for line in processes.splitlines())


def claim():
    with open(ROOT / "queue.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        queue = json.loads((ROOT / "queue.json").read_text())
        for job in queue:
            if job["phase"] == "pending":
                job.update(phase="claimed", pid=os.getpid(), claimed_utc=now())
                write_json(ROOT / "queue.json", queue)
                return job["job"]
    return None


def mark(job, phase):
    with open(ROOT / "queue.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        queue = json.loads((ROOT / "queue.json").read_text())
        for entry in queue:
            if entry["job"] == job:
                entry.update(phase=phase, updated_utc=now())
        write_json(ROOT / "queue.json", queue)


def worker(index, first):
    guard = lock(Path(f"/tmp/nanochat-looping-gpu{index}.lock"))
    if not gpu_empty(index):
        raise RuntimeError(f"GPU {index} acquired by another process; not launching")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), NANOCHAT_DTYPE="bfloat16",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    job = first
    while job:
        verify_source()
        if not gpu_empty(index):
            raise RuntimeError(f"Unexpected GPU process before {job}; not launching")
        write_json(ROOT / f"worker_{index}.json", dict(
            phase="running", time=now(), pid=os.getpid(), gpu=index, job=job))
        log = ROOT / "logs" / f"{job}.log"
        with open(log, "x") as output:
            child = subprocess.Popen([PYTHON, "-u", str(SOURCE / "experiments.py"), job],
                                     cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            write_json(ROOT / f"worker_{index}.json", dict(
                phase="running", time=now(), pid=os.getpid(), child_pid=child.pid, gpu=index, job=job))
            code = child.wait()
        if job not in ("gradient", "sensitivity"):
            mark(job, "complete" if code == 0 else "failed")
        if code:
            write_json(ROOT / f"worker_{index}.json", dict(
                phase="failed", time=now(), pid=os.getpid(), gpu=index, job=job, returncode=code))
            return 1
        job = claim()
    write_json(ROOT / f"worker_{index}.json", dict(
        phase="complete", time=now(), pid=os.getpid(), gpu=index))
    guard.close()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--first")
    args = parser.parse_args()
    if args.worker is not None:
        sys.exit(worker(args.worker, args.first))
    ROOT.mkdir(parents=True, exist_ok=True)
    supervisor = lock(ROOT / "supervisor.lock")
    verify_source()
    assert json.loads((ROOT / "tests_cpu.json").read_text())["passed"]
    assert json.loads((ROOT / "tests_gpu.json").read_text())["passed"]
    assert (ROOT / "experiment_manifest.json").is_file(), "Run prepare.py --freeze first"
    assert not (ROOT / "queue.json").exists(), "Refusing duplicate launch"
    for gpu in (0, 1, 2):
        if not gpu_empty(gpu):
            raise RuntimeError(f"GPU {gpu} not free")
    (ROOT / "logs").mkdir(exist_ok=True)
    jobs = [f"{g}_{v}" for g in GROUPS for v in VARIANTS] + ["baseline_eval"]
    queue = [dict(job=j, phase="pending") for j in jobs]
    queue[0].update(phase="claimed", claimed_utc=now())
    write_json(ROOT / "queue.json", queue)
    children = [
        subprocess.Popen([PYTHON, "-u", str(SOURCE / "launch.py"), "--worker", str(i), "--first", first])
        for i, first in [(0, "gradient"), (1, "sensitivity"), (2, jobs[0])]
    ]
    write_json(ROOT / "status.json", dict(
        phase="running", pid=os.getpid(), time=now(), workers=[c.pid for c in children], gpus=[0, 1, 2]))
    codes = [c.wait() for c in children]
    queue = json.loads((ROOT / "queue.json").read_text())
    passed = all(c == 0 for c in codes) and all(j["phase"] == "complete" for j in queue)
    if passed:
        with open(ROOT / "logs/plots.log", "x") as log:
            code = subprocess.call([PYTHON, str(SOURCE / "plot_results.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        passed = code == 0
    write_json(ROOT / "status.json", dict(
        phase="complete" if passed else "needs_attention", time=now(),
        worker_returncodes=codes, all_plots_generated=passed))
    supervisor.close()


if __name__ == "__main__":
    main()
