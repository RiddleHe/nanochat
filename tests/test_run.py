"""Shared run tools must fail before GPU work/training when prerequisites fail."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("run_tools", Path(__file__).resolve().parents[1] / "scripts/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    code = directory / "code"
    code.mkdir(parents=True)
    (code / "model.py").write_text("value = 1\n")
    data = tmp_path / "shard.parquet"
    data.write_bytes(b"data")
    token = tmp_path / "tokenizer.pkl"
    token.write_bytes(b"tokenizer")
    monkeypatch.setattr(run, "environment", lambda: {"python": "test"})
    run.write(directory / "environment.json", run.environment())
    manifest = dict(run_tools_version=1, repo=str(code), snapshot_sha256={"model.py": run.sha(code / "model.py")},
                    artifact_sha256={"environment.json": run.sha(directory / "environment.json")},
                    tokenizer_sha256={str(token): run.sha(token)}, external_sha256={},
                    dataset_inventory=[dict(path=str(data), size=4, mtime_ns=data.stat().st_mtime_ns)],
                    training_log=str(tmp_path / "training.log"), checkpoint_dir=str(tmp_path / "checkpoints"),
                    expected_steps=5, gpu_count=2, command=["python", "--model-type=example"],
                    model_config={"n_layer": 4}, env={})
    run.write(directory / "manifest.json", manifest)
    (directory / "manifest.sha256").write_text(run.sha(directory / "manifest.json"))
    run.status(directory, "prepared", training_started=False)
    monkeypatch.setattr(run, "dataset_paths", lambda m: sorted(str(p) for p in tmp_path.glob("*.parquet")))
    return directory, manifest


def test_verify_and_tampered_code(saved):
    directory, _ = saved
    run.verify(directory)
    (directory / "code/model.py").write_text("changed")
    with pytest.raises(ValueError, match="Changed input"):
        run.verify(directory)


@pytest.mark.parametrize("change", ["new_source", "data", "new_shard", "tokenizer", "environment", "manifest", "outputs", "started"])
def test_verify_refuses_changed_inputs_or_existing_outputs(saved, change):
    directory, manifest = saved
    if change == "new_source":
        (directory / "code/shadow.py").write_text("pass")
    elif change == "data":
        Path(manifest["dataset_inventory"][0]["path"]).write_bytes(b"changed data")
    elif change == "new_shard":
        (directory.parent / "added.parquet").write_bytes(b"new shard")
    elif change == "tokenizer":
        Path(next(iter(manifest["tokenizer_sha256"]))).write_bytes(b"changed tokenizer")
    elif change == "environment":
        run.write(directory / "environment.json", {"python": "different"})
    elif change == "manifest":
        run.write(directory / "manifest.json", {})
    elif change == "outputs":
        Path(manifest["checkpoint_dir"]).mkdir()
    else:
        (directory / "started.json").write_text("{}")
    with pytest.raises((ValueError, FileExistsError)):
        run.verify(directory)


def test_run_lock_prevents_duplicate_controller(tmp_path):
    with run.locked(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="Another controller"):
            with run.locked(tmp_path / "lock"):
                pytest.fail("Second controller acquired lock")


def test_idle_checks_compute_processes_even_at_zero_utilization():
    gpu = dict(compute_process=False, memory_mib=4, utilization=0)
    assert run.idle([gpu])
    for change in (dict(compute_process=True), dict(memory_mib=101), dict(utilization=6)):
        assert not run.idle([dict(gpu, **change)])


def test_gpu_aliases_cannot_select_same_device_twice(monkeypatch):
    monkeypatch.setattr(run, "gpu_inventory", lambda: {"0": {"index": "0", "uuid": "GPU-a"}})
    with pytest.raises(ValueError, match="distinct"):
        run.select_gpus("0,GPU-a", 2)
    with pytest.raises(ValueError, match="Unknown"):
        run.select_gpus("9", 1)


def complete_checkpoint(manifest):
    directory = Path(manifest["checkpoint_dir"])
    directory.mkdir()
    for name in ("model_000005.pt", "optim_000005_rank0.pt", "optim_000005_rank1.pt"):
        (directory / name).write_bytes(b"checkpoint")
    run.write(directory / "meta_000005.json", dict(step=5, val_bpb=1.2, model_type="example", model_config={"n_layer": 4}))


def test_dependency_requires_success_and_complete_checkpoint(saved):
    directory, manifest = saved
    assert not run.dependency_ready(directory)
    run.status(directory, "completed", exit_code=0)
    with pytest.raises(ValueError, match="incomplete"):
        run.dependency_ready(directory)
    complete_checkpoint(manifest)
    assert run.dependency_ready(directory)
    (Path(manifest["checkpoint_dir"]) / "optim_000005_rank1.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        run.dependency_ready(directory)


def test_dependency_failure_is_terminal(saved):
    directory, _ = saved
    run.status(directory, "failed", exit_code=1)
    with pytest.raises(RuntimeError, match="failed"):
        run.dependency_ready(directory)


def test_nonfinite_final_validation_is_not_success(saved):
    _, manifest = saved
    complete_checkpoint(manifest)
    meta = Path(manifest["checkpoint_dir"]) / "meta_000005.json"
    value = run.read(meta)
    value["val_bpb"] = float("nan")
    meta.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="nonfinite"):
        run.final_checkpoint(manifest)


def test_failed_validation_never_starts_training(saved, monkeypatch):
    directory, manifest = saved
    def fail(*args):
        raise RuntimeError("failed check")
    monkeypatch.setattr(run, "validate", fail)
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: pytest.fail("Training started"))
    with pytest.raises(RuntimeError, match="failed check"):
        run.train(directory, manifest, [])
    assert not (directory / "started.json").exists()
    assert not Path(manifest["training_log"]).exists()


def test_successful_launch_records_completion_and_cannot_repeat(saved, monkeypatch):
    directory, manifest = saved
    complete_checkpoint(manifest)
    # A fake child writes the checkpoint on exit, like the trainer does.
    checkpoint = Path(manifest["checkpoint_dir"])
    checkpoint.rename(directory / "pending_checkpoint")
    monkeypatch.setattr(run, "validate", lambda *a: None)
    monkeypatch.setattr(run, "select_gpus", lambda *a: [dict(uuid="GPU-test", compute_process=False,
                                                          memory_mib=0, utilization=0)])
    class Child:
        pid = 123
        returncode = 0
        def wait(self):
            (directory / "pending_checkpoint").rename(checkpoint)
            return 0
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: Child())
    run.train(directory, manifest, [dict(uuid="GPU-test")])
    assert run.read(directory / "status.json")["status"] == "completed"
    with pytest.raises(FileExistsError):
        run.verify(directory)


def test_failed_wait_dependency_stops_without_gpu_checks(saved, monkeypatch):
    directory, _ = saved
    monkeypatch.setattr(run, "select_gpus", lambda *a: [dict(uuid="GPU-test")])
    def fail(path):
        raise RuntimeError("Dependency failed")
    monkeypatch.setattr(run, "dependency_ready", fail)
    monkeypatch.setattr(run, "train", lambda *a: pytest.fail("Training started"))
    args = argparse.Namespace(action="wait", run_dir=directory, gpus="0,1", after=[directory.parent / "earlier"],
                              poll_seconds=1, idle_samples=2, timeout_seconds=30)
    with pytest.raises(RuntimeError, match="Dependency failed"):
        run.operate(args)
    assert run.read(directory / "status.json")["status"] == "failed"


def test_clean_environment_clears_inherited_rank_and_gpu(monkeypatch):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    env = run.clean_env()
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_OFFLINE"] == "1"
    assert all(key not in env for key in ("RANK", "WORLD_SIZE", "TORCH_COMPILE_DISABLE"))


def test_gpu_becoming_busy_during_cpu_checks_blocks_next_phase(saved, monkeypatch):
    directory, manifest = saved
    modes = []
    def checked(command, **kwargs):
        mode = command[command.index("--mode") + 1]
        modes.append(mode)
        run.write(command[command.index("--report") + 1], dict(status="passed", mode=mode))
    monkeypatch.setattr(run, "run_checked", checked)
    monkeypatch.setattr(run, "select_gpus", lambda *a: [dict(uuid="GPU-a", compute_process=True,
                                                          memory_mib=10000, utilization=100)])
    with pytest.raises(RuntimeError, match="became occupied before gpu"):
        run.validate(directory, manifest, [dict(uuid="GPU-a"), dict(uuid="GPU-b")])
    assert modes == ["cpu"]
    assert run.read(directory / "validation.json")["status"] == "failed"


def test_interrupted_validation_stops_its_process_group(monkeypatch):
    signals = []
    class Child:
        pid = 12345
        calls = 0
        def wait(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            return 0
        def poll(self):
            return None
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        run.run_checked(["test"], cwd="/tmp", env={}, output=None)
    assert signals == [(12345, run.signal.SIGTERM)]


def test_busy_gpu_blocks_validation_and_training(saved, monkeypatch):
    directory, _ = saved
    monkeypatch.setattr(run, "select_gpus", lambda *a: [dict(compute_process=True, memory_mib=4, utilization=0)])
    monkeypatch.setattr(run, "validate", lambda *a: pytest.fail("Validation touched occupied GPU"))
    args = argparse.Namespace(action="launch", run_dir=directory, gpus="0,1")
    with pytest.raises(RuntimeError, match="occupied"):
        run.operate(args)
    assert run.read(directory / "status.json")["status"] == "failed"


def test_snapshot_includes_local_edits_and_untracked_source(tmp_path):
    source, dest = tmp_path / "repo", tmp_path / "saved"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "model.py").write_text("original")
    (source / ".gitignore").write_text("ignored.py\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    (source / "model.py").write_text("local edit")
    (source / "new.py").write_text("new code")
    (source / "ignored.py").write_text("ignored")
    (source / "README.md").write_text("no documentation copies")
    hashes = run.snapshot(source, dest)
    assert (dest / "model.py").read_text() == "local edit"
    assert (dest / "new.py").is_file()
    assert not (dest / "ignored.py").exists()
    assert not (dest / "README.md").exists()
    assert "new.py" in hashes


def test_wait_requires_consecutive_idle_samples(saved, monkeypatch):
    directory, _ = saved
    samples = iter([True, True, False, True, True, True, True, True])
    monkeypatch.setattr(run, "select_gpus", lambda *a: [dict(uuid="GPU-test", index="0", compute_process=False,
                                                          memory_mib=4, utilization=0)])
    calls = []
    monkeypatch.setattr(run, "idle", lambda *a: next(samples))
    monkeypatch.setattr(run.time, "sleep", lambda seconds: calls.append(seconds))
    monkeypatch.setattr(run, "train", lambda *a: calls.append("launch"))
    args = argparse.Namespace(action="wait", run_dir=directory, gpus="0,1", after=[],
                              poll_seconds=1, idle_samples=3, timeout_seconds=30)
    run.operate(args)
    assert calls == [1, 1, 1, 1, 1, "launch"]
