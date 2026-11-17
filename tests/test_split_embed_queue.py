"""A queued launch requires successful dependencies and idle GPUs, not just exited PIDs."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def queue():
    path = Path(__file__).resolve().parents[1] / 'work/queue_split_after_perloop.py'
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / 'queue_split_after_perloop.py'
    spec = importlib.util.spec_from_file_location('queue_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exited_process_without_completion_fails(queue, tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'process_start', lambda pid: None)
    dep = dict(pid=99999, start_ticks='100', run_dir=str(tmp_path), final_step=19074)
    assert queue.dependency_state(dep)[0] == 'failed'
    (tmp_path/'done.json').write_text(json.dumps(dict(final_step=19074,val_bpb=.8)))
    assert queue.dependency_state(dep)[0] == 'failed'  # Missing checkpoint.
    (tmp_path/'checkpoints').mkdir()
    for name in ('model_019074.pt','meta_019074.json','state_019074.pt'):
        (tmp_path/'checkpoints'/name).write_text('data')
    assert queue.dependency_state(dep)[0] == 'complete'
    monkeypatch.setattr(queue, 'process_start', lambda pid: '100')
    assert queue.dependency_state(dep)[0] == 'waiting'


def test_waits_for_both_then_two_idle_checks_and_launches_once(queue, monkeypatch):
    phase = [0]
    monkeypatch.setattr(queue.time, 'time', lambda: phase[0])
    def dependency(dep):
        return ('waiting','running') if dep['name']=='v2' and phase[0] == 0 else ('complete','done')
    monkeypatch.setattr(queue, 'dependency_state', dependency)
    idle = Mock(side_effect=[False, True, True])
    monkeypatch.setattr(queue, 'idle_gpus', idle)
    verify = Mock()
    monkeypatch.setattr(queue, 'verify_target', verify)
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(queue.subprocess, 'run', run)
    config = dict(deadline_epoch=10, dependencies=[dict(name='v1'),dict(name='v2')],
                  gpus=[], target_run='/frozen/split', poll_seconds=1)
    queue.wait_and_launch(config, Mock(), sleep=lambda _: phase.__setitem__(0, phase[0]+1))
    assert phase[0] == 3
    verify.assert_called_once_with(config)
    run.assert_called_once()
    assert run.call_args.args[0] == ['bash','/frozen/split/launch.sh','--launch','--gpus','4,7']


def test_dependency_failure_never_launches(queue, monkeypatch):
    monkeypatch.setattr(queue, 'dependency_state', lambda d: ('failed','missing final result'))
    run = Mock()
    monkeypatch.setattr(queue.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Dependency failed'):
        queue.wait_and_launch(dict(deadline_epoch=float('inf'),dependencies=[dict(name='v1')]),Mock())
    run.assert_not_called()
