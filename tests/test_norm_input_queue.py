"""An idle GPU pair and a verified single-head target are required to launch."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest

@pytest.fixture
def queue():
    p=Path(__file__).resolve().parents[1]/'work/queue_norm_input.py'
    if not p.exists():p=Path(__file__).resolve().parents[2]/'queue_norm_input.py'
    spec=importlib.util.spec_from_file_location('norm_queue',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def test_waits_for_both_gpus_and_two_consecutive_observations(queue,monkeypatch):
    monkeypatch.setattr(queue,'idle_gpus',Mock(side_effect=[False,True,False,True,True]))
    verify=Mock();monkeypatch.setattr(queue,'verify_target',verify)
    launch=Mock(return_value=Mock(returncode=0));monkeypatch.setattr(queue.subprocess,'run',launch)
    sleep=Mock();publish=Mock()
    queue.wait_and_launch(dict(deadline_epoch=float('inf'),target_run='/run',poll_seconds=60,gpus=[{'index':1},{'index':2}]),publish,sleep)
    assert sleep.call_count==4;verify.assert_called_once();launch.assert_called_once()
    assert launch.call_args.args[0][-1]=='1,2'
    assert publish.call_args.args==('finished',)

def test_failed_verification_never_launches(queue,monkeypatch):
    monkeypatch.setattr(queue,'idle_gpus',lambda _:True)
    monkeypatch.setattr(queue,'verify_target',Mock(side_effect=RuntimeError('changed')))
    launch=Mock();monkeypatch.setattr(queue.subprocess,'run',launch)
    with pytest.raises(RuntimeError,match='changed'):
        queue.wait_and_launch(dict(deadline_epoch=float('inf'),target_run='/run',poll_seconds=0,gpus=[]),Mock(),Mock())
    launch.assert_not_called()

def test_busy_compute_process_prevents_idle_even_at_zero_utilization(queue,monkeypatch):
    monkeypatch.setattr(queue.subprocess,'check_output',Mock(side_effect=['1, GPU-one, 4, 0\n2, GPU-two, 4, 0\n','GPU-two, 123\n']))
    assert not queue.idle_gpus([{'index':1,'uuid':'GPU-one'},{'index':2,'uuid':'GPU-two'}])
