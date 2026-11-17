"""A held run must never start training before explicit launch and GPU checks."""
from pathlib import Path
import importlib.util
import json
import subprocess
from unittest.mock import Mock

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root/'work/split_embed_launch.py'
    if not path.exists():
        path = root.parent/'launcher_core.py'
        if not path.exists():
            path = root.parent/'launch.py'  # Older frozen run's copy.
    spec = importlib.util.spec_from_file_location('split_launcher_under_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'HERE',tmp_path)
    config = dict(model_type='split',selected_gpus=[],expected_steps=8329,
        global_batch_tokens=1048576,gradient_accumulation_steps=128,
        validation=dict(status='passed',distributed_status='pending_automatic_preflight'),
        gpu_count=2,command=['/python','train'],repo=str(tmp_path),env={})
    (tmp_path/'manifest.json').write_text(json.dumps(config))
    monkeypatch.setattr(module,'verify',Mock())
    monkeypatch.setattr(module,'verify_gpus',Mock())
    monkeypatch.setattr(module.subprocess,'Popen',Mock(side_effect=AssertionError('Dataset training must not start')))
    return module


def test_default_only_verifies(launcher,monkeypatch):
    monkeypatch.setattr('sys.argv',['launch.py'])
    monkeypatch.setattr(launcher.subprocess,'run',Mock(side_effect=AssertionError('No synthetic or training process without --launch')))
    launcher.main()
    launcher.verify.assert_called_once()
    launcher.verify_gpus.assert_not_called()
    launcher.subprocess.Popen.assert_not_called()


def test_occupied_gpu_prevents_all_execution(launcher,monkeypatch):
    monkeypatch.setattr('sys.argv',['launch.py','--launch','--gpus','2,3'])
    monkeypatch.setattr(launcher.subprocess,'check_output',lambda *a,**k:'2, GPU-two\n3, GPU-three\n')
    launcher.verify_gpus.side_effect=RuntimeError('GPU is busy')
    monkeypatch.setattr(launcher.subprocess,'run',Mock(side_effect=AssertionError('No synthetic process on busy GPUs')))
    with pytest.raises(RuntimeError,match='busy'):
        launcher.main()
    launcher.subprocess.Popen.assert_not_called()
    assert not (launcher.HERE/'launch_request.json').exists()


def test_failed_preflight_prevents_training_and_allows_retry(launcher,monkeypatch):
    monkeypatch.setattr('sys.argv',['launch.py','--launch','--gpus','2,3'])
    monkeypatch.setattr(launcher.subprocess,'check_output',lambda *a,**k:'2, GPU-two\n3, GPU-three\n')
    monkeypatch.setattr(launcher.subprocess,'run',Mock(side_effect=subprocess.CalledProcessError(1,'synthetic test')))
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main()
    launcher.subprocess.Popen.assert_not_called()
    assert not (launcher.HERE/'launch_request.json').exists()
