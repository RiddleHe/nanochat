"""Pending GPU validation must never be bypassed by an explicit launch."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess
import pytest

@pytest.fixture
def launcher(tmp_path,monkeypatch):
    p=Path(__file__).resolve().parents[1]/'work/norm_input_launch.py'
    if not p.exists():p=Path(__file__).resolve().parents[2]/'launch.py'
    spec=importlib.util.spec_from_file_location('norm_launcher',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'HERE',tmp_path)
    config={'validation':{'status':'cpu_passed_pending_gpu','cpu':{'status':'passed'}},'research_name':'norm',
            'expected_steps':8779,'command':['/python'],'env':{},'repo':str(tmp_path)}
    (tmp_path/'manifest.json').write_text(json.dumps(config))
    core=SimpleNamespace(verify=Mock(),verify_gpus=Mock(),main=Mock())
    monkeypatch.setattr(m.importlib.util,'spec_from_file_location',lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _:None)))
    monkeypatch.setattr(m.importlib.util,'module_from_spec',lambda _:core)
    monkeypatch.setattr(m.subprocess,'check_output',Mock(return_value='1, GPU-one\n2, GPU-two\n'))
    monkeypatch.setattr(m.subprocess,'run',Mock())
    return m,core

def test_default_standby_does_not_execute_gpu_checks(launcher,monkeypatch):
    m,core=launcher;monkeypatch.setattr('sys.argv',['launch.py']);m.main()
    core.verify.assert_called_once();core.verify_gpus.assert_not_called();core.main.assert_not_called()
    m.subprocess.run.assert_not_called();m.subprocess.check_output.assert_not_called()

def test_busy_gpu_blocks_preflight_and_training(launcher,monkeypatch):
    m,core=launcher;monkeypatch.setattr('sys.argv',['launch.py','--launch','--gpus','1,2'])
    core.verify_gpus.side_effect=RuntimeError('busy')
    with pytest.raises(RuntimeError,match='busy'):m.main()
    core.main.assert_not_called();m.subprocess.run.assert_not_called()

def test_failed_gpu_preflight_blocks_training_and_finalization(launcher,monkeypatch):
    m,core=launcher;monkeypatch.setattr('sys.argv',['launch.py','--launch','--gpus','1,2'])
    m.subprocess.run.side_effect=subprocess.CalledProcessError(1,'synthetic')
    with pytest.raises(subprocess.CalledProcessError):m.main()
    assert m.subprocess.run.call_count==1;core.main.assert_not_called()
