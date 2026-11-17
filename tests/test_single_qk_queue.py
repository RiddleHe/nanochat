import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(params=['queue_single_after_split.py', 'queue_head_out_after_single.py'])
def queue(request):
    path = Path(__file__).resolve().parents[1]/'work'/request.param
    spec = importlib.util.spec_from_file_location('single_queue_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requires_success_and_all_final_shards(queue,tmp_path,monkeypatch):
    monkeypatch.setattr(queue.runner,'process_start',lambda pid:None)
    dep=dict(run_dir=str(tmp_path),pid=1,start_ticks='1',launcher_pid=2,
             launcher_start_ticks='2',final_step=8329,checkpoint_dir=str(tmp_path),gpu_count=2)
    def status(**kw):
        (tmp_path/'status.json').write_text(json.dumps(kw))
    status(status='running',training_started=True)
    assert queue.dependency_state(dep)[0]=='failed'
    monkeypatch.setattr(queue.runner,'process_start',lambda pid:str(pid))
    assert queue.dependency_state(dep)[0]=='waiting'
    status(status='failed',returncode=1,training_started=True)
    assert queue.dependency_state(dep)[0]=='failed'
    status(status='exited',returncode=0,training_started=True)
    assert queue.dependency_state(dep)[0]=='waiting'
    monkeypatch.setattr(queue.runner,'process_start',lambda pid:None)
    for name in ('model_008329.pt','optim_008329_rank0.pt'):
        (tmp_path/name).write_text('checkpoint')
    (tmp_path/'meta_008329.json').write_text(json.dumps(dict(step=8329,val_bpb=.7)))
    assert queue.dependency_state(dep)[0]=='failed'
    (tmp_path/'optim_008329_rank1.pt').write_text('checkpoint')
    assert queue.dependency_state(dep)[0]=='complete'
    (tmp_path/'meta_008329.json').write_text(json.dumps(dict(step=8328,val_bpb=.7)))
    assert queue.dependency_state(dep)[0]=='failed'
