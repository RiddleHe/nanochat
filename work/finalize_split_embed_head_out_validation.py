"""Seal the split-embedding/per-head-output run after validation; never launch training."""
from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent
manifest = json.loads((here/'manifest.json').read_text())
assert not (here/'launch_request.json').exists()
cpu = json.loads((here/'cpu_validation.json').read_text())
gpu = json.loads((here/'gpu_validation.json').read_text())
assert cpu['status'] == gpu['status'] == 'passed'
assert gpu['parameter_count'] == manifest['parameter_counts']['total'] == 754188288
assert gpu['architecture_training_flops_per_token'] == manifest['estimated_flops_per_token'] == 4619501568
assert gpu['architecture_forward_flops_per_token'] == 1539833856
assert gpu['output_updates_verified'] and gpu['cached_generation_verified']
assert gpu['full_model_optimizer_steps'] == 1
assert gpu['full_shape'] == dict(batch=2,sequence_len=2048,depth=24,width=1536)
ddp_file = here/'distributed_validation.json'
ddp = json.loads(ddp_file.read_text()) if ddp_file.exists() else None
if ddp:
    assert ddp['status'] == 'passed' and ddp['world_size'] == 2
    assert ddp['parameters_identical_after_update'] and ddp['parameter_count'] == 754188288
    assert ddp['architecture_training_flops_per_token'] == 4619501568
manifest['validation'] = dict(status='passed',cpu=cpu,
    gpu='BF16 eager/compiled gradients, output matrix optimizer updates, cached generation, full D24 B2 T2048 forward/backward/update',
    distributed_status='passed' if ddp else 'pending_automatic_preflight',
    training_started=False)
files = [p for p in here.iterdir() if p.is_file() and p.name not in
         ('manifest.json','status.json','launch.lock')]
files += list((here/'code/tests').rglob('*.py'))
manifest['artifact_sha256'] = {str(p.relative_to(here)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(files)}
(here/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(here/'status.json').write_text(json.dumps(dict(status='ready',run_id=manifest['model_tag'],
    validation='cpu_gpu_passed',distributed_validation=manifest['validation']['distributed_status'],
    training_started=False),indent=2)+'\n')
print(json.dumps(manifest['validation'],indent=2))
