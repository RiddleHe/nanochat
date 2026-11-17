"""Seal a norm-input run after CPU and single-GPU validation; never launch."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
manifest=json.loads((here/'manifest.json').read_text())
assert not (here/'launch_request.json').exists()
cpu=json.loads((here/'cpu_validation.json').read_text());gpu=json.loads((here/'gpu_validation.json').read_text())
assert cpu['status']==gpu['status']=='passed'
assert gpu['parameter_count']==manifest['parameter_counts']['total']
assert gpu['architecture_training_flops_per_token']==manifest['estimated_flops_per_token']
assert gpu['cached_generation_verified'] and gpu['full_model_optimizer_steps']==1
assert gpu['full_shape']==dict(batch=2,sequence_len=2048,depth=24,width=1536)
assert manifest['model_config']['norm_value_input']
manifest['estimated_forward_flops_per_token']=gpu['architecture_forward_flops_per_token']
manifest['validation']=dict(status='passed',cpu=cpu,gpu='gpu_validation.json',
    distributed_status='pending_automatic_preflight',training_started=False)
names=['queue_norm_input.py','launcher_core.py','launch.py','launch.sh','gpu_smoke.py','distributed_smoke.py','finalize_validation.py',
       'prepare_snapshot.py','environment_packages.json','cpu_validation.json','cpu_validation.log','gpu_validation.json']
files=[here/n for n in names]+list((here/'code/tests').rglob('*.py'))
manifest['artifact_sha256']={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(here/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(here/'status.json').write_text(json.dumps(dict(status='ready',run_id=manifest['model_tag'],
    validation='cpu_gpu_passed',distributed_validation='pending_automatic_preflight',training_started=False),indent=2)+'\n')
print(json.dumps(manifest['validation'],indent=2))
