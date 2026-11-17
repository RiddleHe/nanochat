"""Freeze the split-embedding/shared-output experiment; does not start training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess

repo = Path(__file__).resolve().parents[1]
python = '/hdd/mh3897/nanochat/.venv/bin/python'
prior = Path('/local-ssd/mh3897/run_queues/bov_d24_f3p91e19_gpt_base_head_scaled_embed_no_embed_norm_nosmear_s42_20261111T031430Z')
model_type = 'gpt_base_split_embed_shared_out_no_embed_norm'
rid = f'bov_d24_f3p91e19_{model_type}_nosmear_s42_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
run = Path('/local-ssd/mh3897/run_queues') / rid
run.mkdir()
code = run/'code'
code.mkdir()
for name in ('nanochat','scripts','tasks','configs','runs','tests'):
    shutil.copytree(repo/name, code/name, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('pyproject.toml','uv.lock'):
    shutil.copy2(repo/name, code/name)
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
(run/'parent_git_revision.txt').write_text(commit+'\n')
env = dict(os.environ, NANOCHAT_BASE_DIR='/local-ssd/mh3897', PYTHON=python,
           PYTHONPATH=str(code), HF_HUB_OFFLINE='1')
with (run/'prepare.log').open('w') as output:
    subprocess.run(['bash',str(code/'runs/bov/d24_f3p91e19'/f'{model_type}.sh'),
        '--dry-run','--gpu-count','2','--run-id',rid,'--source-commit',commit,
        '--manifest',str(run/'manifest.json')],cwd=code,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
manifest=json.loads((run/'manifest.json').read_text())
manifest.update(model_type=model_type, selected_gpus=[], estimated_forward_flops_per_token=1492287104,
                validation={'status':'pending','training_started':False})
manifest['env']['HF_HUB_OFFLINE']='1'
kernel=Path(json.loads((prior/'manifest.json').read_text())['kernel_cache_path'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()
manifest['kernel_cache_path']=str(kernel)
manifest['kernel_cache_sha256']={str(p.relative_to(kernel)):sha(p) for p in sorted(kernel.rglob('*')) if p.is_file()}
manifest['artifact_sha256']={}
(run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ('launch.py','launch.sh'):
    shutil.copy2(prior/name,run/name)
shutil.copy2(repo/'work/split_embed_launch.py',run/'launch.py')
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
(run/'environment_packages.json').write_text(json.dumps(packages,sort_keys=True,indent=2)+'\n')
smoke=(prior/'gpu_smoke.py').read_text().replace('head_scaled_embed','split_embed_shared_out').replace('HeadScaledEmbed','SplitEmbedSharedOut')
smoke=smoke.replace('UUID = "GPU-909c9309-e370-e959-2373-9c06aabd0d5e"', 'UUID = os.environ["CUDA_VISIBLE_DEVICES"]')
smoke=smoke.replace('including chunk checkpoint backward', 'including FA3 target backward')
(run/'gpu_smoke.py').write_text(smoke)
shutil.copy2(Path(__file__),run/'prepare_snapshot.py')
shutil.copy2(repo/'work/split_embed_distributed_smoke.py',run/'distributed_smoke.py')
shutil.copy2(repo/'work/split_embed_gpu_extra.py',run/'gpu_extra.py')
shutil.copy2(repo/'work/finalize_split_embed_validation.py',run/'finalize_validation.py')
(run/'status.json').write_text(json.dumps(dict(status='prepared',run_id=rid,training_started=False),indent=2)+'\n')
(repo/'work/split-embed-prepared-run.txt').write_text(str(run)+'\n')
print(run,flush=True)
