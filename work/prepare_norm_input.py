"""Freeze all three norm-input variants, without launching training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/hdd/mh3897/nanochat/.venv/bin/python'
VARIANTS = {
    'pooled_qk_norm_input_sum': ('gpt_base_pooled_qk_embed_no_embed_norm', '+No W_v +norm input + sum attn'),
    'single_qk_norm_input': ('gpt_base_single_qk_embed_no_embed_norm', '+No W_v +single head +norm input'),
    'multi_head_norm_input': ('gpt_base_split_embed_head_out_no_embed_norm', '+No W_v +multi-head + norm input'),
    'single_qk_value_proj_norm_input': ('gpt_base_single_qk_value_proj_no_embed_norm', '+W_v +single head +norm input'),
}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8<<20),b''):h.update(c)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', choices=VARIANTS)
    parser.add_argument('--index', type=Path, default=REPO/'work/norm-input-prepared-runs.json')
    args=parser.parse_args()
    names=args.variant or list(VARIANTS)[:3]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prior=Path((REPO/'work/single-qk-prepared-run.txt').read_text().strip())
    kernel=Path(json.loads((prior/'manifest.json').read_text())['kernel_cache_path'])
    kernel_hashes={str(p.relative_to(kernel)):sha(p) for p in sorted(kernel.rglob('*')) if p.is_file()}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    prepared={}
    for name in names:
        key,label=VARIANTS[name]
        rid=f'bov_d24_f3p91e19_{name}_nosmear_s42_{stamp}'
        run=Path('/local-ssd/mh3897/run_queues')/rid;run.mkdir()
        code=run/'code';code.mkdir()
        for directory in ('nanochat','scripts','tasks','configs','runs','tests'):
            shutil.copytree(REPO/directory,code/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for file in ('pyproject.toml','uv.lock'):shutil.copy2(REPO/file,code/file)
        env=dict(os.environ,NANOCHAT_BASE_DIR='/local-ssd/mh3897',PYTHON=PYTHON,PYTHONPATH=str(code),HF_HUB_OFFLINE='1')
        with (run/'prepare.log').open('w') as log:
            subprocess.run(['bash',str(code/'runs/bov/d24_f3p91e19'/f'{name}.sh'),'--dry-run','--gpu-count','2',
                            '--run-id',rid,'--source-commit',commit,'--manifest',str(run/'manifest.json')],
                           cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads((run/'manifest.json').read_text())
        manifest.update(model_type=key,research_name=label,selected_gpus=[],kernel_cache_path=str(kernel),
                        kernel_cache_sha256=kernel_hashes,artifact_sha256={},
                        validation={'status':'pending','training_started':False})
        manifest['env']['HF_HUB_OFFLINE']='1'
        (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (run/'environment_packages.json').write_text(json.dumps(packages,sort_keys=True,indent=2)+'\n')
        for source,dest in [('norm_input_launch.py','launch.py'),('split_embed_launch.py','launcher_core.py'),('norm_input_gpu_smoke.py','gpu_smoke.py'),
                            ('norm_input_distributed_smoke.py','distributed_smoke.py'),('queue_norm_input.py','queue_norm_input.py'),
                            ('finalize_norm_input_validation.py','finalize_validation.py'),('prepare_norm_input.py','prepare_snapshot.py')]:
            shutil.copy2(REPO/'work'/source,run/dest)
        shutil.copy2(prior/'launch.sh',run/'launch.sh')
        (run/'status.json').write_text(json.dumps({'status':'prepared','run_id':rid,'training_started':False},indent=2)+'\n')
        (run/'parent_git_revision.txt').write_text(commit+'\n')
        prepared[name]=str(run)
        print(name,run,flush=True)
    args.index.write_text(json.dumps(prepared,indent=2)+'\n')

if __name__=='__main__':main()
