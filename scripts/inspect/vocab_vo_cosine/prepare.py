"""Freeze inputs and select strictly Emb-dominant middle/late V receivers."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,shutil,subprocess,sys
import numpy as np
import torch
from .common import read,write,sha,hashes,select

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--gpu',type=int,required=True);p.add_argument('--output-root',type=Path,default=Path('/local-ssd/mh3897/analysis_runs/d24_vocab_vo_cosine'));a=p.parse_args()
 c=read(a.config);training=Path(c['training_run']);tm=read(training/'manifest.json')
 assert read(training/'status.json')['returncode']==0 and tm['expected_steps']==c['checkpoint_step']
 gpu=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 uuid,gpu_device_name,memory,util=[v.strip() for v in gpu];assert int(memory)<1000 and int(util)<=5
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');r=a.output_root.resolve()/f"{c['model_type']}_step{c['checkpoint_step']}_{stamp}";r.mkdir(parents=True,exist_ok=False)
 for name in ['code','inputs/checkpoint','inputs/tokenizer','derived','validation','logs','visualization']:(r/name).mkdir(parents=True,exist_ok=True)
 shutil.copytree(Path(__file__).parent,r/'code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 write(r/'inputs/config.json',c)
 checkpoint_dir=Path(tm['checkpoint_dir']);step=c['checkpoint_step']
 for name in [f'model_{step:06d}.pt',f'meta_{step:06d}.json']:
  shutil.copy2(checkpoint_dir/name,r/'inputs/checkpoint'/name);assert sha(checkpoint_dir/name)==sha(r/'inputs/checkpoint'/name)
 meta=read(r/'inputs/checkpoint'/f'meta_{step:06d}.json');assert meta['model_type']==c['model_type'] and meta['model_config']==tm['model_config']
 for path,digest in tm['tokenizer_sha256'].items():
  assert sha(path)==digest;shutil.copy2(path,r/'inputs/tokenizer'/Path(path).name)
 for name in ['gpt_base_scalar_res_qkv_no_embed_norm.py','gpt_base_attn_res_qkv.py']:
  relative='nanochat/model/'+name;source=Path(tm['repo'])/relative;assert sha(source)==tm['source_sha256'][relative]
  target=r/'code/model_reference'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
 state=torch.load(r/'inputs/checkpoint'/f'model_{step:06d}.pt',map_location='cpu',weights_only=True,mmap=True);state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
 rows=select(state,meta['model_config'],c['candidate_layers']);layers=[v['layer'] for v in rows if v['selected']];assert layers
 reference=Path(c['coefficient_reference']);rm=read(reference/'manifest.json');schema=read(reference/'summaries/scalar_coefficients.json')
 assert rm['checkpoint_sha256']==sha(r/'inputs/checkpoint'/f'model_{step:06d}.pt')
 assert schema['statistics_sha256']==sha(reference/'summaries/scalar_coefficients.npz')
 with np.load(reference/'summaries/scalar_coefficients.npz') as z:
  for row in rows:
   ref=next(v for v in schema['receivers'] if v['kind']=='V' and v['layer']==row['layer'])
   np.testing.assert_array_equal(row['coefficients'],z['coefficients_bf16_as_fp32'][ref['start']:ref['stop']])
 write(r/'inputs/selected_layers.json',dict(rule='Emb BF16-applied V coefficient strictly exceeds every other available source',selected=layers,candidates=rows))
 write(r/'inputs/reference.json',dict(run=str(reference),manifest_sha256=sha(reference/'manifest.json'),coefficient_sha256=schema['statistics_sha256'],exact_match=True))
 write(r/'manifest.json',dict(created_utc=stamp,config=c,model_config=meta['model_config'],selected_layers=layers,gpu_uuid=uuid,gpu_name=gpu_device_name,gpu_index=a.gpu,python_executable=sys.executable,training_manifest_sha256=sha(training/'manifest.json'),input_hashes=hashes(r/'inputs'),code_hashes=hashes(r/'code'),formula='y = (e @ W_V.T) @ W_O.T; cosine(e,y)',population='all valid vocabulary rows equally weighted, including special tokens; exclude padding',environment_overrides=dict(CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(c['cpu_threads']))))
 write(r/'status.json',dict(state='prepared'));print('RUN_DIR='+str(r),flush=True)
if __name__=='__main__':main()
