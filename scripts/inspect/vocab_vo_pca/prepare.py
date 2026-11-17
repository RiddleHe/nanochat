"""Freeze a PCA run that matches the completed vocabulary VO cosine probe."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,shutil,subprocess,sys
from .common import read,write,sha,hashes

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--gpu',type=int,required=True);p.add_argument('--output-root',type=Path,default=Path('/local-ssd/mh3897/analysis_runs/d24_vocab_vo_pca'));a=p.parse_args();c=read(a.config);parent=Path(c['parent_run']);pm=read(parent/'manifest.json');pr=read(parent/'results.json')
 assert pr['state']=='complete' and pr['manifest_sha256']==sha(parent/'manifest.json')
 assert hashes(parent/'inputs')==pm['input_hashes'] and hashes(parent/'code')==pm['code_hashes']
 for name,digest in pr['output_sha256'].items():assert sha(parent/name)==digest
 assert read(parent/'validation/summary.json')['passed']
 gpu=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');uuid,gpu_name,mem,util=[s.strip() for s in gpu];assert int(mem)<1000 and int(util)<=5
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');r=a.output_root.resolve()/f"gpt_base_scalar_res_qkv_no_embed_norm_step{pm['config']['checkpoint_step']}_{stamp}";r.mkdir(parents=True,exist_ok=False)
 for d in ['code','inputs','derived','validation','logs','visualization']:(r/d).mkdir()
 shutil.copytree(Path(__file__).parent,r/'code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for d in ['checkpoint','tokenizer']:shutil.copytree(parent/'inputs'/d,r/'inputs'/d)
 shutil.copy2(parent/'inputs/selected_layers.json',r/'inputs/selected_layers.json')
 shutil.copy2(parent/'derived/statistics.npz',r/'inputs/parent_statistics.npz')
 shutil.copy2(parent/'derived/schema.json',r/'inputs/parent_schema.json')
 write(r/'inputs/parent_reference.json',dict(run=str(parent),manifest_sha256=sha(parent/'manifest.json'),results_sha256=sha(parent/'results.json'),statistics_sha256=sha(parent/'derived/statistics.npz')))
 write(r/'inputs/config.json',c)
 layers=c.get('layers',pm['selected_layers']);assert layers==sorted(set(layers)) and all(0<=l<pm['model_config']['n_layer'] for l in layers)
 inputs=hashes(r/'inputs')
 for k,v in inputs.items():
  if k.startswith(('checkpoint/','tokenizer/')) or k=='selected_layers.json':assert v==pm['input_hashes'][k]
 write(r/'manifest.json',dict(created_utc=stamp,config=c,model_type=pm['config']['model_type'],model_config=pm['model_config'],checkpoint_step=pm['config']['checkpoint_step'],layers=layers,target_layers=pm['selected_layers'],parent_batch_size=pm['config']['batch_size'],gpu_uuid=uuid,gpu_name=gpu_name,gpu_index=a.gpu,python_executable=sys.executable,input_hashes=inputs,code_hashes=hashes(r/'code'),population=pm['population'],formula=pm['formula'],environment_overrides=dict(CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(c['cpu_threads']))))
 write(r/'status.json',dict(state='prepared'));print('RUN_DIR='+str(r),flush=True)
if __name__=='__main__':main()
