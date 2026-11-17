"""Freeze inputs, model, FlashAttention kernel, and probe into a fresh run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,shutil,subprocess,sys
from .common import read,write,sha,hashes

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--output-root',type=Path,default=Path('/local-ssd/mh3897/analysis_runs/d24_self_attention_mass'));a=p.parse_args();c=read(a.config)
 parent=Path(c['parent_run']);pm=read(parent/'manifest.json');selection=Path(c['selection_run']);sm=read(selection/'manifest.json')
 for f,h in pm['input_hashes'].items():assert sha(parent/f)==h
 for f,h in pm['source_hashes'].items():assert sha(parent/'code'/f)==h
 assert pm['checkpoint_sha256']==sm['input_hashes']['checkpoint/model_007773.pt']
 selected=selection/'inputs/selected_layers.json';assert sha(selected)==sm['input_hashes']['selected_layers.json'];target_layers=read(selected)['selected'];layers=c.get('layers',target_layers);assert layers==sorted(set(layers)) and all(0<=l<24 for l in layers)
 kernel=Path(pm['kernel_cache_path'])
 for f,h in pm['kernel_cache_sha256'].items():assert sha(kernel/f)==h
 gpu=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');uuid,name,mem,util=[s.strip() for s in gpu];assert int(mem)<1000 and int(util)<=5
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');r=a.output_root.resolve()/f'gpt_base_scalar_res_qkv_no_embed_norm_step7773_{stamp}';r.mkdir(parents=True,exist_ok=False)
 for folder in ['code','inputs','measurements','derived','validation','logs','visualization']:(r/folder).mkdir()
 shutil.copytree(Path(__file__).parent,r/'code/probe',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(parent/'code/nanochat',r/'code/nanochat',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for folder in ['checkpoint','runtime_base']:shutil.copytree(parent/'inputs'/folder,r/'inputs'/folder)
 shutil.copy2(parent/'inputs/tokens.npy',r/'inputs/tokens.npy');shutil.copy2(selected,r/'inputs/selected_layers.json');write(r/'inputs/config.json',c)
 shutil.copytree(kernel,r/'kernel_cache'/kernel.name,symlinks=True)
 write(r/'inputs/parent_reference.json',dict(run=str(parent),manifest_sha256=sha(parent/'manifest.json'),selection_run=str(selection),selection_manifest_sha256=sha(selection/'manifest.json')))
 env=dict(CUDA_VISIBLE_DEVICES=uuid,NANOCHAT_DTYPE='bfloat16',NANOCHAT_BASE_DIR=str(r/'inputs/runtime_base'),OMP_NUM_THREADS=str(c['cpu_threads']),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 for k in ['HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','KERNELS_CACHE','HF_KERNELS_CACHE']:env[k]=str(r/'kernel_cache')
 write(r/'manifest.json',dict(created_utc=stamp,config=c,layers=layers,target_layers=target_layers,checkpoint_step=7773,model_type=pm['config']['model_type'],model_config=read(r/'inputs/checkpoint/meta_007773.json')['model_config'],tokens_shape=pm['tokens_shape'],gpu_uuid=uuid,gpu_name=name,gpu_index=a.gpu,python_executable=sys.executable,environment_overrides=env,input_hashes=hashes(r/'inputs'),code_hashes=hashes(r/'code'),kernel_hashes=hashes(r/'kernel_cache'),formula=c.get('measurement','self_attention')+': softmax probabilities from actual BF16 Q/K and FA3 LSE; exclude-prefix mode computes A[t,t]/(1-sum(A[t,0:4])) per query t>=4 with direct FP64 remaining-key fallback for small retained mass',population='same first 128 deterministic BOS-bestfit validation sequences, length 2048; occurrence weighted'))
 write(r/'status.json',dict(state='prepared'));print('RUN_DIR='+str(r),flush=True)
if __name__=='__main__':main()
