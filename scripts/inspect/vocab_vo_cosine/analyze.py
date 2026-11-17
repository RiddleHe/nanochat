"""Run only from frozen code: full vocabulary, full V and O matrices, one GPU."""
from pathlib import Path
import argparse,os,sys,time,platform,subprocess
import numpy as np
from common import read,write,sha,hashes,select

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();m=read(r/'manifest.json');c=m['config']
 assert Path(__file__).resolve()==r/'code/analyze.py'
 if any(os.environ.get(k)!=v for k,v in m['environment_overrides'].items()):
  env=dict(os.environ,**m['environment_overrides']);env.pop('PYTHONPATH',None);os.execve(m['python_executable'],[m['python_executable'],str(__file__),'--run-dir',str(r)],env)
 assert hashes(r/'inputs')==m['input_hashes'] and hashes(r/'code')==m['code_hashes']
 assert not (r/'results.json').exists(),'Use a fresh run for new analysis'
 import torch
 torch.set_num_threads(c['cpu_threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 assert torch.cuda.device_count()==1
 row=subprocess.check_output(['nvidia-smi','--id='+m['gpu_uuid'],'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(row[0])<1000 and int(row[1])<=5,'Selected GPU is no longer idle'
 write(r/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,platform=platform.platform(),gpu_uuid=m['gpu_uuid'],gpu_name=torch.cuda.get_device_name(0),tf32=False))
 write(r/'status.json',dict(state='running'))
 state=torch.load(r/'inputs/checkpoint'/f"model_{c['checkpoint_step']:06d}.pt",map_location='cpu',weights_only=True,mmap=True);state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
 cfg=m['model_config'];n,d=cfg['vocab_size'],cfg['n_embd'];layers=m['selected_layers'];ids=np.arange(n,dtype=np.int64)
 assert select(state,cfg,c['candidate_layers'])==read(r/'inputs/selected_layers.json')['candidates']
 assert cfg['n_head']==cfg['n_kv_head'] and cfg['scalar_res_gate']=='sqrt'
 emb=state['transformer.wte.weight'][:n].to(torch.bfloat16).float().cuda();assert emb.shape==(n,d) and torch.isfinite(emb).all()
 inp=emb.norm(dim=-1);out_norm=np.empty((len(layers),n),np.float32);cosines=np.empty_like(out_norm);checks=[];summaries=[];t=time.monotonic()
 sample_ids=np.unique(np.r_[np.arange(8),np.linspace(0,n-1,24,dtype=int)]);sample=emb[sample_ids]
 def cosine(x,y):return (x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1))
 with torch.inference_mode():
  # Analytic direction checks include opposite and orthogonal vectors.
  x=torch.tensor([[1.,0.],[0.,1.]],device='cuda');torch.testing.assert_close(cosine(x,x),torch.ones(2,device='cuda'));torch.testing.assert_close(cosine(x,-x),-torch.ones(2,device='cuda'));assert torch.equal(cosine(x,x.flip(-1)),torch.zeros(2,device='cuda'))
  for li,l in enumerate(layers):
   v=state[f'transformer.h.{l}.attn.c_v.weight'].to(torch.bfloat16).float().cuda();o=state[f'transformer.h.{l}.attn.c_proj.weight'].to(torch.bfloat16).float().cuda();assert v.shape==o.shape==(d,d)
   seq=(sample@v.T)@o.T;composed=sample@(o@v).T
   ref=(sample.double()@v.double().T)@o.double().T
   def rel(a,b):return float(((a.double()-b.double()).norm(dim=-1)/b.double().norm(dim=-1).clamp_min(1e-30)).max())
   composition_error=rel(composed,seq);fp64_error=rel(seq,ref);cos_error=float((cosine(sample,seq).double()-cosine(sample.double(),ref)).abs().max())
   assert composition_error<c['validation_relative_output_tolerance'] and fp64_error<c['validation_relative_output_tolerance'] and cos_error<c['validation_absolute_cosine_tolerance']
   h=cfg['n_head'];hd=d//h;parts=sum((sample@v[j*hd:(j+1)*hd].T)@o[:,j*hd:(j+1)*hd].T for j in range(h));head_error=rel(parts,seq);assert head_error<c['validation_relative_output_tolerance']
   for start in range(0,n,c['batch_size']):
    x=emb[start:start+c['batch_size']];y=(x@v.T)@o.T;yn=y.norm(dim=-1);den=inp[start:start+len(x)]*yn
    value=torch.where(den>0,(x*y).sum(-1)/den,torch.full_like(den,float('nan')))
    assert torch.isfinite(y).all();finite=value[torch.isfinite(value)];assert (finite.abs()<=1+2e-6).all()
    out_norm[li,start:start+len(x)]=yn.cpu().numpy();cosines[li,start:start+len(x)]=value.clamp(-1,1).cpu().numpy()
   values=cosines[li];valid=np.isfinite(values);z=values[valid].astype(np.float64)
   summaries.append(dict(layer=l,valid_count=int(valid.sum()),undefined_count=int((~valid).sum()),mean=float(z.mean()),std=float(z.std()),minimum=float(z.min()),maximum=float(z.max()),median=float(np.median(z)),positive_fraction=float((z>0).mean())))
   checks.append(dict(layer=l,composition_relative_error=composition_error,fp64_relative_output_error=fp64_error,fp64_absolute_cosine_error=cos_error,head_sum_relative_error=head_error))
   print(f'L{l}: mean cosine {z.mean():.6f}, range [{z.min():.6f}, {z.max():.6f}]',flush=True)
 valid=np.isfinite(cosines);expected=(inp.cpu().numpy()[None,:]>0)&(out_norm>0);assert np.array_equal(valid,expected)
 order=np.stack([np.lexsort((ids,-np.where(valid[j],cosines[j],-np.inf))) for j in range(len(layers))])
 for j in range(len(layers)):
  assert np.array_equal(np.sort(order[j]),ids)
  v=cosines[j,order[j]][:valid[j].sum()];assert np.all(np.diff(v)<=0)
 np.savez_compressed(r/'derived/statistics.npz',token_ids=ids,layers=np.array(layers),input_norm=inp.cpu().numpy(),output_norm=out_norm,cosine=cosines,valid_mask=valid,sort_indices=order)
 write(r/'derived/schema.json',dict(axes=dict(cosine=['selected_layer','token_id'],output_norm=['selected_layer','token_id'],input_norm=['token_id'],sort_indices=['selected_layer','descending_rank']),formula=m['formula'],population=m['population'],vocab_size=n,dimensions=d,embedding='raw lookup cast BF16 then FP32',weights='forward-cast BF16 then FP32',operations='two sequential FP32 linear projections; TF32 disabled; no intermediate BF16 rounding',scalar='positive coefficient cancels from cosine; not multiplied',attention=False,residual_addition=False,receiver_norm=False,centering=False,undefined='NaN and valid_mask false when either norm is zero; sorted after valid tokens',sorting='descending cosine independently per layer; ascending token ID breaks ties',head_combination='all heads concatenated then full O projection'))
 write(r/'derived/layer_summary.json',summaries)
 write(r/'validation/summary.json',dict(passed=True,checks=checks,sample_token_ids=sample_ids.tolist(),analytic_direction_checks=True,all_tokens_checked=True,cosine_bounds_checked=True,sort_permutations_checked=True,undefined_mask_checked=True,coefficient_selection_reverified=True,tolerances={k:v for k,v in c.items() if k.startswith('validation_')}))
 torch.cuda.synchronize()
 write(r/'results.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),seconds=time.monotonic()-t,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,vocab_size=n,layers=layers,output_sha256={str(p.relative_to(r)):sha(p) for folder in ['derived','validation'] for p in (r/folder).iterdir()}))
 write(r/'status.json',dict(state='complete'))
if __name__=='__main__':main()
