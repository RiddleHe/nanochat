"""Observe FlashAttention LSE and diagonal logits without changing model outputs."""
from pathlib import Path
import argparse,os,sys,subprocess,platform,time,traceback,importlib.metadata
import numpy as np
from common import read,write,sha,hashes

def collect(r):
 m=read(r/'manifest.json');c=m['config'];assert Path(__file__).resolve()==r/'code/probe/collect_exclude_prefix.py'
 if any(os.environ.get(k)!=v for k,v in m['environment_overrides'].items()):
  env=dict(os.environ,**m['environment_overrides']);env.pop('PYTHONPATH',None);os.execve(m['python_executable'],[m['python_executable'],str(__file__),'--run-dir',str(r)],env)
 assert not (r/'collection.json').exists()
 for folder,key in [('inputs','input_hashes'),('code','code_hashes'),('kernel_cache','kernel_hashes')]:assert hashes(r/folder)==m[key]
 gpu=subprocess.check_output(['nvidia-smi','--id='+m['gpu_uuid'],'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(gpu[0])<1000 and int(gpu[1])<=5
 sys.path.insert(0,str(r/'code'))
 import torch
 from nanochat.checkpoint_manager import build_model
 import nanochat.flash_attention as fa
 torch.set_num_threads(c['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 assert torch.cuda.device_count()==1 and fa.USE_FA3 and fa._fa3 is not None
 kernel_path=Path(fa._fa3.__file__).resolve();assert kernel_path.is_relative_to(r/'kernel_cache')
 write(r/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,platform=platform.platform(),gpu_uuid=m['gpu_uuid'],gpu_name=torch.cuda.get_device_name(0),tf32=False,backend='FA3',backend_file=str(kernel_path)))
 (r/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
 model,_,_=build_model(str(r/'inputs/checkpoint'),m['checkpoint_step'],torch.device('cuda:0'),'eval');model.requires_grad_(False)
 tokens=np.load(r/'inputs/tokens.npy');assert list(tokens.shape)==m['tokens_shape'] and c['batch_size']==1
 layers=m['layers'];nh=m['model_config']['n_head'];nt=tokens.shape[1];original=fa.flash_attn.flash_attn_func
 fallback_rows=0;minimum_retained=1.;max_adjusted=0.;seq=-1;current_layer=None;values={};checks=[];handles=[];windows={};min_p=1.;max_p=0.;max_first_error=0.;start=time.monotonic()
 for l,block in enumerate(model.transformer.h):
  def enter(module,args,l=l):
   nonlocal current_layer
   current_layer=l
  handles.append(block.attn.register_forward_pre_hook(enter))
 def observe(q,k,v,causal=False,window_size=(-1,-1)):
  nonlocal min_p,max_p,max_first_error,fallback_rows,minimum_retained,max_adjusted
  l=current_layer
  if l not in layers:return original(q,k,v,causal=causal,window_size=window_size)
  assert l not in values and causal and q.shape==k.shape==v.shape==(1,nt,nh,128) and q.dtype==torch.bfloat16
  out,lse=fa._fa3.flash_attn_func(q,k,v,causal=causal,window_size=window_size,return_attn_probs=True)
  assert lse.shape==(1,nh,nt) and torch.isfinite(lse).all()
  qh=q[0].transpose(0,1).double();kh=k[0].transpose(0,1).double();scale=q.shape[-1]**-.5
  diagonal=(qh*kh).sum(-1)*scale;raw=torch.exp(diagonal-lse[0].double())
  low=float(raw.min());high=float(raw.max());first_error=float((raw[:,0]-1).abs().max())
  assert low>=0 and high<=1+c['probability_atol'] and first_error<c['probability_atol']
  min_p=min(min_p,low);max_p=max(max_p,high);max_first_error=max(max_first_error,first_error)
  t=torch.arange(nt,device='cuda');prefix=torch.arange(4,device='cuda');prefix_mask=prefix[None,:]<=t[:,None]
  if window_size[0]>=0:prefix_mask &= t[:,None]-prefix[None,:]<=window_size[0]
  prefix_logits=(qh@kh[:,:4].transpose(-1,-2))*scale
  prefix_logits.masked_fill_(~prefix_mask[None],float('-inf'))
  log_removed_mass=torch.logsumexp(prefix_logits,-1)-lse[0].double()
  retained=-torch.expm1(log_removed_mass)
  adjusted=raw/retained;adjusted[:,:4]=float('nan')
  fallback=(retained<c['retained_mass_fallback_threshold'])|(~torch.isfinite(adjusted))|(adjusted>1+c['adjusted_probability_atol']);fallback[:,:4]=False
  fallback_rows+=int(fallback.sum());positions=torch.where(fallback.any(0))[0]
  keys=torch.arange(nt,device='cuda')
  # FP64 remaining-key logsumexp prevents subtraction error when the prefix dominates.
  for start_pos in range(0,len(positions),64):
   pos=positions[start_pos:start_pos+64];mask=(keys[None,:]<=pos[:,None])&(keys[None,:]>=4)
   if window_size[0]>=0:mask &= pos[:,None]-keys[None,:]<=window_size[0]
   scores=(qh[:,pos]@kh.transpose(-1,-2))*scale;scores.masked_fill_(~mask[None],float('-inf'))
   remaining_lse=torch.logsumexp(scores,-1);direct=torch.exp(diagonal[:,pos]-remaining_lse)
   adjusted[:,pos]=torch.where(fallback[:,pos],direct,adjusted[:,pos])
  valid=adjusted[:,4:];assert torch.isfinite(valid).all() and float(valid.min())>=0 and float(valid.max())<=1+c['adjusted_probability_atol']
  assert float((adjusted[:,4]-1).abs().max())<c['adjusted_probability_atol']
  minimum_retained=min(minimum_retained,float(retained[:,4:].min()));max_adjusted=max(max_adjusted,float(valid.max()))
  values[l]=dict(raw_self_probability=raw.T.float().cpu().numpy(),retained_mass=retained.T.cpu().numpy(),adjusted_self_probability=adjusted.T.float().cpu().numpy(),fallback=fallback.T.cpu().numpy());windows[l]=list(window_size)
  if seq in c['validation_sequences']:
   base=original(q,k,v,causal=causal,window_size=window_size);assert torch.equal(out,base)
   pos=torch.tensor(c['validation_positions'],device='cuda');keys=torch.arange(nt,device='cuda');mask=keys[None,:]<=pos[:,None]
   if window_size[0]>=0:mask &= pos[:,None]-keys[None,:]<=window_size[0]
   scores=q[0,pos].transpose(0,1).double()@k[0].transpose(0,1).double().transpose(-1,-2)*(q.shape[-1]**-.5)
   scores.masked_fill_(~mask[None],float('-inf'));ref_lse=torch.logsumexp(scores,-1);ref_p=torch.softmax(scores,-1)
   ref_diag=ref_p.gather(-1,pos[None,:,None].expand(nh,-1,1)).squeeze(-1)
   p_error=float((raw[:,pos]-ref_diag).abs().max());lse_error=float((lse[0,:,pos].double()-ref_lse).abs().max())
   remaining_scores=scores.clone();remaining_scores[:,:,0:4]=float('-inf');remaining_probs=torch.softmax(remaining_scores,-1)
   ref_adjusted=remaining_probs.gather(-1,pos[None,:,None].expand(nh,-1,1)).squeeze(-1)
   adjusted_error=float((adjusted[:,pos]-ref_adjusted).abs().max())
   assert p_error<c['probability_atol'] and lse_error<c['lse_atol'] and adjusted_error<c['adjusted_probability_atol']
   row_sum_error=float((ref_p.sum(-1)-1).abs().max());assert row_sum_error<1e-12 and torch.count_nonzero(ref_p.masked_select(~mask[None].expand(nh,-1,-1)))==0
   checks.append(dict(sequence=seq,layer=l,heads=list(range(nh)),positions=c['validation_positions'],window_size=list(window_size),adjusted_probability_max_abs_error=adjusted_error,probability_max_abs_error=p_error,lse_max_abs_error=lse_error,reference_row_sum_max_error=row_sum_error,attention_output_bitwise_equal=True))
  return out
 write(r/'status.json',dict(state='running'))
 try:
  with torch.inference_mode():
   for seq in range(len(tokens)):
    ids=torch.from_numpy(tokens[seq:seq+1].astype(np.int64)).cuda();values={}
    fa.flash_attn.flash_attn_func=original;baseline=model(ids)
    fa.flash_attn.flash_attn_func=observe;logits=model(ids)
    assert torch.isfinite(logits).all() and torch.equal(logits,baseline) and sorted(values)==layers
    arrays={name:np.stack([values[l][name] for l in layers]) for name in ['raw_self_probability','retained_mass','adjusted_self_probability','fallback']}
    assert all(a.shape==(len(layers),nt,nh) for a in arrays.values())
    np.savez(r/'measurements'/f'sequence_{seq:03d}.npz',**arrays)
    del baseline,logits
    if (seq+1)%8==0:
     print(f'{seq+1}/{len(tokens)} sequences complete; logits bitwise equal',flush=True);write(r/'status.json',dict(state='running',completed_sequences=seq+1,total_sequences=len(tokens)))
 finally:
  fa.flash_attn.flash_attn_func=original
  for handle in handles:handle.remove()
 torch.cuda.synchronize()
 write(r/'measurements/schema.json',dict(shape_per_chunk=[len(layers),nt,nh],axes=['layer','sequence_position','head'],layers=layers,dtype='probabilities float32; retained_mass float64; fallback bool',token_ids='../inputs/tokens.npy',formula=m['formula'],windows=windows,clipping='none; small numerical overshoot retained',query_positions='0..3 excluded: adjusted probabilities are NaN; position 4 included',removed_keys=[0,1,2,3],fallback='FP64 direct remaining-key logsumexp for retained mass below 0.1 or invalid adjusted ratio',retained_mass='1-prefix probability mass from FA3 LSE; recorded before any fallback; tiny negative numerical errors possible in excluded queries',head_dim=128))
 write(r/'validation/collection.json',dict(passed=True,logits_bitwise_equal_all_sequences=True,sequences_checked=len(tokens),probability_min=min_p,probability_max=max_p,first_position_max_abs_error=max_first_error,tolerances=dict(probability_atol=c['probability_atol'],lse_atol=c['lse_atol']),explicit_softmax_checks=checks,fallback_query_head_count=fallback_rows,minimum_retained_mass_valid_queries=minimum_retained,maximum_adjusted_probability=max_adjusted,adjusted_probability_atol=c['adjusted_probability_atol']))
 write(r/'collection.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),seconds=time.monotonic()-start,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,output_sha256={str(p.relative_to(r)):sha(p) for folder in ['measurements','validation'] for p in sorted((r/folder).iterdir())}))
 write(r/'status.json',dict(state='collected'))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve()
 try:collect(r)
 except BaseException as e:write(r/'status.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
