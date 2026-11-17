"""Observe FlashAttention LSE and diagonal and first-key logits without changing model outputs."""
from pathlib import Path
import argparse,os,sys,subprocess,platform,time,traceback,importlib.metadata
import numpy as np
from common import read,write,sha,hashes

def collect(r):
 m=read(r/'manifest.json');c=m['config'];assert Path(__file__).resolve()==r/'code/probe/collect_sink.py'
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
 seq=-1;current_layer=None;values={};checks=[];handles=[];windows={};min_p=1.;max_p=0.;max_first_error=0.;start=time.monotonic()
 for l,block in enumerate(model.transformer.h):
  def enter(module,args,l=l):
   nonlocal current_layer
   current_layer=l
  handles.append(block.attn.register_forward_pre_hook(enter))
 def observe(q,k,v,causal=False,window_size=(-1,-1)):
  nonlocal min_p,max_p,max_first_error
  l=current_layer
  if l not in layers:return original(q,k,v,causal=causal,window_size=window_size)
  assert l not in values and causal and q.shape==k.shape==v.shape==(1,nt,nh,128) and q.dtype==torch.bfloat16
  out,lse=fa._fa3.flash_attn_func(q,k,v,causal=causal,window_size=window_size,return_attn_probs=True)
  assert lse.shape==(1,nh,nt) and torch.isfinite(lse).all()
  diagonal=(q.float()*k.float()).sum(-1).transpose(1,2)*(q.shape[-1]**-.5)
  prob=torch.exp(diagonal-lse.float());assert torch.isfinite(prob).all()
  low=float(prob.min());high=float(prob.max());first_error=float((prob[:,:,0]-1).abs().max())
  assert low>=0 and high<=1+c['probability_atol'] and first_error<c['probability_atol']
  min_p=min(min_p,low);max_p=max(max_p,high);max_first_error=max(max_first_error,first_error)
  first_logit=(q.float()*k[:,0:1].float()).sum(-1).transpose(1,2)*(q.shape[-1]**-.5)
  available=torch.ones(nt,device='cuda',dtype=torch.bool) if window_size[0]<0 else torch.arange(nt,device='cuda')<=window_size[0]
  first_prob=torch.exp(first_logit.masked_fill(~available[None,None,:],float('-inf'))-lse.float())
  assert torch.isfinite(first_prob).all() and float(first_prob.min())>=0 and float(first_prob.max())<=1+c['probability_atol']
  assert torch.count_nonzero(first_prob[:,:,~available])==0
  values[l]=dict(self_probability=prob[0].T.cpu().numpy(),first_probability=first_prob[0].T.cpu().numpy());windows[l]=list(window_size)
  if seq in c['validation_sequences']:
   base=original(q,k,v,causal=causal,window_size=window_size);assert torch.equal(out,base)
   pos=torch.tensor(c['validation_positions'],device='cuda');keys=torch.arange(nt,device='cuda');mask=keys[None,:]<=pos[:,None]
   if window_size[0]>=0:mask &= pos[:,None]-keys[None,:]<=window_size[0]
   scores=q[0,pos].transpose(0,1).double()@k[0].transpose(0,1).double().transpose(-1,-2)*(q.shape[-1]**-.5)
   scores.masked_fill_(~mask[None],float('-inf'));ref_lse=torch.logsumexp(scores,-1);ref_p=torch.softmax(scores,-1)
   ref_diag=ref_p.gather(-1,pos[None,:,None].expand(nh,-1,1)).squeeze(-1)
   p_error=float((prob[0,:,pos].double()-ref_diag).abs().max());lse_error=float((lse[0,:,pos].double()-ref_lse).abs().max())
   first_error_ref=float((first_prob[0,:,pos].double()-ref_p[:,:,0]).abs().max())
   assert max(p_error,first_error_ref)<c['probability_atol'] and lse_error<c['lse_atol']
   row_sum_error=float((ref_p.sum(-1)-1).abs().max());assert row_sum_error<1e-12 and torch.count_nonzero(ref_p.masked_select(~mask[None].expand(nh,-1,-1)))==0
   checks.append(dict(sequence=seq,layer=l,heads=list(range(nh)),positions=c['validation_positions'],window_size=list(window_size),probability_max_abs_error=p_error,first_probability_max_abs_error=first_error_ref,lse_max_abs_error=lse_error,reference_row_sum_max_error=row_sum_error,attention_output_bitwise_equal=True))
  return out
 write(r/'status.json',dict(state='running'))
 try:
  with torch.inference_mode():
   for seq in range(len(tokens)):
    ids=torch.from_numpy(tokens[seq:seq+1].astype(np.int64)).cuda();values={}
    fa.flash_attn.flash_attn_func=original;baseline=model(ids)
    fa.flash_attn.flash_attn_func=observe;logits=model(ids)
    assert torch.isfinite(logits).all() and torch.equal(logits,baseline) and sorted(values)==layers
    arrays={name:np.stack([values[l][name] for l in layers]) for name in ['self_probability','first_probability']}
    assert all(a.shape==(len(layers),nt,nh) for a in arrays.values())
    np.savez(r/'measurements'/f'sequence_{seq:03d}.npz',**arrays)
    del baseline,logits
    if (seq+1)%8==0:
     print(f'{seq+1}/{len(tokens)} sequences complete; logits bitwise equal',flush=True);write(r/'status.json',dict(state='running',completed_sequences=seq+1,total_sequences=len(tokens)))
 finally:
  fa.flash_attn.flash_attn_func=original
  for handle in handles:handle.remove()
 torch.cuda.synchronize()
 write(r/'measurements/schema.json',dict(shape_per_chunk=[len(layers),nt,nh],axes=['layer','sequence_position','head'],layers=layers,dtype='float32',token_ids='../inputs/tokens.npy',formula=m['formula'],windows=windows,clipping='none; small numerical overshoot retained',query_first_positions='saved but excluded from summaries',first_key='absolute sequence position zero; includes zero weight where outside the sliding window',arrays=['self_probability','first_probability'],head_dim=128))
 write(r/'validation/collection.json',dict(passed=True,logits_bitwise_equal_all_sequences=True,sequences_checked=len(tokens),probability_min=min_p,probability_max=max_p,first_position_max_abs_error=max_first_error,tolerances=dict(probability_atol=c['probability_atol'],lse_atol=c['lse_atol']),explicit_softmax_checks=checks))
 write(r/'collection.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),seconds=time.monotonic()-start,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,output_sha256={str(p.relative_to(r)):sha(p) for folder in ['measurements','validation'] for p in sorted((r/folder).iterdir())}))
 write(r/'status.json',dict(state='collected'))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve()
 try:collect(r)
 except BaseException as e:write(r/'status.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
