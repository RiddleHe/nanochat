"""Reproduce VO outputs then compute centered and uncentered FP64 spectra."""
from pathlib import Path
import argparse,os,sys,subprocess,platform,time,traceback
import numpy as np
from common import read,write,sha,hashes

def analyze(r):
 m=read(r/'manifest.json');c=m['config'];tol=c['validation']
 assert Path(__file__).resolve()==r/'code/analyze.py'
 if any(os.environ.get(k)!=v for k,v in m['environment_overrides'].items()):
  env=dict(os.environ,**m['environment_overrides']);env.pop('PYTHONPATH',None);os.execve(m['python_executable'],[m['python_executable'],str(__file__),'--run-dir',str(r)],env)
 assert hashes(r/'inputs')==m['input_hashes'] and hashes(r/'code')==m['code_hashes']
 assert not (r/'results.json').exists()
 import torch
 torch.set_num_threads(c['cpu_threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 assert torch.cuda.device_count()==1
 gpu=subprocess.check_output(['nvidia-smi','--id='+m['gpu_uuid'],'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(gpu[0])<1000 and int(gpu[1])<=5
 write(r/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,platform=platform.platform(),gpu_uuid=m['gpu_uuid'],gpu_name=torch.cuda.get_device_name(0),tf32=False))
 write(r/'status.json',dict(state='running'))
 state=torch.load(r/'inputs/checkpoint'/f"model_{m['checkpoint_step']:06d}.pt",map_location='cpu',weights_only=True,mmap=True);state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
 n,d=m['model_config']['vocab_size'],m['model_config']['n_embd'];emb=state['transformer.wte.weight'][:n].to(torch.bfloat16).float().cuda();assert emb.shape==(n,d)
 parent=np.load(r/'inputs/parent_statistics.npz');np.testing.assert_array_equal(parent['layers'],m['target_layers']);parent_index={int(l):i for i,l in enumerate(parent['layers'])};np.testing.assert_array_equal(parent['token_ids'],np.arange(n))
 np.testing.assert_array_equal(emb.norm(dim=-1).cpu().numpy(),parent['input_norm'])
 summaries=[];validation=[];start=time.monotonic();sample_ids=np.unique(np.r_[np.arange(8),np.linspace(0,n-1,56,dtype=int)])
 def spectrum(x,name):
  x=x.double();mu=x.mean(0);raw_energy=float(x.square().sum());mean_energy=float(n*mu.square().sum());arrays=dict(mean_vector=mu.cpu().numpy(),mean_energy_fraction=np.array(mean_energy/raw_energy),token_count=np.array(n),dimensions=np.array(d));summary=dict(source=name,mean_energy_fraction=mean_energy/raw_energy);checks=dict(source=name)
  for mode in ['uncentered','centered']:
   z=x if mode=='uncentered' else x-mu
   gram=z.T@z;gram=(gram+gram.T)*.5
   vals,vecs=torch.linalg.eigh(gram);vals=vals.flip(0);directions=vecs.flip(1).T
   minimum=float(vals.min());maximum=float(vals.max());negative_tolerance=32*d*torch.finfo(torch.float64).eps*maximum
   assert minimum>=-negative_tolerance
   vals=vals.clamp_min(0);energy=float(z.square().sum());eigen_energy=float(vals.sum())
   trace_error=abs(eigen_energy-energy)/energy
   orth=float((directions@directions.T-torch.eye(d,device='cuda',dtype=torch.float64)).abs().max())
   sample=z[sample_ids];recon=(sample@directions.T)@directions
   recon_error=float(((recon-sample).norm(dim=-1)/sample.norm(dim=-1).clamp_min(1e-30)).max())
   eigen_res=float((gram@directions.T-directions.T*vals).norm()/gram.norm())
   # Independent direct SVD on a row subset checks the Gram spectral method.
   small_s=torch.linalg.svdvals(sample);small_gram=sample@sample.T
   small_e=torch.linalg.eigvalsh((small_gram+small_gram.T)*.5).flip(0).clamp_min(0)
   svd_error=float((small_s.square()-small_e).norm()/small_s.square().norm())
   assert max(trace_error,recon_error,eigen_res,svd_error)<tol['spectral_relative_tolerance'] and orth<tol['orthogonality_atol']
   e=vals.cpu().numpy();fraction=e/e.sum();cumulative=np.minimum(np.cumsum(fraction),1.);cumulative[-1]=1.
   p=fraction[fraction>0];effective=float(np.exp(-(p*np.log(p)).sum()));participation=float(1/(fraction@fraction))
   counts={str(t):int(np.searchsorted(cumulative,t)+1) for t in c['thresholds']}
   arrays.update({mode+'_singular_values':np.sqrt(e),mode+'_fractions':fraction,mode+'_cumulative':cumulative,mode+'_directions':directions.cpu().numpy()})
   if mode=='centered':arrays['pc_variances']=e/(n-1)
   summary[mode]=dict(direction_counts=counts,effective_rank=effective,participation_rank=participation,total_energy=energy)
   checks[mode]=dict(trace_relative_error=trace_error,orthogonality_max_abs_error=orth,reconstruction_relative_error=recon_error,eigen_residual_relative_error=eigen_res,subset_direct_svd_relative_error=svd_error,minimum_unclipped_eigenvalue=minimum,negative_eigenvalue_tolerance=negative_tolerance)
  identity_error=abs(raw_energy-summary['centered']['total_energy']-mean_energy)/raw_energy;assert identity_error<tol['spectral_relative_tolerance'];checks['centering_energy_identity_relative_error']=identity_error
  filename='embedding.npz' if name=='Emb' else f'layer_{int(name[1:]):02d}.npz';np.savez_compressed(r/'derived'/filename,**arrays)
  print(f"{name}: centered k90={summary['centered']['direction_counts']['0.9']}, uncentered k90={summary['uncentered']['direction_counts']['0.9']}, mean-energy fraction={summary['mean_energy_fraction']:.5f}",flush=True)
  return summary,checks
 with torch.inference_mode():
  s,v=spectrum(emb,'Emb');summaries.append(s);validation.append(v)
  for j,l in enumerate(m['layers']):
   wv=state[f'transformer.h.{l}.attn.c_v.weight'].to(torch.bfloat16).float().cuda();wo=state[f'transformer.h.{l}.attn.c_proj.weight'].to(torch.bfloat16).float().cuda();assert wv.shape==wo.shape==(d,d)
   out=torch.empty_like(emb);norms=np.empty(n,np.float32);cos=np.empty(n,np.float32)
   for a in range(0,n,m['parent_batch_size']):
    x=emb[a:a+m['parent_batch_size']];y=(x@wv.T)@wo.T;out[a:a+len(x)]=y
    yn=y.norm(dim=-1);den=x.norm(dim=-1)*yn
    cc=torch.where(den>0,(x*y).sum(-1)/den,torch.full_like(den,float('nan'))).clamp(-1,1)
    norms[a:a+len(x)]=yn.cpu().numpy();cos[a:a+len(x)]=cc.cpu().numpy()
   # An independent FP64 composition checks all layers, including new ones.
   reference=emb[sample_ids].double()@(wo.double()@wv.double()).T
   projection_error=float(((out[sample_ids].double()-reference).norm(dim=-1)/reference.norm(dim=-1)).max())
   assert projection_error<2e-6
   checks=dict(parent_reference_available=l in parent_index,fp64_composed_projection_max_relative_error=projection_error)
   if l in parent_index:
    idx=parent_index[l]
    np.testing.assert_allclose(norms,parent['output_norm'][idx],rtol=tol['parent_norm_rtol'],atol=0)
    np.testing.assert_allclose(cos,parent['cosine'][idx],rtol=0,atol=tol['parent_cosine_atol'],equal_nan=True)
    checks['parent_norm_max_relative_error']=float(np.max(np.abs(norms-parent['output_norm'][idx])/np.maximum(parent['output_norm'][idx],np.finfo(np.float32).tiny)))
    checks['parent_cosine_max_absolute_error']=float(np.nanmax(np.abs(cos-parent['cosine'][idx])))
   s,v=spectrum(out,'L'+str(l));v.update(checks);summaries.append(s);validation.append(v)
   write(r/'status.json',dict(state='running',completed_layers=j+1,total_layers=len(m['layers'])))
 parent.close();torch.cuda.synchronize()
 write(r/'derived/summary.json',summaries)
 write(r/'derived/schema.json',dict(population=m['population'],vocab_size=n,dimensions=d,layers=m['layers'],target_layers=m['target_layers'],matrix='raw Emb or sequential FP32 output E @ W_V.T @ W_O.T with BF16-cast weights',centering='subtract FP64 mean vector; no row normalization or coordinate standardization',spectral_method=c['spectrum_method'],directions='rows are right singular vectors / PC directions in descending spectral order',pc_variance='centered singular value squared / (vocab_size-1)',mean_energy_fraction='N*||mean_vector||^2 / ||matrix||_F^2',fractions='squared singular values / sum of squared singular values',effective_rank='exp(-sum(p*log(p)))',participation_rank='1/sum(p^2)',thresholds='smallest number of directions reaching threshold; no tie extension',numerical_tail='Gram eigenvalues below zero within recorded roundoff tolerance clipped to zero; tiny singular values less accurate than direct SVD',scalar='positive per-layer Emb scalar omitted; spectral fractions unaffected'))
 write(r/'validation/summary.json',dict(passed=True,sample_token_ids=sample_ids.tolist(),checks=validation,tolerances=tol))
 write(r/'results.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),seconds=time.monotonic()-start,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,output_sha256={str(p.relative_to(r)):sha(p) for folder in ['derived','validation'] for p in sorted((r/folder).iterdir())}))
 write(r/'status.json',dict(state='complete'))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve()
 try:analyze(r)
 except BaseException as e:
  write(r/'status.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
