"""Summarize attention to the first sequence token, with explicit comparison denominators."""
from pathlib import Path
import argparse
import numpy as np
from common import read,write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);r=p.parse_args().run_dir.resolve();m=read(r/'manifest.json');collection=read(r/'collection.json');assert collection['state']=='complete' and collection['manifest_sha256']==sha(r/'manifest.json')
 assert not (r/'results.json').exists()
 for f,h in collection['output_sha256'].items():assert sha(r/f)==h
 assert read(r/'validation/collection.json')['passed']
 nseq,nt=m['tokens_shape'];nh=m['model_config']['n_head'];layers=m['layers'];schema=read(r/'measurements/schema.json');positions=np.arange(1,nt)
 available=np.array([positions<=schema['windows'][str(l)][0] if schema['windows'][str(l)][0]>=0 else np.ones(nt-1,dtype=bool) for l in layers])
 counts=np.array([np.minimum(positions+1,schema['windows'][str(l)][0]+1) if schema['windows'][str(l)][0]>=0 else positions+1 for l in layers])
 assert np.all(counts>=2)
 names=['first','self','uniform','other'];shape=(len(layers),nh);s={name:np.zeros(shape) for name in names};ss={name:np.zeros(shape) for name in names};sequence_means={name:[] for name in names};conditional_sum=np.zeros(shape);conditional_sumsq=np.zeros(shape)
 for i in range(nseq):
  with np.load(r/'measurements'/f'sequence_{i:03d}.npz') as z:
   first=z['first_probability'][:,1:,:].astype(np.float64);self_p=z['self_probability'][:,1:,:].astype(np.float64);assert first.shape==(len(layers),nt-1,nh)
  assert np.count_nonzero(first[~available])==0
  metrics=dict(first=first,self=self_p,uniform=np.broadcast_to(1/counts[:,:,None],first.shape),other=(1-first)/(counts-available)[:,:,None])
  for name,a in metrics.items():s[name]+=a.sum(1);ss[name]+=(a*a).sum(1);sequence_means[name].append(a.mean(1))
  conditional_sum+=(first*available[:,:,None]).sum(1);conditional_sumsq+=(first*first*available[:,:,None]).sum(1)
 n=nseq*(nt-1);cn=nseq*available.sum(1);arrays=dict(layers=np.array(layers),heads=np.arange(nh),query_count=np.full(shape,n),first_available_query_count=cn,available_key_count=counts,first_key_available=available,sequence_lengths=np.full(nseq,nt))
 for name in names:
  mean=s[name]/n;std=np.sqrt(np.maximum(ss[name]/n-mean*mean,0));arrays[name+'_mean']=mean;arrays[name+'_std']=std;arrays[name+'_sequence_means']=np.stack(sequence_means[name]);np.testing.assert_allclose(mean,arrays[name+'_sequence_means'].mean(0),atol=1e-14,rtol=1e-12)
 arrays['first_available_mean']=conditional_sum/cn[:,None];arrays['first_available_std']=np.sqrt(np.maximum(conditional_sumsq/cn[:,None]-arrays['first_available_mean']**2,0))
 np.savez(r/'derived/statistics.npz',**arrays)
 summary=dict(model=m['model_type'],checkpoint_step=m['checkpoint_step'],sequences=nseq,mean_sequence_length=float(nt),min_sequence_length=nt,max_sequence_length=nt,query_count_per_head=n,excluded_query_positions=nseq,rows=[])
 for i,l in enumerate(layers):
  row=dict(layer=l,window_size=schema['windows'][str(l)],first_available_fraction=float(available[i].mean()),first_available_count=int(cn[i]),heads=[])
  for name in names:row[name+'_mean']=float(arrays[name+'_mean'][i].mean())
  for h in range(nh):
   head=dict(head=h,count=n,first_available_mean=float(arrays['first_available_mean'][i,h]))
   for name in names:head[name]=dict(mean=float(arrays[name+'_mean'][i,h]),std=float(arrays[name+'_std'][i,h]))
   row['heads'].append(head)
  summary['rows'].append(row)
 write(r/'derived/summary.json',summary)
 write(r/'derived/schema.json',dict(first='A[t,0], averaged over all query positions t>0 and sequences; zero when first key is masked',self='A[t,t], same averaging',uniform='1 / number of eligible keys, averaged over query positions t>0 and sequences; actual average softmax score over eligible keys',other='(1-A[t,0]) / (eligible_key_count - first_key_is_eligible), same averaging',std='population SD over query occurrences',sequence_means='saved separately; equal sequence mean equals occurrence mean because lengths match',first_available='conditional summary over queries where position zero is visible, saved separately',sequence_length='actual input window length, not underlying packed document length',population=m['population'],curve='equal mean over twelve head means'))
 write(r/'results.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),collection_sha256=sha(r/'collection.json'),output_sha256={str(p.relative_to(r)):sha(p) for p in sorted((r/'derived').iterdir())}));write(r/'status.json',dict(state='complete'))
 print('Layer / first-token % / average-eligible-token %:',[(row['layer'],round(row['first_mean']*100,3),round(row['uniform_mean']*100,3)) for row in summary['rows']])
if __name__=='__main__':main()
