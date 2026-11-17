"""Average per-query self-attention renormalized over eligible keys at positions >=4."""
from pathlib import Path
import argparse
import numpy as np
from common import read,write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);r=p.parse_args().run_dir.resolve();m=read(r/'manifest.json');collection=read(r/'collection.json');assert collection['state']=='complete' and collection['manifest_sha256']==sha(r/'manifest.json')
 assert not (r/'results.json').exists()
 for f,h in collection['output_sha256'].items():assert sha(r/f)==h
 assert read(r/'validation/collection.json')['passed']
 ns,nt=m['tokens_shape'];nh=m['model_config']['n_head'];layers=m['layers'];schema=read(r/'measurements/schema.json');pos=np.arange(4,nt)
 counts=np.array([pos-np.maximum(4,pos-schema['windows'][str(l)][0])+1 if schema['windows'][str(l)][0]>=0 else pos-3 for l in layers]);assert counts.min()==1
 names=['adjusted','uniform','raw'];shape=(len(layers),nh);s={name:np.zeros(shape) for name in names};ss={name:np.zeros(shape) for name in names};seqmeans={name:[] for name in names};fallback_counts=np.zeros(shape,dtype=np.int64)
 for i in range(ns):
  with np.load(r/'measurements'/f'sequence_{i:03d}.npz') as z:
   assert np.isnan(z['adjusted_self_probability'][:,:4]).all()
   a=z['adjusted_self_probability'][:,4:].astype(np.float64);raw=z['raw_self_probability'][:,4:].astype(np.float64);fallback_counts+=z['fallback'][:,4:].sum(1)
  assert np.isfinite(a).all() and np.max(np.abs(a[:,0]-1))<m['config']['adjusted_probability_atol']
  for name,b in dict(adjusted=a,raw=raw,uniform=np.broadcast_to(1/counts[:,:,None],a.shape)).items():s[name]+=b.sum(1);ss[name]+=(b*b).sum(1);seqmeans[name].append(b.mean(1))
 n=ns*(nt-4);arrays=dict(layers=np.array(layers),target_layers=np.array(m['target_layers']),heads=np.arange(nh),query_count=np.full(shape,n),remaining_key_counts=counts,query_positions=pos,fallback_counts=fallback_counts)
 for name in names:
  mean=s[name]/n;arrays[name+'_mean']=mean;arrays[name+'_std']=np.sqrt(np.maximum(ss[name]/n-mean*mean,0));arrays[name+'_sequence_means']=np.stack(seqmeans[name]);np.testing.assert_allclose(mean,arrays[name+'_sequence_means'].mean(0),atol=1e-14,rtol=1e-12)
 np.savez(r/'derived/statistics.npz',**arrays)
 summary=dict(model=m['model_type'],checkpoint_step=m['checkpoint_step'],sequences=ns,mean_sequence_length=nt,query_count_per_head=n,query_start=4,target_layers=m['target_layers'],rows=[])
 for i,l in enumerate(layers):
  row=dict(layer=l,target=l in m['target_layers'],window_size=schema['windows'][str(l)],heads=[])
  for name in names:row[name+'_mean']=float(arrays[name+'_mean'][i].mean())
  for h in range(nh):
   head=dict(head=h,count=n,fallback_count=int(fallback_counts[i,h]))
   for name in names:head[name]=dict(mean=float(arrays[name+'_mean'][i,h]),std=float(arrays[name+'_std'][i,h]))
   row['heads'].append(head)
  summary['rows'].append(row)
 write(r/'derived/summary.json',summary)
 write(r/'derived/schema.json',dict(metric='A[t,t] / sum(A[t,j] for eligible j>=4), computed per query t>=4 before averaging',removed_keys=[0,1,2,3],query_start=4,head_statistic='mean across 128 sequences and positions 4..2047',uniform='1/remaining eligible key count, averaged identically',std='population SD across query occurrences, not uncertainty of mean',curve='equal mean of twelve head means',target_layers=m['target_layers'],model_forward='unchanged; observational conditional probabilities, not an intervention'))
 write(r/'results.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),collection_sha256=sha(r/'collection.json'),output_sha256={str(p.relative_to(r)):sha(p) for p in sorted((r/'derived').iterdir())}));write(r/'status.json',dict(state='complete'))
 print([(row['layer'],round(row['adjusted_mean']*100,3),round(row['uniform_mean']*100,3)) for row in summary['rows']])
if __name__=='__main__':main()
