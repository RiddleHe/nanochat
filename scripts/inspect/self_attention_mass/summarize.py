"""Average saved per-token probabilities, excluding each sequence's first position."""
from pathlib import Path
import argparse
import numpy as np
from common import read,write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);r=p.parse_args().run_dir.resolve();m=read(r/'manifest.json');collection=read(r/'collection.json');assert collection['state']=='complete' and collection['manifest_sha256']==sha(r/'manifest.json')
 assert not (r/'results.json').exists()
 for f,h in collection['output_sha256'].items():assert sha(r/f)==h
 assert read(r/'validation/collection.json')['passed'] and m['config']['exclude_position_zero']
 nseq,nt=m['tokens_shape'];nh=m['model_config']['n_head'];shape=(len(m['layers']),nh);s=np.zeros(shape);ss=np.zeros(shape);n=0
 chunks=sorted((r/'measurements').glob('sequence_*.npy'));assert [p.name for p in chunks]==[f'sequence_{i:03d}.npy' for i in range(nseq)]
 for p in chunks:
  a=np.load(p);assert a.shape==(len(m['layers']),nt,nh) and a.dtype==np.float32
  a=a[:,1:,:].astype(np.float64);s+=a.sum(1);ss+=(a*a).sum(1);n+=a.shape[1]
 mean=s/n;variance=ss/n-mean*mean;assert variance.min()>-1e-12;sd=np.sqrt(np.maximum(variance,0));layer_mean=mean.mean(1)
 np.savez(r/'derived/statistics.npz',layers=np.array(m['layers']),heads=np.arange(nh),token_mean=mean,token_std=sd,count=np.full(shape,n),layer_mean=layer_mean)
 summary=dict(model=m['model_type'],checkpoint_step=m['checkpoint_step'],token_count_per_head=n,excluded_positions=nseq,rows=[dict(layer=l,mean=float(layer_mean[i]),heads=[dict(head=h,mean=float(mean[i,h]),std=float(sd[i,h]),count=n) for h in range(nh)]) for i,l in enumerate(m['layers'])])
 write(r/'derived/summary.json',summary)
 write(r/'derived/schema.json',dict(metric='softmax attention probability assigned to current sequence position',head_statistic='mean over token occurrences at positions 1..2047',std='population SD across those occurrences; not uncertainty of the mean',layer_statistic='equal mean of all 12 head means',positions='zero based',layers='zero based; target selection preserved from vocabulary VO run',clipping='none'))
 write(r/'results.json',dict(state='complete',manifest_sha256=sha(r/'manifest.json'),collection_sha256=sha(r/'collection.json'),output_sha256={str(p.relative_to(r)):sha(p) for p in sorted((r/'derived').iterdir())}));write(r/'status.json',dict(state='complete'))
 print('Layer mean self-attention:',[(s['layer'],round(s['mean']*100,3)) for s in summary['rows']])
if __name__=='__main__':main()
