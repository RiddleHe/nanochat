"""Render independently sorted cosine curves, retaining full measurements on disk."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from common import read,write,sha

def simplify(y,tolerance):
 keep={0,len(y)-1};stack=[(0,len(y)-1)]
 while stack:
  a,b=stack.pop()
  if b-a<2:continue
  expected=y[a]+(y[b]-y[a])*np.arange(1,b-a)/(b-a)
  error=np.abs(y[a+1:b]-expected);k=int(np.argmax(error))+a+1
  if error[k-a-1]>tolerance:keep.add(k);stack.extend([(a,k),(k,b)])
 return np.array(sorted(keep),dtype=np.int64)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();result=read(r/'results.json');m=read(r/'manifest.json')
 assert result['state']=='complete' and result['manifest_sha256']==sha(r/'manifest.json')
 for name,digest in result['output_sha256'].items():assert sha(r/name)==digest
 assert read(r/'validation/summary.json')['passed']
 series=[];validation=[]
 with np.load(r/'derived/statistics.npz') as z:
  for j,l in enumerate(z['layers']):
   n=int(z['valid_mask'][j].sum());assert n>=2
   order=z['sort_indices'][j,:n];values=z['cosine'][j,order].astype(np.float64)
   indices=simplify(values,m['config']['plot_max_absolute_cosine_error'])
   error=float(np.max(np.abs(np.interp(np.arange(n),indices,values[indices])-values)))
   assert error<=m['config']['plot_max_absolute_cosine_error']+1e-12
   points=[dict(x=100*int(i)/(n-1),y=float(values[i]),rank=int(i)+1,token_id=int(z['token_ids'][order[i]])) for i in indices]
   series.append(dict(id='L'+str(l),label='L'+str(l),layer=int(l),count=n,points=points))
   validation.append(dict(layer=int(l),points=len(points),maximum_display_error=error))
 data=dict(run_id=r.name,series=series,vocab_size=result['vocab_size'],formula='cosine(e, W_O W_V e)',sorting='independent descending cosine per layer',maximum_display_error=m['config']['plot_max_absolute_cosine_error'])
 template=Path(__file__).with_name('view.html');fragment=template.read_text().replace('__DATA__',json.dumps(data,separators=(',',':')));assert len(fragment.encode())<1000000
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(fragment)
 archive=r/'visualization/sorted_cosine_dark';archive.mkdir(parents=True,exist_ok=True)
 for p in [Path(__file__),Path(__file__).with_name('common.py'),template,a.output]:shutil.copy2(p,archive/p.name)
 write(archive/'data.json',data);write(archive/'settings.json',dict(source_manifest_sha256=result['manifest_sha256'],source_statistics_sha256=sha(r/'derived/statistics.npz'),sorting=data['sorting'],y_domain=[-1,1],x='100*(descending_rank-1)/(valid_vocab_count-1)',theme='dark',decimation='adaptive linear interpolation; raw measurements remain complete',maximum_display_error=data['maximum_display_error']))
 write(archive/'data_validation.json',dict(passed=True,layers=validation,fragment_bytes=len(fragment.encode())))
 write(archive/'hashes.json',{p.name:sha(p) for p in archive.iterdir() if p.is_file() and p.name!='hashes.json'})
 print(json.dumps(dict(archive=str(archive),validation=validation,fragment_bytes=len(fragment.encode()))))
if __name__=='__main__':main()
