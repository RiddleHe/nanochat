"""Plot all centered/uncentered cumulative spectra and export concentration table."""
from pathlib import Path
import argparse,csv,json,shutil
import numpy as np
from common import read,write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();m=read(r/'manifest.json');result=read(r/'results.json')
 assert result['state']=='complete' and sha(r/'manifest.json')==result['manifest_sha256']
 for f,h in result['output_sha256'].items():assert sha(r/f)==h
 assert read(r/'validation/summary.json')['passed']
 archive=r/'visualization/cumulative_spectra_dark';archive.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
 summary=read(r/'derived/summary.json');all_data={}
 for mode in ['centered','uncentered']:
  series=[]
  for s in summary:
   f='embedding.npz' if s['source']=='Emb' else f"layer_{int(s['source'][1:]):02d}.npz"
   with np.load(r/'derived'/f) as z:
    y=z[mode+'_cumulative'];assert len(y)==1536 and np.all(np.diff(y)>=-1e-12) and abs(y[-1]-1)<1e-12
    # Cumulative sums can overshoot unity by roundoff before the exact endpoint.
    y=np.minimum(np.maximum.accumulate(y),1.0)
    series.append(dict(id=s['source'],label=s['source'],points=[dict(x=i+1,y=round(float(v),10)) for i,v in enumerate(y)]))
  data=dict(run_id=r.name,mode=mode,series=series,dimensions=1536,vocab_size=m['model_config']['vocab_size']);all_data[mode]=data
  template=Path(__file__).with_name('view.html').read_text()
  text=template.replace('__MODE__',mode).replace('__TITLE__','Centered PCA · cumulative variance' if mode=='centered' else 'Uncentered spectrum · cumulative energy').replace('__Y_LABEL__','Cumulative variance explained' if mode=='centered' else 'Cumulative energy captured').replace('__DATA__',json.dumps(data,separators=(',',':')))
  assert len(text.encode())<1000000
  out=a.output_dir/f'vocab-vo-pca-{mode}.html';out.write_text(text);shutil.copy2(out,archive/f'{mode}.html')
  write(archive/f'{mode}-data.json',data)
 columns=['source','mean_energy_fraction','centered_k50','centered_k90','centered_k99','centered_effective_rank','uncentered_k50','uncentered_k90','uncentered_k99','uncentered_effective_rank']
 with (archive/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for s in summary:
   row=dict(source=s['source'],mean_energy_fraction=s['mean_energy_fraction'])
   for mode in ['centered','uncentered']:
    for threshold,label in [('0.5','50'),('0.9','90'),('0.99','99')]:row[mode+'_k'+label]=s[mode]['direction_counts'][threshold]
    row[mode+'_effective_rank']=s[mode]['effective_rank']
   w.writerow(row)
 write(archive/'data.json',all_data)
 write(archive/'settings.json',dict(manifest_sha256=result['manifest_sha256'],inputs_sha256=result['output_sha256'],x='number of directions, logarithmic axis 1..1536',y='cumulative squared-singular-value fraction, linear 0..1',population=m['population'],theme='dark',smoothing=False,all_directions_plotted=True,display_rounding_absolute_error_at_most=5e-11))
 for f in ['plot.py','common.py','view.html']:shutil.copy2(Path(__file__).with_name(f),archive/f)
 write(archive/'hashes.json',{p.name:sha(p) for p in archive.iterdir() if p.is_file() and p.name!='hashes.json'})
 print(archive)
if __name__=='__main__':main()
