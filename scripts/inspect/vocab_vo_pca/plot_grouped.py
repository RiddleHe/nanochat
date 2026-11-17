"""Centered PCA for all layers, grouped by the original target selection."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from common import read,write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();m=read(r/'manifest.json');result=read(r/'results.json')
 assert result['state']=='complete' and sha(r/'manifest.json')==result['manifest_sha256']
 for f,h in result['output_sha256'].items():assert sha(r/f)==h
 assert read(r/'validation/summary.json')['passed'] and m['layers']==list(range(m['model_config']['n_layer']))
 archive=r/'visualization/centered_grouped_dark';archive.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
 series=[];d=m['model_config']['n_embd']
 for s in read(r/'derived/summary.json'):
  name=s['source'];layer=None if name=='Emb' else int(name[1:]);f='embedding.npz' if layer is None else f'layer_{layer:02d}.npz'
  with np.load(r/'derived'/f) as z:
   y=z['centered_cumulative'];assert len(y)==d and np.all(np.diff(y)>=-1e-12) and abs(y[-1]-1)<1e-12
   y=np.minimum(np.maximum.accumulate(y),1.)
   series.append(dict(id=name,label=name,group='Emb' if layer is None else 'target' if layer in m['target_layers'] else 'non-target',values=np.round(y,10).tolist()))
 data=dict(run_id=r.name,mode='centered',series=series,dimensions=d,vocab_size=m['model_config']['vocab_size'],target_layers=m['target_layers'])
 template=Path(__file__).with_name('grouped_view.html').read_text()
 html=template.replace('__MODE__','centered').replace('__TITLE__','Centered PCA · all layers').replace('__Y_LABEL__','Cumulative variance explained').replace('__DATA__',json.dumps(data,separators=(',',':')))
 assert len(html.encode())<1000000
 out=a.output_dir/'vocab-vo-pca-all-layers-centered.html';out.write_text(html);shutil.copy2(out,archive/'centered.html');write(archive/'data.json',data)
 write(archive/'settings.json',dict(manifest_sha256=result['manifest_sha256'],inputs_sha256=result['output_sha256'],target_layers=m['target_layers'],non_target_layers=[l for l in m['layers'] if l not in m['target_layers']],target_definition='Original strict Emb-largest selection within candidate layers L8–23; non-target means every other layer',theme='dark',legend=['Emb','Target','Non-target'],references=[.5,.9],x='Number of directions, logarithmic',y='Cumulative centered explained variance',all_directions_plotted=True,smoothing=False,display_rounding_absolute_error_at_most=5e-11))
 for f in ['plot_grouped.py','grouped_view.html','common.py']:shutil.copy2(Path(__file__).with_name(f),archive/f)
 write(archive/'hashes.json',{p.name:sha(p) for p in archive.iterdir() if p.is_file() and p.name!='hashes.json'})
 print(out)

if __name__=='__main__':main()
