"""Render head means and a mean-across-heads curve from validated measurements."""
from pathlib import Path
import argparse,json,shutil
from common import read,write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();results=read(r/'results.json');assert results['state']=='complete' and results['manifest_sha256']==sha(r/'manifest.json')
 assert results['collection_sha256']==sha(r/'collection.json') and read(r/'validation/collection.json')['passed']
 for f,h in results['output_sha256'].items():assert sha(r/f)==h
 data=read(r/'derived/summary.json');data['run_id']=r.name;template=Path(__file__).with_name('exclude_prefix_view.html').read_text();html=template.replace('__DATA__',json.dumps(data,separators=(',',':')));assert len(html.encode())<1000000
 a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'self-attention-exclude-prefix.html';out.write_text(html);archive=r/'visualization/self_attention_exclude_prefix_target_color_dark';archive.mkdir(parents=True,exist_ok=True);shutil.copy2(out,archive/out.name)
 write(archive/'data.json',data);write(archive/'settings.json',dict(manifest_sha256=results['manifest_sha256'],summary_sha256=sha(r/'derived/summary.json'),theme='dark',x='zero-based attention layer; dots have small fixed offsets by group and head',y='mean softmax attention probability, percent; zero baseline and padded data maximum',metric='self-attention conditional on eligible keys at absolute sequence positions >=4',target_layers=data['target_layers'],shapes='target triangle, other circle, uniform square',colors=dict(target='--viz-series-3',other='--viz-series-1',uniform='--viz-series-2'),comparison='1 / remaining eligible-key count per query',mean_sequence_length=data['mean_sequence_length'],curve='equal mean of 12 head token means',query_positions='0..3 excluded; 4 included',std='token population SD, hover only',smoothing=False))
 for f in ['plot_exclude_prefix.py','exclude_prefix_view.html','common.py']:shutil.copy2(Path(__file__).with_name(f),archive/f)
 write(archive/'hashes.json',{p.name:sha(p) for p in archive.iterdir() if p.is_file() and p.name!='hashes.json'});print(out)
if __name__=='__main__':main()
