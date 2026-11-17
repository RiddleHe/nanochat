"""Plot raw-source per-head output energy collected alongside input alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, require, sha256, verify_run, write_json
from .plot import load_data


def plot(run, output, layers, kind='V', family='attention'):
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest,meta=verify_run(run)
    require(kind in manifest['config'].get('raw_projected_energy_kinds',[]), 'Raw output energy was not collected')
    # Use the shared validated source selection and provenance checks.
    data,results=load_data(run,layers,family=family,threshold=manifest['config']['squared_energy_thresholds'][0])
    sources=read_json(run/'inputs/source_index.json')
    allowed={'attention':('embedding','attention'),'mlp':('mlp',)}[family]
    data['panels']=[p for p in data['panels'] if p['kind']==kind]
    with np.load(run/'summaries/statistics.npz') as stats:
        for panel in data['panels']:
            layer,h=panel['layer'],panel['head']
            indices=[s['index'] for s in sources[:2*layer+1] if s['source_family'] in allowed]
            key=f'raw_energy_{kind}_{layer:02d}'
            panel.update(mean=stats[key+'_token_mean'][h,indices].tolist(),
                         std=stats[key+'_token_std'][h,indices].tolist(),
                         counts=stats[key+'_valid_count'][h,indices].tolist())
    maximum=max(v for p in data['panels'] for v in p['mean'])
    data.update(view=f'raw_projected_energy_{kind}_{layers}_{family}',layers=layers,kind=kind,
                layout='layer_grid',transpose=True,theme='light',raw_sources=True,
                head_dimension=meta['n_embd']//meta['n_head'],color_domain=[0,maximum if maximum>0 else 1])
    here=Path(__file__).parent
    template=here.parent/'qkv_source_contributions/projected_energy_view.html'
    code=[Path(__file__),here/'common.py',here/'plot.py',template]
    inputs=dict(manifest_sha256=sha256(run/'manifest.json'),statistics_sha256=results['statistics_sha256'],
                svd_index_sha256=results['svd_index_sha256'],layers=layers,kind=kind,family=family,
                code_sha256={p.name:sha256(p) for p in code})
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    archive=run/'visualization'/f'raw_energy_{kind}_{"_".join(map(str,layers))}_{family}_{fingerprint}'
    fragment=template.read_text().replace('__ENERGY_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    require(len(fragment.encode())<1_000_000,'Inline size limit exceeded')
    if output.exists(): require(output.read_text()==fragment,'Preserve existing plot; choose a fresh path')
    if archive.exists(): require(read_json(archive/'settings.json')['inputs']==inputs,'Archive mismatch')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(fragment)
    archive.mkdir(parents=True,exist_ok=True)
    for p in [*code,output]: shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),
        units='squared_L2_norm',routing_weighting=False,receiving_normalization=False,
        singular_directions='all',head_aggregation=False,aggregation='square_per_token_then_mean',
        color_domain=data['color_domain'],color_scale='shared_linear_all_displayed_cells'))
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',type=int,nargs='+',required=True)
    p.add_argument('--kind',choices=['Q','K','V'],default='V')
    p.add_argument('--family',choices=['attention','mlp'],default='attention')
    a=p.parse_args();print(plot(a.run_dir,a.output,a.layers,a.kind,a.family))


if __name__=='__main__':main()
