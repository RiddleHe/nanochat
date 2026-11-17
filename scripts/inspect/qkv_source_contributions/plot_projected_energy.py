"""Plot token-mean squared projected norms, preserving individual heads and sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .prepare import sha256, write_json


def plot(derived,output,kind='V',family='attention'):
    derived,output=Path(derived).resolve(),Path(output).resolve()
    manifest=json.loads((derived/'manifest.json').read_text());validation=json.loads((derived/'validation.json').read_text())
    assert validation['passed'] and sha256(derived/'statistics.npz')==manifest['statistics_sha256']
    assert kind in manifest['inputs']['kinds']
    layers=manifest['inputs']['layers'];meta=manifest['model_config'];heads=meta['n_head'] if kind=='Q' else meta['n_kv_head'];panels=[]
    families={'attention':('embedding','attention'),'mlp':('mlp',)}[family]
    with np.load(derived/'statistics.npz') as z:
        for layer in layers:
            selected=[s for s in manifest['source_index'][:2*layer+1] if s['source_family'] in families]
            idx=[s['index'] for s in selected]
            labels=['Emb' if s['source_family']=='embedding' else ('A' if s['source_family']=='attention' else 'M')+str(s['source_layer']) for s in selected]
            for h in range(heads):
                key=f'abs_{kind}_{layer:02d}'
                panels.append(dict(layer=layer,kind=kind,head=h,labels=labels,
                    mean=z[key+'_squared_token_mean'][h,idx].tolist(),std=z[key+'_squared_token_std'][h,idx].tolist(),
                    counts=[manifest['tokens']]*len(idx)))
    maximum=max(v for p in panels for v in p['mean'])
    data=dict(run_id=Path(manifest['inputs']['run_dir']).name,view=f'projected_energy_{kind}_{layers}_{family}',
        layers=layers,kind=kind,layout='layer_grid',family=family,transpose=True,theme='light',tokens=manifest['tokens'],
        head_dimension=meta['n_embd']//meta['n_head'],panels=panels,color_domain=[0,maximum if maximum>0 else 1])
    here=Path(__file__).parent;template=here/'projected_energy_view.html'
    fragment=template.read_text().replace('__ENERGY_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    assert len(fragment.encode())<1_000_000
    inputs=dict(derived_dir=str(derived),derived_manifest_sha256=sha256(derived/'manifest.json'),statistics_sha256=manifest['statistics_sha256'],
        kind=kind,family=family,layers=layers,theme='light',plot_sha256=sha256(Path(__file__)),template_sha256=sha256(template),prepare_sha256=sha256(here/'prepare.py'))
    fp=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    run=Path(manifest['inputs']['run_dir']);archive=run/'visualization'/f'projected_energy_{kind}_{"_".join(map(str,layers))}_{family}_{fp}'
    if archive.exists():assert json.loads((archive/'settings.json').read_text())['inputs']==inputs
    if output.exists():assert output.read_text()==fragment,'Choose a fresh output path'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(fragment);archive.mkdir(parents=True,exist_ok=True)
    for p in [Path(__file__),here/'prepare.py',template,output]:shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),color_domain=data['color_domain'],
        color_scale='shared_linear_across_all_panels',aggregation=manifest['aggregation'],units='squared_L2_norm',
        projection=manifest['projection'],input=manifest['input'],singular_directions='all'))
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--derived-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['Q','K','V'],default='V');p.add_argument('--family',choices=['attention','mlp'],default='attention')
    a=p.parse_args();print(plot(a.derived_dir,a.output,a.kind,a.family))


if __name__=='__main__':main()
