"""Render alignment by source for selected receiving layers and heads; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, require, sha256, verify_run, verify_svd, write_json


def nullable(array):
    return [float(x) if np.isfinite(x) else None for x in array]


def load_data(run, layers, heads=None, family='attention', threshold=0.9):
    run=Path(run).resolve()
    manifest,meta=verify_run(run)
    require(manifest['config'].get('measurement','sources')=='sources', 'This view requires per-source measurements')
    verify_svd(run,manifest)
    results=read_json(run/'results.json')
    require(results['state']=='analysis_complete','Summaries incomplete')
    require(results['manifest_sha256']==sha256(run/'manifest.json') and results['svd_index_sha256']==sha256(run/'svd/index.json'),'Summary provenance mismatch')
    for relative,key in [('summaries/statistics.npz','statistics_sha256'),('summaries/schema.json','schema_sha256'),('validation/summary.json','validation_sha256')]:
        require(sha256(run/relative)==results[key],f'Summary changed: {relative}')
    require(read_json(run/'validation/summary.json')['passed'],'Validation failed')
    thresholds=manifest['config']['squared_energy_thresholds']
    matches=[i for i,t in enumerate(thresholds) if abs(t-threshold)<1e-12]
    require(len(matches)==1,'Requested squared-energy threshold was not collected')
    ti=matches[0]
    require(layers and len(set(layers))==len(layers) and all(0<=l<meta['n_layer'] for l in layers),'Invalid receiving layers')
    if heads is None: heads=list(range(min(meta['n_head'],meta['n_kv_head'])))
    require(heads and len(set(heads))==len(heads) and all(0<=h<min(meta['n_head'],meta['n_kv_head']) for h in heads),'Head must exist in Q, K, and V')
    families={'attention':('embedding','attention'),'mlp':('mlp',),'all':('embedding','attention','mlp')}[family]
    sources=read_json(run/'inputs/source_index.json');panels=[]
    with np.load(run/'summaries/statistics.npz') as stats:
        for layer in layers:
            selected=[s for s in sources[:2*layer+1] if s['source_family'] in families]
            for head in heads:
                for kind in manifest['config'].get('kinds', list('QKV')):
                    key=f'alignment_{kind}_{layer:02d}'
                    idx=[s['index'] for s in selected]
                    labels=['Emb' if s['source_family']=='embedding' else ('A' if s['source_family']=='attention' else 'M')+str(s['source_layer']) for s in selected]
                    with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as bank:
                        k=int(bank['k'][head,ti]);valid=bool(bank['matrix_valid'][head])
                        captured=float(bank['captured_squared_fraction'][head,ti])
                    panels.append(dict(layer=layer,head=head,kind=kind,title=f'L{layer} · head {head} · {kind}',
                        k=k,captured=captured,reference=k/meta['n_embd'] if valid else None,
                        labels=labels,mean=nullable(stats[key+'_token_mean'][head,idx,ti]),
                        std=nullable(stats[key+'_token_std'][head,idx,ti]),counts=stats[key+'_valid_count'][head,idx,ti].tolist()))
    data=dict(run_id=manifest['run_id'],tokens=results['tokens'],threshold=threshold,family=family,
        input_dimension=meta['n_embd'],panels=panels,view=f'alignment_{layers}_{heads}_{family}_{threshold}')
    data['model_label']='AttnRes QKV' if manifest['config'].get('model_type')=='gpt_base_attn_res_qkv' else 'GPTBase'
    if manifest['config'].get('model_type') == 'gpt_base_no_embed_norm':
        data['model_label'] = 'GPTBase · no norm(x0)'
    return data, results


def plot(run, output, layers, heads, family='attention', threshold=0.9):
    run,output=Path(run).resolve(),Path(output).resolve()
    data,results=load_data(run,layers,heads,family,threshold)
    template=Path(__file__).with_name('view.html')
    fragment=template.read_text().replace('__ALIGNMENT_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    require(len(fragment.encode())<1_000_000,'Select fewer layers/heads to stay within inline size limit')
    tag=f'{family}_layers_{"_".join(map(str,layers))}_heads_{"_".join(map(str,heads))}_sq{threshold:g}'
    inputs=dict(statistics_sha256=results['statistics_sha256'],svd_index_sha256=results['svd_index_sha256'],
        layers=layers,heads=heads,family=family,squared_energy_threshold=threshold,
        plot_sha256=sha256(Path(__file__)),common_sha256=sha256(Path(__file__).with_name('common.py')),template_sha256=sha256(template))
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    archive=run/'visualization'/f'{tag}_{fingerprint}'
    # A changed presentation gets a distinct archive; identical regeneration is safe.
    if archive.exists():
        previous=read_json(archive/'settings.json')
        require(previous['inputs']==inputs,'Presentation already exists with different settings/code; preserve it and use a new variant')
    if output.exists(): require(output.read_text()==fragment,'Output exists with different content; choose a fresh output path')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(fragment)
    archive.mkdir(parents=True,exist_ok=True)
    for p in (Path(__file__),Path(__file__).with_name('common.py'),template,output): shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),
        y_domain=[0,1],scale_scope='shared_fraction_axis_all_panels',head_aggregation=False))
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',type=int,nargs='+',default=[12]);p.add_argument('--heads',type=int,nargs='+',default=[1,7])
    p.add_argument('--family',choices=['attention','mlp','all'],default='attention');p.add_argument('--threshold',type=float,default=0.9)
    a=p.parse_args();print(plot(a.run_dir,a.output,a.layers,a.heads,a.family,a.threshold))


if __name__=='__main__': main()
