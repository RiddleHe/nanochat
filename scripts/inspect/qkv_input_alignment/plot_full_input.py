"""Plot actual undivided Q/K/V input alignment: receiving layer X, query/head Y."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, require, sha256, verify_run, verify_svd, write_json


def plot(run, output, threshold=.9, kinds='QKV', theme='dark', emb_reference_run=None, box_kinds=None):
    require(box_kinds is None or (emb_reference_run is not None and box_kinds and set(box_kinds) <= set(kinds)),
            'Box kinds require an embedding reference and must be displayed projections')
    box_kinds = kinds if box_kinds is None else box_kinds
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest,meta=verify_run(run);verify_svd(run,manifest)
    require(manifest['config'].get('measurement')=='full_input','Requires actual full-input collection')
    results=read_json(run/'results.json')
    require(results['state']=='analysis_complete','Summary incomplete')
    require(results['manifest_sha256']==sha256(run/'manifest.json') and results['svd_index_sha256']==sha256(run/'svd/index.json'),'Summary provenance mismatch')
    for path,key in [('summaries/statistics.npz','statistics_sha256'),('summaries/schema.json','schema_sha256'),('validation/summary.json','validation_sha256')]:
        require(results[key]==sha256(run/path),'Summary hash mismatch')
    require(read_json(run/'validation/summary.json')['passed'],'Summary validation failed')
    ts=manifest['config']['squared_energy_thresholds'];matches=[i for i,t in enumerate(ts) if abs(t-threshold)<1e-12]
    require(len(matches)==1,'Threshold not collected');ti=matches[0];rows=[]
    with np.load(run/'summaries/statistics.npz') as stats:
        for kind in kinds:
            heads=meta['n_head'] if kind=='Q' else meta['n_kv_head']
            group=[dict(kind=kind,head=h,values=[]) for h in range(heads)]
            for layer in range(meta['n_layer']):
                key=f'alignment_{kind}_{layer:02d}'
                with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as bank:
                    for h,row in enumerate(group):
                        mean=float(stats[key+'_token_mean'][h,0,ti]);std=float(stats[key+'_token_std'][h,0,ti])
                        k=int(bank['k'][h,ti]);valid=bool(bank['matrix_valid'][h])
                        row['values'].append(dict(layer=layer,mean=mean if np.isfinite(mean) else None,std=std if np.isfinite(std) else None,
                            count=int(stats[key+'_valid_count'][h,0,ti]),k=k,captured=float(bank['captured_squared_fraction'][h,ti]) if valid else None,
                            reference=k/meta['n_embd'] if valid else None))
            rows.extend(group)
    comparison=None
    if emb_reference_run is not None:
        from .plot import load_data
        emb_run=Path(emb_reference_run).resolve()
        emb_manifest,emb_meta=verify_run(emb_run)
        require(meta==emb_meta and manifest['checkpoint_sha256']==emb_manifest['checkpoint_sha256'] and
                manifest['tokens_sha256']==emb_manifest['tokens_sha256'], 'Embedding comparison model/tokens mismatch')
        emb_data,emb_results=load_data(emb_run,list(range(meta['n_layer'])),family='attention',threshold=threshold)
        require(read_json(run/'svd/index.json')['files']==read_json(emb_run/'svd/index.json')['files'], 'Comparison singular subspaces differ')
        lookup={(p['kind'],p['head'],p['layer']):p for p in emb_data['panels']}
        for row in rows:
            for v in row['values']:
                ref=lookup[row['kind'],row['head'],v['layer']]
                require(ref['labels'][0]=='Emb' and ref['k']==v['k'] and ref['counts'][0]==v['count'], 'Embedding comparison axes/counts mismatch')
                v['emb_mean']=ref['mean'][0]
                v['delta']=v['mean']-v['emb_mean'] if v['mean'] is not None and v['emb_mean'] is not None else None
                v['exceeds_emb']=v['delta'] is not None and v['delta']>0
                v['boxed']=v['exceeds_emb'] and row['kind'] in box_kinds
        comparison=dict(run_dir=str(emb_run),manifest_sha256=sha256(emb_run/'manifest.json'),
                        statistics_sha256=emb_results['statistics_sha256'],svd_index_sha256=emb_results['svd_index_sha256'],
                        criterion='token_mean_full_input_alignment > token_mean_embedding_alignment; strict comparison, no tolerance')
    maximum=max((v['mean'] for r in rows for v in r['values'] if v['mean'] is not None),default=0)
    data=dict(run_id=manifest['run_id'],view=f'actual_full_input_{kinds}_sq{threshold:g}',threshold=threshold,tokens=results['tokens'],
              input_dimension=meta['n_embd'],rows=rows,sources=[str(i) for i in range(meta['n_layer'])],
              color_domain=[0,max(.01,math.ceil(maximum*100)/100)],theme=theme,kinds=kinds)
    if comparison:
        data['view']+='_boxed_above_emb'
        if box_kinds != kinds: data['view']+=f'_boxes_{box_kinds}'
        data['embedding_comparison']=comparison
        data['box_kinds']=box_kinds
        data['boxed_count']=sum(v['boxed'] for row in rows for v in row['values'])
    here=Path(__file__).parent;template=here/'full_input_view.html'
    fragment=template.read_text().replace('__FULL_INPUT_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    require(len(fragment.encode())<1_000_000,'Inline size limit exceeded')
    sources=[Path(__file__),here/'common.py',template]
    if comparison: sources.append(here/'plot.py')
    inputs=dict(statistics_sha256=results['statistics_sha256'],svd_index_sha256=results['svd_index_sha256'],
                threshold=threshold,measurement='actual_full_input',theme=theme,kinds=kinds,code_sha256={p.name:sha256(p) for p in sources})
    if comparison:
        inputs['embedding_comparison']=comparison
        inputs['box_kinds']=box_kinds
    fp=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    archive=run/'visualization'/f'{data["view"]}_{theme}_{fp}'
    if archive.exists():require(read_json(archive/'settings.json')['inputs']==inputs,'Archive mismatch')
    if output.exists():require(output.read_text()==fragment,'Preserve previous figure; choose a new path')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(fragment);archive.mkdir(parents=True,exist_ok=True)
    for p in [*sources,output]:shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),
        color_domain=data['color_domain'],row_order=['Q_K_V','head_index'],x_axis='receiving_layer',head_aggregation=False))
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--kind',choices=['QKV','Q','K','V'],default='QKV')
    p.add_argument('--theme',choices=['dark','light'],default='dark')
    p.add_argument('--emb-reference-run',type=Path,help='Outline cells above Emb-only token mean from a matching source-alignment run')
    p.add_argument('--box-kinds',help='Restrict comparison boxes to these displayed projections, e.g. V')
    a=p.parse_args();print(plot(a.run_dir,a.output,a.threshold,a.kind,a.theme,a.emb_reference_run,a.box_kinds))


if __name__=='__main__':main()
