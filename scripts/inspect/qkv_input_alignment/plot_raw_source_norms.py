"""Summarize saved source L2 norms and render Emb plus every attention output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, require, sha256, verify_chunk, verify_run, write_json


def plot(run,output,theme='light'):
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest,meta=verify_run(run)
    results=read_json(run/'results.json')
    require(results['state']=='analysis_complete' and read_json(run/'validation/summary.json')['passed'],'Run incomplete')
    mh,sh=sha256(run/'manifest.json'),sha256(run/'svd/index.json')
    require(results['manifest_sha256']==mh and results['svd_index_sha256']==sh,'Summary provenance mismatch')
    nseq,length=manifest['tokens_shape'];total=np.zeros(2*meta['n_layer']+1);squares=total.copy();hashes={}
    for i in range(nseq):
        path=run/'tokens'/f'chunk_{i:05d}.npz'
        rec=verify_chunk(run,path,mh,sh)
        require(rec['sequence_id']==i and rec['tokens']==length and rec['tokens_sha256']==manifest['tokens_sha256'],'Chunk identity mismatch')
        require(rec['sha256']==results['chunk_sha256'][path.name],'Summary chunk mismatch')
        with np.load(path) as z: raw=z['source_norm'].astype('float64')
        require(raw.shape==(length,len(total)) and np.isfinite(raw).all() and (raw>=0).all(),'Invalid source norms')
        total+=raw.sum(0);squares+=(raw*raw).sum(0);hashes[path.name]=rec['sha256']
    count=nseq*length;mean=total/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0))
    sources=read_json(run/'inputs/source_index.json')
    selected=[s for s in sources if s['source_family'] in ('embedding','attention')]
    require([s['source_layer'] for s in selected]==list(range(-1,meta['n_layer'])),'Source selection mismatch')
    points=[dict(**s,label='Emb' if s['source_family']=='embedding' else f'A{s["source_layer"]}',
                 mean=float(mean[s['index']]),std=float(std[s['index']])) for s in selected]
    data=dict(run_id=manifest['run_id'],model_label='AttnRes QKV' if manifest['config'].get('model_type')=='gpt_base_attn_res_qkv' else 'GPTBase',
              tokens=count,dimensions=meta['n_embd'],points=points,theme=theme,y_max=max(p['mean'] for p in points)*1.05)
    template=Path(__file__).parent.parent/'qkv_source_contributions/raw_source_norms_view.html'
    code=[Path(__file__),Path(__file__).with_name('common.py'),template]
    inputs=dict(manifest_sha256=mh,chunk_sha256=hashes,theme=theme,code_sha256={p.name:sha256(p) for p in code})
    fp=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    archive=run/'visualization'/f'raw_source_norms_attention_embedding_{theme}_{fp}'
    fragment=template.read_text().replace('__RAW_SOURCE_NORMS_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    if output.exists(): require(output.read_text()==fragment,'Choose a fresh output path')
    if archive.exists(): require(read_json(archive/'settings.json')['inputs']==inputs,'Archive mismatch')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(fragment);archive.mkdir(parents=True,exist_ok=True)
    for p in [*code,output]:shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),
        measurement='mean_over_tokens_of_full_raw_source_L2_norm',std_ddof=0,source_families=['embedding','attention'],
        routing_weighting=False,receiving_normalization=False,qkv_projection=False,scale='linear',y_domain=[0,data['y_max']]))
    write_json(archive/'validation.json',dict(passed=True,chunk_hashes_verified=True,token_count=count,points=len(points),finite_nonnegative=True))
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--theme',choices=['light','dark'],default='light')
    a=p.parse_args();print(plot(a.run_dir,a.output,a.theme))


if __name__=='__main__':main()
