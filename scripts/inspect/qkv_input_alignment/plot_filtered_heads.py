"""Stack Emb + attention alignment for all Q/K/V heads below a strict k cutoff."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from .common import read_json, require, sha256, verify_run, write_json
from .plot import load_data


def plot(run, output, cutoff=80, threshold=0.9):
    run, output = Path(run).resolve(), Path(output).resolve()
    _, meta = verify_run(run)
    full, results = load_data(run, list(range(meta['n_layer'])), family='attention', threshold=threshold)
    rows = sorted((p for p in full['panels'] if p['reference'] is not None and p['k']<cutoff),
                  key=lambda p:(p['layer'], 'QKV'.index(p['kind']), p['head']))
    require(bool(rows), 'No valid matrices meet the strict k cutoff')
    maximum=max((v for p in rows for v in p['mean'] if v is not None), default=0)
    data=dict(run_id=full['run_id'],view=f'filtered_heads_k_lt_{cutoff}_sq{threshold:g}',
              cutoff=cutoff,threshold=threshold,tokens=full['tokens'],input_dimension=full['input_dimension'],
              rows=rows,sources=['Emb']+[str(i) for i in range(max(p['layer'] for p in rows))],
              color_domain=[0,max(.01,math.ceil(maximum*100)/100)],theme='dark')
    here=Path(__file__).parent;template=here/'filtered_heads_view.html'
    fragment=template.read_text().replace('__FILTERED_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    require(len(fragment.encode())<1_000_000,'Inline size limit exceeded')
    sources=[Path(__file__),here/'plot.py',here/'common.py',template]
    inputs=dict(statistics_sha256=results['statistics_sha256'],svd_index_sha256=results['svd_index_sha256'],
                cutoff=cutoff,comparison='strict_less_than',threshold=threshold,family='embedding_and_attention',
                theme='dark',code_sha256={p.name:sha256(p) for p in sources})
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    archive=run/'visualization'/f'{data["view"]}_dark_{fingerprint}'
    if archive.exists():require(read_json(archive/'settings.json')['inputs']==inputs,'Archive mismatch')
    if output.exists():require(output.read_text()==fragment,'Preserve previous figure; choose a new path')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(fragment);archive.mkdir(parents=True,exist_ok=True)
    for p in [*sources,output]:shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),
        color_domain=data['color_domain'],row_order=['receiving_layer','Q_K_V','head_index'],
        row_entries=[dict(layer=p['layer'],kind=p['kind'],head=p['head'],k=p['k']) for p in rows],
        availability='Emb plus attention source_layer < receiving_layer',head_aggregation=False))
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cutoff',type=int,default=80);p.add_argument('--threshold',type=float,default=.9)
    a=p.parse_args();print(plot(a.run_dir,a.output,a.cutoff,a.threshold))


if __name__=='__main__':main()
