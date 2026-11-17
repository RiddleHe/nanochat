"""Render validated source cosine statistics as an inline light-mode heatmap."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .prepare import sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--family',choices=['attention-embedding','mlp'],default='attention-embedding')
    args=parser.parse_args()
    run=args.run_dir.resolve()
    results=json.loads((run/'results.json').read_text())
    assert results['state']=='analysis_complete' and results['validation']['passed']
    for name, expected in results['summary_sha256'].items():
        assert sha256(run/'summaries'/name)==expected
    schema=json.loads((run/'summaries/schema.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    depth=len(schema['source_index'])//2
    families = ('mlp',) if args.family == 'mlp' else ('embedding','attention')
    first_source = 0 if args.family == 'mlp' else -1
    cells=[]
    with np.load(run/'summaries/statistics.npz') as stats:
        for layer in range(depth):
            for qi,kind in enumerate(schema['query_types']):
                for source in schema['source_index'][:2*layer+1]:
                    if source['source_family'] not in families:
                        continue
                    si=source['index']
                    mean=stats[f'cosine_{layer:02d}_token_mean'][qi,si]
                    std=stats[f'cosine_{layer:02d}_token_std'][qi,si]
                    count=int(stats[f'cosine_{layer:02d}_valid_count'][qi,si])
                    assert (count==0 and np.isnan(mean) and np.isnan(std)) or (count>0 and -1<=mean<=1 and std>=0)
                    cells.append(dict(kind=kind,layer=layer,source_layer=source['source_layer'],
                        mean=float(mean) if count else None,std=float(std) if count else None,count=count,
                        query_norm=float(stats[f'query_norm_{layer:02d}'][qi])))
    assert len(cells)==3*depth*(depth-1 if args.family == 'mlp' else depth+1)//2
    assert all(c['source_layer'] < c['layer'] for c in cells)
    data=dict(run_id=manifest['run_id'],depth=depth,tokens=results['tokens'],cells=cells,
        family=args.family,first_source=first_source,source_label='MLP' if args.family=='mlp' else 'Attention')
    template=Path(__file__).with_name('view_with_embedding.html')
    html=template.read_text().replace('__COSINE_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    html=html.replace('__FAMILY__',args.family).replace('__SOURCE_TITLE__','MLP source' if args.family=='mlp' else 'embedding + attention')
    html=html.replace('__SOURCE_NOTE__','Raw MLP outputs · embedding and attention sources omitted' if args.family=='mlp' else 'Emb = normalized token embedding · raw attention outputs · MLP sources omitted')
    html=html.replace('__UNDEFINED_LEGEND__','' if args.family=='mlp' else '<span class="undefined-swatch" aria-hidden="true"></span><span>Undefined · zero query norm at layer 0</span>')
    assert '__COSINE_DATA__' not in html and len(html.encode())<1_000_000
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html)
    archive=run/'visualization'/('mlp_cosine' if args.family=='mlp' else 'attention_embedding_cosine')
    archive.mkdir(parents=True,exist_ok=True)
    for path in (Path(__file__),template,args.output):
        shutil.copy2(path,archive/path.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(source_families=list(families),omitted=['embedding','attention'] if args.family=='mlp' else ['mlp'],query_types=['Q','K','V'],
        color_domain=[-1,1],color_scale='linear_diverging',center=0,theme='fixed_light',layout='1x3_standard_inline_mobile_stacked',stack_below_px=600,
        measurement=schema['measurement'],aggregation=schema['aggregation'],std_ddof=0,undefined='solid_gray',unavailable='diagonal_hatch',
        summary_sha256=results['summary_sha256'],inline_path=str(args.output.resolve()),
        files_sha256={p.name:sha256(p) for p in (Path(__file__),template,args.output)}))
    print(json.dumps(dict(output=str(args.output),archive=str(archive),available_cells=len(cells),defined_cells=sum(c['count']>0 for c in cells),
        displayed_mean_range=[min(c['mean'] for c in cells if c['count']),max(c['mean'] for c in cells if c['count'])])))


if __name__=='__main__':
    main()
