"""Plot full input-component norms against individual Q/K/V head output norms."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 7])
    parser.add_argument('--family', choices=['attention', 'mlp'], default='attention')
    args = parser.parse_args()
    run, layer = args.run_dir, args.layer
    manifest = json.loads((run/'manifest.json').read_text())
    results = json.loads((run/'results.json').read_text())
    config = json.loads((run/'inputs/checkpoint_metadata.json').read_text())['model_config']
    assert manifest['config']['experiment'] == 'd24_qkv_separate_sources'
    assert json.loads((run/'validation/summary.json').read_text())['passed']
    stats_path = run/'summaries/statistics.npz'
    assert digest(stats_path) == results['statistics_sha256']
    sources = json.loads((run/'inputs/source_index.json').read_text())
    assert sources == manifest['source_index']
    assert 0 < layer < config['n_layer']
    assert len(set(args.heads)) == len(args.heads)
    assert all(0 <= h < min(config['n_head'], config['n_kv_head']) for h in args.heads)
    families = ('embedding','attention') if args.family == 'attention' else ('mlp',)
    start = -1 if args.family == 'attention' else 0
    selected = [s for s in sources[:2*layer+1] if s['source_family'] in families]
    assert [s['source_layer'] for s in selected] == list(range(start,layer))
    idx = [s['index'] for s in selected]
    panels = []
    with np.load(stats_path) as z:
        for head in args.heads:
            for kind in 'QKV':
                nheads = config['n_head'] if kind == 'Q' else config['n_kv_head']
                series = []
                for name, key in [('Input',f'input_{layer:02d}'),('Projected',f'abs_{kind}_{layer:02d}')]:
                    values = []
                    for stat in ('mean','std'):
                        a = z[f'{key}_token_{stat}']
                        assert a.shape == ((2*layer+1,) if name == 'Input' else (nheads,2*layer+1))
                        a = a[idx] if name == 'Input' else a[head,idx]
                        assert np.isfinite(a).all() and (a>=0).all()
                        values.append(a.tolist())
                    series.append(dict(name=name,values=values[0],std=values[1]))
                panels.append(dict(layer=layer,head=head,kind=kind,title=f'Head {head} · {kind}',series=series))
    data = dict(run_id=manifest['run_id'],view=f'head_input_L{layer}_{args.heads}_{args.family}',
        title=f'GPTBase · layer {layer} · input vs head output',tokens=results['tokens'],
        family_label='Emb + attention' if args.family == 'attention' else 'MLP',x_start=start,
        input_dimensions=config['n_embd'],output_dimensions=config['n_embd']//config['n_head'],
        panels=panels,source_index=selected,
        comparison_max=max(v for p in panels for s in p['series'] for v in s['values']))
    template = Path(__file__).with_name('head_input_view.html')
    html = template.read_text().replace('__SOURCE_NORMS_DATA__',json.dumps(data,separators=(',',':')))
    assert len(html.encode()) < 1_000_000
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html)
    archive = run/'visualization'/f'{args.family}_head_input_layer_{layer}_heads_{"_".join(map(str,args.heads))}'
    archive.mkdir(parents=True,exist_ok=True)
    files = (Path(__file__),template,args.output)
    for p in files: shutil.copy2(p,archive/p.name)
    settings = dict(statistics_sha256=digest(stats_path),manifest_sha256=digest(run/'manifest.json'),
        source_family=args.family,layer=layer,heads=args.heads,indexing='zero_based',
        input='source_vector_divided_by_actual_complete_residual_RMSNorm_denominator',
        output='individual_head_projected_norm_before_RoPE_and_QK_normalization',
        aggregation='token_mean_only_no_head_averaging_or_concatenation',scaling=False,
        input_dimensions=data['input_dimensions'],output_dimensions=data['output_dimensions'],
        y_domain=[0,data['comparison_max']*1.05],scale_scope='shared_across_all_panels',
        inline_path=str(args.output),files_sha256={p.name:digest(p) for p in files})
    for name,value in [('data',data),('settings',settings)]:
        (archive/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),archive=str(archive),panels=len(panels))))


if __name__ == '__main__':
    main()
