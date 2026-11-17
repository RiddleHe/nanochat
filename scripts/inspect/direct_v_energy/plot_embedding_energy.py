"""Plot embedding component energy, equally averaged across all twelve V heads."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metric', choices=['energy', 'share'], default='energy')
    parser.add_argument('--uniform-reference', action='store_true', help='Add equal component-energy share 1/(2L+1)')
    parser.add_argument('--include-first-attention', action='store_true', help='Add the no-norm model attention-0 energy share')
    args = parser.parse_args()
    if args.uniform_reference and args.metric != 'share':
        parser.error('--uniform-reference requires --metric share')
    if args.include_first_attention and args.metric != 'share':
        parser.error('--include-first-attention requires --metric share')
    root = args.run_dir.resolve(); parent = read(root/'manifest.json')
    comparison = read(root/'comparison.json'); series = []; inputs = {}
    for name, label in [('baseline','GPTBase baseline'),('no_embed_norm','No norm(x0)')]:
        run = root/name; result = read(run/'results.json')
        assert sha(run/'results.json') == comparison['model_result_hashes'][name]
        assert result['state'] == 'complete' and result['tokens_sha256'] == parent['tokens_sha256']
        assert sha(run/'manifest.json') == result['manifest_sha256'] == parent['child_manifest_hashes'][name]
        assert sha(run/'validation/summary.json') == result['validation_sha256'] and read(run/'validation/summary.json')['passed']
        assert sha(run/'derived/statistics.npz') == result['statistics_sha256']
        inputs[name] = dict(results_sha256=sha(run/'results.json'), statistics_sha256=result['statistics_sha256'])
        assert read(run/'inputs/source_index.json')[0]['family'] == 'embedding'
        points = []
        with np.load(run/'derived/statistics.npz') as z:
            for layer, previous in zip(parent['layers'],comparison['layers']):
                assert layer == previous['layer']
                field = 'energy_share' if args.metric == 'share' else 'energy'
                heads = z[f'{field}_V_{layer:02d}_mean'][:,0]
                assert heads.shape == (12,) and np.isfinite(heads).all()
                key = 'baseline_embedding_energy' if name=='baseline' else 'variant_embedding_energy'
                if args.metric == 'share':
                    key += '_share'
                    assert ((heads >= 0) & (heads <= 1)).all()
                np.testing.assert_array_equal(heads, previous[key])
                points.append(dict(x=layer, y=float(heads.mean())))
        series.append(dict(id=name,label=label,points=points))
    if args.uniform_reference:
        series.append(dict(id='uniform', label='Uniform share · 1/(2L+1)',
                           points=[dict(x=layer, y=1/(2*layer+1)) for layer in parent['layers']]))
    if args.include_first_attention:
        run = root/'no_embed_norm'
        source = read(run/'inputs/source_index.json')[1]
        assert source == dict(index=1, family='attention', layer=0)
        assert all(layer >= 1 for layer in parent['layers'])
        points = []
        with np.load(run/'derived/statistics.npz') as z:
            for layer in parent['layers']:
                heads = z[f'energy_share_V_{layer:02d}_mean'][:,1]
                assert heads.shape == (12,) and np.isfinite(heads).all() and ((heads >= 0) & (heads <= 1)).all()
                points.append(dict(x=layer, y=float(heads.mean())))
        for s in series[:2]: s['label'] += ' · Emb'
        series.append(dict(id='no_embed_norm_attn0',label='No norm(x0) · A0',points=points))
    data = dict(run_id=root.name, tokens=262144, heads=12, metric=args.metric, uniform_reference=args.uniform_reference, include_first_attention=args.include_first_attention, series=series)
    here = Path(__file__).parent; template = here/'embedding_energy_view.html'
    fragment = template.read_text().replace('__EMBEDDING_ENERGY__',json.dumps(data,separators=(',',':'),allow_nan=False))
    output = args.output.resolve()
    assert '__EMBEDDING_ENERGY__' not in fragment and len(fragment.encode())<1000000
    assert not output.exists() or output.read_text()==fragment
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(fragment)
    archive = root/'visualization'/('embedding_'+args.metric+'_'+sha(output)[:12]); archive.mkdir(parents=True,exist_ok=True)
    for p in (Path(__file__),here/'common.py',template,output): shutil.copy2(p,archive/p.name)
    write(archive/'data.json',data)
    write(archive/'settings.json',dict(source_run=str(root),inputs=inputs,
        code_hashes={p.name:sha(p) for p in (Path(__file__),here/'common.py',template)},
        inline_path=str(output),inline_sha256=sha(output),theme='Codex dark',x_scale='linear',y_scale='linear from zero, fixed across toggles',
        metric=args.metric, uniform_reference=('1/(2L+1): equal energy for Emb and every preceding attention and MLP source' if args.uniform_reference else None),
        aggregation=('equal arithmetic mean across 12 heads of token-mean selected source energy / sum of all available Emb+attention+MLP component energies; ratio computed before token averaging' if args.metric == 'share' else 'equal arithmetic mean of the 12 per-head token-mean squared V-projection norms'),
        sources=({'baseline':[0], 'no_embed_norm':[0,1]} if args.include_first_attention else {'baseline':[0], 'no_embed_norm':[0]}),
        normalization='source / actual complete receiver residual RMSNorm denominator',layers=parent['layers']))
    print(archive)


if __name__ == '__main__': main()
