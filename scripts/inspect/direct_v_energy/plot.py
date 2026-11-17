"""Render the matched six-layer heatmaps with one linear scale across both models."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('both','no_embed_norm','baseline'), default='both')
    a = p.parse_args()
    root = a.run_dir.resolve(); parent = read(root/'manifest.json'); comparison = read(root/'comparison.json')
    datasets = []; result_hashes = {}
    model_names = ('no_embed_norm', 'baseline') if a.model == 'both' else (a.model,)
    for name in model_names:
        run = root/name; result = read(run/'results.json')
        assert sha(run/'results.json') == comparison['model_result_hashes'][name]
        assert result['state'] == 'complete' and result['tokens_sha256'] == parent['tokens_sha256']
        assert sha(run/'manifest.json') == result['manifest_sha256'] == parent['child_manifest_hashes'][name]
        assert sha(run/'validation/summary.json') == result['validation_sha256'] and read(run/'validation/summary.json')['passed']
        assert sha(run/'derived/statistics.npz') == result['statistics_sha256']
        result_hashes[name] = sha(run/'results.json')
        panels = []; sources = read(run/'inputs/source_index.json')
        with np.load(run/'derived/statistics.npz') as z:
            for l in parent['layers']:
                selected = [s for s in sources[:2*l+1] if s['family'] in ('embedding','attention')]
                idx = [s['index'] for s in selected]; labels = ['Emb' if s['family']=='embedding' else f"A{s['layer']}" for s in selected]
                for h in range(12):
                    panels.append(dict(layer=l, kind='V', head=h, labels=labels,
                        mean=z[f'energy_V_{l:02d}_mean'][h,idx].tolist(), std=z[f'energy_V_{l:02d}_std'][h,idx].tolist(),
                        shares=z[f'energy_share_V_{l:02d}_mean'][h,idx].tolist(), counts=[result['tokens']]*len(idx)))
        datasets.append(dict(run_id=root.name+'/'+name, view='direct-v-energy', model_label='GPTBase · no norm(x0)' if name=='no_embed_norm' else 'GPTBase · baseline',
            layers=parent['layers'], kind='V', layout='layer_grid', family='attention', transpose=True, theme='light', tokens=result['tokens'], head_dimension=128, panels=panels))
    maximum = max(v for d in datasets for p in d['panels'] for v in p['mean'])
    for d in datasets:
        d['color_domain'] = [0, maximum]
        d['scale_label'] = 'same linear scale for both models' if len(datasets)>1 else 'shared linear scale across all six layers'
    here = Path(__file__).parent; template = (here/'view.html').read_text()
    fragments = []
    for i, data in enumerate(datasets):
        fragment = template.replace('qkv-projected-energy', 'direct-v-energy-'+str(i)).replace('__ENERGY_DATA__', json.dumps(data,separators=(',',':'),allow_nan=False))
        if i: fragment = fragment.replace('  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>\n','')
        fragments.append(fragment)
    fragment = '\n'.join(fragments); output = a.output.resolve()
    assert len(fragment.encode()) < 1000000 and '__ENERGY_DATA__' not in fragment
    assert not output.exists() or output.read_text() == fragment
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(fragment)
    settings = dict(result_hashes=result_hashes, comparison_sha256=sha(root/'comparison.json'), inline_path=str(output), inline_sha256=sha(output),
        code_hashes={n:sha(here/n) for n in ('plot.py','view.html','common.py')}, color_domain=[0,maximum],
        theme='same light blue sequential heatmap as reference', models=list(model_names), color_scope='all displayed models/layers/heads/sources',
        units='mean per-token squared L2 norm', layers=parent['layers'], heads=list(range(12)), source_family='embedding+attention',
        normalization='source / actual complete receiver residual RMSNorm denominator', directions='all; no SVD', data_tokens=262144)
    archive = root/'visualization'/sha(output)[:12]; archive.mkdir(parents=True,exist_ok=True)
    for file in (here/'plot.py',here/'view.html',here/'common.py',output): shutil.copy2(file,archive/file.name)
    write(archive/'data.json',datasets); write(archive/'settings.json',settings)
    print(archive)


if __name__ == '__main__': main()
