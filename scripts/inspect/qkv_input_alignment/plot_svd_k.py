"""Plot each actual head's SVD cutoff k across receiving layers; saved SVD only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, require, sha256, verify_run, verify_svd, write_json


def plot(run, output, kind='V', threshold=0.9):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest, meta = verify_run(run)
    verify_svd(run, manifest)
    thresholds = manifest['config']['squared_energy_thresholds']
    matches = [i for i, t in enumerate(thresholds) if abs(t-threshold)<1e-12]
    require(len(matches)==1, 'Threshold not present in saved SVD')
    ti = matches[0]
    series = []
    for layer in range(meta['n_layer']):
        with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as bank:
            if not series:
                series = [dict(head=h, values=[]) for h in range(len(bank['k']))]
            require(len(series)==len(bank['k']), 'Head counts differ across layers')
            for h, s in enumerate(series):
                valid = bool(bank['matrix_valid'][h])
                s['values'].append(dict(layer=layer, k=int(bank['k'][h, ti]) if valid else None, rank=int(bank['rank'][h]),
                    captured=float(bank['captured_squared_fraction'][h, ti]) if valid else None))
    data = dict(run_id=manifest['run_id'], view=f'{kind}_svd_k_rank_sq{threshold:g}',
                kind=kind, threshold=threshold, series=series, theme='dark')
    here = Path(__file__).parent
    template = here/'svd_k_view.html'
    fragment = template.read_text().replace('__SVD_K_DATA__', json.dumps(data, separators=(',', ':'), allow_nan=False))
    require(len(fragment.encode())<1_000_000, 'Inline size limit exceeded')
    sources = [Path(__file__), here/'common.py', template]
    inputs = dict(manifest_sha256=sha256(run/'manifest.json'), svd_index_sha256=sha256(run/'svd/index.json'),
                  kind=kind, threshold=threshold, theme='dark', code_sha256={p.name:sha256(p) for p in sources})
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    archive = run/'visualization'/f'{data["view"]}_dark_{fingerprint}'
    if archive.exists(): require(read_json(archive/'settings.json')['inputs']==inputs, 'Archive mismatch')
    if output.exists(): require(output.read_text()==fragment, 'Preserve previous figure; choose a new path')
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(fragment)
    archive.mkdir(parents=True, exist_ok=True)
    for p in [*sources, output]: shutil.copy2(p, archive/p.name)
    write_json(archive/'data.json', data)
    write_json(archive/'settings.json', dict(inputs=inputs, inline_path=str(output), inline_sha256=sha256(output),
        head_aggregation=False, quantity='cutoff_k_and_numerical_rank', rank_tolerance='max(matrix_shape) * float64_epsilon * largest_singular_value',
        y_scale='linear_padded_observed_extent_all_heads'))
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['Q','K','V'], default='V'); p.add_argument('--threshold', type=float, default=0.9)
    a = p.parse_args(); print(plot(a.run_dir, a.output, a.kind, a.threshold))


if __name__=='__main__': main()
