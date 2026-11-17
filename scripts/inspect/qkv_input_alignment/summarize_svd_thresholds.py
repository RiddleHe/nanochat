"""Derive squared-energy cutoffs for every saved head spectrum; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, require, sha256, verify_run, verify_svd, write_json


def summarize(run, thresholds=(.25, .5, .75, .9)):
    run = Path(run).resolve()
    manifest, meta = verify_run(run)
    verify_svd(run, manifest)
    thresholds = sorted(set(float(t) for t in thresholds))
    require(bool(thresholds) and all(0 < t <= 1 for t in thresholds), 'Invalid thresholds')
    tolerance = manifest['config']['tie_rtol']
    entries = []
    for layer in range(meta['n_layer']):
        for kind in 'QKV':
            with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as bank:
                for head, sigma in enumerate(bank['sigma']):
                    require(np.isfinite(sigma).all() and (sigma >= 0).all() and (np.diff(sigma) <= 0).all(), 'Invalid spectrum')
                    cumulative = np.cumsum(sigma.astype(np.float64)**2)
                    valid = bool(cumulative[-1] > 0)
                    points = []
                    for t in thresholds:
                        minimum = count = 0
                        captured = None
                        if valid:
                            fractions = cumulative/cumulative[-1]
                            minimum = int(np.searchsorted(fractions, t, side='left')) + 1
                            count = minimum
                            while count < len(sigma) and abs(sigma[count]-sigma[minimum-1]) <= tolerance*sigma[minimum-1]:
                                count += 1
                            captured = float(fractions[count-1])
                            require(captured >= t, 'Threshold not reached')
                            require(minimum == 1 or fractions[minimum-2] < t, 'Not minimum count')
                        for i, old in enumerate(manifest['config']['squared_energy_thresholds']):
                            if abs(old-t) < 1e-12:
                                require(count == int(bank['k'][head, i]), 'Saved cutoff mismatch')
                        points.append(dict(threshold=t, minimum_count=minimum, count=count, captured=captured))
                    entries.append(dict(layer=layer, kind=kind, head=head, rank=int(bank['rank'][head]), directions=len(sigma), valid=valid, points=points))
    inputs = dict(manifest_sha256=sha256(run/'manifest.json'), svd_index_sha256=sha256(run/'svd/index.json'), thresholds=thresholds, tie_rtol=tolerance,
                  code_sha256={p.name:sha256(p) for p in [Path(__file__), Path(__file__).with_name('common.py')]})
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    out = run/'derived'/f'svd_thresholds_{fingerprint}'
    if out.exists():
        require(read_json(out/'manifest.json')['inputs'] == inputs, 'Existing derivation mismatch')
        require(read_json(out/'statistics.json')['entries'] == entries, 'Existing data mismatch')
        return out
    out.mkdir(parents=True)
    write_json(out/'statistics.json', dict(run_id=manifest['run_id'], thresholds=thresholds, entries=entries))
    write_json(out/'manifest.json', dict(source_run=str(run), inputs=inputs, quantity='cumulative_squared_singular_values', statistics_sha256=sha256(out/'statistics.json')))
    write_json(out/'validation.json', dict(passed=True, heads=len(entries), saved_cutoffs_verified=True, minimal_counts_verified=True, svd_hashes_verified=True))
    for name in ['summarize_svd_thresholds.py', 'common.py']:
        shutil.copy2(Path(__file__).with_name(name), out/name)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[.25,.5,.75,.9])
    a = p.parse_args()
    print(summarize(a.run_dir, a.thresholds))
