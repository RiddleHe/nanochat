"""Reuse validated singular vectors verbatim, deriving new squared-energy cutoffs."""
import argparse
from pathlib import Path
import platform
import sys
import numpy as np
from .common import now, read_json, require, sha256, stage_entry, verify_run, verify_svd, write_json, write_npz


def cutoffs(sigma, rank, thresholds, tie_rtol):
    require(np.isfinite(sigma).all() and (sigma >= 0).all() and (np.diff(sigma) <= 0).all(), 'Invalid spectrum')
    cumulative = np.cumsum(sigma.astype(np.float64)**2)
    total = float(cumulative[-1])
    counts, captured = [], []
    for t in thresholds:
        if not total:
            counts.append(0); captured.append(0.); continue
        k = min(int(np.searchsorted(cumulative/total,t,side='left'))+1, int(rank))
        require(k > 0, 'Nonzero matrix has zero rank')
        cutoff = sigma[k-1]
        while k < rank and abs(sigma[k]-cutoff) <= tie_rtol*cutoff:
            k += 1
        counts.append(k); captured.append(float(cumulative[k-1]/total))
    return np.array(counts,dtype=np.int64), np.array(captured,dtype=np.float64)


def source_run(run, manifest):
    ref = read_json(run/'inputs/svd_reference.json')
    source = Path(ref['run'])
    require(sha256(source/'manifest.json') == ref['manifest_sha256'], 'Source manifest changed')
    require(sha256(source/'svd/index.json') == ref['svd_index_sha256'], 'Source SVD index changed')
    sm, sc = verify_run(source)
    verify_svd(source, sm)
    require(sm['checkpoint_sha256'] == manifest['checkpoint_sha256'] and sm['tokens_sha256'] == manifest['tokens_sha256'], 'Reference checkpoint/tokens mismatch')
    require(sc == read_json(run/'inputs/checkpoint_metadata.json')['model_config'], 'Model configuration mismatch')
    for name in ('forward_dtype','svd_dtype','basis_dtype','tie_rtol'):
        require(sm['config'][name] == manifest['config'][name], f'SVD convention mismatch: {name}')
    return source, sm, ref


def reuse(run):
    run = Path(run)
    manifest, _ = verify_run(run)
    source, sm, ref = source_run(run,manifest)
    if (run/'svd/index.json').exists():
        verify_svd(run,manifest); print('Verified reused SVD bank.'); return
    require(not any((run/'tokens').glob('chunk_*.json')), 'Cannot replace committed SVDs')
    config = manifest['config']; thresholds = config['squared_energy_thresholds']
    files = {}
    old_index = read_json(source/'svd/index.json')
    for name in old_index['files']:
        path = run/'svd'/name
        with np.load(source/'svd'/name) as z:
            arrays = {key:z[key].copy() for key in z.files}
        old_k, old_captured = arrays['k'], arrays['captured_squared_fraction']
        selections = [cutoffs(s,r,thresholds,config['tie_rtol']) for s,r in zip(arrays['sigma'],arrays['rank'])]
        arrays['k'] = np.stack([s[0] for s in selections])
        arrays['captured_squared_fraction'] = np.stack([s[1] for s in selections])
        for i,t in enumerate(sm['config']['squared_energy_thresholds']):
            if t in thresholds:
                j=thresholds.index(t)
                require(np.array_equal(arrays['k'][:,j],old_k[:,i]), 'Shared cutoff mismatch')
                require(np.array_equal(arrays['captured_squared_fraction'][:,j],old_captured[:,i]), 'Shared captured fraction mismatch')
        rec = read_json((source/'svd'/name).with_suffix('.json'))
        for check in rec['checks']:
            require(check['reconstruction_relative_error'] <= config['validation']['svd_relative_error'], 'Inherited reconstruction error')
            require(check['stored_basis_orthogonality_error'] <= config['validation']['stored_basis_orthogonality_error'], 'Inherited basis error')
        write_npz(path,arrays)
        with np.load(path) as new, np.load(source/'svd'/name) as old:
            for key in ('sigma','vh','rank','matrix_valid'):
                require(new[key].dtype == old[key].dtype and new[key].tobytes() == old[key].tobytes(), 'Reused SVD array changed')
        files[name] = sha256(path)
        write_json(path.with_suffix('.json'),dict(rec,manifest_sha256=sha256(run/'manifest.json'),sha256=files[name],
            reused_from=str(source/'svd'/name),source_sha256=old_index['files'][name],basis_unchanged=True,shared_cutoffs_identical=True))
    write_json(run/'svd/environment.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
        operation='reuse_saved_singular_vectors_and_values; CPU_threshold_derivation_only',source=ref))
    write_json(run/'svd/index.json',dict(manifest_sha256=sha256(run/'manifest.json'),files=files,
        validation_sha256={Path(n).with_suffix('.json').name:sha256((run/'svd'/n).with_suffix('.json')) for n in files},
        environment_sha256=sha256(run/'svd/environment.json'),thresholds=thresholds,
        threshold_definition='squared_singular_values',created_utc=now(),reused_from=ref))
    verify_svd(run,manifest)
    write_json(run/'status.json',dict(state='svd_complete',completed_chunks=0))
    print(f'Reused {len(files)} matrices without SVD; thresholds={thresholds}.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();reuse(stage_entry(a.run_dir,'reuse_svd'))
