"""Validate chunks and aggregate per-token energy and full-bank energy shares in FP64."""
import argparse
from pathlib import Path
import numpy as np
from .common import read, sha, verify, write


def summarize(run):
    m = verify(run); mh = sha(run/'manifest.json')
    collection = read(run/'collection.json'); assert collection['state'] == 'complete' and collection['manifest_sha256'] == mh
    n = 0; means = {}; m2 = {}; maxima = {}; chunk_hashes = {}; oracle_error = 0.
    for i in range(128):
        p = run/'tokens'/f'chunk_{i:05d}.npz'; record = read(p.with_suffix('.json'))
        assert record['sequence_id'] == i and record['tokens'] == 2048 and record['manifest_sha256'] == mh
        assert record['tokens_sha256'] == m['tokens_sha256'] and record['sha256'] == sha(p)
        v = run/'validation'/p.with_suffix('.json').name; assert record['validation_sha256'] == sha(v)
        validation = read(v)
        assert validation['passed'] and validation['unchanged_logits'] and validation['embedding_identity'] and validation['actual_V_input_identity']
        if m['config']['model_type'] == 'gpt_base': assert validation['baseline_reference_matched']
        for check in validation['checks']:
            for key, value in check.items():
                if key.endswith('rms'): maxima[key] = max(maxima.get(key, 0), value)
        chunk_hashes[p.name] = record['sha256']
        with np.load(p) as z:
            arrays = {'source_norm':z['source_norm'].astype(np.float64)}
            for l in m['config']['layers']:
                key = f'energy_V_{l:02d}'; e = z[key].astype(np.float64)
                assert e.shape == (2048, 12, 2*l+1) and np.isfinite(e).all() and (e>=0).all()
                oracle = z[f'norm_V_{l:02d}'].astype(np.float64)**2
                np.testing.assert_allclose(e, oracle, rtol=1e-6, atol=1e-8)
                oracle_error = max(oracle_error, float(np.max(np.abs(e-oracle)/np.maximum(e,1e-12))))
                total = e.sum(-1, keepdims=True); assert (total>0).all()
                arrays[key] = e; arrays[f'energy_share_V_{l:02d}'] = e/total
                arrays[f'denominator_{l:02d}'] = z[f'denominator_{l:02d}'].astype(np.float64)
            for key, a in arrays.items():
                assert a.shape[0] == 2048 and np.isfinite(a).all()
                mean = a.mean(0); part_m2 = ((a-mean)**2).sum(0)
                if n == 0: means[key] = mean; m2[key] = part_m2
                else:
                    delta = mean-means[key]
                    m2[key] += part_m2+delta**2*n*2048/(n+2048)
                    means[key] += delta*2048/(n+2048)
        n += 2048
    assert n == collection['tokens'] == 262144
    arrays = {}
    for key in means:
        arrays[key+'_mean'] = means[key]; arrays[key+'_std'] = np.sqrt(m2[key]/n)
        arrays[key+'_variance'] = m2[key]/n
    np.savez(run/'derived/statistics.npz', **arrays)
    write(run/'validation/summary.json', dict(passed=True, tokens=n, sequences=128, unchanged_logits_tokens=n,
        maxima=maxima, energy_norm_identity_max_relative_error=oracle_error, baseline_reference_matched=m['config']['model_type']=='gpt_base'))
    write(run/'results.json', dict(state='complete', tokens=n, manifest_sha256=mh, tokens_sha256=m['tokens_sha256'],
        statistics_sha256=sha(run/'derived/statistics.npz'), validation_sha256=sha(run/'validation/summary.json'), chunk_hashes=chunk_hashes))
    write(run/'status.json', dict(state='complete', tokens=n))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', type=Path, required=True)
    root = p.parse_args().run_dir.resolve(); parent = read(root/'manifest.json')
    for name in parent['models']:
        assert sha(root/name/'manifest.json') == parent['child_manifest_hashes'][name]
        summarize(root/name)
    comparison = []
    with np.load(root/'baseline/derived/statistics.npz') as b, np.load(root/'no_embed_norm/derived/statistics.npz') as a:
        for l in parent['layers']:
            key = f'energy_V_{l:02d}_mean'; sk = f'energy_share_V_{l:02d}_mean'
            comparison.append(dict(layer=l, baseline_embedding_energy=b[key][:,0].tolist(), variant_embedding_energy=a[key][:,0].tolist(),
                embedding_energy_ratio=(a[key][:,0]/b[key][:,0]).tolist(), baseline_embedding_energy_share=b[sk][:,0].tolist(),
                variant_embedding_energy_share=a[sk][:,0].tolist()))
    write(root/'comparison.json', dict(layers=comparison, model_result_hashes={n:sha(root/n/'results.json') for n in parent['models']},
        share_definition='mean of per-token embedding energy / sum of all embedding+attention+MLP source energies; not output-energy fraction'))
    print(root/'comparison.json')


if __name__ == '__main__': main()
