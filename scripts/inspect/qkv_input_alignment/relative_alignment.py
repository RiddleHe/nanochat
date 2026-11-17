"""Derive source-average-normalized squared projection ratios on CPU.

Each token/head uses all finite available source alignments in its denominator.
The implementation is copied into a fresh output directory before execution.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .common import (matrix_specs, read_json, require, sha256, verify_chunk,
                     verify_run, verify_svd, write_json, write_npz)


def normalize_alignment(alignment, metric='squared'):
    """Return q, denominator, and valid-source count; source is the last axis.

    NaN denotes an undefined source alignment, not zero projection. A finite
    zero participates in the source average. An all-zero average is undefined.
    """
    a = np.asarray(alignment, dtype=np.float64)
    require(a.ndim >= 1 and a.shape[-1] > 0, 'Missing source axis')
    valid = np.isfinite(a)
    require(np.all(valid | np.isnan(a)), 'Infinite alignment')
    require(np.all((a[valid] >= 0) & (a[valid] <= 1)), 'Alignment outside [0,1]')
    require(metric in ('squared', 'length'), 'Unknown alignment metric')
    r = a if metric == 'squared' else np.sqrt(a)
    count = valid.sum(axis=-1, dtype=np.int64)
    denominator = np.divide(np.where(valid, r, 0).sum(axis=-1), count,
                            out=np.full(count.shape, np.nan), where=count > 0)
    q = np.divide(r, denominator[..., None], out=np.full_like(r, np.nan),
                  where=valid & (denominator[..., None] > 0))
    return q, denominator, count


class Moments:
    """Merge population moments over the token axis in FP64, excluding NaN."""
    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def add(self, values):
        valid = np.isfinite(values)
        n = valid.sum(axis=0, dtype=np.int64)
        mean = np.divide(np.where(valid, values, 0).sum(axis=0), n,
                         out=np.zeros_like(self.mean), where=n > 0)
        centered = np.where(valid, values - mean, 0)
        m2 = np.square(centered).sum(axis=0)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * np.divide(n, total, out=np.zeros_like(mean), where=total > 0)
        self.m2 += m2 + delta**2 * np.divide(self.count * n, total,
                                            out=np.zeros_like(mean), where=total > 0)
        self.count = total

    def arrays(self):
        mean = np.where(self.count > 0, self.mean, np.nan)
        variance = np.divide(self.m2, self.count, out=np.full_like(mean, np.nan),
                             where=self.count > 0)
        return dict(token_mean=mean, token_std=np.sqrt(np.maximum(variance, 0)),
                    valid_count=self.count)


def derive(run, output, threshold, metric):
    manifest, meta = verify_run(run)
    verify_svd(run, manifest)
    require(manifest['config'].get('measurement', 'sources') == 'sources', 'Needs source measurements')
    results = read_json(run/'results.json')
    require(results['state'] == 'analysis_complete', 'Source collection incomplete')
    mh, sh = sha256(run/'manifest.json'), sha256(run/'svd/index.json')
    require(results['manifest_sha256'] == mh and results['svd_index_sha256'] == sh, 'Provenance mismatch')
    require(results['statistics_sha256'] == sha256(run/'summaries/statistics.npz'), 'Source summary changed')
    require(results['validation_sha256'] == sha256(run/'validation/summary.json') and
            read_json(run/'validation/summary.json')['passed'], 'Source validation failed')
    thresholds = manifest['config']['squared_energy_thresholds']
    require(threshold in thresholds, 'Threshold not collected')
    ti = thresholds.index(threshold)
    specs = [s for s in matrix_specs(meta, manifest['config']) if s[1] == 'V']
    require(bool(specs), 'No V data')
    moments = {l: Moments((h, 2*l+1)) for l, _, h, _, _ in specs}
    denominator_moments = {l: Moments((h,)) for l, _, h, _, _ in specs}
    source_count_min = {l: np.full(h, 2*l+1, dtype=np.int64) for l, _, h, _, _ in specs}
    source_count_max = {l: np.zeros(h, dtype=np.int64) for l, _, h, _, _ in specs}
    undefined_denominators = {l: np.zeros(h, dtype=np.int64) for l, _, h, _, _ in specs}
    chunk_hashes = {}
    identity_error = 0.
    nseq, length = manifest['tokens_shape']
    for i in range(nseq):
        path = run/'tokens'/f'chunk_{i:05d}.npz'
        record = verify_chunk(run, path, mh, sh)
        require(record['sequence_id'] == i and record['tokens'] == length and
                record['tokens_sha256'] == manifest['tokens_sha256'], 'Token identity mismatch')
        require(record['sha256'] == results['chunk_sha256'][path.name], 'Chunk not in validated results')
        chunk_hashes[path.name] = record['sha256']
        with np.load(path) as z:
            for layer, _, heads, _, _ in specs:
                a = z[f'alignment_V_{layer:02d}'][..., ti]
                require(a.shape == (length, heads, 2*layer+1), 'Wrong source bank shape')
                q, denominator, n = normalize_alignment(a, metric=metric)
                usable = np.isfinite(denominator) & (denominator > 0)
                normalized_mean = np.divide(np.nansum(q, axis=-1), n,
                                            out=np.full(n.shape, np.nan), where=n > 0)
                if np.any(usable):
                    identity_error = max(identity_error, float(np.max(np.abs(normalized_mean[usable]-1))))
                require(identity_error < 5e-14, 'Source-average identity failed')
                require(np.all(np.where(np.isfinite(q), (q >= 0) & (q <= n[..., None]+1e-12), True)), 'Invalid relative alignment')
                if layer == 0:
                    require(np.all(q[np.isfinite(q)] == 1), 'L0 must equal one')
                moments[layer].add(q)
                denominator_moments[layer].add(denominator)
                source_count_min[layer] = np.minimum(source_count_min[layer], n.min(axis=0))
                source_count_max[layer] = np.maximum(source_count_max[layer], n.max(axis=0))
                undefined_denominators[layer] += (~usable).sum(axis=0)
        if (i+1) % 32 == 0:
            print(f'Relative alignment: {i+1}/{nseq} sequences', flush=True)
    arrays = {}
    panels = [dict(label=label, points=[]) for label in ('Emb', 'A0', 'Previous attention output')]
    for layer, _, heads, _, _ in specs:
        statistics = moments[layer].arrays()
        key = f'relative_alignment_V_{layer:02d}'
        arrays.update({key+'_'+name: value for name, value in statistics.items()})
        arrays.update({f'denominator_V_{layer:02d}_'+name: value
                       for name, value in denominator_moments[layer].arrays().items()})
        arrays[key+'_valid_sources_min'] = source_count_min[layer]
        arrays[key+'_valid_sources_max'] = source_count_max[layer]
        arrays[key+'_undefined_denominator_count'] = undefined_denominators[layer]
        with np.load(run/'svd'/f'V_{layer:02d}.npz') as bank:
            ks = bank['k'][:, ti]
            captured = bank['captured_squared_fraction'][:, ti]
        for si, index in enumerate([0] if layer == 0 else [0, 1, 2*layer-1]):
            for head in range(heads):
                count = int(statistics['valid_count'][head, index])
                panels[si]['points'].append(dict(layer=layer, head=head,
                    source=('Emb', 'A0', f'A{layer-1}')[si],
                    mean=float(statistics['token_mean'][head, index]) if count else None,
                    std=float(statistics['token_std'][head, index]) if count else None,
                    count=count, k=int(ks[head]), captured=float(captured[head]),
                    valid_sources_min=int(source_count_min[layer][head]),
                    valid_sources_max=int(source_count_max[layer][head])))
    if 0 in moments:
        require(np.all(arrays['relative_alignment_V_00_token_mean'] == 1) and
                np.all(arrays['relative_alignment_V_00_token_std'] == 0), 'Invalid L0 moments')
    write_npz(output/'statistics.npz', arrays)
    write_json(output/'data.json', dict(layers=[s[0] for s in specs], tokens=nseq*length,
        threshold=threshold, metric=f'projection_{metric}_ratio_divided_by_token_head_source_mean', panels=panels))
    write_json(output/'schema.json', dict(
        metric=metric,
        formula=('q_s = alignment_s / mean_valid_sources(alignment)' if metric == 'squared'
                 else 'r_s = sqrt(alignment_s); q_s = r_s / mean_valid_sources(r)'),
        denominator_sources='Emb plus all prior attention and MLP outputs, including the measured source',
        normalization_order='per token and head, then aggregate over token occurrences',
        zero_source='undefined alignment excluded from denominator and valid-source count',
        zero_projection='finite zero alignment included in denominator and valid-source count',
        zero_denominator='all q values undefined; excluded from moments',
        reference=1, ddof=0, analysis_dtype='float64', summary_axes=['head', 'available_source'],
        source_index=read_json(run/'inputs/source_index.json'), threshold=threshold,
        note='Relative geometric alignment, not projected V-output energy or causal contribution'))
    validation = dict(passed=True, input_hashes_verified=True, all_sources_in_denominator=True,
        source_average_identity_max_abs_error=identity_error, layer0_mean_one_sd_zero=True,
        undefined_denominator_total=sum(int(x.sum()) for x in undefined_denominators.values()),
        chunks=nseq, tokens=nseq*length)
    write_json(output/'validation.json', validation)
    write_json(output/'results.json', dict(state='complete', run=str(run),
        source_manifest_sha256=mh, source_results_sha256=sha256(run/'results.json'),
        source_svd_index_sha256=sh, source_chunk_sha256=chunk_hashes,
        analysis_manifest_sha256=sha256(output/'manifest.json'),
        output_sha256={name: sha256(output/name) for name in
                       ('statistics.npz', 'data.json', 'schema.json', 'validation.json')}))
    print(f'Derived and validated {len(specs)} layers; source-average error {identity_error:.3g}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--metric', choices=('squared', 'length'), default='squared',
                        help='Squared alignment (default); length reproduces the older square-root metric')
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    default_name = ('source_mean_relative_squared_alignment' if args.metric == 'squared'
                    else 'source_mean_relative_alignment')
    output = (args.output_dir or run/'derived'/default_name).resolve()
    if not args.frozen:
        output.mkdir(parents=True, exist_ok=False)
        package = output/'code/scripts/inspect/qkv_input_alignment'
        package.mkdir(parents=True)
        for name in ('relative_alignment.py', 'common.py'):
            shutil.copy2(Path(__file__).parent/name, package/name)
        write_json(output/'manifest.json', dict(run=str(run), threshold=args.threshold, metric=args.metric,
            source_manifest_sha256=sha256(run/'manifest.json'),
            source_results_sha256=sha256(run/'results.json'),
            python_executable=sys.executable, python=sys.version, numpy=np.__version__,
            code_sha256={str(p.relative_to(output)): sha256(p) for p in package.iterdir()}))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        env.pop('PYTHONPATH', None)
        command = [sys.executable, '-m', 'scripts.inspect.qkv_input_alignment.relative_alignment',
                   '--run-dir', str(run), '--output-dir', str(output),
                   '--threshold', str(args.threshold), '--metric', args.metric, '--frozen']
        with (output/'analysis.log').open('w') as log:
            process = subprocess.Popen(command, cwd=output/'code', env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            require(process.wait() == 0, 'Frozen derivation failed; see analysis.log')
        return
    frozen = read_json(output/'manifest.json')
    require(Path(__file__).resolve().is_relative_to(output/'code'), 'Analysis escaped frozen code')
    require(frozen['run'] == str(run) and frozen['threshold'] == args.threshold, 'Frozen settings changed')
    require(frozen['metric'] == args.metric, 'Frozen metric changed')
    require(frozen['python'] == sys.version and frozen['numpy'] == np.__version__, 'Environment changed')
    for path, digest in frozen['code_sha256'].items():
        require(sha256(output/path) == digest, 'Frozen derivation code changed')
    require(sha256(run/'manifest.json') == frozen['source_manifest_sha256'] and
            sha256(run/'results.json') == frozen['source_results_sha256'], 'Source run changed')
    derive(run, output, args.threshold, args.metric)


if __name__ == '__main__':
    main()
