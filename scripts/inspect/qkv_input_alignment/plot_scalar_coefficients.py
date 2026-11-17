"""Render validated static ScalarRes Q/K/V coefficients, without forward replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', choices=['shared-linear', 'per-query-log', 'per-query-linear'], default='shared-linear')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    schema_path = run / 'summaries/scalar_coefficients.json'
    stats_path = run / 'summaries/scalar_coefficients.npz'
    schema = json.loads(schema_path.read_text())
    validation = json.loads((run / 'validation/scalar_coefficients.json').read_text())
    assert validation['passed']
    assert sha(run / 'manifest.json') == schema['manifest_sha256'] == validation['manifest_sha256']
    assert sha(stats_path) == schema['statistics_sha256'] == validation['statistics_sha256']
    assert not schema['token_dependent'] and not schema['head_dependent'] and not schema['normalized']
    receivers = [r for r in schema['receivers'] if r['kind'] in ('Q', 'K', 'V')]
    depth = max(r['layer'] for r in receivers) + 1
    sources = schema['source_index'][:2 * depth - 1]
    cells = []
    with np.load(stats_path) as stats:
        for r in receivers:
            assert r['stop'] - r['start'] == r['available_sources'] == 2 * r['layer'] + 1
            assert tuple(stats['offsets'][r['receiver_index']:r['receiver_index'] + 2]) == (r['start'], r['stop'])
            for s in sources[:r['available_sources']]:
                i = r['start'] + s['index']
                cells.append(dict(kind=r['kind'], layer=r['layer'], source=s['index'],
                    value=float(stats['coefficients_fp32'][i]),
                    applied=float(stats['coefficients_bf16_as_fp32'][i]),
                    z=float(stats['logits'][i])))
    assert len(cells) == 3 * depth ** 2
    assert all(np.isfinite(c['value']) and c['value'] > 0 for c in cells)
    assert len({(c['kind'], c['layer'], c['source']) for c in cells}) == len(cells)
    data = dict(run=str(run), depth=depth, sources=sources, cells=cells,
        color_domain=[0, max(c['value'] for c in cells)], coefficient='FP32 gate',
        source_order='Emb, Attn 0, MLP 0, Attn 1, MLP 1, ...', normalized=False)
    data['scale'] = args.scale
    data['color_domains'] = {kind: [min(c['value'] for c in cells if c['kind'] == kind),
        max(c['value'] for c in cells if c['kind'] == kind)] for kind in ('Q', 'K', 'V')}
    if args.scale == 'per-query-linear':
        for domain in data['color_domains'].values():
            domain[0] = 0
    template = Path(__file__).with_name('view_scalar_coefficients.html')
    html = template.read_text().replace('__SCALAR_DATA__', json.dumps(data, separators=(',', ':')))
    assert '__SCALAR_DATA__' not in html and len(html.encode()) < 1_000_000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    archive = run / 'visualization/qkv_scalar_coefficients_all_sources'
    if args.scale != 'shared-linear':
        archive = archive.with_name(archive.name + '_' + args.scale.replace('-', '_'))
    archive.mkdir(parents=True, exist_ok=True)
    for src in (Path(__file__), template, args.output):
        shutil.copy2(src, archive / src.name)
    (archive / 'data.json').write_text(json.dumps(data, indent=2) + '\n')
    settings = dict(theme='fixed_light', layout='stacked_Q_K_V', scale=args.scale,
        color_domains=data['color_domains'] if args.scale != 'shared-linear' else None,
        color_domain=data['color_domain'] if args.scale == 'shared-linear' else None,
        renormalized=False, source_order=data['source_order'],
        plotted='coefficients_fp32', tooltip=['coefficients_fp32', 'coefficients_bf16_as_fp32', 'logits'],
        unavailable='hatched', valid_cells=len(cells), inline_path=str(args.output.resolve()),
        input_sha256={p.name:sha(p) for p in (schema_path, stats_path, run / 'manifest.json')},
        files_sha256={p.name:sha(p) for p in (Path(__file__), template, args.output)})
    (archive / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), archive=str(archive), cells=len(cells),
        domains=settings['color_domains'] or settings['color_domain'])))


if __name__ == '__main__':
    main()
