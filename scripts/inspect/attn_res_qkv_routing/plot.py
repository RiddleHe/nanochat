"""Render saved source-family routing summaries; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['attention', 'mlp'], default='attention')
    parser.add_argument('--theme', choices=['auto', 'light'], default='auto')
    args = parser.parse_args()
    summary = args.run_dir / 'summaries/statistics.npz'
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    results = json.loads((args.run_dir / 'results.json').read_text())
    assert digest == results['summary_sha256'][summary.name]
    schema_path = args.run_dir / 'summaries/schema.json'
    assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == results['summary_sha256'][schema_path.name]
    schema = json.loads(schema_path.read_text())
    reference = json.loads((args.run_dir / 'summaries/attention_heatmaps.json').read_text())
    cells, rows = [], []
    with np.load(summary) as stats:
        for layer in range(24):
            for query_index, kind in enumerate(schema['query_types']):
                rows.append(dict(layer=layer, kind=kind, **{
                    f'{args.family}_mass_mean': float(stats[f'{args.family}_mass_{layer:02d}_token_mean'][query_index]),
                    f'{args.family}_mass_std': float(stats[f'{args.family}_mass_{layer:02d}_token_std'][query_index])}))
                for source in schema['source_index'][:2*layer+1]:
                    if source['source_family'] != args.family:
                        continue
                    cells.append(dict(layer=layer, kind=kind, source_layer=source['source_layer'],
                        mean=float(stats[f'weights_{layer:02d}_token_mean'][query_index, source['index']]),
                        std=float(stats[f'weights_{layer:02d}_token_std'][query_index, source['index']]),
                        count=results['tokens']))
    data = dict(run_id=reference['run_id'], tokens=results['tokens'], cells=cells, rows=rows,
                source_family=args.family, renormalized=False, std_ddof=0, measurement=reference['measurement'])
    if args.family == 'attention':
        assert data == reference
    assert len(data['cells']) == 828 and len(data['rows']) == 72
    expected = {(kind, layer, source) for kind in 'QKV' for layer in range(24) for source in range(layer)}
    assert {(c['kind'], c['layer'], c['source_layer']) for c in data['cells']} == expected
    for c in data['cells']:
        assert 0 <= c['mean'] <= 1 and c['std'] >= 0 and c['count'] == data['tokens']
    for row in data['rows']:
        total = sum(c['mean'] for c in data['cells'] if (c['kind'], c['layer']) == (row['kind'], row['layer']))
        assert abs(total - row[f'{args.family}_mass_mean']) < 1e-10
    template = Path(__file__).with_name('view_light.html' if args.theme == 'light' else 'view.html')
    html = template.read_text().replace('__ROUTING_DATA__', json.dumps(data, separators=(',', ':')))
    html = html.replace('__FAMILY__', args.family).replace('__FAMILY_TITLE__', 'MLP' if args.family == 'mlp' else 'Attention')
    html = html.replace('__OMITTED__', 'attention' if args.family == 'mlp' else 'MLP')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    archive = args.run_dir / f'visualization/{args.family}_weights{ "_light" if args.theme == "light" else ""}'
    archive.mkdir(parents=True, exist_ok=True)
    for src in (Path(__file__), template, args.output):
        shutil.copy2(src, archive / src.name)
    (archive / f'{args.family}_heatmaps.json').write_text(json.dumps(data, indent=2) + '\n')
    settings = dict(source_family=args.family, queries=['Q', 'K', 'V'], layout='1x3', theme=args.theme,
                    color_domain=[0, max(c['mean'] for c in data['cells'])], color_scale='linear',
                    color_ceiling='global_max_displayed_token_mean', renormalized=False,
                    x=f'source_{args.family}_layer', y='receiving_layer_increasing_downward',
                    std_ddof=0, summary_sha256=digest, inline_path=str(args.output.resolve()),
                    files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), template, args.output)})
    (archive / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'archive': str(archive), 'cells': len(data['cells'])}))


if __name__ == '__main__':
    main()
