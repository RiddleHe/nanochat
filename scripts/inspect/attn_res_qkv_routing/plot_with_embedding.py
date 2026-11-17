"""Render embedding plus attention routing from verified saved statistics, CPU only."""
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
    args = parser.parse_args()
    run = args.run_dir
    results = json.loads((run / 'results.json').read_text())
    for name in ('statistics.npz', 'schema.json'):
        assert hashlib.sha256((run / 'summaries' / name).read_bytes()).hexdigest() == results['summary_sha256'][name]
    schema = json.loads((run / 'summaries/schema.json').read_text())
    depth = json.loads((run / 'inputs/checkpoint_metadata.json').read_text())['model_config']['n_layer']
    cells, rows = [], []
    with np.load(run / 'summaries/statistics.npz') as stats:
        for layer in range(depth):
            for qi, kind in enumerate(schema['query_types']):
                rows.append(dict(layer=layer, kind=kind, attention_mass_mean=float(stats[f'attention_mass_{layer:02d}_token_mean'][qi]), attention_mass_std=float(stats[f'attention_mass_{layer:02d}_token_std'][qi])))
                for source in schema['source_index'][:2*layer+1]:
                    if source['source_family'] not in ('embedding', 'attention'):
                        continue
                    cells.append(dict(layer=layer, kind=kind, source_layer=source['source_layer'],
                        mean=float(stats[f'weights_{layer:02d}_token_mean'][qi, source['index']]),
                        std=float(stats[f'weights_{layer:02d}_token_std'][qi, source['index']])))
    assert len(cells) == 3 * depth * (depth+1) // 2
    assert all(0 <= c['mean'] <= 1 and c['std'] >= 0 for c in cells)
    assert all(c['source_layer'] == -1 or c['source_layer'] < c['layer'] for c in cells)
    for row in rows:
        family = [c for c in cells if c['kind'] == row['kind'] and c['layer'] == row['layer']]
        assert abs(sum(c['mean'] for c in family if c['source_layer'] >= 0) - row['attention_mass_mean']) < 1e-10
        assert sum(c['source_layer'] == -1 for c in family) == 1
    manifest = json.loads((run / 'manifest.json').read_text())
    data = dict(run_id=manifest['run_id'], depth=depth, tokens=results['tokens'], cells=cells, rows=rows, renormalized=False)
    template = Path(__file__).with_name('view_with_embedding.html')
    html = template.read_text().replace('__ROUTING_DATA__', json.dumps(data, separators=(',', ':')))
    html = html.replace('__FAMILY__', 'attention-embedding').replace('__FAMILY_TITLE__', 'Embedding + attention')
    assert '__ROUTING_DATA__' not in html
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    archive = run / 'visualization/attention_embedding_weights'
    archive.mkdir(parents=True, exist_ok=True)
    for src in (Path(__file__), template, args.output):
        shutil.copy2(src, archive / src.name)
    (archive / 'data.json').write_text(json.dumps(data, indent=2)+'\n')
    settings = dict(source_families=['embedding', 'attention'], omitted='mlp', queries=['Q','K','V'],
        color_domain=[0,max(c['mean'] for c in cells)], color_scale='linear', color_ceiling='global_max_displayed_token_mean', theme='fixed_light',
        renormalized=False, embedding_source_layer=-1, layout='1x3', std_ddof=0,
        summary_sha256=results['summary_sha256']['statistics.npz'], inline_path=str(args.output),
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),template,args.output)})
    (archive / 'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), archive=str(archive), valid_cells=len(cells), scale=settings['color_domain'])))


if __name__ == '__main__':
    main()
