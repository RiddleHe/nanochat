"""Render all-head input alignment: Emb + attention above MLP, Q/K/V columns."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from .common import read_json, require, sha256, write_json
from .plot import load_data


def plot(run, output, layer=12, threshold=0.9, transpose=False, layers=None, kind=None, family='all', color_max_floor=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    selected_layers = layers if layers is not None else [layer]
    require(layers is None or kind in ('Q', 'K', 'V'), 'Select a query kind for the layer grid')
    data, results = load_data(run, selected_layers, family=family, threshold=threshold)
    if kind is not None:
        data['panels'] = [p for p in data['panels'] if p['kind'] == kind]
    data.update(layer=layer, view=f'heatmap_family_rows_L{layer}_sq{threshold:g}', theme='light', transpose=transpose)
    if layers is not None:
        data.update(layers=layers, kind=kind, layout='layer_grid',
                    view=f'heatmap_layers_{"_".join(map(str, layers))}_{kind}_sq{threshold:g}')
    data['view'] += '_' + family
    if transpose: data['view'] += '_transposed'
    maximum = max((v for p in data['panels'] for v in p['mean'] if v is not None), default=0)
    data['color_domain'] = [0, max(0.01, math.ceil(maximum*100)/100)]
    if color_max_floor is not None:
        require(math.isfinite(color_max_floor) and 0 < color_max_floor <= 1, 'Invalid color maximum floor')
        data['color_domain'][1] = max(color_max_floor, data['color_domain'][1])
    here = Path(__file__).parent
    template = here / 'heatmap_view.html'
    fragment = template.read_text().replace('__ALIGNMENT_DATA__', json.dumps(data, separators=(',', ':'), allow_nan=False))
    require(len(fragment.encode()) < 1_000_000, 'Inline size limit exceeded')
    sources = [Path(__file__), here/'plot.py', here/'common.py', template]
    inputs = dict(statistics_sha256=results['statistics_sha256'], svd_index_sha256=results['svd_index_sha256'],
                  layers=selected_layers, kind=kind, family=family, squared_energy_threshold=threshold, theme='light', transpose=transpose, color_max_floor=color_max_floor,
                  code_sha256={p.name: sha256(p) for p in sources})
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    archive = run/'visualization'/f'{data["view"]}_light_{fingerprint}'
    if archive.exists():
        require(read_json(archive/'settings.json')['inputs'] == inputs, 'Archive provenance mismatch')
    if output.exists():
        require(output.read_text() == fragment, 'Preserve existing figure; choose a fresh output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)
    archive.mkdir(parents=True, exist_ok=True)
    for p in [*sources, output]:
        shutil.copy2(p, archive/p.name)
    write_json(archive/'data.json', data)
    write_json(archive/'settings.json', dict(inputs=inputs, inline_path=str(output), inline_sha256=sha256(output),
               color_domain=data['color_domain'], scale_scope='shared_all_displayed_cells', head_aggregation=False,
               layout=data.get('layout', 'family_rows'), layers=selected_layers,
               source_families={'all':['embedding_and_attention','mlp'],'attention':['embedding_and_attention'],'mlp':['mlp']}[family], query_kinds=[kind] if kind else ['Q', 'K', 'V']))
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, default=12)
    p.add_argument('--layers', type=int, nargs='+', help='Receiving layers in row-major order, three columns')
    p.add_argument('--kind', choices=['Q', 'K', 'V'], help='Query kind for a multiple-layer grid')
    p.add_argument('--family', choices=['all', 'attention', 'mlp'], default='all')
    p.add_argument('--threshold', type=float, default=0.9)
    p.add_argument('--transpose', action='store_true', help='Source layers on X, head indices on Y')
    p.add_argument('--color-max-floor', type=float, help='Minimum color maximum as a fraction; expands to fit data')
    a = p.parse_args()
    print(plot(a.run_dir, a.output, a.layer, a.threshold, a.transpose, a.layers, a.kind, a.family, a.color_max_floor))


if __name__ == '__main__':
    main()
