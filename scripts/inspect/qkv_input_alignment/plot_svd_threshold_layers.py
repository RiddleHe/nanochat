"""Plot derived singular-energy cutoff counts across layers for Q/K/V heads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .common import read_json, require, sha256, write_json


def plot(derived, output, threshold=.5, kind='QKV', show_rank=False, show_uniform=False):
    derived, output = Path(derived).resolve(), Path(output).resolve()
    manifest = read_json(derived/'manifest.json')
    require(sha256(derived/'statistics.json') == manifest['statistics_sha256'], 'Summary hash mismatch')
    require(read_json(derived/'validation.json')['passed'], 'Summary not validated')
    summary = read_json(derived/'statistics.json')
    require(any(abs(t-threshold)<1e-12 for t in summary['thresholds']), 'Threshold missing')
    panels = []
    require(kind in ('QKV','Q','K','V'), 'Invalid projection selection')
    kinds = kind
    for kind in kinds:
        entries = [e for e in summary['entries'] if e['kind']==kind]
        series = []
        for head in sorted({e['head'] for e in entries}):
            values = []
            for e in sorted((e for e in entries if e['head']==head), key=lambda e:e['layer']):
                require(e['valid'], 'Undefined matrix')
                matches = [p for p in e['points'] if abs(p['threshold']-threshold)<1e-12]
                require(len(matches)==1, 'Missing/duplicate threshold')
                p = matches[0]
                values.append(dict(layer=e['layer'], k=p['count'], captured=p['captured'], rank=e['rank'], uniform_count=threshold*e['rank']))
            require([v['layer'] for v in values]==list(range(len(values))), 'Layer sequence incomplete')
            series.append(dict(head=head, values=values))
        panels.append(dict(kind=kind, series=series))
    require(all([s['head'] for s in p['series']]==[s['head'] for s in panels[0]['series']] for p in panels), 'Head counts differ')
    data = dict(run_id=summary['run_id'], view=f'{kinds.lower()}_k_sq{threshold:g}_layers'+('_rank' if show_rank else '')+('_uniform' if show_uniform else ''), threshold=threshold, panels=panels, show_rank=show_rank, show_uniform=show_uniform)
    here = Path(__file__).parent
    sources = [Path(__file__), here/'svd_threshold_layers_view.html', here/'common.py']
    fragment = sources[1].read_text().replace('__SVD_LAYER_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    inputs = dict(derived_dir=str(derived), manifest_sha256=sha256(derived/'manifest.json'), threshold=threshold, kind=kinds, show_rank=show_rank, show_uniform=show_uniform, uniform_definition='threshold * numerical_rank (unrounded)', theme='dark', code_sha256={p.name:sha256(p) for p in sources})
    fingerprint = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    archive = Path(manifest['source_run'])/'visualization'/f'{data["view"]}_dark_{fingerprint}'
    require(not output.exists() or output.read_text()==fragment, 'Preserve previous plot')
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(fragment)
    archive.mkdir(parents=True,exist_ok=True)
    for p in [*sources,output]: shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),head_aggregation=False,y_scale='shared_zero_to_nice_padded_maximum',quantity='squared_singular_energy_cutoff_count',rank_tolerance='max(matrix_shape) * float64_epsilon * largest_singular_value' if show_rank else None))
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--derived-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--kind',choices=['QKV','Q','K','V'],default='QKV')
    p.add_argument('--show-rank',action='store_true')
    p.add_argument('--show-uniform',action='store_true')
    a=p.parse_args()
    print(plot(a.derived_dir,a.output,a.threshold,a.kind,a.show_rank,a.show_uniform))
