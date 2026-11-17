"""Plot selected heads' direction counts against squared-energy thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .common import read_json, require, sha256, write_json


def plot(derived, output, layer=12, kind='V', heads=(0,1,2), layers=None):
    derived, output = Path(derived).resolve(), Path(output).resolve()
    manifest = read_json(derived/'manifest.json')
    require(sha256(derived/'statistics.json') == manifest['statistics_sha256'], 'Summary hash mismatch')
    require(read_json(derived/'validation.json')['passed'], 'Summary not validated')
    summary = read_json(derived/'statistics.json')
    entries = []
    layers = list(layers) if layers is not None else [layer]
    for receiving_layer in layers:
        for head in heads:
            matches = [e for e in summary['entries'] if (e['layer'],e['kind'],e['head']) == (receiving_layer,kind,head)]
            require(len(matches)==1 and matches[0]['valid'], 'Missing or undefined head')
            entries.append(matches[0])
    data = dict(run_id=summary['run_id'], layers=layers, kind=kind, entries=entries, thresholds=summary['thresholds'], max_count=max(e['directions'] for e in entries))
    here = Path(__file__).parent
    sources = [Path(__file__), here/'svd_thresholds_view.html', here/'common.py']
    fragment = sources[1].read_text().replace('__SVD_THRESHOLD_DATA__', json.dumps(data, separators=(',',':'), allow_nan=False))
    inputs = dict(derived_dir=str(derived), manifest_sha256=sha256(derived/'manifest.json'), layers=layers, kind=kind, heads=list(heads), theme='dark', code_sha256={p.name:sha256(p) for p in sources})
    fingerprint = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    layer_label = '_'.join(map(str,layers))
    archive = Path(manifest['source_run'])/'visualization'/f'svd_thresholds_L{layer_label}_{kind}_dark_{fingerprint}'
    require(not output.exists() or output.read_text()==fragment, 'Preserve previous plot; choose another output')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(fragment)
    archive.mkdir(parents=True,exist_ok=True)
    for p in [*sources, output]:
        shutil.copy2(p,archive/p.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(inputs=inputs,inline_path=str(output),inline_sha256=sha256(output),y_scale=[0,data['max_count']],head_aggregation=False))
    return archive


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--derived-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,default=12)
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--kind',choices=['Q','K','V'],default='V')
    p.add_argument('--heads',type=int,nargs='+',default=[0,1,2])
    a=p.parse_args()
    print(plot(a.derived_dir,a.output,a.layer,a.kind,a.heads,a.layers))
