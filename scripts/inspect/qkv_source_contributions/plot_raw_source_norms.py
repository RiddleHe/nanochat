"""Plot saved raw full-vector norms for GPTBase embedding and attention outputs."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from .prepare import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir
    manifest = json.loads((run/'manifest.json').read_text())
    results = json.loads((run/'results.json').read_text())
    assert manifest['config']['experiment'] == 'd24_qkv_separate_sources'
    assert json.loads((run/'validation/summary.json').read_text())['passed']
    stats_path = run/'summaries/statistics.npz'
    assert sha256(stats_path) == results['statistics_sha256']
    sources = json.loads((run/'inputs/source_index.json').read_text())
    assert sources == manifest['source_index']
    metadata = json.loads((run/'inputs/checkpoint_metadata.json').read_text())['model_config']
    depth = metadata['n_layer']
    selected = [s for s in sources if s['source_family'] in ('embedding','attention')]
    assert [s['source_layer'] for s in selected] == list(range(-1,depth))
    points = []
    with np.load(stats_path) as stats:
        mean, std = stats['source_norm_token_mean'], stats['source_norm_token_std']
        assert mean.shape == std.shape == (2*depth+1,)
        for source in selected:
            i = source['index']
            assert np.isfinite(mean[i]) and np.isfinite(std[i]) and mean[i]>=0 and std[i]>=0
            points.append(dict(**source, label='Emb' if source['source_family']=='embedding' else f'A{source["source_layer"]}',
                               mean=float(mean[i]), std=float(std[i])))
    data = dict(run_id=manifest['run_id'],tokens=results['tokens'],dimensions=metadata['n_embd'],points=points,
                y_max=max(p['mean'] for p in points)*1.05)
    template = Path(__file__).with_name('raw_source_norms_view.html')
    fragment = template.read_text().replace('__RAW_SOURCE_NORMS_DATA__',json.dumps(data,separators=(',',':')))
    assert len(fragment.encode())<1_000_000
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(fragment)
    archive = run/'visualization/raw_source_norms_attention_embedding'
    archive.mkdir(parents=True,exist_ok=True)
    for src in (Path(__file__),template,args.output):
        shutil.copy2(src,archive/src.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(statistics_sha256=sha256(stats_path),inline_path=str(args.output),
        measurement='token_mean_of_full_source_vector_L2_norm_before_receiving_layer_normalization',
        embedding='normalized_token_embedding_at_residual_stream_input',
        attention='module_output_after_output_projection_before_residual_addition',
        source_families=['embedding','attention'],layers=list(range(depth)),dimensions=metadata['n_embd'],
        std_ddof=0,tokens=results['tokens'],theme='dark',scale='linear',y_domain=[0,data['y_max']],
        applied_scaling=False,receiving_layer_normalization=False,qkv_projection=False,
        files_sha256={src.name:sha256(src) for src in (Path(__file__),template,args.output)}))
    print(json.dumps(dict(output=str(args.output),archive=str(archive),points=len(points),maximum=max(p['mean'] for p in points))))


if __name__ == '__main__':
    main()
