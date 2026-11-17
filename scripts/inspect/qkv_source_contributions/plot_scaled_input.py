"""Compare full projected norms with a best-fit scalar times input norms."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .prepare import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=('attention', 'mlp'), default='attention')
    args = parser.parse_args()
    run = args.run_dir
    manifest = json.loads((run/'manifest.json').read_text())
    results = json.loads((run/'results.json').read_text())
    stats_path = run/'summaries/statistics.npz'
    assert sha256(stats_path) == results['statistics_sha256']
    assert json.loads((run/'validation/summary.json').read_text())['passed']
    panels = []
    families = ('embedding', 'attention') if args.family == 'attention' else ('mlp',)
    source_start = -1 if args.family == 'attention' else 0
    family_label = 'Emb + attention' if args.family == 'attention' else 'MLP'
    with np.load(stats_path) as z:
        for layer in [6,12,18,23]:
            sources = [s for s in manifest['source_index'][:2*layer+1] if s['source_family'] in families]
            idx = [s['index'] for s in sources]
            assert [s['source_layer'] for s in sources] == list(range(source_start,layer))
            x = z[f'input_{layer:02d}_token_mean'][idx]
            xs = z[f'input_{layer:02d}_token_std'][idx]
            for kind in 'QKV':
                y = z[f'full_{kind}_{layer:02d}_token_mean'][idx]
                ys = z[f'full_{kind}_{layer:02d}_token_std'][idx]
                factor = float(x@y/(x@x))
                scaled = factor*x
                residual = float(np.linalg.norm(y-scaled)/np.linalg.norm(y))
                assert np.isfinite(scaled).all() and factor>0
                assert abs(float(x@(y-scaled))) < 1e-9
                panels.append(dict(layer=layer,kind=kind,factor=factor,residual=residual,
                    title=f'L{layer} · {kind}', source_index=sources, original_input=x.tolist(),
                    series=[dict(name='Input',values=scaled.tolist(),std=(factor*xs).tolist()),
                            dict(name='Projected',values=y.tolist(),std=ys.tolist())]))
    data = dict(run_id=manifest['run_id'],view='scaled_input' if args.family == 'attention' else 'scaled_input_mlp',tokens=results['tokens'],panels=panels,
        title=f'GPTBase · {family_label} · scaled input vs projection',x_start=source_start,
        family=args.family, family_label=family_label,
        comparison_max=max(v for p in panels for s in p['series'] for v in s['values']))
    template = Path(__file__).with_name('scaled_input_view.html')
    html = template.read_text().replace('__SOURCE_NORMS_DATA__',json.dumps(data,separators=(',',':')))
    if args.family == 'mlp':
        html = html.replace('gpt-base-scaled-input', 'gpt-base-mlp-scaled-input')
    assert len(html.encode())<1_000_000
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html)
    archive_name = 'attention_embedding' if args.family == 'attention' else 'mlp'
    archive = run/f'visualization/{archive_name}_scaled_input_comparison'
    archive.mkdir(parents=True,exist_ok=True)
    for src in (Path(__file__),template,args.output):
        shutil.copy2(src,archive/src.name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(statistics_sha256=sha256(stats_path),
        fit='one_positive_least_squares_multiplier_per_receiving_layer_and_query_no_intercept',
        source_family=args.family,
        fitting_data=f'token_mean_norms_over_{family_label}_sources_equal_source_weight',
        scaled_std='original_input_population_token_std_times_fitted_multiplier',
        raw_projected_values_unchanged=True,scale_scope='one_shared_linear_y_axis_across_all_12_panels',
        y_domain=[0,data['comparison_max']*1.05],theme='dark',inline_path=str(args.output),
        files_sha256={src.name:sha256(src) for src in (Path(__file__),template,args.output)}))
    print(json.dumps(dict(output=str(args.output),archive=str(archive),maximum=data['comparison_max'])))


if __name__ == '__main__':
    main()
