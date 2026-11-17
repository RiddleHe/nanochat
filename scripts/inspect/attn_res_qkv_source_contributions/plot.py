"""Plot one family's weighted input and projected norms from saved summaries."""
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
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 23])
    parser.add_argument('--family', choices=['attention', 'mlp'], default='attention')
    parser.add_argument('--include-embedding', action='store_true')
    parser.add_argument('--theme', choices=['auto', 'dark'], default='auto')
    parser.add_argument('--mirrored', action='store_true', help='Projected above zero, input below; identical magnitude scale')
    args = parser.parse_args()
    run = args.run_dir
    manifest = json.loads((run/'manifest.json').read_text())
    results = json.loads((run/'results.json').read_text())
    assert manifest['config']['experiment'] == 'd24_attn_res_qkv_source_contributions'
    assert results['validation']['passed'] and results['state'] == 'analysis_complete'
    stats_path = run/'summaries/statistics.npz'
    assert sha256(stats_path) == results['summary_sha256']['statistics.npz']
    assert len(args.layers) == 4 and len(set(args.layers)) == 4
    assert all(1 <= layer < 24 for layer in args.layers)
    panels = []
    comparison_max = 0.
    with np.load(stats_path) as stats:
        for layer in args.layers:
            available = manifest['source_index'][:2*layer+1]
            selected = [s for s in available if s['source_family'] == args.family or (args.include_embedding and s['source_family'] == 'embedding')]
            indices = [s['index'] for s in selected]
            shared_indices = [s['index'] for s in available if args.include_embedding or s['source_family'] in ('attention', 'mlp')]
            assert [s['source_layer'] for s in selected] == ([-1] if args.include_embedding else []) + list(range(layer))
            series = []
            for kind in 'QKV':
                for label, prefix in [('input', 'input'), ('projected', 'full')]:
                    key = f'{prefix}_{kind}_{layer:02d}'
                    mean, std = stats[key+'_token_mean'], stats[key+'_token_std']
                    assert np.isfinite(mean).all() and np.isfinite(std).all() and (mean >= 0).all() and (std >= 0).all()
                    comparison_max = max(comparison_max, float(mean[shared_indices].max()))
                    series.append(dict(name=f'{kind} {label}', values=mean[indices].tolist(), std=std[indices].tolist()))
            panels.append(dict(layer=layer, title=f'Receiving layer {layer}', series=series, source_index=selected))
    data = dict(run_id=manifest['run_id'], family=args.family, tokens=results['tokens'], panels=panels,
                title=f'AttnRes · {"embedding + " if args.include_embedding else ""}{"MLP" if args.family == "mlp" else "attention"}-source contributions',
                x_start=-1 if args.include_embedding else 0, comparison_max=comparison_max,
                embedding_displayed=args.include_embedding, theme=args.theme,
                source_caption='Emb = normalized token embedding; MLP sources omitted' if args.include_embedding and args.family == 'attention' else '')
    if args.mirrored:
        data['input_max'] = max(max(s['values']) for p in panels for s in p['series'] if s['name'].endswith('input'))
        data['projected_max'] = max(max(s['values']) for p in panels for s in p['series'] if s['name'].endswith('projected'))
        data['mirrored'] = True
    template = Path(__file__).with_name('view_mirrored.html' if args.mirrored else 'view.html')
    fragment = template.read_text().replace('__SOURCE_NORMS_DATA__', json.dumps(data, separators=(',', ':')))
    assert len(fragment.encode()) < 1_000_000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    archive = run/'visualization'/f'{args.family}_sources_layers_{"_".join(map(str,args.layers))}{"_with_embedding" if args.include_embedding else ""}{"_dark" if args.theme == "dark" else ""}'
    if args.mirrored:
        archive = archive.with_name(archive.name + '_mirrored_shared_scale')
    archive.mkdir(parents=True, exist_ok=True)
    for path, name in [(args.output,'inline.html'),(template,template.name),(Path(__file__),Path(__file__).name)]:
        shutil.copy2(path,archive/name)
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(statistics_sha256=sha256(stats_path), inline_path=str(args.output),
        inline_sha256=sha256(args.output), layers=args.layers, source_family=args.family,
        x='embedding_at_minus_one_then_zero_based_source_layer' if args.include_embedding else 'zero_based_source_layer', y='mean_per_token_absolute_L2_norm', layout='2x2_desktop_stacked_mobile', theme=args.theme,
        input='raw_source * actual_applied_query_weight / actual_complete_query_mixture_RMS_denominator',
        projected='full_vector_norm_combined_across_heads_per_token_then_mean_pre_RoPE_pre_QKnorm',
        input_dimensions=1536, projected_dimensions=1536, std_ddof=0, tokens=results['tokens'],
        embedding_displayed=args.include_embedding,
        y_domain=({'projected':[0,comparison_max*1.05], 'input':[0,comparison_max*1.05]} if args.mirrored else [0,comparison_max*1.05]),
        input_direction='increasing_downward_from_center_zero' if args.mirrored else 'increasing_upward',
        scale_scope='identical_input_projected_magnitude_scale_across_selected_layers_and_all_source_families' if args.mirrored else 'all_six_series_across_selected_receiving_layers_and_attention_mlp'+('_embedding_sources' if args.include_embedding else '_sources'),
        source_hashes={p.name:sha256(p) for p in (Path(__file__),template)}))
    print(json.dumps(dict(archive=str(archive),data_max=comparison_max,output=str(args.output))))


if __name__ == '__main__':
    main()
