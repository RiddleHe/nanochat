"""Plot one source family's full input/QKV norms in a 2x2 grid."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .prepare import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23])
    parser.add_argument("--family", choices=["attention", "mlp", "interleaved"], default="attention")
    args = parser.parse_args()
    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text())
    results = json.loads((run / "results.json").read_text())
    assert manifest["config"]["experiment"] == "d24_qkv_separate_sources"
    stats_path = run / "summaries/statistics.npz"
    assert sha256(stats_path) == results["statistics_sha256"]
    assert json.loads((run / "validation/summary.json").read_text())["passed"]
    panels = []
    families = ["embedding", "attention"] if args.family == "attention" else ["mlp"]
    if args.family == "interleaved":
        families = ["attention", "mlp"]
    source_start = -1 if args.family == "attention" else 0
    with np.load(stats_path) as stats:
        for layer in args.layers:
            available = manifest["source_index"][:2*layer+1]
            selected = [s for s in available if s["source_family"] in families]
            indices = [s["index"] for s in selected]
            expected_layers = [j for j in range(layer) for _ in range(2)] if args.family == "interleaved" else list(range(source_start, layer))
            assert [s["source_layer"] for s in selected] == expected_layers
            series = []
            for name, key in [("Input", f"input_{layer:02d}")] + [(kind, f"full_{kind}_{layer:02d}") for kind in "QKV"]:
                mean, std = stats[key+"_token_mean"][indices], stats[key+"_token_std"][indices]
                assert np.isfinite(mean).all() and (mean >= 0).all()
                series.append(dict(name=name, values=np.round(mean, 7).tolist(), std=np.round(std, 7).tolist()))
            panels.append(dict(layer=layer, title=f"Receiving layer {layer}", series=series,
                               source_index=selected))
            if args.family == "interleaved":
                panels[-1]["source_labels"] = [f"{s['source_layer']}.{'a' if s['source_family'] == 'attention' else 'm'}" for s in selected]
    data = dict(run_id=manifest["run_id"], tokens=results["tokens"], panels=panels,
                title=f"GPTBase · {args.family.upper() if args.family == 'mlp' else 'attention'}-source contributions",
                source_caption="Emb = embedding reference" if args.family == "attention" else "MLP outputs only",
                x_start=source_start,
                comparison_max=max(v for p in panels for s in p["series"] for v in s["values"]))
    if args.family == "interleaved":
        data.update(title="GPTBase · attention vs MLP sources", columns=1,
                    source_caption="a = attention output · m = MLP output")
    template = Path(__file__).with_name("source_norms_view.html")
    fragment = template.read_text().replace("__SOURCE_NORMS_DATA__", json.dumps(data, separators=(",", ":")))
    assert len(fragment.encode()) < 1_000_000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    artifact = run / "visualization" / (args.family + "_sources_layers_" + "_".join(map(str, args.layers)))
    artifact.mkdir(parents=True, exist_ok=True)
    for path, name in [(args.output, "inline.html"), (template, template.name), (Path(__file__), Path(__file__).name)]:
        shutil.copy2(path, artifact / name)
    write_json(artifact / "data.json", data)
    write_json(artifact / "settings.json", dict(statistics_sha256=sha256(stats_path),
        inline_sha256=sha256(args.output), inline_path=str(args.output), layers=args.layers,
        source_families=families, embedding_is_reference=args.family == "attention",
        x="zero_based_source_layer_index", y="mean_per_token_L2_norm",
        layout="one_column" if args.family == "interleaved" else "2_columns_desktop_stacked_mobile", input_dimensions=1536, projected_dimensions=1536,
        input="source_norm / actual_receiving_layer_RMSNorm_denominator_per_token_then_mean",
        projected="sqrt(sum_squared_head_norms)_per_token_then_mean_before_RoPE_and_QK_normalization",
        std="population_token_std_in_hover", y_domain=[0, data["comparison_max"]*1.05],
        token_count=results["tokens"], source_hashes={p.name:sha256(p) for p in (Path(__file__), template)}))
    print(json.dumps(dict(artifact=str(artifact), maximum=data["comparison_max"], output=str(args.output))))


if __name__ == "__main__":
    main()
