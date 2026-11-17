"""Plot heads by source: token-mean absolute norms and token-mean relative shares."""
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
    parser.add_argument("--metrics", nargs="+", choices=["magnitude", "share"], default=["magnitude", "share"])
    args = parser.parse_args()
    statistics_path = args.run_dir / "summaries/statistics.npz"
    summary = json.loads((args.run_dir / "summaries/heatmaps.json").read_text())
    lookup = {(c["kind"], c["layer"], c["source"]): c for c in summary["cells"]}
    panels = []
    share_sum_errors = []
    with np.load(statistics_path) as statistics:
        for layer in args.layers:
            for kind in "QKV":
                values = statistics[f"abs_{kind}_{layer:02d}_token_mean"]
                assert values.ndim == 2 and values.shape[1] == layer + 1
                assert np.isfinite(values).all() and (values >= 0).all()
                for source in range(layer + 1):
                    np.testing.assert_array_equal(values[:, source], lookup[kind, layer, source-1]["heads"])
                panels.append({"layer": layer, "kind": kind, "metric": "magnitude",
                               "values": np.round(values, 7).tolist()})
                shares = statistics[f"share_{kind}_{layer:02d}_token_mean"]
                assert shares.shape == values.shape
                assert np.isfinite(shares).all() and (shares >= 0).all()
                np.testing.assert_allclose(shares.sum(axis=1), 1, rtol=0, atol=2e-6)
                share_sum_errors.append(float(np.max(np.abs(shares.sum(axis=1)-1))))
                panels.append({"layer": layer, "kind": kind, "metric": "share",
                               "values": np.round(shares, 9).tolist()})
    metrics = list(dict.fromkeys(args.metrics))
    panels = [panel for panel in panels if panel["metric"] in metrics]
    data = {"run_id": summary["run_id"], "tokens": summary["tokens"],
            "layers": args.layers, "metrics": metrics, "panels": panels}
    template = Path(__file__).with_name("heads_view.html")
    fragment = template.read_text().replace("__HEAD_DATA__", json.dumps(data, separators=(",", ":")))
    assert len(fragment.encode()) < 1_000_000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    metric_tag = "both" if len(metrics) == 2 else metrics[0]
    artifact_dir = args.run_dir / "visualization" / (f"heads_transposed_{metric_tag}_layers_" + "_".join(map(str, args.layers)))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for source, name in [(args.output, "inline.html"), (Path(__file__), "plot_heads.py"), (template, template.name)]:
        shutil.copy2(source, artifact_dir / name)
    write_json(artifact_dir / "data.json", data)
    write_json(artifact_dir / "settings.json", {
        "statistics_sha256": sha256(statistics_path), "inline_sha256": sha256(args.output),
        "layers": args.layers, "layer_numbering": "zero_based",
        "x": "source_layer_with_embedding_column", "y": "head_index",
        "values": [{"magnitude": "mean_absolute_norm_per_token", "share": "mean_relative_share_per_token"}[m] for m in metrics], "head_averaging": False,
        "relative_share_definition": "per_token_source_norm_divided_by_sum_of_source_norms_within_head_then_mean_across_tokens",
        "color_scale": "linear_shared_across_layers_and_QKV_separately_for_each_metric",
        "token_count": summary["tokens"], "inline_path": str(args.output),
        "validation": "absolute_norms_match_existing_summary; relative_shares_read_directly_from_saved_per_token_share_means",
        "maximum_mean_share_row_sum_error": max(share_sum_errors),
    })
    print(f"Saved {len(panels)} panels to {args.output}")


if __name__ == "__main__":
    main()
