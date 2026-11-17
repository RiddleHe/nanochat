"""Plot raw source norms and dimension-matched input/QKV component norms."""
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
    args = parser.parse_args()
    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text())
    results = json.loads((run / "results.json").read_text())
    stats_path = run / "summaries/statistics.npz"
    assert sha256(stats_path) == results["statistics_sha256"]
    assert json.loads((run / "validation/summary.json").read_text())["passed"]
    panels = []
    with np.load(stats_path) as stats:
        for layer in manifest["config"]["replay_layers"]:
            series = []
            for name, key in [("Input", f"input_{layer:02d}")] + [(kind, f"full_{kind}_{layer:02d}") for kind in "QKV"]:
                mean, std = stats[key + "_token_mean"], stats[key + "_token_std"]
                assert mean.shape == (layer + 1,) and np.isfinite(mean).all() and (mean >= 0).all()
                series.append(dict(name=name, values=np.round(mean, 7).tolist(), std=np.round(std, 7).tolist()))
            panels.append(dict(layer=layer, title=f"Receiving layer {layer}", series=series))
        raw = dict(title="Raw source outputs", layer=None, series=[dict(name="Raw source",
                   values=np.round(stats["source_norm_token_mean"], 7).tolist(),
                   std=np.round(stats["source_norm_token_std"], 7).tolist())])
    data = dict(run_id=manifest["run_id"], tokens=results["tokens"], panels=panels, raw=raw,
                comparison_max=max(value for panel in panels for series in panel["series"] for value in series["values"]),
                raw_max=max(raw["series"][0]["values"]))
    template = Path(__file__).with_name("source_norms_view.html")
    fragment = template.read_text().replace("__SOURCE_NORMS_DATA__", json.dumps(data, separators=(",", ":")))
    assert len(fragment.encode()) < 1_000_000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    artifact = run / "visualization/source_norms"
    artifact.mkdir(parents=True, exist_ok=True)
    for path, name in [(args.output, "inline.html"), (template, template.name), (Path(__file__), Path(__file__).name)]:
        shutil.copy2(path, artifact / name)
    write_json(artifact / "data.json", data)
    write_json(artifact / "settings.json", dict(
        statistics_sha256=sha256(stats_path), inline_sha256=sha256(args.output), inline_path=str(args.output),
        parent_run=manifest["parent_run"], token_count=results["tokens"], input_dimensions=1536, output_dimensions=1536,
        projection_location="before_rope_qk_norm_and_scaling", source_index="Emb_then_zero_based_layer_write",
        input_definition="mean_over_tokens(source_norm / actual_receiving_layer_RMSNorm_denominator)",
        projected_definition="mean_over_tokens(sqrt(sum_over_heads(per_token_head_norm ** 2)))",
        std="population_token_std_stored_and_available_on_hover_not_confidence_intervals",
        y_scale="shared_linear_zero_to_1.05_times_comparison_max_across_four_receiving_layers; raw_plot_separate_scale",
        comparison_max=data["comparison_max"], raw_max=data["raw_max"]))
    print(json.dumps(dict(output=str(args.output), comparison_max=data["comparison_max"], raw_max=data["raw_max"])))


if __name__ == "__main__":
    main()
