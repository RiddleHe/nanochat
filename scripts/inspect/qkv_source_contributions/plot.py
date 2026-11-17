"""Build the inline exploration view from saved summaries, without GPU work."""
import argparse
import json
from pathlib import Path
import shutil
from .prepare import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run_dir / "summaries/heatmaps.json"
    data = json.loads(summary_path.read_text())
    # Retain full precision in saved scientific summaries; compact display data only.
    for cell in data["cells"]:
        for key in ("mean", "std", "min", "max"):
            cell[key] = round(cell[key], 6)
        for key in ("heads", "token_std"):
            cell[key] = [round(value, 6) for value in cell[key]]
    template_path = Path(__file__).with_name("view.html")
    template = template_path.read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    fragment = template.replace("__EXPERIMENT_DATA__", payload)
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError("Inline visualization exceeds 1 MB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    destination = args.run_dir / "visualization"
    destination.mkdir(exist_ok=True)
    shutil.copy2(args.output, destination / "qkv-source-contributions.html")
    shutil.copy2(template_path, destination / "view.html")
    shutil.copy2(Path(__file__), destination / "plot.py")
    write_json(destination / "settings.json", {
        "summary_sha256": sha256(summary_path), "fragment_sha256": sha256(args.output),
        "plot_source_sha256": sha256(Path(__file__)), "template_sha256": sha256(template_path),
        "inline_path": str(args.output), "display_decimal_places": 6,
        "mean_scale": "sqrt_color_shared_across_QKV_zero_to_global_max",
        "std_scale": "sqrt_color_shared_across_QKV_zero_to_global_max",
        "layer_numbering": "zero_based", "mean_measurement": "absolute_norm",
        "std_measurement": "population_std_across_per_head_token_means",
    })
    print(f"Visualization saved: {args.output} ({len(fragment.encode())} bytes)")


if __name__ == "__main__":
    main()
