"""Render and archive an inline centered-PCA spectrum chart."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("cumulative", "std", "variance-share"), default="cumulative")
    parser.add_argument("--x-scale", choices=("log", "linear"), default="log")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output.resolve()
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    hashes = json.loads((run / "output_hashes.json").read_text())
    for path, digest in hashes.items():
        assert sha256(run / path) == digest, path
    assert json.loads((run / "validation/summary.json").read_text())["passed"]
    with np.load(run / "derived/spectrum.npz") as arrays:
        fractions = arrays["cumulative_explained_variance"].copy()
        variance_share = arrays["explained_variance_ratio"].copy()
        np.testing.assert_allclose(np.cumsum(variance_share), fractions, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(variance_share, arrays["explained_variance"] / arrays["explained_variance"].sum(), rtol=1e-14)
        std = np.sqrt(arrays["explained_variance"])
        np.testing.assert_allclose(std, arrays["centered_singular_values"] / np.sqrt(
            json.loads((run / "results.json").read_text())["shape"][0] - 1), rtol=1e-14)
    results = json.loads((run / "results.json").read_text())
    assert len(fractions) == results["shape"][1]
    assert np.all(np.diff(fractions) >= 0) and fractions[-1] == 1
    is_std = args.metric == "std"
    assert np.all(np.isfinite(std)) and np.all(std >= 0) and np.all(np.diff(std) <= 0)
    data = {"run_id": run.name, "metric": args.metric,
            "values": std.tolist() if is_std else (fractions * 100).tolist(),
            "title": "PCA standard deviation" if is_std else "Vocabulary embedding PCA",
            "y_axis": "Standard deviation (embedding units)" if is_std else "Cumulative variance explained (%)",
            "description": ("Standard deviation for each individual principal component, in descending order. "
                            "Square root of sample variance across all 32,768 vocabulary embeddings; denominator N minus 1. "
                            "X is logarithmic; Y is linear.") if is_std else
                           "Cumulative centered PCA variance explained. X is logarithmic; Y is linear from 0 to 100 percent.",
            "thresholds": [{"q": float(q), "k": k}
                           for q, k in results["pca_minimum_directions"].items()] if not is_std else []}
    if args.metric == "variance-share":
        data.update(values=(variance_share * 100).tolist(),
                    title="Variance explained per direction",
                    y_axis="Variance explained (%)",
                    description="Non-cumulative centered PCA variance explained by each direction, as a percentage of total variance. These percentages sum to the cumulative variance curve. X is logarithmic; Y is linear.",
                    thresholds=[])
    data["x_scale"] = args.x_scale
    if args.x_scale == "linear":
        data["description"] = data["description"].replace("X is logarithmic", "X is linear")
    template = Path(__file__).with_name("pca_view.html")
    fragment = template.read_text().replace("__PCA_DATA__", json.dumps(data, separators=(",", ":")))
    fragment = fragment.replace("__PCA_TITLE__", data["title"]).replace("__PCA_DESCRIPTION__", data["description"])
    fragment = fragment.replace("vocab-pca-variance", f"vocab-pca-{args.metric}-{args.x_scale}")
    assert len(fragment.encode()) < 1_000_000
    assert not output.exists() or output.read_text() == fragment
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)
    settings = {"source_run": str(run), "source_spectrum_sha256": hashes["derived/spectrum.npz"],
                "x_scale": args.x_scale, "y_scale": "linear", "quantity": data["y_axis"],
                "metric": args.metric, "std_denominator": results["shape"][0] - 1,
                "theme": "Codex dark", "vocabulary_weighting": "uniform",
                "source_code_hashes": {p.name: sha256(p) for p in (Path(__file__), template)},
                "inline_path": str(output), "inline_sha256": sha256(output)}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    archive = run / "visualization" / f"pca_{args.x_scale}_directions_{fingerprint}"
    archive.mkdir(parents=True, exist_ok=True)
    for source in (Path(__file__), template, output):
        shutil.copy2(source, archive / source.name)
    (archive / "data.json").write_text(json.dumps(data) + "\n")
    (archive / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(archive)


if __name__ == "__main__":
    main()
