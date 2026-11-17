"""Compute direct FP64 SVD rank and centered PCA of the frozen embedding table."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_info, threadpool_limits


def metrics(s, shape):
    energy = s * s
    total = float(energy.sum())
    tolerance = max(shape) * np.finfo(np.float64).eps * float(s[0])
    if total == 0:
        return {"numerical_rank": 0, "rank_tolerance": tolerance,
                "stable_rank": None, "energy_entropy_rank": None,
                "participation_ratio": None}
    p = energy / total
    positive = p > 0
    return {"numerical_rank": int(np.count_nonzero(s > tolerance)),
            "rank_tolerance": tolerance, "largest_singular_value": float(s[0]),
            "smallest_singular_value": float(s[-1]),
            "stable_rank": total / float(energy[0]),
            "energy_entropy_rank": float(np.exp(-np.sum(p[positive] * np.log(p[positive])))),
            "participation_ratio": float(1 / np.sum(p * p))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    for relative, expected in manifest["hashes"].items():
        digest = hashlib.sha256()
        with open(run / relative, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, relative
    (run / "status.json").write_text('{"status": "running"}\n')
    start = time.monotonic()
    torch.set_num_threads(args.threads)
    x = torch.load(run / "inputs/embedding.pt", weights_only=True).double().numpy()
    n, d = x.shape
    with threadpool_limits(limits=args.threads):
        runtime = threadpool_info()
        print("Computing uncentered singular values", flush=True)
        raw_s = scipy.linalg.svdvals(x, check_finite=False)
        raw_energy = float(np.sum(x * x))
        mean = x.mean(axis=0)
        x -= mean
        centered_energy = float(np.sum(x * x))
        # Keep independent original rows for reconstruction validation.
        row_ids = np.linspace(0, n - 1, min(128, n), dtype=int)
        reference = x[row_ids].copy()
        print("Computing centered PCA by direct thin SVD", flush=True)
        u, s, vh = scipy.linalg.svd(x, full_matrices=False, check_finite=False,
                                    overwrite_a=True, lapack_driver="gesdd")
        variance = s * s / (n - 1)
        total_variance = float(variance.sum())
        if total_variance == 0:
            raise ValueError("All centered rows are zero; explained variance is undefined")
        ratio = variance / total_variance
        cumulative = np.cumsum(ratio)
        cumulative[-1] = 1.0
        reconstruction = (u[row_ids] * s) @ vh
        checks = {
            "raw_energy_relative_error": abs(float(raw_s @ raw_s) - raw_energy) / raw_energy,
            "centered_energy_relative_error": abs(float(s @ s) - centered_energy) / centered_energy,
            "mean_energy_identity_relative_error": abs(raw_energy - centered_energy - n * float(mean @ mean)) / raw_energy,
            "component_orthogonality_max_error": float(np.max(np.abs(vh @ vh.T - np.eye(d)))),
            "reconstruction_relative_error_128_rows": float(np.linalg.norm(reference - reconstruction) / np.linalg.norm(reference)),
            "singular_values_descending": bool(np.all(np.diff(s) <= 0)),
            "validation_row_ids": row_ids.tolist(),
        }
        numeric = [v for k, v in checks.items() if "error" in k]
        checks["passed"] = all(v < 1e-10 for v in numeric) and checks["singular_values_descending"]
        np.savez(run / "derived/spectrum.npz", raw_singular_values=raw_s,
                 centered_singular_values=s, explained_variance=variance,
                 explained_variance_ratio=ratio, cumulative_explained_variance=cumulative,
                 mean=mean, components=vh)
        results = {
            "shape": [n, d], "representation": manifest["representation"],
            "raw_rank_metrics": metrics(raw_s, x.shape),
            "centered_rank_metrics": metrics(s, x.shape),
            "pca_minimum_directions": {str(q): int(np.searchsorted(cumulative, q) + 1)
                                       for q in (0.5, 0.8, 0.9, 0.95, 0.99)},
            "pca_cumulative_variance": {str(k): float(cumulative[k - 1])
                                        for k in (1, 10, 50, 100, 256, 512, 768, 1024, 1536) if k <= d},
            "mean_fraction_of_uncentered_energy": n * float(mean @ mean) / raw_energy,
            "elapsed_seconds": time.monotonic() - start, "threadpools": runtime,
        }
    (run / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (run / "validation/summary.json").write_text(json.dumps(checks, indent=2) + "\n")
    (run / "status.json").write_text(json.dumps({"status": "complete" if checks["passed"] else "failed"}) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    assert checks["passed"], checks


if __name__ == "__main__":
    main()
