"""Render one dark mean-norm chart with population-standard-deviation bands."""
import argparse
import hashlib
from pathlib import Path
import json
import shutil
import numpy as np
from .common import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--families", choices=("all", "attention"), default="all")
    parser.add_argument("--embedding-references", action="store_true")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output.resolve()
    manifest = read_json(run/'manifest.json')
    no_embed_norm = manifest['config'].get('model_type') == 'gpt_base_no_embed_norm'
    result = read_json(run / "results.json")
    assert result["state"] == "complete"
    assert sha256(run / "derived/statistics.npz") == result["statistics_sha256"]
    assert sha256(run / "derived/schema.json") == result["schema_sha256"]
    assert sha256(run / "validation/summary.json") == result["validation_sha256"]
    assert read_json(run / "validation/summary.json")["passed"]
    schema = read_json(run / "derived/schema.json")
    points = []
    with np.load(run / "derived/statistics.npz") as stats:
        np.testing.assert_allclose(stats["std"]**2, stats["variance"], rtol=1e-14)
        assert np.all(stats["count"] == result["tokens"])
        for source in schema["sources"]:
            i = source["index"]
            point = dict(source, x=source["layer"], mean=float(stats["mean"][i]),
                         std=float(stats["std"][i]), variance=float(stats["variance"][i]), count=int(stats["count"][i]))
            point["lower"] = max(0.0, point["mean"]-point["std"])
            point["upper"] = point["mean"]+point["std"]
            points.append(point)
    assert len(points) == 49
    if args.families == "attention":
        points = [p for p in points if p["family"] != "mlp"]
    data = dict(run_id=run.name,tokens=result["tokens"],documents=result["documents"],dimensions=1536,
        points=points, families=args.families, y_scale=args.y_scale, clipped_lower_band=any(p["std"]>p["mean"] for p in points),
        model_label='GPTBase · no norm(x0)' if no_embed_norm else 'GPTBase', no_embed_norm=no_embed_norm)
    data["references"] = [
        dict(id="embedding-mean", label="Emb mean", value=next(p["mean"] for p in points if p["family"] == "embedding"), dash="7 4"),
        dict(id="embedding-rmsnorm", label="RMSNorm ref. ≈" if no_embed_norm else "RMSNorm ≈", value=float(np.sqrt(data["dimensions"])), dash="3 4",
             meaning="Ideal post-RMSNorm L2 norm sqrt(d); epsilon and BF16 rounding cause small departures"),
    ] if args.embedding_references else []
    template = Path(__file__).with_name("view.html")
    fragment = template.read_text().replace("__NORM_DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False))
    assert len(fragment.encode()) < 1_000_000
    assert not output.exists() or output.read_text() == fragment
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)
    settings = dict(source_run=str(run),statistics_sha256=result["statistics_sha256"],
        source_code_hashes={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("common.py"),template)},
        inline_path=str(output),inline_sha256=sha256(output),theme="Codex dark",
        band="mean ± one population standard deviation over token occurrences, lower bound clipped at zero",
        units="full-vector L2 norm",variance_in_hover=True,x_scale="linear",y_scale=args.y_scale,
        y_domain="shared across visible series; rescales on legend toggle",
        references=data["references"],
        log_floor="decade at or below min positive band bound / 1.04; nonpositive lower bounds clipped to this labeled floor" if args.y_scale == "log" else None,
        embedding="raw lookup; no initial RMSNorm" if no_embed_norm else "pre-RMSNorm lookup",layers=list(range(24)),families=["attention"] if args.families == "attention" else ["attention","mlp"],tokens=result["tokens"])
    fingerprint = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
    archive = run / "visualization" / f"mean_norm_with_std_{fingerprint}"
    archive.mkdir(parents=True,exist_ok=True)
    for source in (Path(__file__),Path(__file__).with_name("common.py"),template,output):
        shutil.copy2(source, archive/source.name)
    write_json(archive / "data.json",data)
    write_json(archive / "settings.json",settings)
    print(archive)


if __name__ == "__main__":
    main()
