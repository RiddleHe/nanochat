"""Freeze the checkpoint embedding tensor and analysis code for a CPU run."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for folder in ("inputs", "code", "derived", "validation"):
        (run / folder).mkdir()
    meta = json.loads(args.metadata.read_text())
    assert meta["model_type"] == "gpt_base"
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    vocab = meta["model_config"]["vocab_size"]
    embedding = state["transformer.wte.weight"][:vocab].clone()
    assert embedding.shape == (vocab, meta["model_config"]["n_embd"])
    assert torch.isfinite(embedding).all()
    torch.save(embedding, run / "inputs/embedding.pt")
    np.save(run / "inputs/token_ids.npy", np.arange(vocab, dtype=np.int64))
    shutil.copy2(args.metadata, run / "inputs/metadata.json")
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, run / "code" / source.name)
    shutil.copy2("nanochat/model/gpt_base.py", run / "code/gpt_base_reference.py")
    (run / "code/tracked_changes.patch").write_text(subprocess.check_output(
        ["git", "diff", "HEAD"], text=True))
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "step": meta["step"], "shape": list(embedding.shape),
        "stored_dtype": str(embedding.dtype), "analysis_dtype": "float64",
        "representation": "stored input embedding; no RMSNorm; no coordinate standardization",
        "weighting": "one equal-weight row per vocabulary ID, including special tokens",
        "excluded_rows": "padding beyond model_config.vocab_size only",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__},
        "hashes": {str(p.relative_to(run)): sha256(p)
                   for directory in (run / "inputs", run / "code")
                   for p in directory.iterdir() if p.is_file()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(run, flush=True)


if __name__ == "__main__":
    main()
