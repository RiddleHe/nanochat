import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def tree_hashes(path):
    return {str(p.relative_to(path)): sha256(p) for p in sorted(Path(path).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def verify_run(run):
    manifest = read_json(run / "manifest.json")
    for section in ("inputs", "code", "kernel_cache"):
        for relative, expected in manifest[section + "_hashes"].items():
            if sha256(run / section / relative) != expected:
                raise ValueError(f"Frozen {section} hash mismatch: {relative}")
    return manifest
