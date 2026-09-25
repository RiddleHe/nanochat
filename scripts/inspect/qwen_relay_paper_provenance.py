"""Validate the archived baseline and the explicitly repackaged source files.

Relocated helpers change file hashes, not intervention function bodies. The
manifest binds both the archived source hashes and the reviewed package hashes;
it does not rewrite or relax the archived baseline/selection/output checks.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('qwen_relay_paper_sources.json')


def validate_baseline_sources(baseline):
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest['baseline_source_hashes'].items():
        if baseline['hashes'].get(name) != expected:
            raise ValueError(f'Archived baseline source mismatch: {name}')
    for relative, expected in manifest['package_source_hashes'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Packaged source changed; review a new manifest/run: {relative}')
