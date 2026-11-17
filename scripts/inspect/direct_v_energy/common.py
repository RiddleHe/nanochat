import hashlib
import json
from pathlib import Path

LAYERS = [12, 14, 16, 18, 20, 22]
REFERENCE = Path('/local-ssd/mh3897/analysis_runs/d24_qkv_separate_sources/gpt_base_step7809_20261022T005641Z')
VARIANT = 'bov_d24_f3p91e19_gpt_base_no_embed_norm_nosmear_s42_20261030T190903Z'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def verify(run):
    m = read(run/'manifest.json')
    for section in ('code', 'inputs', 'kernel_cache'):
        assert hashes(run/section) == m[section+'_hashes'], section
    return m


def sources():
    return [dict(index=0, family='embedding', layer=-1)] + [
        dict(index=1+2*l+i, family=f, layer=l) for l in range(24)
        for i, f in enumerate(('attention', 'mlp'))]
