"""Step 1.5 shared settings. Frozen final checkpoint; MLP expansion only; one location per run.

Reuses Step 1 (baseline loading, data, evaluation, LoopLoRA) and Step 2 (per-use gradient
probe and A/B accumulator) without modifying either directory.
"""
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STEP1 = Path('/hdd/mh3897/cc/looping-experiments/step1_20261020')
STEP2 = Path('/hdd/mh3897/cc/looping-experiments/step2_early_20261025')
STEP1_HASH = 'f0d4708bff22bf428f865ac78e0b5d406e27a9e14124535904c6cf6b0f520084'
STEP2_HASH = '34685015d4e8d3fa8eeed7ee629dd31595f490f4f29b19fd79423fdfac3b948e'
os.environ.setdefault('NANOCHAT_DTYPE', 'bfloat16')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
sys.dont_write_bytecode = True
# Append (not insert) so this directory's *15.py modules always win name lookups.
sys.path.append(str(STEP1))
import common as previous            # noqa: E402  Step 1 common.py
import interventions as step1_int    # noqa: E402  Step 1 interventions.py (LoopLoRA)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load('step2_probe', STEP2 / 'probe.py')

VARIANTS = {'shared_small': (8, False), 'independent': (8, True), 'shared_large': (32, False)}
EXPECTED_PARAMETERS = {'shared_small': 61440, 'independent': 245760, 'shared_large': 245760}

PROTOCOL = dict(
    checkpoint=19074, matrix='mlp.c_fc (MLP expansion, 1536 -> 6144)', locations=24, rounds=4,
    selections=['D1', 'D2'], primary_selection='D1', batches=32, batch_size=2, sequence_length=2048,
    score='C = minus mean over six round pairs of B, B = cosine of FP32 batch-mean per-round gradients',
    secondary='A = mean over batches of within-batch cosine (reported, not used for selection)',
    select='two highest C, two lowest C, two random of remaining 20 via numpy default_rng(2026); ties by layer',
    variants={k: dict(rank=r, independent=i) for k, (r, i) in VARIANTS.items()},
    expected_parameters=EXPECTED_PARAMETERS,
    adapter_seed=42, adapter_A_seed_rule='42 + 100*layer_index + 4 (Step 1 rule for up)',
    updates=200, microbatch=2, accumulation=128, tokens_per_update=524288, train_tokens=104857600,
    learning_rate=1e-4, final_learning_rate=1e-5, warmup=20, adam_betas=[0.9, 0.95], adam_eps=1e-8,
    weight_decay=0.0, dropout=0.0, gradient_clip=None, train_order_seed=42,
    data='Step 1 train/development reused unchanged; NEW confirmation from shard 6542 excluding all old registries',
    eval_steps=[0, 50, 100, 200], development_sequences=2048, confirmation_sequences=2048,
    endpoint='update 200 only; no checkpoint selection',
    scaling=dict(scales=[0.9, 1.1], rounds=4, dev_sequences=256,
                 rule='per location choose the lowest-dev-BPB (round, scale); evaluate only that on confirmation'),
    gradient_sum_relative_tolerance=2e-4,
    primary_gain='BPB(shared_large) - BPB(independent) on confirmation at update 200; positive favors independent',
    practical_benefit_threshold='unresolved: agree with Muyu before opening confirmation results',
    compute_note='shared_large uses 4x adapter-forward matmul FLOPs of independent; total FLOPs not matched',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sources():
    return {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}


def layer_index(location):
    assert location.startswith('L') and location.endswith('.up'), location
    i = int(location[1:3]) - 1
    assert 0 <= i < 24
    return i


def verify():
    assert sha(STEP1 / 'experiment_manifest.json') == STEP1_HASH, 'Step 1 manifest drift'
    assert sha(STEP2 / 'manifest.json') == STEP2_HASH, 'Step 2 manifest drift'
    s2 = json.loads((STEP2 / 'manifest.json').read_text())
    assert sha(STEP2 / 'probe.py') == s2['source_hashes']['probe.py'], 'Step 2 probe drift'
    previous.verify_source()
    path = ROOT / 'manifest.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    assert manifest['protocol'] == PROTOCOL, 'Protocol drift'
    assert manifest['source_hashes'] == sources(), 'Step 1.5 source drift'
    assert sha(ROOT / 'data/manifest.json') == manifest['data_manifest_sha256']
    dm = json.loads((ROOT / 'data/manifest.json').read_text())
    assert sha(ROOT / 'data/confirmation.npy') == dm['confirmation']['sha256']
    return manifest


def selection_data(name):
    """D1/D2 exactly as used by Step 2."""
    import numpy as np
    m = json.loads((STEP2 / 'data/manifest.json').read_text())['selections'][name]
    assert sha(m['path']) == m['sha256'], name
    a = np.load(m['path'], mmap_mode='r')
    assert a.shape == (64, 2049) and a.dtype == np.uint16
    return a


def confirmation():
    import numpy as np
    dm = json.loads((ROOT / 'data/manifest.json').read_text())['confirmation']
    path = ROOT / 'data/confirmation.npy'
    assert sha(path) == dm['sha256']
    a = np.load(path, mmap_mode='r')
    assert a.shape == (2048, 2049)
    return a


def frozen_selection():
    """The six locations, frozen by the gradient job before any adapter training."""
    report = json.loads((ROOT / 'results/gradient/results.json').read_text())
    path = ROOT / 'selection.json'
    assert sha(path) == report['selection_sha256'], 'Selection changed after freezing'
    return json.loads(path.read_text()), sha(path)


def select(scores, seed=2026):
    """scores: {location: C}. Deterministic; ties broken by layer number."""
    import numpy as np
    bad = [k for k, v in scores.items() if v is None or not math.isfinite(v)]
    if bad:
        raise ValueError(f'Undefined score(s), resolve before selection: {bad}')
    names = sorted(scores, key=layer_index)
    assert len(names) == 24
    high = sorted(names, key=lambda n: (-scores[n], layer_index(n)))[:2]
    low = sorted(names, key=lambda n: (scores[n], layer_index(n)))[:2]
    assert not set(high) & set(low)
    rest = [n for n in names if n not in high and n not in low]
    picks = np.random.default_rng(seed).choice(len(rest), size=2, replace=False)
    rand = [rest[i] for i in sorted(int(p) for p in picks)]
    return dict(high=high, low=low, random=rand, locations=high + low + rand,
                scores={n: scores[n] for n in names}, seed=seed)


def jobs(locations):
    return [f'{loc.split(".")[0]}_{v}' for loc in locations for v in VARIANTS]
