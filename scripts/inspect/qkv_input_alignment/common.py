"""Artifact contracts shared by the independent alignment stages; no model imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

PACKAGE = 'scripts.inspect.qkv_input_alignment'


class Tee:
    """Keep CLI output visible and in the stage's execution record."""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_npz(path, arrays):
    path = Path(path)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_config(config):
    require(config.get('measurement', 'sources') in ('sources', 'full_input'), 'Unsupported measurement')
    energy_kinds=config.get('raw_projected_energy_kinds', [])
    require(isinstance(energy_kinds,list) and len(set(energy_kinds))==len(energy_kinds) and
            all(k in ('Q','K','V') for k in energy_kinds), 'Invalid projected-energy kinds')
    require(not energy_kinds or config.get('measurement','sources')=='sources', 'Raw energy requires sources mode')
    thresholds = config['squared_energy_thresholds']
    require(isinstance(thresholds, list) and len(thresholds) > 0, 'Specify squared energy thresholds')
    require(all(isinstance(t, (float, int)) and np.isfinite(t) and 0 < t <= 1 for t in thresholds), 'Thresholds must be in (0,1]')
    require(thresholds == sorted(set(thresholds)), 'Thresholds must be unique and increasing')
    for key, expected in [('forward_dtype','bfloat16'),('svd_dtype','float64'),('basis_dtype','float32'),
                          ('analysis_dtype','float32'),('storage_dtype','float32'),('summary_dtype','float64')]:
        require(config[key] == expected, f'Unsupported {key}; expected {expected}')
    require(isinstance(config['source_batch_size'], int) and config['source_batch_size'] > 0, 'Invalid source batch size')
    require(isinstance(config['cpu_threads'], int) and config['cpu_threads'] > 0, 'Invalid CPU threads')
    require(np.isfinite(config['tie_rtol']) and 0 <= config['tie_rtol'] < 1, 'Invalid tie tolerance')
    for key in ('svd_relative_error','stored_basis_orthogonality_error','alignment_bound_tolerance','source_norm_rtol','source_norm_atol'):
        require(np.isfinite(config['validation'][key]) and config['validation'][key] > 0, f'Invalid tolerance: {key}')


def source_index(depth):
    sources = [dict(index=0, source_family='embedding', source_layer=-1)]
    for layer in range(depth):
        for family in ('attention','mlp'):
            sources.append(dict(index=len(sources), source_family=family, source_layer=layer))
    return sources


def matrix_specs(model_config, config=None):
    config = config or {}
    d = model_config['n_embd']
    require(d % model_config['n_head'] == 0, 'Nonintegral head dimension')
    layers = config.get('layers', list(range(model_config['n_layer'])))
    kinds = config.get('kinds', list('QKV'))
    require(layers and list(layers) == sorted(set(layers)) and all(isinstance(l, int) and 0 <= l < model_config['n_layer'] for l in layers), 'Invalid selected layers')
    require(kinds and len(set(kinds)) == len(kinds) and all(k in ('Q','K','V') for k in kinds), 'Invalid selected projection kinds')
    for layer in layers:
        for kind in kinds:
            yield layer, kind, model_config['n_head'] if kind == 'Q' else model_config['n_kv_head'], d // model_config['n_head'], d


def verify_run(run, checkpoint=False):
    run = Path(run)
    m = read_json(run/'manifest.json')
    check_config(m['config'])
    for relative, expected in m['source_hashes'].items():
        require(sha256(run/'code'/relative) == expected, f'Frozen code changed: {relative}')
    for relative, expected in m['input_hashes'].items():
        require(sha256(run/relative) == expected, f'Input changed: {relative}')
    meta = read_json(run/'inputs/checkpoint_metadata.json')
    require(meta['model_type'] in ('gpt_base', 'gpt_base_attn_res_qkv', 'gpt_base_no_embed_norm', 'gpt_base_scalar_res_qkv_no_embed_norm'), 'Unsupported alignment model')
    require(meta['model_type'] == m['config'].get('model_type', 'gpt_base'), 'Configured model type mismatch')
    require(meta['model_type'] == 'gpt_base' or m['config'].get('measurement', 'sources') == 'sources',
            'AttnRes currently supports per-source alignment only')
    require(read_json(run/'inputs/source_index.json') == source_index(meta['model_config']['n_layer']), 'Invalid source index')
    if checkpoint:
        require(sha256(m['checkpoint_path']) == m['checkpoint_sha256'], 'Checkpoint changed')
    return m, meta['model_config']


def stage_entry(run, stage):
    """Route CLI stages into the frozen package and recorded environment."""
    run = Path(run).resolve()
    m = read_json(run/'manifest.json')
    frozen = run/'code/scripts/inspect/qkv_input_alignment'
    if Path(__file__).resolve().parent != frozen or any(
            os.environ.get(k) != v for k,v in m['environment_overrides'].items()):
        env = dict(os.environ, **m['environment_overrides'])
        for name in ('RANK','LOCAL_RANK','WORLD_SIZE','PYTHONPATH'):
            env.pop(name, None)
        os.chdir(run/'code')
        args = [m['python_executable'], '-m', f'{PACKAGE}.{stage}', '--run-dir', str(run)]
        # All stage paths are absolute; extra arguments such as --device are preserved.
        extra = sys.argv[1:]
        for i, value in enumerate(extra):
            if value == '--run-dir':
                del extra[i:i+2]
                break
        else:
            extra = [v for v in extra if not v.startswith('--run-dir=')]
        os.execve(args[0], args + extra, env)
    for name, value in m['environment_overrides'].items():
        require(os.environ.get(name) == value, f'Environment mismatch for {name}; use recorded stage command')
    log = (run/'logs'/f'{stage}.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(f'{now()} stage={stage}')
    return run


def verify_svd(run, manifest):
    run = Path(run)
    index = read_json(run/'svd/index.json')
    require(index['manifest_sha256'] == sha256(run/'manifest.json'), 'SVD manifest mismatch')
    meta = read_json(run/'inputs/checkpoint_metadata.json')['model_config']
    expected = {f'{kind}_{layer:02d}.npz' for layer,kind,*_ in matrix_specs(meta, manifest['config'])}
    require(set(index['files']) == expected, 'Incomplete SVD matrix bank')
    require(index['thresholds'] == manifest['config']['squared_energy_thresholds'], 'SVD threshold mismatch')
    require(index['environment_sha256'] == sha256(run/'svd/environment.json'), 'SVD environment changed')
    for name, digest in index['files'].items():
        require(sha256(run/'svd'/name) == digest, f'SVD changed: {name}')
        marker = (run/'svd'/name).with_suffix('.json')
        require(index['validation_sha256'][marker.name] == sha256(marker), 'SVD validation changed')
        require(read_json(marker)['passed'], 'SVD validation failed')
    return index


def verify_chunk(run, path, manifest_hash, svd_hash):
    record = read_json(path.with_suffix('.json'))
    require(record['manifest_sha256'] == manifest_hash and record['svd_index_sha256'] == svd_hash, 'Chunk provenance mismatch')
    require(record['sha256'] == sha256(path), 'Chunk checksum mismatch')
    validation = Path(run)/'validation'/path.with_suffix('.json').name
    require(record['validation_sha256'] == sha256(validation), 'Chunk validation changed')
    require(read_json(validation)['passed'], 'Chunk failed validation')
    return record
