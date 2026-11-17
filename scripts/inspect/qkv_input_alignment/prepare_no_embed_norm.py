"""Adapt the validated direct-V run into a frozen alignment experiment."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .common import PACKAGE, check_config, matrix_specs, read_json, require, sha256, source_index, write_json


def tree_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def prepare(config, output_root, gpu, run_id=None):
    check_config(config)
    require(config['model_type'] == 'gpt_base_no_embed_norm' and config.get('measurement', 'sources') == 'sources', 'Unsupported model or boundary')
    require(not config.get('svd_reference_run'), 'This adapter computes model-specific SVDs')
    ref = Path(config['reference_run']).resolve()
    parent = read_json(ref/'manifest.json'); result = read_json(ref/'results.json')
    require(parent['config']['model_type'] == config['model_type'], 'Reference model mismatch')
    require(result['state'] == 'complete' and result['manifest_sha256'] == sha256(ref/'manifest.json'), 'Reference incomplete or changed')
    for section in ('code', 'inputs', 'kernel_cache'):
        require(tree_hashes(ref/section) == parent[section+'_hashes'], f'Reference {section} changed')
    require(result['statistics_sha256'] == sha256(ref/'derived/statistics.npz'), 'Reference statistics changed')
    require(result['validation_sha256'] == sha256(ref/'validation/summary.json') and read_json(ref/'validation/summary.json')['passed'], 'Reference validation failed')
    tokens = np.load(ref/'inputs/tokens.npy', mmap_mode='r')
    require(sha256(ref/'inputs/tokens.npy') == parent['tokens_sha256'] == result['tokens_sha256'], 'Reference tokens changed')
    require(tokens.ndim == 2 and tokens.size == result['tokens'], 'Reference token count mismatch')
    meta = read_json(ref/'inputs/checkpoint/meta_007809.json')
    require(meta['model_type'] == config['model_type'] and meta['step'] == 7809, 'Wrong checkpoint')
    require(not any(v for v in meta['model_config'].values() if isinstance(v, bool)), 'Unsupported mixing flags')
    list(matrix_specs(meta['model_config'], config))
    expected_sources = source_index(meta['model_config']['n_layer'])
    normalized_sources = [dict(index=s['index'], source_family=s['family'], source_layer=s['layer'])
                          for s in read_json(ref/'inputs/source_index.json')]
    require(normalized_sources == expected_sources, 'Reference source index mismatch')
    records = []
    for i in range(len(tokens)):
        path = ref/'tokens'/f'chunk_{i:05d}.npz'; rec = read_json(path.with_suffix('.json'))
        require(rec['sequence_id'] == i and rec['tokens'] == tokens.shape[1] and rec['tokens_sha256'] == parent['tokens_sha256'], 'Reference chunk identity mismatch')
        require(rec['manifest_sha256'] == sha256(ref/'manifest.json'), 'Reference chunk manifest mismatch')
        require(rec['sha256'] == result['chunk_hashes'][path.name] == sha256(path), 'Reference chunk changed')
        validation = ref/'validation'/path.with_suffix('.json').name
        require(rec['validation_sha256'] == sha256(validation) and read_json(validation)['passed'], 'Reference chunk validation failed')
        records.append(dict(sequence_id=i, path=str(path), sha256=rec['sha256']))
    description = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    fields = [v.strip() for v in description.split(',')]
    require(len(fields) == 5 and fields[1].startswith('GPU-') and int(fields[3]) < 1000 and int(fields[4]) <= 5, 'Selected GPU is not idle: '+description)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = run_id or f'gpt_base_no_embed_norm_step7809_{stamp}'
    require(Path(run_id).name == run_id and run_id not in ('.', '..'), 'Invalid run id')
    run = Path(output_root).resolve()/run_id; run.mkdir(parents=True, exist_ok=False)
    for directory in ('code','inputs','svd','tokens','summaries','validation','visualization','logs'):
        (run/directory).mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(ref/'code/nanochat', run/'code/nanochat', ignore=ignore)
    shutil.copytree(Path(__file__).parent, run/'code/scripts/inspect/qkv_input_alignment', ignore=ignore)
    shutil.copytree(ref/'inputs/checkpoint', run/'inputs/checkpoint')
    shutil.copy2(ref/'inputs/checkpoint/meta_007809.json', run/'inputs/checkpoint_metadata.json')
    for name in ('tokens.npy', 'provenance.json'):
        shutil.copy2(ref/'inputs'/name, run/'inputs'/name)
    shutil.copytree(ref/'inputs/runtime_base', run/'inputs/runtime_base', ignore=ignore)
    shutil.copytree(ref/'kernel_cache', run/'kernel_cache', ignore=ignore)
    write_json(run/'inputs/source_index.json', expected_sources)
    write_json(run/'inputs/reference_chunks.json', records)
    write_json(run/'inputs/reference.json', dict(run=str(ref), manifest_sha256=sha256(ref/'manifest.json'), results_sha256=sha256(ref/'results.json')))
    env = {k:v.replace(str(ref), str(run)) for k,v in parent['environment_overrides'].items()}
    env['CUDA_VISIBLE_DEVICES'] = fields[1]
    env['OMP_NUM_THREADS'] = str(config['cpu_threads'])
    kernel_dirs = list((run/'kernel_cache').iterdir())
    require(len(kernel_dirs) == 1 and kernel_dirs[0].is_dir(), 'Unexpected kernel layout')
    checkpoint = run/'inputs/checkpoint/model_007809.pt'
    require(sha256(checkpoint) == parent['inputs_hashes']['checkpoint/model_007809.pt'], 'Copied checkpoint mismatch')
    require(sha256(run/'inputs/tokens.npy') == parent['tokens_sha256'], 'Copied tokens mismatch')
    require(tree_hashes(run/'code/nanochat') == tree_hashes(ref/'code/nanochat'), 'Copied model mismatch')
    require(tree_hashes(run/'inputs/runtime_base') == tree_hashes(ref/'inputs/runtime_base'), 'Copied tokenizer mismatch')
    require(tree_hashes(run/'kernel_cache') == parent['kernel_cache_hashes'], 'Copied kernel mismatch')
    repo = Path(__file__).resolve().parents[3]
    manifest = dict(schema_version=1, run_id=run_id, config=config, created_utc=stamp,
        reference_run=str(ref), reference_manifest_sha256=sha256(ref/'manifest.json'),
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint), checkpoint_step=meta['step'],
        tokens_shape=list(tokens.shape), tokens_sha256=parent['tokens_sha256'],
        input_hashes={'inputs/'+k:v for k,v in tree_hashes(run/'inputs').items()}, source_hashes=tree_hashes(run/'code'),
        gpu_uuid=fields[1], selected_gpu=description, environment_overrides=env, python_executable=str(Path(sys.executable).absolute()),
        tokenizer_sha256={str(run/'inputs/runtime_base/tokenizer'/k):v for k,v in tree_hashes(run/'inputs/runtime_base/tokenizer').items()},
        kernel_cache_path=str(kernel_dirs[0]), kernel_cache_sha256=tree_hashes(kernel_dirs[0]),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        commands={stage:[sys.executable,'-m',f'{PACKAGE}.{stage}','--run-dir',str(run)] for stage in ('compute_svd','collect','summarize')},
        collection_definition='per_token_full_source_energy_fraction_in_top_input_singular_subspace; raw_embedding_at_first_block_input',
        threshold_definition='cumulative_squared_singular_values_only; extend_cutoff_ties')
    write_json(run/'manifest.json', manifest)
    write_json(run/'status.json', dict(state='prepared', completed_chunks=0))
    return run
