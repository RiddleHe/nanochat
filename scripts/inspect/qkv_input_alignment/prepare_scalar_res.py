"""Freeze final ScalarRes weights/code and validated, matched reference tokens."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .common import PACKAGE, check_config, matrix_specs, read_json, require, sha256, source_index, write_json
from .prepare_no_embed_norm import tree_hashes


def prepare(config, output_root, gpu, run_id=None):
    check_config(config)
    require(config['model_type'] == 'gpt_base_scalar_res_qkv_no_embed_norm', 'Wrong model')
    require(config.get('measurement', 'sources') == 'sources', 'Only source alignment supported')
    require(not config.get('svd_reference_run'), 'Compute model-specific SVDs')
    training = Path(config['training_run']).resolve()
    tm = read_json(training/'manifest.json')
    require(read_json(training/'status.json').get('returncode') == 0, 'Training did not complete')
    step = config['checkpoint_step']
    require(step == tm['expected_steps'], 'Use the final checkpoint')
    checkpoint_dir = Path(tm['checkpoint_dir'])
    meta = read_json(checkpoint_dir/f'meta_{step:06d}.json')
    require(meta['step'] == step and meta['model_type'] == config['model_type'], 'Checkpoint mismatch')
    require(meta['model_config'] == tm['model_config'], 'Training model config mismatch')
    list(matrix_specs(meta['model_config'], config))
    model_code = Path(tm['repo'])/'nanochat'
    for name, digest in tm['source_sha256'].items():
        if name.startswith('nanochat/'):
            require(sha256(Path(tm['repo'])/name) == digest, 'Frozen training code changed: '+name)
    reference = Path(config['token_reference']).resolve()
    rm = read_json(reference/'manifest.json'); rr = read_json(reference/'results.json')
    require(rr['state'] == 'analysis_complete' and rr['manifest_sha256'] == sha256(reference/'manifest.json'),
            'Token reference incomplete or changed')
    require(rr['validation_sha256'] == sha256(reference/'validation/summary.json') and
            read_json(reference/'validation/summary.json')['passed'], 'Token reference validation failed')
    require(sha256(reference/'inputs/tokens.npy') == rm['tokens_sha256'], 'Reference tokens changed')
    provenance = reference/'inputs/provenance.json'
    require(sha256(provenance) == rm['input_hashes']['inputs/provenance.json'], 'Token provenance changed')
    tokens = np.load(reference/'inputs/tokens.npy', mmap_mode='r')
    require(tokens.shape == (128, 2048) and list(tokens.shape) == rm['tokens_shape'], 'Wrong reference population')
    description = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    fields = [v.strip() for v in description.split(',')]
    require(len(fields) == 5 and int(fields[3]) < 1000 and int(fields[4]) <= 5, 'GPU is not idle: '+description)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = run_id or f'{config["model_type"]}_step{step}_{stamp}'
    require(Path(run_id).name == run_id and run_id not in ('.', '..'), 'Invalid run id')
    run = Path(output_root).resolve()/run_id; run.mkdir(parents=True, exist_ok=False)
    for folder in ('code','inputs','svd','tokens','summaries','validation','visualization','logs'):
        (run/folder).mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(model_code, run/'code/nanochat', ignore=ignore)
    shutil.copytree(Path(__file__).parent, run/'code/scripts/inspect/qkv_input_alignment', ignore=ignore)
    frozen_checkpoint = run/'inputs/checkpoint'; frozen_checkpoint.mkdir()
    for name in (f'model_{step:06d}.pt', f'meta_{step:06d}.json'):
        shutil.copy2(checkpoint_dir/name, frozen_checkpoint/name)
        require(sha256(checkpoint_dir/name) == sha256(frozen_checkpoint/name), 'Checkpoint copy mismatch')
    shutil.copy2(frozen_checkpoint/f'meta_{step:06d}.json', run/'inputs/checkpoint_metadata.json')
    for name in ('tokens.npy','provenance.json'):
        shutil.copy2(reference/'inputs'/name, run/'inputs'/name)
    write_json(run/'inputs/source_index.json', source_index(meta['model_config']['n_layer']))
    write_json(run/'inputs/reference.json', dict(token_reference=str(reference),
        manifest_sha256=sha256(reference/'manifest.json'), results_sha256=sha256(reference/'results.json'),
        training_run=str(training), training_manifest_sha256=sha256(training/'manifest.json'),
        source_norm_validation='independent module-output hooks on this model, not reference-model norms'))
    tokenizer = run/'inputs/runtime_base/tokenizer'; tokenizer.mkdir(parents=True)
    for path, digest in tm['tokenizer_sha256'].items():
        require(sha256(path) == digest, 'Training tokenizer changed')
        shutil.copy2(path, tokenizer/Path(path).name)
    require(tree_hashes(tokenizer) == tree_hashes(reference/'inputs/runtime_base/tokenizer'),
            'Reference and training tokenizers differ')
    cache = Path(tm['kernel_cache_path'])
    frozen_cache = run/'kernel_cache'/cache.name
    shutil.copytree(cache, frozen_cache, ignore=ignore)
    require(tree_hashes(frozen_cache) == tm['kernel_cache_sha256'], 'Kernel cache changed')
    require(tree_hashes(run/'code/nanochat') == tree_hashes(model_code), 'Model copy mismatch')
    require(sha256(run/'inputs/tokens.npy') == rm['tokens_sha256'], 'Token copy mismatch')
    env = dict(CUDA_VISIBLE_DEVICES=fields[1], NANOCHAT_BASE_DIR=str(run/'inputs/runtime_base'),
        NANOCHAT_DTYPE='bfloat16', OMP_NUM_THREADS=str(config['cpu_threads']),
        HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_CACHE=str(run/'kernel_cache'), HUGGINGFACE_HUB_CACHE=str(run/'kernel_cache'),
        KERNELS_CACHE=str(run/'kernel_cache'), HF_KERNELS_CACHE=str(run/'kernel_cache'))
    repo = Path(__file__).resolve().parents[3]
    python = str(Path(sys.executable).absolute())
    checkpoint = frozen_checkpoint/f'model_{step:06d}.pt'
    manifest = dict(schema_version=1, run_id=run_id, config=config, created_utc=stamp,
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint), checkpoint_step=step,
        tokens_shape=list(tokens.shape), tokens_sha256=rm['tokens_sha256'],
        input_hashes={'inputs/'+k:v for k,v in tree_hashes(run/'inputs').items()}, source_hashes=tree_hashes(run/'code'),
        gpu_uuid=fields[1], selected_gpu=description, environment_overrides=env, python_executable=python,
        tokenizer_sha256={str(tokenizer/k):v for k,v in tree_hashes(tokenizer).items()},
        kernel_cache_path=str(frozen_cache), kernel_cache_sha256=tree_hashes(frozen_cache),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        commands={stage:[python,'-m',f'{PACKAGE}.{stage}','--run-dir',str(run)]
                  for stage in ('compute_svd','collect','summarize')},
        collection_definition='raw unweighted bank sources; raw embedding; per-token squared subspace projection / full source squared norm',
        threshold_definition='cumulative_squared_singular_values_only; extend_cutoff_ties')
    write_json(run/'manifest.json', manifest)
    write_json(run/'status.json', dict(state='prepared', completed_chunks=0))
    return run
