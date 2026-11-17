"""Freeze a new source-alignment run. Preparation does not execute SVD or replay."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .common import PACKAGE, check_config, read_json, require, sha256, source_index, verify_run, verify_svd, write_json


def prepare(config, output_root, gpu, run_id=None):
    check_config(config)
    if config.get('model_type') == 'gpt_base_scalar_res_qkv_no_embed_norm':
        from .prepare_scalar_res import prepare as prepare_variant
        return prepare_variant(config, output_root, gpu, run_id)
    if config.get('model_type') == 'gpt_base_no_embed_norm':
        from .prepare_no_embed_norm import prepare as prepare_variant
        return prepare_variant(config, output_root, gpu, run_id)
    reference = Path(config['reference_run']).resolve()
    parent = read_json(reference/'manifest.json')
    model_type = config.get('model_type', 'gpt_base')
    reference_types = {'gpt_base': 'd24_qkv_separate_sources',
                       'gpt_base_attn_res_qkv': 'd24_attn_res_qkv_source_contributions'}
    require(model_type in reference_types, 'Unsupported alignment model')
    require(parent['config']['experiment'] == reference_types[model_type], 'Reference attribution model mismatch')
    require(model_type == 'gpt_base' or config.get('measurement','sources') == 'sources',
            'AttnRes currently supports per-source alignment only')
    require(read_json(reference/'validation/summary.json')['passed'], 'Reference validation failed')
    results = read_json(reference/'results.json')
    require(results['state'] == 'analysis_complete', 'Reference collection incomplete')
    if model_type == 'gpt_base':
        require(sha256(reference/'summaries/statistics.npz') == results['statistics_sha256'], 'Reference summary changed')
    for relative, digest in parent['source_hashes'].items():
        require(sha256(reference/'code'/relative) == digest, f'Reference code changed: {relative}')
    checkpoint = Path(parent['checkpoint_path']).resolve()
    require(sha256(checkpoint) == parent['checkpoint_sha256'], 'Reference checkpoint changed')
    meta = read_json(reference/'inputs/checkpoint_metadata.json')
    require(sha256(reference/'inputs/checkpoint_metadata.json') == parent['metadata_sha256'], 'Reference metadata changed')
    require(meta['model_type'] == model_type, 'Checkpoint model type mismatch')
    require(not any(v for v in meta['model_config'].values() if isinstance(v, bool)), 'BoV/optional source mixing is unsupported')
    provenance = read_json(reference/'inputs/provenance.json')
    require(sha256(reference/'inputs/tokens.npy') == parent['tokens_sha256'] == provenance['tokens_sha256'], 'Reference tokens changed')
    token_shape = np.load(reference/'inputs/tokens.npy', mmap_mode='r').shape
    require(token_shape == (parent['config']['num_sequences'], parent['config']['sequence_length']), 'Reference token shape mismatch')
    require(read_json(reference/'inputs/source_index.json') == source_index(meta['model_config']['n_layer']), 'Reference source indexing mismatch')
    cache = Path(parent['kernel_cache_path'])
    for name, digest in parent['kernel_cache_sha256'].items():
        require(sha256(cache/name) == digest, f'Kernel cache changed: {name}')
    for path, digest in provenance['tokenizer_sha256'].items():
        require(sha256(path) == digest, f'Tokenizer changed: {path}')
    gpu_description = subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=index,uuid,name','--format=csv,noheader,nounits'], text=True).strip()
    fields = [s.strip() for s in gpu_description.split(',')]
    require(len(fields) >= 3 and fields[1].startswith('GPU-'), 'GPU must resolve to one physical device')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = run_id or f'{model_type}_step{meta["step"]}_{stamp}'
    require(Path(run_id).name == run_id and run_id not in ('.','..'), 'Invalid run ID')
    run = Path(output_root).resolve()/run_id
    run.mkdir(parents=True, exist_ok=False)
    for directory in ('code','inputs','svd','tokens','summaries','validation','visualization','logs'):
        (run/directory).mkdir()
    ignore = shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(reference/'code/nanochat', run/'code/nanochat', ignore=ignore)
    shutil.copytree(Path(__file__).parent, run/'code/scripts/inspect/qkv_input_alignment', ignore=ignore)
    for name in ('checkpoint_metadata.json','tokens.npy','source_index.json','provenance.json'):
        shutil.copy2(reference/'inputs'/name, run/'inputs'/name)
    if config.get('svd_reference_run'):
        svd_reference = Path(config['svd_reference_run']).resolve()
        sm, sc = verify_run(svd_reference)
        verify_svd(svd_reference, sm)
        require(sc == meta['model_config'] and sm['checkpoint_sha256'] == parent['checkpoint_sha256'], 'SVD reference model mismatch')
        require(sm['tokens_sha256'] == parent['tokens_sha256'], 'SVD reference tokens mismatch')
        require(sm['config'].get('measurement','sources') == config.get('measurement','sources'), 'SVD reference measurement mismatch')
        write_json(run/'inputs/svd_reference.json', dict(run=str(svd_reference),
            manifest_sha256=sha256(svd_reference/'manifest.json'), svd_index_sha256=sha256(svd_reference/'svd/index.json')))
    records = []
    for i in range(token_shape[0]):
        p = reference/'tokens'/f'chunk_{i:05d}.npz'
        rec = read_json(p.with_suffix('.json'))
        require(rec['sequence_id'] == i and rec['tokens_sha256'] == parent['tokens_sha256'], 'Reference record mismatch')
        require(rec['manifest_sha256'] == sha256(reference/'manifest.json'), 'Reference manifest mismatch')
        require(rec['sha256'] == sha256(p), 'Reference chunk changed')
        if model_type == 'gpt_base_attn_res_qkv':
            validation=reference/'validation'/p.with_suffix('.json').name
            require(sha256(validation)==rec['validation_sha256'] and read_json(validation)['passed'], 'Reference chunk validation failed')
        records.append(dict(sequence_id=i, path=str(p), sha256=rec['sha256']))
    write_json(run/'inputs/reference_chunks.json', records)
    runtime = run/'runtime_cache'
    hub = cache.parent
    env = dict(CUDA_VISIBLE_DEVICES=fields[1], NANOCHAT_BASE_DIR=parent['base_dir'], NANOCHAT_DTYPE='bfloat16',
        HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS=str(config['cpu_threads']), XDG_CACHE_HOME=str(runtime), HF_HOME=str(runtime/'huggingface'),
        HF_HUB_CACHE=str(hub), HUGGINGFACE_HUB_CACHE=str(hub), KERNELS_CACHE=str(hub), HF_KERNELS_CACHE=str(hub),
        TORCH_HOME=str(runtime/'torch'), CUDA_CACHE_PATH=str(runtime/'cuda'), TRITON_CACHE_DIR=str(runtime/'triton'))
    python = str(Path(sys.executable).absolute())
    repo = Path(__file__).resolve().parents[3]
    manifest = dict(schema_version=1,run_id=run_id,config=config,created_utc=stamp,
        reference_run=str(reference), reference_manifest_sha256=sha256(reference/'manifest.json'),
        checkpoint_path=str(checkpoint),checkpoint_sha256=parent['checkpoint_sha256'],checkpoint_step=meta['step'],
        tokens_shape=list(token_shape),tokens_sha256=parent['tokens_sha256'],
        input_hashes={str(p.relative_to(run)):sha256(p) for p in (run/'inputs').iterdir()},
        source_hashes={str(p.relative_to(run/'code')):sha256(p) for p in (run/'code').rglob('*') if p.is_file()},
        gpu_uuid=fields[1],selected_gpu=gpu_description,environment_overrides=env,python_executable=python,
        tokenizer_sha256=provenance['tokenizer_sha256'],kernel_cache_path=str(cache),kernel_cache_sha256=parent['kernel_cache_sha256'],
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        commands={stage:[python,'-m',f'{PACKAGE}.{stage}','--run-dir',str(run)] for stage in
                  (('reuse_svd','collect','summarize','compare_replay') if config.get('svd_reference_run') else ('compute_svd','collect','summarize'))},
        collection_definition=('per_token_actual_undivided_projection_input_energy_fraction_in_top_input_singular_subspace; captured_after_receiving_RMSNorm'
            if config.get('measurement','sources')=='full_input' else
            'per_token_full_source_energy_fraction_in_top_input_singular_subspace; no weighting or receiver RMS needed'),
        threshold_definition='cumulative_squared_singular_values_only; extend_cutoff_ties')
    write_json(run/'manifest.json',manifest)
    write_json(run/'status.json',dict(state='prepared',completed_chunks=0))
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--output-root',type=Path,default=Path('/local-ssd/mh3897/analysis_runs/d24_qkv_input_alignment'))
    p.add_argument('--gpu',type=int,required=True,help='Select a device identity for later collection; no GPU work or reservation here')
    p.add_argument('--run-id')
    args=p.parse_args()
    run=prepare(read_json(args.config),args.output_root,args.gpu,args.run_id)
    stages = ', then '.join(read_json(run/'manifest.json')['commands'])
    print(f'RUN_DIR={run}\nPrepared only. Run {stages} separately.')


if __name__=='__main__':
    main()
