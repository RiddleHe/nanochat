"""Freeze two model snapshots and identical tokens; prepare without launching GPU work."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .common import LAYERS, REFERENCE, VARIANT, hashes, read, sha, sources, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=LAYERS)
    a = p.parse_args()
    assert a.layers == sorted(set(a.layers)) and all(0 <= l < 24 for l in a.layers), 'Layers must be ordered unique indices 0–23'
    base = Path('/local-ssd/mh3897')
    ref = read(REFERENCE/'manifest.json')
    result = read(REFERENCE/'results.json')
    assert result['state'] == 'analysis_complete' and read(REFERENCE/'validation/summary.json')['passed']
    assert sha(REFERENCE/'summaries/statistics.npz') == result['statistics_sha256']
    assert sha(REFERENCE/'inputs/tokens.npy') == ref['tokens_sha256']
    assert np.load(REFERENCE/'inputs/tokens.npy').shape == (128, 2048)
    gpu = subprocess.check_output(['nvidia-smi', f'--id={a.gpu}', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in gpu.split(',')]
    assert int(fields[-2]) < 1000 and int(fields[-1]) <= 5, gpu
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = base/'analysis_runs/d24_no_embed_norm_v_comparison'/f'step7809_{stamp}'
    root.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    package = Path(__file__).parent
    configs = []
    for name, model_type, ckpt, code in [
        ('baseline', 'gpt_base', Path(ref['checkpoint_path']).parent, REFERENCE/'code/nanochat'),
        ('no_embed_norm', 'gpt_base_no_embed_norm', base/'base_checkpoints'/VARIANT, base/'run_queues'/VARIANT/'code/nanochat'),
    ]:
        run = root/name
        for d in ('inputs/checkpoint', 'code', 'tokens', 'validation', 'derived', 'logs', 'visualization'):
            (run/d).mkdir(parents=True, exist_ok=True)
        shutil.copytree(code, run/'code/nanochat', ignore=ignore)
        shutil.copytree(package, run/'code/scripts/inspect/direct_v_energy', ignore=ignore)
        for filename in ('model_007809.pt', 'meta_007809.json'):
            shutil.copy2(ckpt/filename, run/'inputs/checkpoint'/filename)
        meta = read(run/'inputs/checkpoint/meta_007809.json')
        assert meta['model_type'] == model_type and meta['step'] == 7809
        assert all(not v for v in meta['model_config'].values() if isinstance(v, bool))
        configs.append(meta['model_config'])
        if name == 'baseline':
            assert sha(run/'inputs/checkpoint/model_007809.pt') == ref['checkpoint_sha256']
            for file, digest in ref['source_hashes'].items():
                if file.startswith('nanochat/'):
                    assert sha(run/'code'/file) == digest
        for file in ('tokens.npy', 'provenance.json'):
            shutil.copy2(REFERENCE/'inputs'/file, run/'inputs'/file)
        shutil.copytree(base/'tokenizer', run/'inputs/runtime_base/tokenizer', ignore=ignore)
        provenance = read(run/'inputs/provenance.json')
        for file, digest in provenance['tokenizer_sha256'].items():
            assert sha(run/'inputs/runtime_base/tokenizer'/Path(file).name) == digest
        kernel = Path(ref['kernel_cache_path'])
        shutil.copytree(kernel, run/'kernel_cache'/kernel.name, ignore=ignore)
        for file, digest in ref['kernel_cache_sha256'].items():
            assert sha(run/'kernel_cache'/kernel.name/file) == digest
        write(run/'inputs/source_index.json', sources())
        config = dict(layers=a.layers, step=7809, source_batch_size=4, seed=42, model_type=model_type,
            validation=ref['config']['validation'], head_dimension=128, heads=12,
            energy='sum_squared_FP32_V_projection_of_source_divided_by_actual_complete_residual_denominator',
            directions='all; no SVD', std_ddof=0, baseline_norm_rtol=2e-5, baseline_norm_atol=1e-6)
        write(run/'inputs/config.json', config)
        env = dict(NANOCHAT_BASE_DIR=str(run/'inputs/runtime_base'), NANOCHAT_DTYPE='bfloat16', CUDA_VISIBLE_DEVICES=fields[1],
            PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            HF_HUB_CACHE=str(run/'kernel_cache'), HUGGINGFACE_HUB_CACHE=str(run/'kernel_cache'), KERNELS_CACHE=str(run/'kernel_cache'),
            HF_KERNELS_CACHE=str(run/'kernel_cache'), XDG_CACHE_HOME=str(run/'runtime_cache'), HF_HOME=str(run/'runtime_cache/huggingface'),
            TORCH_HOME=str(run/'runtime_cache/torch'), CUDA_CACHE_PATH=str(run/'runtime_cache/cuda'), TRITON_CACHE_DIR=str(run/'runtime_cache/triton'))
        (run/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
        write(run/'manifest.json', dict(run_id=root.name+'/'+name, config=config, gpu_uuid=fields[1], gpu=gpu,
            python=sys.executable, environment_overrides=env, model_code_source=str(code), checkpoint_source=str(ckpt),
            reference_run=str(REFERENCE), reference_manifest_sha256=sha(REFERENCE/'manifest.json'),
            reference_statistics_sha256=result['statistics_sha256'], tokens_sha256=ref['tokens_sha256'],
            packages_sha256=sha(run/'environment-packages.txt'),
            code_hashes=hashes(run/'code'), inputs_hashes=hashes(run/'inputs'), kernel_cache_hashes=hashes(run/'kernel_cache')))
        write(run/'status.json', dict(state='prepared'))
    assert configs[0] == configs[1]
    write(root/'manifest.json', dict(run_id=root.name, models=['baseline','no_embed_norm'], layers=a.layers,
        tokens_sha256=ref['tokens_sha256'], child_manifest_hashes={n:sha(root/n/'manifest.json') for n in ('baseline','no_embed_norm')}))
    print(root, flush=True)


if __name__ == '__main__':
    main()
