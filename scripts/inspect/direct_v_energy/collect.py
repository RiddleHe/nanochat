"""Direct per-token V energies with unchanged-forward and reconstruction checks."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from nanochat.checkpoint_manager import build_model
from nanochat.flash_attention import USE_FA3
from nanochat.model.gpt_base import norm
from .common import read, sha, verify, write


def relative(a, b):
    return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))


def capture(model, ids, layers):
    values = dict(inputs={}, attention={}, mlp={}, projection={}, projection_input={})
    handles = []
    for l, block in enumerate(model.transformer.h):
        def before(module, args, l=l): values['inputs'][l] = args[0].detach().clone()
        def attn(module, args, out, l=l): values['attention'][l] = out[0].detach().clone()
        def mlp(module, args, out, l=l): values['mlp'][l] = out.detach().clone()
        handles.extend([block.register_forward_pre_hook(before), block.attn.register_forward_hook(attn), block.mlp.register_forward_hook(mlp)])
        if l in layers:
            def v(module, args, out, l=l):
                values['projection'][l] = out.detach().clone()
                values['projection_input'][l] = args[0].detach().clone()
            handles.append(block.attn.c_v.register_forward_hook(v))
    try:
        logits = model(ids)
    finally:
        for h in handles: h.remove()
    return values, logits


@torch.inference_mode()
def measure(model, ids, config):
    unhooked = model(ids)
    c, hooked = capture(model, ids, config['layers'])
    assert torch.equal(unhooked, hooked) and torch.isfinite(hooked).all(), 'Forward changed'
    del unhooked, hooked
    direct = model.transformer.wte(ids).to(torch.bfloat16)
    if config['model_type'] == 'gpt_base': direct = norm(direct)
    assert torch.equal(c['inputs'][0], direct), 'Embedding boundary mismatch'
    bank = [c['inputs'][0].float()]
    for l in range(24): bank.extend([c['attention'][l].float(), c['mlp'][l].float()])
    arrays = dict(source_norm=torch.stack([s.norm(dim=-1).flatten() for s in bank], -1).cpu().numpy())
    checks = []
    for l in config['layers']:
        actual = c['inputs'][l]
        assert actual.dtype == torch.bfloat16
        assert torch.equal(norm(actual), c['projection_input'][l]), 'Receiving input boundary mismatch'
        eps = torch.finfo(actual.dtype).eps
        denominator = (actual.float().square().mean(-1, keepdim=True)+eps).sqrt()
        available = bank[:2*l+1]
        total = torch.stack(available).sum(0)
        weight = model.transformer.h[l].attn.c_v.weight.to(actual.dtype).float()
        projected_sum = torch.zeros_like(c['projection'][l], dtype=torch.float32)
        energies, norms = [], []
        full_error = 0.
        for start in range(0, len(available), config['source_batch_size']):
            group = torch.stack(available[start:start+config['source_batch_size']])
            projected = F.linear(group/denominator, weight)
            projected_sum += projected.sum(0)
            shaped = projected.reshape(len(group), -1, 12, 128)
            energy = shaped.square().sum(-1)
            full_error = max(full_error, relative(energy.double().sum(-1), projected.double().square().sum(-1).reshape(len(group), -1)))
            energies.append(energy.permute(1, 2, 0).cpu().numpy())
            norms.append(shaped.norm(dim=-1).permute(1, 2, 0).cpu().numpy())
        energy = np.concatenate(energies, axis=-1)
        assert np.isfinite(energy).all() and (energy >= 0).all()
        arrays[f'energy_V_{l:02d}'] = energy
        arrays[f'norm_V_{l:02d}'] = np.concatenate(norms, axis=-1)
        arrays[f'denominator_{l:02d}'] = denominator.flatten().cpu().numpy()
        check = dict(layer=l, rmsnorm_epsilon=eps,
            residual_relative_rms=relative(total, actual),
            normalization_relative_rms=relative(actual.float()/denominator, c['projection_input'][l]),
            linear_reconstruction_relative_rms=relative(projected_sum, F.linear(total/denominator, weight)),
            projection_relative_rms=relative(projected_sum, c['projection'][l]),
            forward_rounding_relative_rms=relative(F.linear(actual.float()/denominator, weight), c['projection'][l]),
            full_norm_relative_rms=full_error)
        for metric in ('residual_relative_rms', 'normalization_relative_rms', 'linear_reconstruction_relative_rms', 'projection_relative_rms', 'full_norm_relative_rms'):
            assert np.isfinite(check[metric]) and check[metric] <= config['validation'][metric+'_max'], check
        checks.append(check)
    return arrays, dict(passed=True, unchanged_logits=True, embedding_identity=True, actual_V_input_identity=True, checks=checks)


def collect(run):
    m = verify(run)
    assert Path(__file__).resolve().parent == run/'code/scripts/inspect/direct_v_energy'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == m['gpu_uuid'] and torch.cuda.device_count() == 1 and USE_FA3
    packages = '\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n'
    assert (run/'environment-packages.txt').read_text() == packages
    torch.manual_seed(42); torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), gpu_uuid=m['gpu_uuid'], use_fa3=USE_FA3, tf32=False)
    if (run/'environment.json').exists(): assert read(run/'environment.json') == environment
    else: write(run/'environment.json', environment)
    model, _, meta = build_model(str(run/'inputs/checkpoint'), 7809, torch.device('cuda:0'), 'eval')
    model.requires_grad_(False)
    assert meta['model_type'] == m['config']['model_type']
    tokens = np.load(run/'inputs/tokens.npy'); assert tokens.shape == (128, 2048)
    ref = Path(m['reference_run']); assert sha(ref/'manifest.json') == m['reference_manifest_sha256']
    mh = sha(run/'manifest.json'); start = time.monotonic()
    for i, sequence in enumerate(tokens):
        path = run/'tokens'/f'chunk_{i:05d}.npz'
        if path.with_suffix('.json').exists():
            record = read(path.with_suffix('.json'))
            assert record['sha256'] == sha(path) and record['manifest_sha256'] == mh and record['sequence_id'] == i
            assert record['validation_sha256'] == sha(run/'validation'/f'chunk_{i:05d}.json')
            continue
        ids = torch.from_numpy(sequence.copy()).unsqueeze(0).cuda()
        arrays, validation = measure(model, ids, m['config'])
        if m['config']['model_type'] == 'gpt_base':
            old = ref/'tokens'/path.name; rec = read(old.with_suffix('.json'))
            assert sha(old) == rec['sha256'] and rec['manifest_sha256'] == m['reference_manifest_sha256'] and rec['tokens_sha256'] == m['tokens_sha256']
            with np.load(old) as z:
                np.testing.assert_allclose(arrays['source_norm'], z['source_norm'], rtol=2e-5, atol=1e-6)
                for l in m['config']['layers']:
                    np.testing.assert_allclose(arrays[f'norm_V_{l:02d}'], z[f'abs_V_{l:02d}'], rtol=2e-5, atol=1e-6)
            validation['baseline_reference_matched'] = True
            validation['reference_chunk_sha256'] = rec['sha256']
        vp = run/'validation'/f'chunk_{i:05d}.json'; write(vp, validation)
        tmp = path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f: np.savez(f, **arrays)
        tmp.replace(path)
        write(path.with_suffix('.json'), dict(sequence_id=i, tokens=2048, sha256=sha(path), manifest_sha256=mh,
            tokens_sha256=m['tokens_sha256'], validation_sha256=sha(vp)))
        elapsed = time.monotonic()-start
        write(run/'status.json', dict(state='collecting', completed_sequences=i+1, total_sequences=128, seconds=elapsed))
        if (i+1)%8 == 0: print(f'{i+1}/128 sequences; {elapsed:.1f}s', flush=True)
    write(run/'collection.json', dict(state='complete', sequences=128, tokens=262144, manifest_sha256=mh,
        seconds=time.monotonic()-start, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
    write(run/'status.json', dict(state='collected', tokens=262144))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', type=Path, required=True)
    run = p.parse_args().run_dir.resolve()
    try: collect(run)
    except BaseException as e:
        write(run/'status.json', dict(state='failed', error=repr(e), traceback=traceback.format_exc())); raise


if __name__ == '__main__': main()
