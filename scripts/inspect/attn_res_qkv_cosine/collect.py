"""Observe source/query cosines without altering BF16 routing or forward outputs."""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from nanochat.model.gpt_base import norm
from scripts.inspect.attn_res_qkv_routing.collect import run_analysis, now
from .prepare import sha256, write_json
from .summarize import summarize


def cosine_and_scores(queries, values, keys):
    """Analyze forward-cast tensors in FP32; preserve undefined zero-norm cases."""
    q, s, k = queries.float(), values.float(), keys.float()
    qn, sn, kn = q.norm(dim=-1), s.norm(dim=-1), k.norm(dim=-1)
    dot_s = torch.einsum('rd,nbtd->rnbt', q, s)
    dot_k = torch.einsum('rd,nbtd->rnbt', q, k)
    denom_s = qn[:, None, None, None] * sn[None]
    denom_k = qn[:, None, None, None] * kn[None]
    cosine = torch.where(denom_s > 0, dot_s / denom_s, torch.nan)
    key_cosine = torch.where(denom_k > 0, dot_k / denom_k, torch.nan)
    return cosine, key_cosine, dot_k, qn, sn, kn


@torch.inference_mode()
def measure(model, ids, config, index, reference):
    record_path = reference / 'tokens' / f'chunk_{index:05d}.npz'
    record = json.loads(record_path.with_suffix('.json').read_text())
    assert sha256(record_path) == record['sha256']
    assert record['manifest_sha256'] == sha256(reference / 'manifest.json')
    ref = np.load(record_path)
    baseline = model(ids)
    assert torch.isfinite(baseline).all()
    original = model._attn_res
    arrays, checks, seen = {}, [], {}
    handles = []
    for layer, block in enumerate(model.transformer.h):
        for family, module in (('attention', block.attn), ('mlp', block.mlp)):
            def capture(module, args, output, layer=layer, family=family):
                seen[family, layer] = output
            handles.append(module.register_forward_hook(capture))

    def observe(queries, sources):
        actual = original(queries, sources)
        if queries.shape[0] != 3:
            return actual
        layer = len(checks)
        assert len(sources) == 2*layer+1
        assert torch.equal(queries, model.attn_res_queries[4*layer:4*layer+3])
        for previous in range(layer):
            assert sources[2*previous+1] is seen['attention', previous]
            assert sources[2*previous+2] is seen['mlp', previous]
        values = torch.stack(sources)
        keys = norm(values)
        q = queries.to(keys.dtype)
        assert values.dtype == torch.bfloat16 and q.dtype == torch.bfloat16
        scores = torch.einsum('rd,nbtd->rnbt', q, keys)
        weights = scores.float().softmax(dim=1).to(values.dtype)
        replay = torch.einsum('rnbt,nbtd->rbtd', weights, values)
        assert torch.equal(replay, actual)
        saved_weights = weights[:, :, 0, :].permute(2, 0, 1).float().cpu().numpy()
        assert np.array_equal(saved_weights, ref[f'weights_{layer:02d}'])
        cosine, kc, fp32_scores, qn, sn, kn = cosine_and_scores(q, values, keys)
        den = qn[:, None, None, None] * kn[None]
        valid = den > 0
        assert torch.equal(torch.isfinite(cosine), (qn[:, None, None, None] * sn[None]) > 0)
        assert torch.equal(torch.isfinite(kc), valid)
        limits = config['validation']
        for c in (cosine, kc):
            finite = c[torch.isfinite(c)]
            assert not finite.numel() or finite.abs().max() <= 1 + limits['cosine_bound_tolerance']
        identity = ((kc * den - fp32_scores).abs()[valid] / den[valid]).max().item() if valid.any() else 0.
        rounding = ((scores.float()-fp32_scores).abs() / den.clamp_min(1e-30)).max().item()
        difference = (cosine-kc).abs()[valid].max().item() if valid.any() else 0.
        assert identity <= limits['score_identity_normalized_max']
        assert rounding <= limits['bf16_score_normalized_max']
        assert difference <= limits['key_source_cosine_difference_max']
        # Check the ideal, epsilon-aware RMS scaling separately from BF16 keys.
        eps = torch.finfo(values.dtype).eps
        ideal_kn = sn / (sn.square()/values.shape[-1]+eps).sqrt()
        norm_error = ((kn-ideal_kn).abs()/ideal_kn.clamp_min(1e-30)).max().item()
        assert norm_error < 0.02
        for name, tensor in [('cosine',cosine), ('key_cosine',kc), ('scores',scores.float())]:
            arrays[f'{name}_{layer:02d}'] = tensor[:, :, 0, :].permute(2,0,1).cpu().numpy()
        arrays[f'query_norm_{layer:02d}'] = qn.cpu().numpy()
        arrays[f'source_norm_{layer:02d}'] = sn[:,0,:].T.cpu().numpy()
        arrays[f'key_norm_{layer:02d}'] = kn[:,0,:].T.cpu().numpy()
        checks.append(dict(layer=layer, source_identity_passed=True, exact_reference_weights=True,
            exact_bf16_mixture_replay=True, score_identity_normalized_error=identity,
            bf16_score_normalized_error=rounding, key_source_cosine_difference=difference,
            rms_key_norm_relative_error=norm_error, rms_epsilon=eps,
            zero_query_indices=torch.where(qn==0)[0].cpu().tolist()))
        return actual

    model._attn_res = observe
    try:
        observed = model(ids)
        assert torch.equal(observed, baseline), 'Instrumentation changed logits'
    finally:
        del model._attn_res
        ref.close()
        for handle in handles:
            handle.remove()
    assert len(checks) == model.config.n_layer
    return arrays, dict(passed=True, exact_forward_replay=True, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run/'manifest.json').read_text())
    frozen = run/'code/scripts/inspect/attn_res_qkv_cosine/collect.py'
    if Path(__file__).resolve() != frozen:
        env = dict(os.environ, **manifest['environment_overrides'])
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'PYTHONPATH'):
            env.pop(key, None)
        os.chdir(manifest['cwd'])
        os.execve(manifest['command'][0], manifest['command'], env)
    reference = Path(manifest['routing_reference'])
    assert sha256(reference/'manifest.json') == manifest['routing_manifest_sha256']
    try:
        run_analysis(run, measure_fn=lambda m,t,c,i: measure(m,t,c,i,reference), summarize_fn=summarize)
        results = json.loads((run/'results.json').read_text())
        results['visualization_status'] = 'requested'
        write_json(run/'results.json',results)
    except BaseException as error:
        write_json(run/'status.json',dict(state='failed',error=repr(error),traceback=traceback.format_exc(),updated_utc=now()))
        raise


if __name__ == '__main__':
    main()
