"""Weighted, separate source attribution to Q/K/V; collect and validate, never plot."""
import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.model.gpt_base import norm
from scripts.inspect.attn_res_qkv_routing.collect import run_analysis, now
from .prepare import sha256, write_json
from .summarize import summarize


def relative_rms(actual, expected):
    return float((actual.float()-expected.float()).square().mean().sqrt() /
                 expected.float().square().mean().sqrt().clamp_min(1e-12))


def check(value, limit, label):
    if not np.isfinite(value) or value > limit:
        raise ValueError(f'{label}: {value} exceeds {limit}')


@torch.inference_mode()
def measure(model, ids, config, reference):
    baseline = model(ids)
    assert torch.isfinite(baseline).all()
    original = model._attn_res
    arrays, checks, seen, pending, denominators = {}, [], {}, {}, []
    handles = []
    depth, limits = model.config.n_layer, config['validation']
    raw_norms = {}
    for layer, block in enumerate(model.transformer.h):
        for family, module in (('attention', block.attn), ('mlp', block.mlp)):
            def capture(module, args, output, layer=layer, family=family):
                seen[family, layer] = output
                raw_norms[2*layer+(1 if family == 'attention' else 2)] = output.float().norm(dim=-1).flatten().cpu()
            handles.append(module.register_forward_hook(capture))
        for kind in 'QKV':
            def projection_check(module, args, output, layer=layer, kind=kind):
                expected, actual_normalized, record = pending.pop((layer, kind))
                record['normalization_relative_rms'] = relative_rms(actual_normalized, args[0])
                record['projection_relative_rms'] = relative_rms(expected, output)
                check(record['normalization_relative_rms'], limits['normalization_relative_rms_max'], f'Normalization {layer} {kind}')
                check(record['projection_relative_rms'], limits['projection_relative_rms_max'], f'Projection {layer} {kind}')
            handles.append(getattr(block.attn, f'c_{kind.lower()}').register_forward_hook(projection_check))

    def observe(queries, sources):
        actual = original(queries, sources)
        if queries.shape[0] != 3:
            return actual
        layer = len(denominators)
        assert len(sources) == 2*layer+1
        assert torch.equal(queries, model.attn_res_queries[4*layer:4*layer+3])
        for previous in range(layer):
            assert sources[2*previous+1] is seen['attention', previous]
            assert sources[2*previous+2] is seen['mlp', previous]
        if layer == 0:
            raw_norms[0] = sources[0].float().norm(dim=-1).flatten().cpu()
        values = torch.stack(sources)
        assert values.dtype == torch.bfloat16
        keys = norm(values)
        scores = torch.einsum('rd,nbtd->rnbt', queries.to(keys.dtype), keys)
        probabilities = scores.float().softmax(dim=1)
        weights = probabilities.to(values.dtype)
        precast_error = float((probabilities.sum(1)-1).abs().max())
        applied_error = float((weights.float().sum(1)-1).abs().max())
        check(precast_error, limits['precast_weight_sum_max_error'], f'Softmax {layer}')
        check(applied_error, limits['applied_weight_sum_max_error'], f'Applied sum {layer}')
        stored = weights[:, :, 0, :].permute(2, 0, 1).float().cpu().numpy()
        assert np.array_equal(stored, reference[f'weights_{layer:02d}']), f'Parent routing differs: {layer}'
        assert torch.equal(torch.einsum('rnbt,nbtd->rbtd', weights, values), actual)
        denominator = (actual.float().square().mean(-1, keepdim=True) + torch.finfo(actual.dtype).eps).sqrt()
        denominators.append(denominator[:, 0, :, 0].T.cpu())
        block = model.transformer.h[layer]
        for qi, kind in enumerate('QKV'):
            weight = getattr(block.attn, f'c_{kind.lower()}').weight.to(actual.dtype).float()
            heads = block.attn.n_head if kind == 'Q' else block.attn.n_kv_head
            linear_mixture = torch.einsum('nbt,nbtd->btd', weights[qi].float(), values.float())
            mixture_error = relative_rms(linear_mixture, actual[qi])
            check(mixture_error, limits['fp32_mixture_relative_rms_max'], f'Mixture {layer} {kind}')
            projected_sum = torch.zeros_like(actual[qi], dtype=torch.float32)
            parts, input_parts = [], []
            full_error = 0.
            for start in range(0, len(sources), config['source_batch_size']):
                end = min(start+config['source_batch_size'], len(sources))
                component = values[start:end].float() * weights[qi, start:end].float().unsqueeze(-1) / denominator[qi]
                input_parts.append(component.norm(dim=-1).reshape(end-start, -1).T.cpu())
                projected = F.linear(component, weight)
                projected_sum += projected.sum(0)
                head_norms = projected.reshape(end-start, -1, heads, block.attn.head_dim).norm(dim=-1)
                full_error = max(full_error, relative_rms(head_norms.double().square().sum(-1).sqrt(), projected.norm(dim=-1).reshape(end-start, -1)))
                parts.append(head_norms.permute(1, 2, 0).cpu())
            absolute = torch.cat(parts, -1).numpy()
            total = absolute.sum(-1, keepdims=True)
            assert np.isfinite(absolute).all() and (total > 0).all()
            share = absolute / total
            share_error = float(np.abs(share.sum(-1)-1).max())
            linear_error = relative_rms(projected_sum, F.linear(linear_mixture / denominator[qi], weight))
            check(linear_error, limits['linear_relative_rms_max'], f'Linearity {layer} {kind}')
            check(full_error, limits['full_norm_relative_rms_max'], f'Full norm {layer} {kind}')
            check(share_error, limits['share_sum_max_error'], f'Share {layer} {kind}')
            arrays[f'abs_{kind}_{layer:02d}'] = absolute
            arrays[f'share_{kind}_{layer:02d}'] = share
            arrays[f'input_{kind}_{layer:02d}'] = torch.cat(input_parts, -1).numpy()
            record = dict(layer=layer, kind=kind, source_identity_passed=True, exact_parent_weights=True,
                          exact_bf16_mixture_replay=True, precast_weight_sum_error=precast_error,
                          applied_weight_sum_error=applied_error, mixture_relative_rms=mixture_error,
                          linear_relative_rms=linear_error, full_norm_relative_rms=full_error, share_sum_error=share_error)
            checks.append(record)
            pending[layer, kind] = projected_sum, actual[qi].float()/denominator[qi], record
        return actual

    model._attn_res = observe
    try:
        observed = model(ids)
        assert torch.equal(observed, baseline), 'Instrumentation changed logits'
    finally:
        del model._attn_res
        for handle in handles:
            handle.remove()
    assert not pending and len(denominators) == depth and len(raw_norms) == 2*depth+1
    arrays['source_norm'] = torch.stack([raw_norms[i] for i in range(2*depth+1)], -1).numpy()
    arrays['input_denominator'] = torch.stack(denominators, 1).numpy()
    return arrays, dict(passed=True, exact_forward_replay=True, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    run = parser.parse_args().run_dir
    try:
        manifest = json.loads((run/'manifest.json').read_text())
        ref_path = run/'inputs/routing_reference.json'
        assert sha256(ref_path) == manifest['routing_reference_sha256']
        reference = json.loads(ref_path.read_text())
        parent = Path(reference['path'])
        assert sha256(parent/'manifest.json') == reference['manifest_sha256']
        def measured(model, ids, config, index):
            path = parent/'tokens'/f'chunk_{index:05d}.npz'
            record = reference['records'][index]
            assert record['sequence_id'] == index and record['tokens_sha256'] == manifest['tokens_sha256']
            assert sha256(path) == record['sha256']
            with np.load(path) as chunk:
                return measure(model, ids, config, chunk)
        run_analysis(run, measure_fn=measured, summarize_fn=summarize)
    except BaseException as error:
        write_json(run/'status.json', dict(state='failed', error=repr(error), traceback=traceback.format_exc(), updated_utc=now()))
        raise


if __name__ == '__main__':
    main()
