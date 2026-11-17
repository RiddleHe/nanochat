"""Observe ScalarRes banks and extract static gates without changing model arithmetic."""
import numpy as np
import torch
from .common import require, sha256, source_index, write_json, write_npz

MODEL_TYPE = 'gpt_base_scalar_res_qkv_no_embed_norm'


def independent_norms(model, ids):
    """Module-output norms from an independent, unmodified forward pass."""
    values = {}
    handles = []
    modules = [(0, model.transformer.wte)]
    for layer, block in enumerate(model.transformer.h):
        modules.extend([(2*layer+1, block.attn), (2*layer+2, block.mlp)])
    for index, module in modules:
        def observe(module, args, output, index=index):
            require(index not in values, 'Repeated source module')
            require(isinstance(output, torch.Tensor), 'ScalarRes source must be a tensor')
            values[index] = output.detach().float().norm(dim=-1).reshape(-1)
        handles.append(module.register_forward_hook(observe))
    try:
        logits = model(ids)
    finally:
        for handle in handles:
            handle.remove()
    require(len(values) == len(modules), 'Missing independent norm hooks')
    return logits, torch.stack([values[i] for i in range(len(modules))], -1).cpu().numpy()


def capture_sources(model, ids):
    """Check every receiver's bank identity and coefficient ordering in the real forward."""
    seen, captured, handles = {}, {}, []
    modules = [(0, model.transformer.wte)]
    for layer, block in enumerate(model.transformer.h):
        modules.extend([(2*layer+1, block.attn), (2*layer+2, block.mlp)])
    for index, module in modules:
        def observe(module, args, output, index=index):
            require(index not in seen, 'Repeated source module')
            seen[index] = output
            captured[index] = output.detach().clone()
        handles.append(module.register_forward_hook(observe))
    original = model._scalar_res
    had_override = '_scalar_res' in model.__dict__
    receiver = 0
    gates = model._sqrt_gate(model.scalar_res_logits_flat[:model.scalar_res_active])

    def observe_mix(coefficients, sources):
        nonlocal receiver
        rows, count = coefficients.shape
        require(rows == (3 if receiver < 4*len(model.transformer.h) and receiver % 4 == 0 else 1),
                'Unexpected scalar receiver order')
        require(len(sources) == count and all(s is seen[i] for i, s in enumerate(sources)),
                'Scalar source-bank identity mismatch')
        for row in range(rows):
            lo, hi = model.scalar_res_offsets[receiver:receiver+2]
            require(hi-lo == count and torch.equal(coefficients[row], gates[lo:hi]),
                    'Scalar receiver coefficient mismatch')
            receiver += 1
        return original(coefficients, sources)

    model._scalar_res = observe_mix
    try:
        logits = model(ids)
    finally:
        if had_override:
            model._scalar_res = original
        else:
            del model._scalar_res
        for handle in handles:
            handle.remove()
    require(receiver == 4*len(model.transformer.h)+1, 'Missing scalar receivers')
    require(len(captured) == len(modules), 'Missing scalar sources')
    return [captured[i].reshape(-1, captured[i].shape[-1]) for i in range(len(modules))], logits


@torch.inference_mode()
def extract_coefficients(model, run):
    """Save all receivers, including MLP/final, so later plots only select data."""
    active = model.scalar_res_active
    z = model.scalar_res_logits_flat[:active].detach()
    g = model._sqrt_gate(z)
    ref_z = z.double()
    root = torch.sqrt(1 + ref_z.square())
    reference = torch.where(ref_z >= 0, root+ref_z, (root-ref_z.clamp_max(0)).reciprocal())
    torch.testing.assert_close(g.double(), reference, rtol=3e-6, atol=1e-8)
    require(bool(torch.isfinite(g).all() and (g > 0).all()), 'Invalid scalar coefficients')
    require(torch.count_nonzero(model.scalar_res_logits_flat[active:]).item() == 0, 'Nonzero padding')
    receivers = []
    for layer in range(len(model.transformer.h)):
        receivers.extend(dict(kind=kind, layer=layer) for kind in ('Q', 'K', 'V', 'MLP'))
    receivers.append(dict(kind='final', layer=None))
    for i, row in enumerate(receivers):
        lo, hi = model.scalar_res_offsets[i:i+2]
        row.update(receiver_index=i, start=lo, stop=hi, available_sources=hi-lo)
    output = run/'summaries/scalar_coefficients.npz'
    write_npz(output, dict(logits=z.cpu().numpy(), coefficients_fp32=g.cpu().numpy(),
                          coefficients_bf16_as_fp32=g.to(torch.bfloat16).float().cpu().numpy(),
                          offsets=np.asarray(model.scalar_res_offsets, dtype=np.int64)))
    write_json(run/'summaries/scalar_coefficients.json', dict(
        model_type=MODEL_TYPE, manifest_sha256=sha256(run/'manifest.json'),
        statistics_sha256=sha256(output), receivers=receivers,
        source_index=source_index(len(model.transformer.h)),
        definition='g(z)=z+sqrt(1+z^2); FP32 master gate and actual BF16 forward coefficient',
        token_dependent=False, head_dependent=False, normalized=False,
        embedding='raw embedding; no initial RMSNorm', visualization_status='not_rendered'))
    write_json(run/'validation/scalar_coefficients.json', dict(passed=True,
        manifest_sha256=sha256(run/'manifest.json'), statistics_sha256=sha256(output),
        active_logits=active, padding_entries=model.scalar_res_logits_flat.numel()-active,
        float64_reference_max_abs_error=float((g.double()-reference).abs().max()),
        padding_zero=True, min_coefficient=float(g.min()), max_coefficient=float(g.max())))
