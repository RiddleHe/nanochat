"""Probe boundaries and data semantics on a real small ScalarRes model."""
from dataclasses import asdict
import json
import numpy as np
import torch
import pytest
import nanochat.model.gpt_base_scalar_res_qkv_no_embed_norm as scalar_model
from nanochat.model.gpt_base_scalar_res_qkv_no_embed_norm import (
    GPTBaseScalarResQKVNoEmbedNorm, GPTBaseScalarResQKVNoEmbedNormConfig)
from scripts.inspect.qkv_input_alignment.scalar_res import (
    MODEL_TYPE, capture_sources, independent_norms, extract_coefficients)
from scripts.inspect.qkv_input_alignment.collect import measure
from scripts.inspect.qkv_input_alignment.compute_svd import decompose
from scripts.inspect.raw_activation_norms.collect import NormCapture


@pytest.fixture(autouse=True)
def bf16_model(monkeypatch):
    monkeypatch.setattr(scalar_model, 'COMPUTE_DTYPE', torch.bfloat16)


def model_and_ids():
    torch.manual_seed(19)
    cfg = GPTBaseScalarResQKVNoEmbedNormConfig(sequence_len=4, vocab_size=64,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=16, window_pattern='L')
    model = GPTBaseScalarResQKVNoEmbedNorm(cfg)
    model.init_weights()
    with torch.no_grad():
        model.scalar_res_logits_flat[:model.scalar_res_active].copy_(
            torch.linspace(-5, 5, model.scalar_res_active))
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
    return model.eval(), torch.randint(0, 64, (2, 4))


@torch.inference_mode()
def test_scalar_bank_matches_uninstrumented_forward_and_independent_norms():
    model, ids = model_and_ids()
    original = model(ids)
    expected_logits, expected_norms = independent_norms(model, ids)
    sources, actual = capture_sources(model, ids)
    assert torch.equal(original, expected_logits) and torch.equal(original, actual)
    assert len(sources) == 5 and all(s.shape == (8, 16) for s in sources)
    assert torch.equal(sources[0], model.transformer.wte(ids).reshape(-1, 16))
    np.testing.assert_array_equal(np.stack([s.float().norm(dim=-1).numpy() for s in sources], -1), expected_norms)
    assert '_scalar_res' not in model.__dict__
    assert all(not m._forward_hooks for m in model.modules())


@torch.inference_mode()
def test_full_norm_capture_preserves_batch_for_tensor_attention():
    model, ids = model_and_ids()
    _, expected = independent_norms(model, ids)
    capture = NormCapture(model)
    try:
        model(ids)
        norms = capture.stack()
        assert norms.shape == (2, 4, 5)
        np.testing.assert_array_equal(norms.reshape(8, 5).numpy(), expected)
        assert all(v is None for v in capture.values)
    finally:
        capture.close()


@torch.inference_mode()
def test_scalar_alignment_and_coefficient_artifacts(tmp_path):
    model, ids = model_and_ids()
    cfg = json.loads(open('scripts/inspect/qkv_input_alignment/scalar_res_config.json').read())
    cfg.update(layers=[1], kinds=['K','V'])
    banks = {}
    for kind in cfg['kinds']:
        weight = getattr(model.transformer.h[1].attn, 'c_'+kind.lower()).weight.bfloat16()
        pieces = [decompose(head, [0.5])[0] for head in weight.reshape(2, 8, 16)]
        banks[1,kind] = {k:np.stack([p[k] for p in pieces]) for k in pieces[0]}
    arrays, checks = measure(model, ids, banks, asdict(model.config), cfg, None)
    assert checks['independent_module_norms_match'] and checks['scalar_receiver_order_verified']
    assert arrays['alignment_K_01'].shape == (8, 2, 3, 1)
    assert np.isfinite(arrays['alignment_V_01']).all()
    (tmp_path/'summaries').mkdir(); (tmp_path/'validation').mkdir()
    (tmp_path/'manifest.json').write_text('{}')
    extract_coefficients(model, tmp_path)
    with np.load(tmp_path/'summaries/scalar_coefficients.npz') as data:
        assert len(data['logits']) == model.scalar_res_active
        np.testing.assert_array_equal(data['coefficients_fp32'], model._sqrt_gate(model.scalar_res_logits_flat[:model.scalar_res_active]).numpy())
        assert data['coefficients_fp32'].max() > 1
    schema = json.loads((tmp_path/'summaries/scalar_coefficients.json').read_text())
    assert schema['model_type'] == MODEL_TYPE and not schema['normalized'] and not schema['token_dependent']
    assert len(schema['receivers']) == 9 and schema['receivers'][-1]['kind'] == 'final'
