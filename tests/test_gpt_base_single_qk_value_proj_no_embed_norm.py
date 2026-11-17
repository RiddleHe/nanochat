"""Projected single-head attention against dense math, live gradients, cache and FLOPs."""
from dataclasses import asdict
import copy

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model.gpt_base_single_qk_value_proj_no_embed_norm import (
    GPTBaseSingleQKValueProjNoEmbedNormConfig as Config,
    GPTBaseSingleQKValueProjNoEmbedNorm as Model,
)
from nanochat.model_registry import get_model


@pytest.fixture(autouse=True)
def cpu_float32(monkeypatch):
    monkeypatch.setattr(flash, "USE_FA3", False)
    monkeypatch.setattr(base, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(no_norm, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)


def make_model():
    cfg = Config(sequence_len=16, vocab_size=31, n_layer=3, n_head=2,
                 n_kv_head=2, n_embd=16, window_pattern="L", attention_chunk_size=2, norm_value_input=True)
    with torch.device("meta"):
        model = Model(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():
        for b in model.transformer.h:
            b.attn.c_proj.weight.normal_(std=.1)
            b.mlp.c_proj.weight.normal_(std=.1)
    return model


@pytest.mark.parametrize("window", [(-1, 0), (2, 0), (0, 0)])
@pytest.mark.parametrize("training", [False, True])
def test_dense_reference_and_gradients(window, training):
    model = make_model()
    attn = model.transformer.h[-1].attn
    attn.window_size = window
    attn.train(training)
    with torch.no_grad():
        attn.a_scale.fill_(-1.7)
    ref = copy.deepcopy(attn)
    x = torch.randn(2, 7, 16, requires_grad=True)
    emb = torch.randn_like(x, requires_grad=True)
    xr, er = x.detach().clone().requires_grad_(), emb.detach().clone().requires_grad_()
    cs = model.cos[:, :7], model.sin[:, :7]
    actual, _ = attn(x, emb, None, cs, None)
    q = base.norm(base.apply_rotary_emb(ref.c_q(xr).unsqueeze(2), *cs)).squeeze(2) * 1.2
    k = base.norm(base.apply_rotary_emb(ref.c_k(xr).unsqueeze(2), *cs)).squeeze(2) * 1.2
    distance = torch.arange(7)[:, None] - torch.arange(7)[None, :]
    mask = distance >= 0
    if window[0] >= 0:
        mask &= distance <= window[0]
    probs = ((q @ k.transpose(-1, -2)) / 8**.5).masked_fill(~mask, -torch.inf).softmax(-1)
    expected = ref.c_proj(probs @ ref.c_v(er)) * ref.a_scale
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a,b in [(x.grad,xr.grad),(emb.grad,er.grad)]:
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    for p,pr in zip(attn.parameters(),ref.parameters()):
        torch.testing.assert_close(p.grad,pr.grad,atol=4e-6,rtol=4e-5)
    assert emb.grad.abs().sum() > 0


def test_registry_shapes_optimizer_and_serialization():
    model = make_model()
    assert get_model('gpt_base_single_qk_value_proj_no_embed_norm') == (Config, Model)
    assert asdict(Config(**asdict(model.config))) == asdict(model.config)
    for i,b in enumerate(model.transformer.h):
        if i in model.config.pooled_qk_target_layers:
            assert b.attn.c_q.weight.shape == b.attn.c_k.weight.shape == (8,16)
            assert b.attn.c_proj.weight.shape == (16,8)
            assert b.attn.c_v.weight.shape == (8,16)
            assert not hasattr(b.attn,'head_scale')
            assert b.attn.a_scale.dtype == torch.float32
            assert b.attn.a_scale.shape == (1,)
            assert b.attn.a_scale.item() == 1.0
        else:
            assert b.attn.c_v.weight.shape == (16,16)
            assert not hasattr(b.attn, 'a_scale')
    opt = model.setup_optimizer()
    ps = [p for g in opt.param_groups for p in g['params']]
    assert len(ps) == len(set(map(id,ps)))
    assert set(map(id,ps)) == set(map(id,model.parameters()))
    scales = [b.attn.a_scale for b in model.transformer.h if hasattr(b.attn, 'a_scale')]
    for scale in scales:
        group = next(g for g in opt.param_groups if any(p is scale for p in g['params']))
        assert group['kind'] == 'adamw' and group['weight_decay'] == 0
        assert group['betas'] == (0.96, 0.95)
    with torch.no_grad():
        for scale in scales:
            scale.fill_(2.3)
    restored = Model(Config(**asdict(model.config)))
    restored.load_state_dict(model.state_dict())
    tokens = torch.randint(31,(2,7))
    torch.testing.assert_close(restored(tokens),model(tokens))


@pytest.mark.parametrize('depth,heads,width',[(3,2,16),(24,12,1536)])
def test_flops(depth,heads,width):
    opts = dict(sequence_len=2048,vocab_size=32768,n_layer=depth,n_head=heads,
                n_kv_head=heads,n_embd=width,window_pattern='SSSL')
    with torch.device('meta'):
        baseline = no_norm.GPTBaseNoEmbedNorm(base.GPTBaseConfig(**opts))
        model = Model(Config(**opts))
    delta = 0
    d = width // heads
    for i in model.config.pooled_qk_target_layers:
        context = min(2048,model.window_sizes[i][0])
        # Shrink all four projections and both score and value products.
        delta += -8*width*(width-d) - 4*(width-d)*context
    scale_width = width * len(model.config.pooled_qk_target_layers)
    assert model.estimate_forward_flops() == baseline.estimate_flops()//3 + delta + scale_width
    assert model.estimate_flops() == baseline.estimate_flops() + 3*delta + 4*scale_width
    if depth == 24:
        assert model.estimate_flops() == 4031318016
        assert model.estimate_forward_flops() == 1343767040
        assert model.num_scaling_params()['total'] == 684982283
        assert model.num_scaling_params()['scalars'] == 11
        assert round(3.91e19 / (model.estimate_flops() * 1048576)) == 9250


def test_compiled_training_and_cache():
    model = make_model()
    for b in model.transformer.h:
        b.attn.window_size = (2,0)
        if hasattr(b.attn, 'a_scale'):
            with torch.no_grad():
                b.attn.a_scale.fill_(-0.7)
    tokens = torch.randint(31,(1,9))
    compiled = torch.compile(model,backend='aot_eager',fullgraph=True)
    loss = compiled(tokens,tokens)
    torch.testing.assert_close(loss,model(tokens,tokens))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.transformer.wte.weight.grad.abs().sum() > 0
    model.eval()
    cache = KVCache(1,2,16,8,3,'cpu',torch.float32)
    expanded = KVCache(2,2,16,8,3,'cpu',torch.float32)
    with torch.no_grad():
        expected = model(tokens)
        prefix = model(tokens[:,:3],kv_cache=cache)
        expanded.prefill(cache)
        torch.testing.assert_close(model(tokens[:,3:].expand(2,-1),kv_cache=expanded),
            expected[:,3:].expand(2,-1,-1),atol=1e-6,rtol=2e-5)
        suffix = model(tokens[:,3:],kv_cache=cache)
        torch.testing.assert_close(torch.cat([prefix,suffix],1),expected,atol=1e-6,rtol=2e-5)
        cache.reset()
        torch.testing.assert_close(model(tokens[:,:2],kv_cache=cache),expected[:,:2],atol=1e-6,rtol=2e-5)
