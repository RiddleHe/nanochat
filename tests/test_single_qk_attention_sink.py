"""Learned zero-value sink: equations, masks, gradients, cache and budget."""
import copy
from dataclasses import asdict
import json

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model.gpt_base_single_qk_embed_no_embed_norm import (
    GPTBaseSingleQKEmbedNoEmbedNormConfig as Config,
    GPTBaseSingleQKEmbedNoEmbedNorm as Model,
)
from nanochat.model_registry import get_model


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(flash, 'USE_FA3', False)
    monkeypatch.setattr(base, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(no_norm, 'COMPUTE_DTYPE', torch.float32)
    torch.set_num_threads(1)
    torch.manual_seed(42)


def make_model(*, sink=True, targets=(1, 2), open_output=True):
    cfg = Config(n_layer=3, n_head=2, n_kv_head=2, n_embd=16,
                 sequence_len=16, vocab_size=31, attention_chunk_size=2,
                 window_pattern='L', norm_value_input=True,
                 attention_sink=sink, pooled_qk_target_layers=targets)
    with torch.device('meta'):
        model = Model(cfg)
    model.to_empty(device='cpu')
    model.init_weights()
    if open_output:
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=.1)
                block.mlp.c_proj.weight.normal_(std=.03)
    return model


@pytest.mark.parametrize('window', [(-1, 0), (2, 0), (0, 0)])
@pytest.mark.parametrize('training', [False, True])
def test_dense_zero_value_reference_and_gradients(window, training):
    attn = make_model().transformer.h[1].attn
    model = make_model()
    attn.window_size = window
    attn.train(training)
    with torch.no_grad():
        attn.sink_logit.fill_(.7)
        attn.a_scale.fill_(-1.3)
    ref = copy.deepcopy(attn)
    x = torch.randn(2, 7, 16, requires_grad=True)
    emb = torch.randn_like(x, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    er = emb.detach().clone().requires_grad_()
    cs = model.cos[:, :7], model.sin[:, :7]
    actual, _ = attn(x, emb, None, cs, None)
    q = base.norm(base.apply_rotary_emb(ref.c_q(xr).unsqueeze(2), *cs)).squeeze(2) * 1.2
    k = base.norm(base.apply_rotary_emb(ref.c_k(xr).unsqueeze(2), *cs)).squeeze(2) * 1.2
    dist = torch.arange(7)[:, None] - torch.arange(7)[None, :]
    allowed = dist >= 0
    if window[0] >= 0:
        allowed &= dist <= window[0]
    logits = (q @ k.transpose(-1, -2) / 8**.5).masked_fill(~allowed, -torch.inf)
    # Independent denominator implementation, with no synthetic value matmul.
    log_z = torch.logaddexp(torch.logsumexp(logits, -1, keepdim=True), ref.sink_logit)
    probabilities = torch.exp(logits - log_z)
    expected = ref.c_proj(probabilities @ er) * ref.a_scale
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    assert torch.all(probabilities.sum(-1) < 1)
    assert torch.count_nonzero(probabilities.masked_select(~allowed)) == 0
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in [(x.grad, xr.grad), (emb.grad, er.grad)]:
        torch.testing.assert_close(a, b, atol=4e-6, rtol=5e-5)
    for (name, p), (_, r) in zip(attn.named_parameters(), ref.named_parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=8e-6, rtol=8e-5, msg=name)
    assert attn.sink_logit.grad.abs().sum() > 0


def test_sink_is_always_available_and_does_not_renormalize():
    attn = make_model().transformer.h[1].attn
    attn.window_size = (0, 0)  # Only self; token zero is unavailable after t=0.
    q = k = torch.zeros(1, 1, 4, 8)
    values = torch.randn(1, 4, 16)
    pos = torch.arange(4)
    # All real scores and the sink score are zero: every self value gets 1/2.
    out = attn._chunk(q, k, values, pos, pos)
    torch.testing.assert_close(out, .5 * values)
    with torch.no_grad():
        attn.sink_logit.fill_(-80)
    torch.testing.assert_close(attn._chunk(q, k, values, pos, pos), values)
    with torch.no_grad():
        attn.sink_logit.fill_(80)
    out = attn._chunk(q, k, values, pos, pos)
    assert torch.isfinite(out).all() and out.abs().max() < 1e-30


def test_initialization_optimizer_and_legacy_config():
    model = make_model(open_output=False)
    sinks = [model.transformer.h[i].attn.sink_logit for i in model.config.pooled_qk_target_layers]
    assert all(p.dtype == torch.float32 and p.shape == (1,) and p.item() == 0 for p in sinks)
    opt = model.setup_optimizer(scalar_lr=.5)
    grouped = [p for group in opt.param_groups for p in group['params']]
    assert len(grouped) == len(set(map(id, grouped))) == len(list(model.parameters()))
    assert set(map(id, grouped)) == set(map(id, model.parameters()))
    for p in sinks:
        group = next(g for g in opt.param_groups if any(q is p for q in g['params']))
        assert group['kind'] == 'adamw'
        assert group['lr'] == group['initial_lr'] == .001
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-10 and group['weight_decay'] == 0
    for i in model.config.pooled_qk_target_layers:
        a = model.transformer.h[i].attn
        assert a.a_scale.item() == 1 and a.c_proj.weight.count_nonzero() == 0
        group = next(g for g in opt.param_groups if any(q is a.a_scale for q in g['params']))
        assert group['lr'] == .5 and group['betas'] == (.96, .95)
    old = make_model(sink=False)
    legacy = asdict(old.config)
    legacy.pop('attention_sink')
    loaded = Model(Config(**legacy))
    loaded.load_state_dict(old.state_dict(), strict=True)
    assert not loaded.config.attention_sink
    assert not any('sink' in n for n in loaded.state_dict())
    assert Config(**json.loads(json.dumps(asdict(model.config)))).attention_sink


@pytest.mark.parametrize('targets', [(1, 2), (1,)])
def test_normalized_branch_cache_checkpoint_and_compile(targets):
    model = make_model(targets=targets)
    for b in model.transformer.h:
        b.attn.window_size = (2, 0)
    raw, values = [], []
    handles = [model.transformer.h[0].register_forward_pre_hook(lambda m,a: raw.append(a[0].detach()))]
    for i in targets:
        handles.append(model.transformer.h[i].attn.register_forward_pre_hook(lambda m,a: values.append(a[1].detach())))
    tokens = torch.randint(31, (2, 9))
    logits = model(tokens)
    torch.testing.assert_close(raw[0], model.transformer.wte(tokens))
    for value in values:
        torch.testing.assert_close(value, base.norm(model.transformer.wte(tokens)))
    for handle in handles:
        handle.remove()
    model(tokens, tokens.roll(-1, 1)).backward()
    eager_grads = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
    compiled(tokens, tokens.roll(-1, 1)).backward()
    for n,p in model.named_parameters():
        torch.testing.assert_close(p.grad, eager_grads[n], atol=2e-6, rtol=2e-4, msg=n)
    restored = Model(Config(**json.loads(json.dumps(asdict(model.config)))))
    restored.init_weights()
    restored.load_state_dict(model.state_dict())
    for b in restored.transformer.h:
        b.attn.window_size = (2, 0)
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens), logits)
        cache = KVCache(2, 2, 16, 8, 3, 'cpu', torch.float32)
        out = torch.cat([model(tokens[:, :3], kv_cache=cache),
                         model(tokens[:, 3:7], kv_cache=cache),
                         model(tokens[:, 7:], kv_cache=cache)], dim=1)
        assert cache.get_pos() == 9
        torch.testing.assert_close(out, logits, atol=2e-6, rtol=2e-4)


def test_budget_excludes_sink_softmax_work_and_counts_parameters():
    args = dict(n_layer=24, n_head=12, n_kv_head=12, n_embd=1536,
                sequence_len=2048, vocab_size=32768, window_pattern='SSSL', norm_value_input=True)
    with torch.device('meta'):
        original = Model(Config(**args))
        sink = Model(Config(**args, attention_sink=True))
    assert sink.estimate_flops() == original.estimate_flops() == 4247586816
    assert sink.estimate_forward_flops() == original.estimate_forward_flops() == 1415856640
    counts = sink.num_scaling_params()
    assert counts['total'] == original.num_scaling_params()['total'] + 11 == 706609174
    assert counts['scalars'] == 22
    assert counts['transformer_matrices'] == original.num_scaling_params()['transformer_matrices']
    assert round(3.91e19 / (sink.estimate_flops() * 1048576)) == 8779


@pytest.mark.parametrize('flags', [dict(attention_sink=1), dict(attention_sink=True, norm_value_input=False)])
def test_reject_invalid_flags(flags):
    with pytest.raises(ValueError, match='attention_sink'):
        Config(**flags)


@pytest.mark.parametrize('name', ['gpt_base_single_qk_value_proj_no_embed_norm',
                                'gpt_base_single_qk_no_out_proj_no_embed_norm'])
def test_reject_unsupported_subclasses(name):
    C, _ = get_model(name)
    with pytest.raises(ValueError, match='attention_sink'):
        C(attention_sink=True, norm_value_input=True)
