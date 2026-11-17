"""Dense-reference, gradient, cache and budget checks for pooled QK attention."""

from dataclasses import asdict
import copy

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model.gpt_base_pooled_qk_embed_no_embed_norm import (
    D24_TARGET_LAYERS, GPTBasePooledQKEmbedNoEmbedNormConfig as Config,
    GPTBasePooledQKEmbedNoEmbedNorm as Model, target_layers_for_depth,
)
from nanochat.model_registry import get_model


@pytest.fixture(autouse=True)
def cpu_float32(monkeypatch):
    monkeypatch.setattr(flash, "USE_FA3", False)
    monkeypatch.setattr(base, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(no_norm, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)


def make_model(**kwargs):
    options = dict(sequence_len=16, vocab_size=31, n_layer=3,
                   n_head=2, n_kv_head=2, n_embd=16, window_pattern="L",
                   attention_chunk_size=2)
    options.update(kwargs)
    with torch.device("meta"):
        model = Model(Config(**options))
    model.to_empty(device="cpu")
    model.init_weights()
    # Expose the actual attention/residual paths, hidden by zero output init.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
    return model


def test_mapping_configuration_and_absent_values():
    assert get_model("gpt_base_pooled_qk_embed_no_embed_norm") == (Config, Model)
    assert target_layers_for_depth(24) == D24_TARGET_LAYERS
    assert target_layers_for_depth(1) == (0,)
    assert target_layers_for_depth(3) == (1, 2)
    assert target_layers_for_depth(12) == (4, 5, 6, 7, 9, 10, 11)
    config = Config()
    assert asdict(Config(**asdict(config))) == asdict(config)
    with pytest.raises(ValueError, match="fixed depth"):
        Config(pooled_qk_target_layers=[0])
    with pytest.raises(ValueError, match="BoV flags"):
        Config(add_init_res=True)
    with pytest.raises(ValueError, match="equal Q/K"):
        Config(n_kv_head=1)
    model = make_model()
    for i, block in enumerate(model.transformer.h):
        assert hasattr(block.attn, "c_v") == (i not in model.config.pooled_qk_target_layers)
    assert not any("smear" in key for key in model.state_dict())


def dense_reference(attn, x, embedding, cos_sin):
    b, t, _ = x.shape
    q = attn.c_q(x).view(b, t, attn.n_head, attn.head_dim)
    k = attn.c_k(x).view(b, t, attn.n_head, attn.head_dim)
    q = (base.norm(base.apply_rotary_emb(q, *cos_sin)) * 1.2).transpose(1, 2)
    k = (base.norm(base.apply_rotary_emb(k, *cos_sin)) * 1.2).transpose(1, 2)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k).float() / attn.head_dim**0.5
    indices = torch.arange(t)
    distance = indices[:, None] - indices[None, :]
    mask = distance >= 0
    if attn.window_size[0] >= 0:
        mask &= distance <= attn.window_size[0]
    probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    # Independent formulation: mix the FULL embedding separately per head,
    # then average the outputs. Linearity makes this equal to pooling first.
    mixed = torch.einsum("bhqk,bkd->bhqd", probabilities, embedding).mean(1)
    return attn.c_proj(mixed)


@pytest.mark.parametrize("window", [(-1, 0), (2, 0), (0, 0)])
@pytest.mark.parametrize("training", [False, True])
def test_dense_reference_outputs_and_live_gradients(window, training):
    model = make_model()
    attn = model.transformer.h[-1].attn
    attn.window_size = window
    attn.train(training)
    reference = copy.deepcopy(attn)
    x = torch.randn(2, 7, 16, requires_grad=True)
    embedding = torch.randn(2, 7, 16, requires_grad=True)
    xr, er = x.detach().clone().requires_grad_(), embedding.detach().clone().requires_grad_()
    cos_sin = model.cos[:, :7], model.sin[:, :7]
    output, _ = attn(x, embedding, None, cos_sin, None)
    expected = dense_reference(reference, xr, er, cos_sin)
    torch.testing.assert_close(output, expected, atol=5e-7, rtol=2e-5)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient)
    for actual, wanted in [(x.grad, xr.grad), (embedding.grad, er.grad)]:
        torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-5)
    assert embedding.grad.abs().sum() > 0
    for p, pr in zip(attn.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, pr.grad, atol=3e-6, rtol=3e-5)


def test_uniform_attention_includes_first_token_and_has_no_future_leakage():
    model = make_model()
    attn = model.transformer.h[-1].attn.eval()
    with torch.no_grad():
        attn.c_q.weight.zero_()
        attn.c_k.weight.zero_()
        attn.c_proj.weight.copy_(torch.eye(16))
    embeddings = torch.randn(1, 7, 16)
    out, _ = attn(torch.randn_like(embeddings), embeddings, None,
                  (model.cos[:, :7], model.sin[:, :7]), None)
    expected = embeddings.cumsum(1) / torch.arange(1, 8)[None, :, None]
    torch.testing.assert_close(out, expected)


def test_model_gradients_cache_copy_reset_and_roundtrip():
    model = make_model()
    # Exercise actual sliding-window boundaries, plus ordinary and target layers.
    for block in model.transformer.h:
        block.attn.window_size = (2, 0)
    optimizer = model.setup_optimizer()
    optimized = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters()}
    tokens = torch.randint(31, (1, 9))
    loss = model(tokens, tokens.roll(-1, 1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in optimized)
    assert model.transformer.wte.weight.grad.abs().sum() > 0
    model.eval()
    def cache(batch):
        return KVCache(batch, 2, 16, 8, 3, "cpu", torch.float32)
    original, expanded = cache(1), cache(2)
    with torch.no_grad():
        expected = model(tokens)
        prefix = model(tokens[:, :3], kv_cache=original)
        expanded.prefill(original)
        rest = model(tokens[:, 3:].expand(2, -1), kv_cache=expanded)
        torch.testing.assert_close(rest, expected[:, 3:].expand(2, -1, -1), atol=1e-6, rtol=2e-5)
        pieces = [prefix, model(tokens[:, 3:4], kv_cache=original),
                  model(tokens[:, 4:7], kv_cache=original), model(tokens[:, 7:], kv_cache=original)]
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-6, rtol=2e-5)
        assert original.get_pos() == expanded.get_pos() == 9
        original.reset()
        torch.testing.assert_close(model(tokens[:, :2], kv_cache=original), expected[:, :2], atol=1e-6, rtol=2e-5)
        # Raw embedding bytes, not projected values, occupy target cache slots.
        raw = model.transformer.wte(tokens[:, :2])
        torch.testing.assert_close(original.v_cache[-1, :, :2].reshape_as(raw), raw)
        restored = Model(Config(**asdict(model.config)))
        restored.load_state_dict(model.state_dict())
        for block in restored.transformer.h:
            block.attn.window_size = (2, 0)
        torch.testing.assert_close(restored.eval()(tokens), expected)


def test_flops_parameter_savings_and_fullgraph_capture():
    model = make_model()
    cfg = model.config
    baseline_cfg = base.GPTBaseConfig(**{f: v for f, v in asdict(cfg).items()
                                       if f in base.GPTBaseConfig.__dataclass_fields__})
    baseline = no_norm.GPTBaseNoEmbedNorm(baseline_cfg)
    removed = len(cfg.pooled_qk_target_layers) * cfg.n_embd**2
    assert sum(p.numel() for p in baseline.parameters()) - sum(p.numel() for p in model.parameters()) == removed
    pooling = len(cfg.pooled_qk_target_layers) * cfg.n_head * cfg.sequence_len
    assert model.estimate_flops() == baseline.estimate_flops() - 6 * removed + 2 * pooling
    assert model.estimate_forward_flops() == baseline.estimate_flops() // 3 - 2 * removed + pooling
    tokens = torch.randint(31, (2, 7))
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    actual = compiled(tokens, tokens)
    expected = model(tokens, tokens)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
