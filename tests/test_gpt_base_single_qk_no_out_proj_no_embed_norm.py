"""Proposal 9 equations, initialization/learning, residuals, cache and FLOPs."""
import copy
from dataclasses import asdict

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model.gpt_base_single_qk_embed_no_embed_norm import (
    GPTBaseSingleQKEmbedNoEmbedNormConfig as PreviousConfig,
    GPTBaseSingleQKEmbedNoEmbedNorm as PreviousModel,
)
from nanochat.model.gpt_base_single_qk_no_out_proj_no_embed_norm import (
    GPTBaseSingleQKNoOutProjNoEmbedNormConfig as Config,
    GPTBaseSingleQKNoOutProjNoEmbedNorm as Model,
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
                 n_kv_head=2, n_embd=16, window_pattern="L", attention_chunk_size=2)
    with torch.device("meta"):
        model = Model(cfg)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))
    model.init_weights()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model


@pytest.mark.parametrize("window", [(-1, 0), (2, 0), (0, 0)])
@pytest.mark.parametrize("training", [False, True])
def test_dense_equation_and_gradients(window, training):
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
    actual, _ = attn(x, model._prepare_embedding_values(emb), None, cs, None)
    q = base.norm(base.apply_rotary_emb(ref.c_q(xr).unsqueeze(2), *cs)).squeeze(2) * 1.2
    k = base.norm(base.apply_rotary_emb(ref.c_k(xr).unsqueeze(2), *cs)).squeeze(2) * 1.2
    distance = torch.arange(7)[:, None] - torch.arange(7)[None, :]
    mask = distance >= 0
    if window[0] >= 0:
        mask &= distance <= window[0]
    probs = ((q @ k.transpose(-1, -2)) / 8**.5).masked_fill(~mask, -torch.inf).softmax(-1)
    expected = (probs @ base.norm(er)) * ref.a_scale
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in [(x.grad, xr.grad), (emb.grad, er.grad)]:
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    for p, pr in zip(attn.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, pr.grad, atol=4e-6, rtol=4e-5)
    assert emb.grad.abs().sum() > 0


def test_initialization_raw_residual_and_learning():
    model = make_model()
    assert get_model("gpt_base_single_qk_no_out_proj_no_embed_norm") == (Config, Model)
    assert asdict(Config(**asdict(model.config))) == asdict(model.config)
    for i, block in enumerate(model.transformer.h):
        if i in model.config.pooled_qk_target_layers:
            assert isinstance(block.attn.c_proj, torch.nn.Identity)
            assert not hasattr(block.attn, "c_v")
            assert block.attn.a_scale.item() == 0
        else:
            assert block.attn.c_proj.weight.count_nonzero() == 0
    captured = []
    handle = model.transformer.h[0].register_forward_pre_hook(
        lambda module, args: captured.append((args[0].detach().clone(), args[1].detach().clone())))
    tokens = torch.randint(31, (2, 7))
    model(tokens)
    handle.remove()
    emb = model.transformer.wte(tokens)
    torch.testing.assert_close(captured[0][0], emb)
    torch.testing.assert_close(captured[0][1], base.norm(emb))
    optimizer = model.setup_optimizer()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len(set(map(id, parameters)))
    assert set(map(id, parameters)) == set(map(id, model.parameters()))
    attn = model.transformer.h[-1].attn
    group = next(g for g in optimizer.param_groups if any(p is attn.a_scale for p in g["params"]))
    assert group["kind"] == "adamw" and group["weight_decay"] == 0
    model(tokens, tokens.roll(-1, 1)).backward()
    assert attn.a_scale.grad.abs().sum() > 0
    assert attn.c_q.weight.grad.count_nonzero() == 0
    optimizer.step()
    assert attn.a_scale.item() != 0
    model.zero_grad(set_to_none=True)
    model(tokens, tokens.roll(-1, 1)).backward()
    assert attn.c_q.weight.grad.abs().sum() > 0
    assert attn.c_k.weight.grad.abs().sum() > 0


def test_normalization_required():
    with pytest.raises(ValueError, match="requires norm_value_input"):
        Config(norm_value_input=False)


def test_cache_with_open_scalar():
    model = make_model().eval()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.window_size = (2, 0)
            if hasattr(block.attn, "a_scale"):
                block.attn.a_scale.fill_(-0.7)
        tokens = torch.randint(31, (1, 9))
        expected = model(tokens)
        cache = KVCache(1, 2, 16, 8, 3, "cpu", torch.float32)
        parts = [model(tokens[:, :3], kv_cache=cache)]
        parts.extend(model(tokens[:, i:i+1], kv_cache=cache) for i in range(3, 9))
        torch.testing.assert_close(torch.cat(parts, 1), expected, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("depth,heads,width", [(3, 2, 16), (24, 12, 1536)])
def test_flops_and_parameters(depth, heads, width):
    opts = dict(sequence_len=2048, vocab_size=32768, n_layer=depth,
                n_head=heads, n_kv_head=heads, n_embd=width, window_pattern="SSSL")
    with torch.device("meta"):
        model = Model(Config(**opts))
        previous = PreviousModel(PreviousConfig(**opts, norm_value_input=True))
    targets = model.config.pooled_qk_target_layers
    count = len(targets)
    # Independent matrix/attention calculation, including padded LM vocabulary.
    d = width // heads
    matrices = model.lm_head.weight.shape[0] * width + depth * 8 * width**2
    matrices += (depth - count) * 4 * width**2 + count * 2 * width * d
    attention = 0
    for i, window in enumerate(model.window_sizes):
        t = 2048 if window[0] < 0 else min(2048, window[0])
        attention += (6 * (width + d) if i in targets else 12 * width) * t
    expected = 6 * matrices + attention + 4 * count * width
    assert model.estimate_flops() == expected
    assert model.estimate_forward_flops() == 2 * matrices + attention // 3 + count * width
    assert previous.estimate_flops() - expected == 6 * count * width**2
    assert previous.num_scaling_params()["total"] - model.num_scaling_params()["total"] == count * width**2
    assert model.num_scaling_params()["scalars"] == count
