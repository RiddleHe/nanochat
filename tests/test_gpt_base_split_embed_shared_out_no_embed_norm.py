"""Check the proposed diagonal-map algebra, gradients, optimizer and budget."""

from dataclasses import asdict
import copy

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model.gpt_base_split_embed_shared_out_no_embed_norm import (
    GPTBaseSplitEmbedSharedOutNoEmbedNormConfig as Config,
    GPTBaseSplitEmbedSharedOutNoEmbedNorm as Model,
)
from nanochat.model_registry import get_model


@pytest.fixture(autouse=True)
def cpu_float32(monkeypatch):
    monkeypatch.setattr(flash, "USE_FA3", False)
    monkeypatch.setattr(base, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(no_norm, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)


def make_model():
    config = Config(sequence_len=16, vocab_size=31, n_layer=3, n_head=2,
                    n_kv_head=2, n_embd=16, window_pattern="L", attention_chunk_size=2)
    with torch.device("meta"):
        model = Model(config)
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
    return model


def dense_reference(attn, x, embedding, cos_sin):
    b, t, _ = x.shape
    q = attn.c_q(x).view(b, t, attn.n_head, attn.head_dim)
    k = attn.c_k(x).view(b, t, attn.n_head, attn.head_dim)
    q = (base.norm(base.apply_rotary_emb(q, *cos_sin)) * 1.2).transpose(1, 2)
    k = (base.norm(base.apply_rotary_emb(k, *cos_sin)) * 1.2).transpose(1, 2)
    distance = torch.arange(t)[:, None] - torch.arange(t)[None, :]
    allowed = distance >= 0
    if attn.window_size[0] >= 0:
        allowed &= distance <= attn.window_size[0]
    outputs = []
    for h in range(attn.n_head):
        scores = (q[:, h] @ k[:, h].transpose(-1, -2)) / attn.head_dim**0.5
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        chunk = embedding[..., h * attn.head_dim:(h + 1) * attn.head_dim]
        # Literal user formula: scale and project each token first, then mix.
        # Independent of the efficient sum-heads-before-projection implementation.
        transformed = (chunk * attn.head_scale[h]) @ attn.c_proj.weight.T
        outputs.append(probabilities @ transformed)
    return torch.stack(outputs).sum(0)



@pytest.mark.parametrize("window", [(-1, 0), (2, 0), (0, 0)])
@pytest.mark.parametrize("training", [False, True])
def test_signed_scales_dense_outputs_and_gradients(window, training):
    model = make_model()
    attn = model.transformer.h[-1].attn
    attn.window_size = window
    attn.train(training)
    with torch.no_grad():
        attn.head_scale.normal_()  # Includes distinct positive and negative maps.
    ref = copy.deepcopy(attn)
    x = torch.randn(2, 7, 16, requires_grad=True)
    emb = torch.randn(2, 7, 16, requires_grad=True)
    xr, er = x.detach().clone().requires_grad_(), emb.detach().clone().requires_grad_()
    cos_sin = model.cos[:, :7], model.sin[:, :7]
    actual, _ = attn(x, emb, None, cos_sin, None)
    expected = dense_reference(ref, xr, er, cos_sin)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in [(x.grad, xr.grad), (emb.grad, er.grad)]:
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    for (_, p), (_, pr) in zip(attn.named_parameters(), ref.named_parameters()):
        torch.testing.assert_close(p.grad, pr.grad, atol=4e-6, rtol=4e-5)
    assert emb.grad.abs().sum() > 0
    assert attn.head_scale.grad.abs().sum() > 0


def test_identity_initialization_registry_optimizer_and_counts():
    model = make_model()
    assert get_model("gpt_base_split_embed_shared_out_no_embed_norm") == (Config, Model)
    assert asdict(Config(**asdict(model.config))) == asdict(model.config)
    scales = [b.attn.head_scale for b in model.transformer.h if hasattr(b.attn, "head_scale")]
    assert all(p.dtype == torch.float32 and torch.all(p == 1) for p in scales)
    count = sum(p.numel() for p in scales)
    assert model.num_scaling_params()["scalars"] == count == 2 * 16
    assert model.num_scaling_params()["total"] == sum(p.numel() for p in model.parameters())
    for i in model.config.pooled_qk_target_layers:
        attn = model.transformer.h[i].attn
        assert not hasattr(attn, "c_v")
        assert attn.head_scale.shape == (2, 8)
        assert attn.c_proj.weight.shape == (16, 8)
    optimizer = model.setup_optimizer(scalar_lr=0.123)
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(parameters) == len(set(map(id, parameters)))
    assert set(map(id, parameters)) == set(map(id, model.parameters()))
    for scale in scales:
        group = next(g for g in optimizer.param_groups if any(p is scale for p in g["params"]))
        assert group["kind"] == "adamw" and group["lr"] == 0.123
        assert group["weight_decay"] == 0 and group["betas"] == (0.96, 0.95)


@pytest.mark.parametrize("depth,heads,width,window", [(3, 2, 16, "L"), (3, 1, 16, "L"), (24, 12, 1536, "SSSL")])
def test_flops_against_baseline_operation_counts(depth, heads, width, window):
    opts = dict(sequence_len=2048, vocab_size=32768, n_layer=depth, n_head=heads,
                n_kv_head=heads, n_embd=width, window_pattern=window)
    with torch.device("meta"):
        baseline = no_norm.GPTBaseNoEmbedNorm(base.GPTBaseConfig(**opts))
        model = Model(Config(**opts))
    forward_delta = train_delta = 0
    for i in model.config.pooled_qk_target_layers:
        d = width // heads
        # Remove Wv and shrink Wo; token-pair work remains unchanged.
        forward_delta += -2 * width**2 - 2 * (width**2 - width*d) + width + (heads-1)*d
        train_delta += -6 * width**2 - 6 * (width**2 - width*d) + 4*width + (heads-1)*d
    assert model.estimate_forward_flops() == baseline.estimate_flops() // 3 + forward_delta
    assert model.estimate_flops() == baseline.estimate_flops() + train_delta


def test_checkpoint_cache_and_compiled_training():
    model = make_model()
    with torch.no_grad():
        for b in model.transformer.h:
            b.attn.window_size = (2, 0)
            if hasattr(b.attn, "head_scale"):
                b.attn.head_scale.normal_()
    tokens = torch.randint(31, (1, 9))
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    loss = compiled(tokens, tokens)
    torch.testing.assert_close(loss, model(tokens, tokens))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.transformer.wte.weight.grad.abs().sum() > 0
    model.eval()
    original = KVCache(1, 2, 16, 8, 3, "cpu", torch.float32)
    expanded = KVCache(2, 2, 16, 8, 3, "cpu", torch.float32)
    with torch.no_grad():
        expected = model(tokens)
        prefix = model(tokens[:, :3], kv_cache=original)
        expanded.prefill(original)
        torch.testing.assert_close(model(tokens[:, 3:].expand(2, -1), kv_cache=expanded),
                                   expected[:, 3:].expand(2, -1, -1), atol=1e-6, rtol=2e-5)
        rest = model(tokens[:, 3:], kv_cache=original)
        torch.testing.assert_close(torch.cat([prefix, rest], 1), expected, atol=1e-6, rtol=2e-5)
        original.reset()
        torch.testing.assert_close(model(tokens[:, :2], kv_cache=original), expected[:, :2], atol=1e-6, rtol=2e-5)
        restored = Model(Config(**asdict(model.config)))
        restored.load_state_dict(model.state_dict())
        for b in restored.transformer.h:
            b.attn.window_size = (2, 0)
        torch.testing.assert_close(restored.eval()(tokens), expected)


def test_d24_budget_and_parameters():
    cfg = Config(sequence_len=2048, vocab_size=32768, n_layer=24,
                 n_head=12, n_kv_head=12, n_embd=1536, window_pattern="SSSL")
    with torch.device("meta"):
        model = Model(cfg)
    assert model.num_scaling_params()["total"] == 730415616
    assert model.num_scaling_params()["scalars"] == 16896
    assert model.estimate_forward_flops() == 1492287104
    assert model.estimate_flops() == 4476847232
    assert round(3.91e19 / (1048576 * model.estimate_flops())) == 8329


def test_cache_rejects_grad_and_overflow():
    model = make_model()
    attn = model.transformer.h[-1].attn
    x = torch.randn(1, 3, 16)
    cache = KVCache(1, 2, 2, 8, 3, "cpu", torch.float32)
    args = (x, x, None, (model.cos[:, :3], model.sin[:, :3]), cache)
    with pytest.raises(ValueError, match="inference-only"):
        attn(*args)
    with torch.no_grad(), pytest.raises(ValueError, match="capacity"):
        attn(*args)


@pytest.mark.parametrize("kwargs", [dict(n_kv_head=1), dict(add_init_v=True),
                                    dict(pooled_qk_target_layers=(0,))])
def test_invalid_architectures(kwargs):
    config = dict(n_layer=3, n_embd=16, n_head=2, n_kv_head=2)
    config.update(kwargs)
    with pytest.raises(ValueError):
        Config(**config)


def test_sharding_guard(monkeypatch):
    with torch.device("meta"):
        model = Model(Config(n_layer=24, n_embd=1536, n_head=12, n_kv_head=12))
    monkeypatch.setattr(base, "get_dist_info", lambda: (True, 0, 0, 5))
    with pytest.raises(ValueError, match="divisible"):
        model.setup_optimizer()
