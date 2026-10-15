"""Smear-free training and cached inference for GPTBase and its BoV variants."""

import pytest
import torch

import nanochat.flash_attention as attention
import nanochat.model.gpt_base as base_module
import nanochat.model.gpt_base_attn_res_qkv as qkv_module
from nanochat.engine import KVCache
from nanochat.model_registry import get_model


@pytest.mark.parametrize("model_type", [
    "gpt_base",
    "gpt_base_v_from_x0_learn",
    "gpt_base_v_from_v1",
    "gpt_base_v_from_value_emb_learn_shared",
    "gpt_base_add_init_value_emb_learn",
    "gpt_base_add_init_value_emb_nanogpt",
    "gpt_base_attn_res_qkv",
])
def test_smear_free_training_and_cache(monkeypatch, model_type):
    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(base_module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(qkv_module, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)
    Config, Model = get_model(model_type)
    config = Config(sequence_len=16, vocab_size=31, n_layer=3,
                    n_head=4, n_kv_head=2, n_embd=32, window_pattern="L")
    with torch.device("meta"):
        model = Model(config)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    model.init_weights()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not any("smear" in name for name in model.state_dict())
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
        if hasattr(model, "attn_res_queries"):
            model.attn_res_queries.normal_(std=0.2)

    counts = model.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["scalars"] == (model.attn_res_queries.numel()
                                  if hasattr(model, "attn_res_queries") else 0)
    assert model.estimate_flops() > 0
    optimizer = model.setup_optimizer()
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    tokens = torch.randint(config.vocab_size, (2, 6))
    loss = model(tokens, tokens.roll(-1, dims=1))
    assert torch.isfinite(loss)
    loss.backward()
    # Some BoV variants deliberately exclude unused V projections from optimization.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in optimized)
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters() if p.grad is not None}

    model.eval()
    cache = KVCache(2, config.n_kv_head, 16, config.n_embd // config.n_head,
                    config.n_layer, "cpu", torch.float32)
    sentinel = object()
    cache.prev_embedding = sentinel
    with torch.no_grad():
        expected = model(tokens)
        parts = [model(tokens[:, :3], kv_cache=cache),
                 model(tokens[:, 3:4], kv_cache=cache),
                 model(tokens[:, 4:], kv_cache=cache)]
        torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model(tokens[:, :1]), expected[:, :1], atol=1e-6, rtol=1e-5)
    assert cache.get_pos() == tokens.size(1)
    assert cache.prev_embedding is sentinel


def test_qkv_no_longer_requires_24_smear_channels(monkeypatch):
    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(qkv_module, "COMPUTE_DTYPE", torch.float32)
    Config, Model = get_model("gpt_base_attn_res_qkv")
    model = Model(Config(sequence_len=8, vocab_size=17, n_layer=2,
                         n_head=2, n_kv_head=1, n_embd=16, window_pattern="L"))
    model.init_weights()
    assert torch.isfinite(model(torch.tensor([[1]]))).all()
