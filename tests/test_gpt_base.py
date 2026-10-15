"""CPU regression checks for GPTBase without auxiliary embedding/residual paths."""

from dataclasses import asdict

import pytest
import torch

import nanochat.flash_attention as attention
import nanochat.model.gpt as gpt_module
import nanochat.model.gpt_base as base_module
from nanochat.engine import KVCache
from nanochat.model.gpt import GPT, GPTConfig
from nanochat.model.gpt_base import GPTBase, GPTBaseConfig


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(base_module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)
    config = GPTBaseConfig(sequence_len=16, vocab_size=31, n_layer=2,
                           n_head=2, n_kv_head=1, n_embd=32, window_pattern="L")
    # Match the training launcher's meta -> to_empty -> init_weights path.
    with torch.device("meta"):
        model = GPTBase(config)
    model.to_empty(device="cpu")
    model.init_weights()
    # Nonzero output projections exercise attention/cache and all gradient paths.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
    return model


def test_parameters_optimizer_and_backward(model):
    forbidden = ("smear", "resid_lambdas", "x0_lambdas", "value_embeds", "ve_gate")
    assert not any(part in name for name in model.state_dict() for part in forbidden)
    counts = model.num_scaling_params()
    assert counts["scalars"] == 0
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    expected_flops = 6 * (counts["total"] - counts["wte"])
    expected_flops += 12 * model.config.n_embd * model.config.sequence_len * model.config.n_layer
    assert model.estimate_flops() == expected_flops

    optimizer = model.setup_optimizer(scalar_lr=0.123)
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == {id(p) for p in model.parameters()}
    tokens = torch.randint(model.config.vocab_size, (2, 6))
    loss = model(tokens, tokens.roll(-1, dims=1))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1), (3, 1, 2)])
def test_cached_logits_match_full_sequence(model, chunks):
    model.eval()
    tokens = torch.randint(model.config.vocab_size, (2, sum(chunks)))
    cache = KVCache(batch_size=2, num_heads=model.config.n_kv_head,
                    seq_len=model.config.sequence_len,
                    head_dim=model.config.n_embd // model.config.n_head,
                    num_layers=model.config.n_layer, device="cpu", dtype=torch.float32)
    # GPTBase must not read or write the smear state shared with other models.
    sentinel = object()
    cache.prev_embedding = sentinel
    with torch.no_grad():
        expected = model(tokens)
        outputs = []
        start = 0
        for size in chunks:
            outputs.append(model(tokens[:, start:start + size], kv_cache=cache))
            start += size
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model(tokens[:, :1]), expected[:, :1], atol=1e-6, rtol=1e-5)
    assert cache.get_pos() == tokens.size(1)
    assert cache.prev_embedding is sentinel


def test_matches_gpt_with_auxiliary_paths_disabled(model):
    """Removing smear preserves the underlying Transformer outputs and gradients."""
    reference = GPT(GPTConfig(**asdict(model.config), use_lambdas=False))
    reference.init_weights()
    reference.value_embeds = torch.nn.ModuleDict()
    for block in reference.transformer.h:
        block.attn.ve_gate = None
    with torch.no_grad():
        reference.smear_lambda.zero_()
        reference.smear_gate.weight.zero_()
    incompatible = reference.load_state_dict(model.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {"smear_lambda", "smear_gate.weight"}
    assert not incompatible.unexpected_keys

    tokens = torch.randint(model.config.vocab_size, (2, 6))
    targets = tokens.roll(-1, dims=1)
    torch.testing.assert_close(model(tokens), reference(tokens), atol=1e-6, rtol=1e-5)
    model(tokens, targets).backward()
    reference(tokens, targets).backward()
    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_params[name].grad,
                                   atol=1e-6, rtol=1e-5)
