"""CPU numerical contracts; these do not substitute for CUDA/FA3 pilot checks."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

import nanochat.flash_attention as attention
import nanochat.model.gpt_base as base
from scripts.train_weighted_loop_ab import (
    WeightedLoop, WeightedLoopConfig, pooled_balance, weighted_residual, weighted_topk,
)


def test_gate_support_ties_and_finite_difference():
    ties = weighted_topk(torch.zeros(2, 24), 12)
    assert torch.equal(ties[:, :12], torch.ones(2, 12))
    assert torch.equal(ties[:, 12:], torch.zeros(2, 12))
    scores = torch.linspace(-1, 1, 24).requires_grad_()
    reward = torch.arange(24).sin()
    gates = weighted_topk(scores, 12)
    assert (gates > 0).sum() == 12
    torch.testing.assert_close(gates.sum(), torch.tensor(12.))
    (gates * reward).sum().backward()
    assert torch.equal(scores.grad[:12], torch.zeros(12))
    assert scores.grad[12:].abs().min() > 0
    for j in (3, 12, 18, 23):
        delta = torch.zeros_like(scores); delta[j] = 1e-3
        numerical = ((weighted_topk(scores.detach() + delta, 12) -
                      weighted_topk(scores.detach() - delta, 12)) * reward).sum() / 2e-3
        torch.testing.assert_close(scores.grad[j], numerical, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nonbinary_residual_forward_and_gradients(dtype):
    h = torch.tensor([[1., -2., 3.]], dtype=dtype, requires_grad=True)
    candidate = torch.tensor([[3., 4., -1.]], dtype=dtype, requires_grad=True)
    gate = torch.tensor([[1.75]], requires_grad=True)
    upstream = torch.tensor([[.5, -1., 2.]], dtype=dtype)
    out = weighted_residual(h, candidate, gate)
    assert out.dtype == dtype
    expected = (h.float() + 1.75 * (candidate.float() - h.float())).to(dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert not torch.equal(out, candidate)
    (out * upstream).sum().backward()
    torch.testing.assert_close(h.grad, (-.75 * upstream.float()).to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(candidate.grad, (1.75 * upstream.float()).to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(gate.grad, ((candidate.float() - h.float()) * upstream).sum().reshape(1, 1))


def test_balance_pooling_normalization_and_all24_gradient():
    # Each loop uses only half the bank, but the shared bank is uniform overall.
    counts = torch.zeros(4, 24)
    counts[:2, :12] = 8; counts[2:, 12:] = 8
    probs = counts / 12
    torch.testing.assert_close(pooled_balance(counts, probs, 8, 12), torch.tensor(1.))
    concentrated = torch.zeros(4, 24); concentrated[:, :12] = 8
    scores = torch.cat([torch.ones(12), -torch.ones(12)]).repeat(4, 8, 1).requires_grad_()
    loss = pooled_balance(concentrated, scores.softmax(-1).sum(1), 8, 12)
    assert 1 < loss < 2
    loss.backward()
    assert (scores.grad[..., :12] > 0).all()
    assert (scores.grad[..., 12:] < 0).all()
    # Equal microbatch units are averaged; duplicating a unit preserves its loss.
    torch.testing.assert_close(
        pooled_balance(concentrated * 2, scores.detach().softmax(-1).sum(1) * 2, 16, 12), loss)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(base, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)
    config = WeightedLoopConfig(sequence_len=8, vocab_size=17, n_layer=4,
                                n_embd=8, n_head=2, n_kv_head=1, num_loops=4,
                                select_k=2, window_pattern="L", balance_alpha=.01)
    net = WeightedLoop(config)
    net.init_weights()
    with torch.no_grad():
        # Expose nonzero residual/task selector gradients beyond zero initialization.
        for block in net.transformer.h:
            block.attn.c_proj.weight.normal_(std=.07)
            block.mlp.c_proj.weight.normal_(std=.07)
    return net


def test_training_aux_eval_ce_isolation_and_explicit_stats(model):
    tokens = torch.randint(17, (4, 4)); targets = tokens.roll(-1, 1)
    initial_keys = set(model.__dict__)
    ce, weighted_aux = model.training_loss(tokens, targets)
    explicit_ce, stats = model(tokens, targets, return_stats=True)
    torch.testing.assert_close(ce, explicit_ce, atol=0, rtol=0)
    torch.testing.assert_close(weighted_aux, .01 * stats["balance"])
    assert stats["selected_counts"].shape == (4, 4)
    torch.testing.assert_close(stats["selected_counts"].sum(1), torch.full((4,), 32.))
    torch.testing.assert_close(stats["gate_mass"].sum(1), torch.full((4,), 32.))
    assert not stats["gate_mass"].requires_grad and not stats["selected_counts"].requires_grad
    ce.backward()
    assert model.selector_weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        reference = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
        torch.testing.assert_close(model(tokens, targets, loss_reduction="none"), reference, atol=0, rtol=0)
        model.config.balance_alpha = 0
        torch.testing.assert_close(model(tokens, targets, loss_reduction="none"), reference, atol=0, rtol=0)
        _, zero_aux = model.training_loss(tokens, targets)
        assert zero_aux.item() == 0
    assert set(model.__dict__) == initial_keys
    with pytest.raises(ValueError, match="four sequences"):
        model.training_loss(tokens[:2], targets[:2])


def test_causal_prefix_and_cached_decoding(model):
    from nanochat.engine import KVCache
    model.eval()
    tokens = torch.randint(17, (2, 6))
    config = model.config
    cache = KVCache(batch_size=2, num_heads=config.n_kv_head, seq_len=8,
                    head_dim=config.n_embd // config.n_head, num_layers=model.cache_layers,
                    device="cpu", dtype=torch.float32)
    with torch.no_grad():
        full = model(tokens)
        torch.testing.assert_close(model(tokens[:, :3]), full[:, :3], atol=2e-7, rtol=2e-5)
        cached = torch.cat([model(tokens[:, :3], kv_cache=cache),
                            model(tokens[:, 3:4], kv_cache=cache),
                            model(tokens[:, 4:], kv_cache=cache)], 1)
    torch.testing.assert_close(cached, full, atol=2e-7, rtol=2e-5)
    assert cache.get_pos() == 6


def test_checkpoint_recomputation_matches_loss_and_parameter_gradients(model):
    checkpointed = deepcopy(model)
    checkpointed.config.checkpoint_blocks = True
    tokens = torch.randint(17, (4, 4)); targets = tokens.roll(-1, 1)
    for net in (model, checkpointed):
        ce, aux = net.training_loss(tokens, targets)
        (ce + aux).backward()
    for (name, param), (_, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, other.grad, atol=0, rtol=0)


def test_ab_initialization_and_frozen_training_unit(model):
    config_b = deepcopy(model.config)
    config_a = deepcopy(config_b); config_a.balance_alpha = 0
    nets = []
    for config in (config_a, config_b):
        torch.manual_seed(42)
        net = WeightedLoop(config); net.init_weights(); nets.append(net)
    for name, tensor in nets[0].state_dict().items():
        torch.testing.assert_close(tensor, nets[1].state_dict()[name], atol=0, rtol=0)
    assert nets[0].estimate_flops() == nets[1].estimate_flops()
    frozen = dict(device_batch_size=4, model=dict(n_layer=24, num_loops=4, select_k=12,
                  sequence_len=2048, n_embd=1536, n_head=12, n_kv_head=12,
                  selector_mode="learned", balance_alpha=.01))
    WeightedLoop.validate_training_config(frozen)
    frozen["device_batch_size"] = 8
    with pytest.raises(ValueError, match="four sequences"):
        WeightedLoop.validate_training_config(frozen)
