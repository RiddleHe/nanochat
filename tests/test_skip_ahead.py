"""Tests for the GPTBase skip-ahead routing variants."""

import torch
import torch.nn as nn

from nanochat.model.gpt_base import Block, GPTBase, GPTBaseConfig
from nanochat.model_registry import get_model


def make_config(**overrides):
    kwargs = dict(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    kwargs.update(overrides)
    return GPTBaseConfig(**kwargs)


def make_model(**overrides):
    model = GPTBase(make_config(**overrides))
    model.init_weights()
    return model


def test_registry_variants():
    expected = {
        "skip_ahead_dense": ("dense", "current"),
        "skip_ahead_sparse": ("sparse", "current"),
        "skip_ahead_dense_x0": ("dense", "x0"),
        "skip_ahead_sparse_x0": ("sparse", "x0"),
    }
    for name, (mode, source) in expected.items():
        config_cls, model_cls = get_model(name)
        config = config_cls()
        assert model_cls is GPTBase
        assert config.skip_ahead_mode == mode
        assert config.skip_gate_source == source


def test_skip_gates_initialize_to_one():
    x = torch.randn(2, 4, 32)
    for mode in ("dense", "sparse"):
        model = make_model(skip_ahead_mode=mode)
        for layer_idx in range(model.config.n_layer):
            gate = model._compute_skip_gate(layer_idx, x, x)
            torch.testing.assert_close(gate, torch.ones_like(gate), rtol=0, atol=0)


def test_dense_gate_is_input_dependent_and_between_zero_and_two():
    model = make_model(skip_ahead_mode="dense")
    with torch.no_grad():
        model.skip_gates[0].weight.fill_(0.1)
    x_positive = torch.ones(1, 2, 32)
    x_negative = -x_positive
    positive_gate = model._compute_skip_gate(0, x_positive, x_positive)
    negative_gate = model._compute_skip_gate(0, x_negative, x_negative)
    assert positive_gate.shape == (1, 2, 1)
    assert torch.all((0 < positive_gate) & (positive_gate < 2))
    assert torch.all((0 < negative_gate) & (negative_gate < 2))
    assert torch.all(positive_gate > negative_gate)


def test_x0_gate_uses_x0_instead_of_current_state():
    model = make_model(skip_ahead_mode="dense", skip_gate_source="x0")
    with torch.no_grad():
        model.skip_gates[0].weight.fill_(0.1)
    current = torch.ones(1, 2, 32)
    x0 = -current
    actual = model._compute_skip_gate(0, current, x0)
    expected = 2 * torch.sigmoid(model.skip_gates[0](x0))
    torch.testing.assert_close(actual, expected)


def test_sparse_gate_is_hard_zero_but_gate_gradient_survives():
    model = make_model(skip_ahead_mode="sparse", skip_threshold=0.5)
    gate_projection = model.skip_gates[0]
    with torch.no_grad():
        gate_projection.weight.fill_(-0.1)
    x = torch.ones(1, 2, 32)
    gate = model._compute_skip_gate(0, x, x)
    torch.testing.assert_close(gate, torch.zeros_like(gate), rtol=0, atol=0)

    gate.sum().backward()
    assert gate_projection.weight.grad is not None
    assert torch.count_nonzero(gate_projection.weight.grad) > 0


def test_sparse_active_gate_keeps_continuous_value():
    model = make_model(skip_ahead_mode="sparse", skip_threshold=0.5)
    with torch.no_grad():
        model.skip_gates[0].weight.fill_(0.01)
    x = torch.ones(1, 2, 32)
    gate = model._compute_skip_gate(0, x, x)
    soft_gate = 2 * torch.sigmoid(model.skip_gates[0](x))
    assert torch.all(gate > 0.5)
    torch.testing.assert_close(gate, soft_gate)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, *args):
        return self.scale * x


def test_block_gate_zero_is_identity_and_one_matches_baseline():
    config = make_config(n_layer=1)
    block = Block(config, layer_idx=0, window_size=(8, 0))
    block.attn = Scale(0.25)
    block.mlp = Scale(0.5)
    x = torch.randn(2, 4, 32)

    baseline = block(x, cos_sin=None, kv_cache=None)
    gate_one = block(x, cos_sin=None, kv_cache=None, gate=torch.ones(2, 4, 1))
    gate_zero = block(x, cos_sin=None, kv_cache=None, gate=torch.zeros(2, 4, 1))

    torch.testing.assert_close(gate_one, baseline)
    torch.testing.assert_close(gate_zero, x, rtol=0, atol=0)


def test_zero_initialized_dense_model_matches_gpt_base():
    baseline = GPTBase(make_config())
    dense = GPTBase(make_config(skip_ahead_mode="dense"))
    torch.manual_seed(1234)
    baseline.init_weights()
    torch.manual_seed(1234)
    dense.init_weights()

    idx = torch.randint(0, 32, (2, 4))
    baseline_logits = baseline(idx)
    dense_logits = dense(idx)
    torch.testing.assert_close(dense_logits, baseline_logits, rtol=0, atol=0)


def test_parameter_accounting_includes_skip_gates():
    baseline = make_model()
    dense = make_model(skip_ahead_mode="dense")
    assert baseline.num_scaling_params()["skip_gates"] == 0
    assert dense.num_scaling_params()["skip_gates"] == dense.config.n_layer * dense.config.n_embd
    assert dense.num_scaling_params()["total"] == sum(p.numel() for p in dense.parameters())


def test_skip_gates_use_adamw_optimizer_group():
    model = make_model(skip_ahead_mode="dense")
    optimizer = model.setup_optimizer()
    skip_gate_ids = {id(parameter) for parameter in model.skip_gates.parameters()}
    matching_groups = [
        group for group in optimizer.param_groups
        if any(id(parameter) in skip_gate_ids for parameter in group["params"])
    ]
    assert len(matching_groups) == 1
    assert matching_groups[0]["kind"] == "adamw"
    assert {id(parameter) for parameter in matching_groups[0]["params"]} == skip_gate_ids
