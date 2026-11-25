"""The same CPU contract applies to every registered architecture."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest
import torch

from nanochat.model_registry import MODELS

spec = importlib.util.spec_from_file_location("shared_model_checks", Path(__file__).with_name("check.py"))
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


# These existing model initializers omit smear_gate/smear_lambda after to_empty.
# Keep their validation failure explicit; do not repair model code in the checker.
INCOMPLETE_INITIALIZERS = {"gpt", "gpt_nolambda", "attn_res", "attn_res_sink"}


@pytest.fixture(autouse=True)
def isolate_compiler_cache():
    # Each frozen-run check runs in its own process. Match that isolation when
    # pytest checks several architectures with different Muon matrix shapes.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("name", sorted(set(MODELS) - INCOMPLETE_INITIALIZERS))
def test_registered_model_contract(name):
    Config, _ = MODELS[name]
    config = asdict(Config(sequence_len=16, vocab_size=128, n_layer=4,
                           n_head=4, n_kv_head=4, n_embd=32))
    result = checks.basic_checks(name, config, torch.device("cpu"))
    assert result["finite_outputs_gradients_update"]


@pytest.mark.parametrize("name", sorted(INCOMPLETE_INITIALIZERS))
def test_rejects_existing_incomplete_initializers(name):
    Config, _ = MODELS[name]
    config = asdict(Config(sequence_len=16, vocab_size=128, n_layer=4,
                           n_head=4, n_kv_head=4, n_embd=32))
    with pytest.raises(AssertionError, match="Uninitialized.*smear"):
        checks.basic_checks(name, config, torch.device("cpu"))


def test_numerical_check_rejects_finite_but_wrong_values():
    with pytest.raises(AssertionError, match="L2 error"):
        checks.compare(torch.ones(8), torch.zeros(8), "wrong values")


def test_small_shape_preserves_architecture():
    original = dict(n_layer=24, n_head=12, n_kv_head=12, n_embd=1536,
                    sequence_len=2048, vocab_size=32768, pooled_qk_target_layers=[9, 10, 23],
                    norm_value_input=True, sum_head_attention=True)
    small = checks.small_config(original)
    for key in ("n_layer", "n_head", "n_kv_head", "pooled_qk_target_layers", "norm_value_input", "sum_head_attention"):
        assert small[key] == original[key]
    assert original["n_embd"] == 1536


def test_no_out_proj_checks_open_zero_scalar():
    name = "gpt_base_single_qk_no_out_proj_no_embed_norm"
    Config, _ = MODELS[name]
    config = asdict(Config(sequence_len=16, vocab_size=128, n_layer=4,
                           n_head=4, n_kv_head=4, n_embd=32))
    with checks.precision(torch.float32):
        initial = checks.make_model(name, config, "cpu", perturb=False)
        checked = checks.make_model(name, config, "cpu")
        for i in initial.config.pooled_qk_target_layers:
            assert initial.transformer.h[i].attn.a_scale.item() == 0
            assert checked.transformer.h[i].attn.a_scale.item() > 0
        optimizer = checked.setup_optimizer()
        checks.backward(checked, checks.tokens(config, 2, "cpu"), optimizer)
        for i in checked.config.pooled_qk_target_layers:
            assert checked.transformer.h[i].attn.c_q.weight.grad.abs().sum() > 0
