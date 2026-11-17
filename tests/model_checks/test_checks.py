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
