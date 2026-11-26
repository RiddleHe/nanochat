"""CPU checks of GPU preflight bookkeeping; these cannot pass the GPU gate."""
import pytest
import torch

import nanochat.flash_attention as attention
import nanochat.model.gpt_base as base
from scripts.preflight_weighted_loop_ab import (
    compare_tensors, exact_state, gradients, make_model, make_reference,
)


def test_comparison_rejects_wrong_scaling_and_nonfinite():
    expected = {"gradient": torch.tensor([1., 2., -3.])}
    assert compare_tensors(expected, expected, .02, .04)["passed"]
    actual = {"gradient": expected["gradient"] / 2}
    assert not compare_tensors(actual, expected, .02, .04)["passed"]
    with pytest.raises(AssertionError, match="Invalid tensor"):
        compare_tensors({"gradient": torch.full((3,), float("nan"))}, expected, .02, .04)
    with pytest.raises(AssertionError, match="bitwise"):
        exact_state(actual, expected)


def test_reference_shards_share_full_weights_and_preserve_gradients():
    parameters = {"large": torch.nn.Parameter(torch.zeros(32, 32)),
                  "small": torch.nn.Parameter(torch.zeros(4)),
                  "matrix0": torch.nn.Parameter(torch.zeros(2, 2)),
                  "matrix1": torch.nn.Parameter(torch.zeros(2, 2))}
    groups = [dict(kind="adamw", params=[parameters["large"], parameters["small"]],
                   lr=.1, betas=(.9, .95), eps=1e-8, weight_decay=0.),
              dict(kind="muon", params=[parameters["matrix0"], parameters["matrix1"]],
                   lr=.1, momentum=.95, ns_steps=5, beta2=.9, weight_decay=0.)]
    averaged = {name: torch.arange(p.numel()).reshape(p.shape).float() for name, p in parameters.items()}
    refs, optimizer, mapping = make_reference(groups, {id(p): n for n, p in parameters.items()},
                                              parameters, averaged, 2)
    shards = mapping[0]["parameters"]["large"]
    assert len(shards) == 2
    assert len(mapping[1]["heads"]) == 2
    assert len(optimizer.param_groups) == 3
    torch.testing.assert_close(shards[0].grad, averaged["large"][:16])
    torch.testing.assert_close(shards[1].grad, averaged["large"][16:])
    with torch.no_grad():
        shards[1].fill_(7.)
    assert (refs["large"][16:] == 7).all()
    assert (refs["large"][:16] == 0).all()
    assert (parameters["large"] == 0).all()


def test_tiny_setup_has_nonzero_task_and_auxiliary_score_gradients(monkeypatch):
    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(base, "COMPUTE_DTYPE", torch.float32)
    model_a = make_model(0., torch.device("cpu"), compiled=False)
    model_b = make_model(.01, torch.device("cpu"), compiled=False)
    exact_state(model_a.state_dict(), model_b.state_dict())
    generator = torch.Generator().manual_seed(9100)
    x = torch.randint(64, (4, 32), generator=generator)
    y = torch.randint(64, (4, 32), generator=generator)
    grad_a, loss_a = gradients(model_a, x, y)
    grad_b, loss_b = gradients(model_b, x, y)
    assert loss_a["weighted_aux"] == 0.
    assert loss_b["weighted_aux"] > 0.
    assert loss_a["ce"] == loss_b["ce"]
    for name in ("selector_weight", "selector_bias"):
        assert grad_a[name].abs().sum() > 0
        assert (grad_b[name] - grad_a[name]).abs().sum() > 0
    assert all(value.abs().sum() > 0 for value in grad_a.values())
