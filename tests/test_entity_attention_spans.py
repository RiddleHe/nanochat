"""CPU checks for span-wide interventions, using analytical attention outputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.inspect import qwen_entity_attention_ablation as core


@pytest.fixture
def attention():
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, 4, 6, 3, generator=generator)
    k = torch.randn(1, 2, 6, 3, generator=generator)
    v = torch.randn(1, 2, 6, 3, generator=generator)
    mask = torch.full((1, 1, 6, 6), float('-inf')).triu(1)
    module = SimpleNamespace(num_key_value_groups=2, layer_idx=0, training=False, is_causal=True)
    return module, q, k, v, mask


def analytical(q, k, v, mask):
    k = k.repeat_interleave(2, dim=1)
    v = v.repeat_interleave(2, dim=1)
    weights = (q @ k.transpose(-2, -1) / 3**0.5 + mask).softmax(-1)
    return weights, v, (weights @ v).transpose(1, 2)


@pytest.mark.parametrize('block_intermediate', [False, True])
@pytest.mark.parametrize('disable_readout', [False, True])
def test_removes_whole_span_without_renormalizing(attention, block_intermediate, disable_readout):
    module, q, k, v, mask = attention
    weights, values, expected = analytical(q, k, v, mask)
    affected = ([3, 4] if block_intermediate else []) + ([5] if disable_readout else [])
    contribution = (weights[..., 1:3] @ values[..., 1:3, :]).transpose(1, 2)
    expected[:, affected] -= contribution[:, affected]
    with core.disable_entity_attention(range(int(disable_readout)), 1, 5, 6,
                                      block_intermediate, entity_positions=(1, 2)) as state:
        actual, _ = core.entity_zero_attention_forward(module, q, k, v, mask, 3**-0.5)
        assert state.intermediate_applications == (2 if block_intermediate else 0)
        assert state.applications == int(disable_readout)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)


def test_generated_query_blocks_span_even_outside_selected_layers(attention):
    module, q, k, v, _ = attention
    # A cached decode query at absolute position 6, after a six-token prompt.
    k = torch.cat([k, k[..., :1, :]], dim=-2)
    v = torch.cat([v, v[..., :1, :]], dim=-2)
    q = q[..., -1:, :]
    mask = torch.zeros(1, 1, 1, 7)
    weights, values, expected = analytical(q, k, v, mask)
    expected -= (weights[..., 1:3] @ values[..., 1:3, :]).transpose(1, 2)
    with core.disable_entity_attention(range(0), 1, 5, 6, False, entity_positions=(1, 2)) as state:
        actual, _ = core.entity_zero_attention_forward(module, q, k, v, mask, 3**-0.5)
        assert state.generated_applications == 1
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)


@pytest.mark.parametrize('positions', [(1,), (1, 2)])
def test_capture_preserves_per_token_attention(attention, positions):
    module, q, k, v, mask = attention
    weights, _, expected = analytical(q, k, v, mask)
    with core.capture_ordinary_entity_attention(range(1), 1, 5, 6, positions) as state:
        actual, _ = core.entity_zero_attention_forward(module, q, k, v, mask, 3**-0.5)
        reference = state.ordinary_entity_attention[0]
        target = weights[:, :, 5, list(positions)]
        if len(positions) == 1:
            target = target.squeeze(-1)
        torch.testing.assert_close(reference, target)
    torch.testing.assert_close(actual, expected)
    coefficients = core._final_entity_attention_by_token(module, q, k, mask, 3**-0.5, 5, positions)
    torch.testing.assert_close(coefficients, weights[:, :, 5, list(positions)])
    if len(positions) == 1:
        legacy = core._final_entity_attention(module, q, k, mask, 3**-0.5, 5, 1)
        torch.testing.assert_close(legacy, coefficients.squeeze(-1))


def test_restore_sums_positive_token_deficits_and_preserves_other_queries(attention):
    module, q, k, v, mask = attention
    weights, values, expected = analytical(q, k, v, mask)
    live = weights[:, :, 5, 1:3]
    ordinary = live.clone()
    ordinary[..., 0] += 0.03
    ordinary[..., 1] *= 0.5  # This token must not receive a negative supplement.
    expected[:, 5] += 0.03 * values[:, :, 1]
    with core.disable_entity_attention(range(0), 1, 5, 6, False,
                                      restore_layers=range(1), ordinary_entity_attention={0: ordinary},
                                      entity_positions=(1, 2)) as state:
        actual, _ = core.entity_zero_attention_forward(module, q, k, v, mask, 3**-0.5)
        trace = state.restore_trace[0]
        torch.testing.assert_close(torch.tensor(trace['added_entity_attention_deficit_by_token']), torch.tensor([[0.03, 0.0]]).expand(4, 2))
        torch.testing.assert_close(torch.tensor(trace['live_entity_attention_by_token']), live[0])
        torch.testing.assert_close(torch.tensor(trace['effective_entity_attention_by_token']), torch.maximum(live, ordinary)[0])
        assert 'ordinary_entity_attention' not in trace
        assert 'live_entity_attention_before_restore' not in trace
        assert 'added_entity_attention_deficit' not in trace
        assert 'effective_entity_attention' not in trace
        assert 'effective_attention_sum' not in trace
        assert state.restore_applications == 1
    torch.testing.assert_close(actual, expected)


def test_legacy_single_token_api_matches_explicit_span(attention):
    module, q, k, v, mask = attention
    results = []
    for kwargs in ({}, {'entity_positions': (1,)}):
        with core.disable_entity_attention(range(1), 1, 5, 6, True, **kwargs):
            results.append(core.entity_zero_attention_forward(module, q, k, v, mask, 3**-0.5)[0])
    assert torch.equal(*results)


def test_single_token_restore_retains_legacy_trace_fields(attention):
    module, q, k, v, mask = attention
    weights, _, _ = analytical(q, k, v, mask)
    reference = weights[:, :, 5, 1] + 0.03
    with core.disable_entity_attention(range(0), 1, 5, 6, False,
                                      restore_layers=range(1), ordinary_entity_attention={0: reference}) as state:
        core.entity_zero_attention_forward(module, q, k, v, mask, 3**-0.5)
        trace = state.restore_trace[0]
        for scalar, per_token in [
            ('ordinary_entity_attention', 'ordinary_entity_attention_by_token'),
            ('live_entity_attention_before_restore', 'live_entity_attention_by_token'),
            ('added_entity_attention_deficit', 'added_entity_attention_deficit_by_token'),
            ('effective_entity_attention', 'effective_entity_attention_by_token'),
        ]:
            assert trace[scalar] == [head[0] for head in trace[per_token]]


@pytest.mark.parametrize('positions', [(), (1, 3), (2,), (-1,), (1, True)])
def test_invalid_spans_are_rejected(positions):
    with pytest.raises(ValueError):
        with core.disable_entity_attention(range(1), 1, 5, 6, True, entity_positions=positions):
            pass
    assert core.ABLATION.entity_position is None


def test_policy_cleanup_after_exception():
    with pytest.raises(RuntimeError):
        with core.disable_entity_attention(range(1), 1, 5, 6, True, entity_positions=(1, 2)):
            raise RuntimeError('test')
    assert core.ABLATION.entity_position is None
    assert core.ABLATION.entity_positions == ()


def test_manifest_and_duplicates(tmp_path):
    manifest = Path(core.__file__).parent / 'entity_attention_data/entities100_balanced12.json'
    entities, digest = core.load_entities(manifest)
    assert len(entities) == 100 and len(set(name for name, _ in entities)) == 100
    assert len(digest) == 64
    path = tmp_path / 'entities.json'
    path.write_text(json.dumps(['Marie Curie', 'Marie Curie']))
    with pytest.raises(ValueError):
        core.load_entities(path)
