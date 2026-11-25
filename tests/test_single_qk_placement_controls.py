"""Placement controls preserve the normalized architecture and literal indices."""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model.gpt_base_pooled_qk_embed_no_embed_norm import target_layers_for_depth
from nanochat.model.gpt_base_single_qk_embed_no_embed_norm import (
    GPTBaseSingleQKEmbedNoEmbedNormConfig as Config,
    GPTBaseSingleQKEmbedNoEmbedNorm as Model,
    SingleQKEmbeddingAttention,
)

CONTROLS = [
    (list(range(1, 22, 2)), 4221634560, 1407205888, 8833),
    (list(range(12, 23)), 4260562944, 1420182016, 8752),
    (list(range(1, 22, 4)), 4508651520, 1502880768, 8270),
    (list(range(3, 20, 4)), 4488198144, 1496063488, 8308),
]


def test_s_and_l_controls_partition_interleaved_targets():
    interleaved = set(CONTROLS[0][0])
    short, long = map(lambda control: set(control[0]), CONTROLS[2:])
    assert short == {i for i in interleaved if i % 4 != 3}
    assert long == {i for i in interleaved if i % 4 == 3}
    assert short.isdisjoint(long) and short | long == interleaved


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(flash, "USE_FA3", False)
    monkeypatch.setattr(base, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(no_norm, "COMPUTE_DTYPE", torch.float32)
    torch.set_num_threads(1)
    torch.manual_seed(42)


@pytest.mark.parametrize("targets", [[], [1, 1], [-1], [24], [True], [1.0], ["1"], "1,3"])
def test_reject_invalid_indices(targets):
    with pytest.raises(ValueError, match="Target layers"):
        Config(n_layer=24, pooled_qk_target_layers=targets)


def test_legacy_mapping_and_literal_depth_validation():
    assert Config(n_layer=12).pooled_qk_target_layers == target_layers_for_depth(12)
    assert Config(n_layer=12, pooled_qk_target_layers=[8, 1]).pooled_qk_target_layers == (1, 8)
    with pytest.raises(ValueError, match="Target layers"):
        Config(n_layer=12, pooled_qk_target_layers=CONTROLS[0][0])


@pytest.mark.parametrize("targets,training_flops,forward_flops,steps", CONTROLS)
def test_d24_budget(targets, training_flops, forward_flops, steps):
    config = Config(n_layer=24, n_head=12, n_kv_head=12, n_embd=1536,
                    sequence_len=2048, vocab_size=32768, window_pattern="SSSL",
                    norm_value_input=True, pooled_qk_target_layers=targets)
    with torch.device("meta"):
        model = Model(config)
        baseline = no_norm.GPTBaseNoEmbedNorm(base.GPTBaseConfig(
            n_layer=24, n_head=12, n_kv_head=12, n_embd=1536,
            sequence_len=2048, vocab_size=32768, window_pattern="SSSL"))
    # Independent delta: remove Wv, narrow Q/K matrices and score products;
    # add the scalar's forward and backward arithmetic.
    width, head_dim = 1536, 128
    delta = sum(-6 * width**2 - 12 * width * (width - head_dim)
                - 6 * (width - head_dim) * (2048 if i % 4 == 3 else 512)
                + 4 * width for i in targets)
    assert model.estimate_flops() == baseline.estimate_flops() + delta == training_flops
    assert model.estimate_forward_flops() == forward_flops
    parameter_delta = len(targets) * (-width**2 - 2 * width * (width - head_dim) + 1)
    assert model.num_scaling_params()["total"] == baseline.num_scaling_params()["total"] + parameter_delta
    assert model.num_scaling_params()["scalars"] == len(targets)
    for i, block in enumerate(model.transformer.h):
        assert isinstance(block.attn, SingleQKEmbeddingAttention) == (i in targets)
        assert block.attn.window_size == ((2048, 0) if i % 4 == 3 else (512, 0))
    assert round(3.91e19 / (training_flops * 1048576)) == steps


@pytest.mark.parametrize("targets,training_flops,forward_flops,steps", CONTROLS)
def test_normalized_control_gradients_cache_and_checkpoint(targets, training_flops, forward_flops, steps):
    config = Config(n_layer=24, n_head=2, n_kv_head=2, n_embd=16,
                    sequence_len=16, vocab_size=31, window_pattern="SSSL",
                    norm_value_input=True, pooled_qk_target_layers=targets)
    model = Model(config)
    model.init_weights()
    for i, block in enumerate(model.transformer.h):
        assert isinstance(block.attn, SingleQKEmbeddingAttention) == (i in targets)
        assert hasattr(block.attn, "c_v") == (i not in targets)
        if i in targets:
            assert block.attn.a_scale.item() == 1
        with torch.no_grad():
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)
        block.attn.window_size = (3, 0) if i % 4 != 3 else (-1, 0)
    tokens = torch.randint(31, (2, 9))
    raw, values = [], []
    handles = [model.transformer.h[0].register_forward_pre_hook(
        lambda m, a: raw.append(a[0].detach()))]
    for i in targets:
        handles.append(model.transformer.h[i].attn.register_forward_pre_hook(
            lambda m, a: values.append(a[1].detach())))
    loss = model(tokens, tokens.roll(-1, 1))
    torch.testing.assert_close(raw[0], model.transformer.wte(tokens))
    expected = base.norm(model.transformer.wte(tokens))
    assert len(values) == len(targets)
    for value in values:
        torch.testing.assert_close(value, expected)
    for handle in handles:
        handle.remove()
    loss.backward()
    assert model.transformer.wte.weight.grad.abs().sum() > 0
    for i in targets:
        for parameter in model.transformer.h[i].attn.parameters():
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        full = model(tokens)
        cache = KVCache(2, 2, 16, 8, 24, "cpu", torch.float32)
        cached = torch.cat([model(tokens[:, :4], kv_cache=cache),
                            model(tokens[:, 4:8], kv_cache=cache),
                            model(tokens[:, 8:], kv_cache=cache)], dim=1)
        assert cache.get_pos() == 9  # Final ordinary layer advances exactly once.
        torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-4)
        restored = Model(Config(**json.loads(json.dumps(asdict(config)))))
        restored.init_weights()
        restored.load_state_dict(model.state_dict())
        for original, loaded in zip(model.transformer.h, restored.transformer.h):
            loaded.attn.window_size = original.attn.window_size
        torch.testing.assert_close(restored(tokens), full)


@pytest.mark.parametrize("targets,training_flops,forward_flops,steps", CONTROLS)
def test_trainer_meta_construction_and_d12_reference(targets, training_flops, forward_flops, steps):
    # Execute the actual trainer builder and reference call without importing its
    # top-level distributed/data/training side effects.
    path = Path(__file__).resolve().parents[1] / "scripts/base_train.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name == "build_model_meta"
             or isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "d12_ref" for t in node.targets)]
    namespace = dict(torch=torch, ConfigClass=Config, ModelClass=Model, vocab_size=32768,
                     args=argparse.Namespace(aspect_ratio=64, head_dim=128, max_seq_len=2048,
                                             window_pattern="SSSL", norm_value_input=True,
                                             sum_head_attention=False, attention_sink=False, target_layers=targets))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    model = namespace["build_model_meta"](24)
    assert model.config.pooled_qk_target_layers == tuple(targets)
    assert model.estimate_flops() == training_flops
    assert namespace["d12_ref"].config.pooled_qk_target_layers == target_layers_for_depth(12)
