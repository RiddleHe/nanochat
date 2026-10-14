"""Run on CPU, then LOOP_TEST_DEVICE=cuda for BF16/FA3 coverage."""
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.model.gpt_base import GPTBase, GPTBaseConfig
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig, GPTLoopUntied
from nanochat.model_registry import get_model
from nanochat.training_state import check_finite_gradients, save_training_state, load_training_state

torch.set_num_threads(2)
DEVICE = os.environ.get("LOOP_TEST_DEVICE", "cpu")


def config(**kwargs):
    return GPTLoopConfig(**(dict(sequence_len=32, vocab_size=64, n_layer=4,
                                num_loops=2, n_embd=64, n_head=1, n_kv_head=1) | kwargs))


def make(cls=GPTLoop, cfg=None, seed=17):
    with torch.device("meta"):
        model = cls(cfg or config())
    model.to_empty(device=DEVICE)
    torch.manual_seed(seed)
    model.init_weights()
    return model


def nontrivial(model):
    # Zero output projections would make a forward-equivalence test vacuous.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.04)
            block.mlp.c_proj.weight.normal_(std=0.04)
        model.smear_lambda.fill_(0.3)
    return model


def matched_pair():
    loop = nontrivial(make())
    untied = make(GPTLoopUntied)
    state = loop.state_dict()
    for key in untied.state_dict():
        if key.startswith("transformer.h."):
            parts = key.split(".")
            parts[2] = str(int(parts[2]) % len(loop.transformer.h))
            state[key] = loop.state_dict()[".".join(parts)]
    untied.load_state_dict(state)
    return loop, untied


def tokens():
    return torch.randint(0, 64, (2, 12), device=DEVICE)


@pytest.mark.parametrize("kwargs", [{"num_loops": 0}, {"num_loops": 3},
                                   {"n_layer": 0}, {"window_pattern": "SSSL"}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize("args,message", [
    (["--model-type", "gpt_loop", "--fp8"], "BF16/FP32 only"),
    (["--model-type", "gpt_loop"], "explicit total batch size"),
    (["--model-type", "gpt_base", "--num-loops", "4"], "requires gpt_loop"),
])
def test_cli_rejects_invalid_recipe(args, message):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NANOCHAT_DTYPE="float32")
    result = subprocess.run([sys.executable, "-m", "scripts.base_train", *args],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and message in result.stderr


def test_registry_and_meta_initialization():
    for name in ("gpt_loop", "gpt_loop_untied"):
        cfg_cls, cls = get_model(name)
        model = make(cls)
        assert isinstance(model.config, cfg_cls)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert model.smear_lambda.item() == 0
        assert model.smear_gate.weight.abs().max() <= 24**-0.5
        duplicate = make(cls)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, duplicate.state_dict()[key], rtol=0, atol=0)


def test_matched_initialization_and_r1():
    loop, untied = make(), make(GPTLoopUntied)
    for i, block in enumerate(untied.transformer.h):
        for key, value in block.state_dict().items():
            torch.testing.assert_close(value, loop.transformer.h[i % 2].state_dict()[key], rtol=0, atol=0)
    x = tokens()
    torch.testing.assert_close(loop(x), untied(x), rtol=0, atol=0)
    cfg = config(num_loops=1)
    r1 = nontrivial(make(cfg=cfg))
    fields = asdict(cfg)
    fields.pop("num_loops")
    base = make(GPTBase, GPTBaseConfig(**fields))
    base.load_state_dict(r1.state_dict())
    torch.testing.assert_close(r1(x), base(x), rtol=0, atol=0)


def test_forward_and_summed_shared_gradients():
    loop, untied = matched_pair()
    x, y = tokens(), tokens()
    torch.testing.assert_close(loop(x), untied(x), rtol=0, atol=0)
    loop(x, y).backward()
    untied(x, y).backward()
    tol = dict(rtol=0.06, atol=2e-5) if COMPUTE_DTYPE == torch.bfloat16 else dict(rtol=2e-4, atol=2e-7)
    independent = dict(untied.named_parameters())
    for name, p in loop.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0, name
        if name.startswith("transformer.h."):
            parts = name.split(".")
            grads = []
            for offset in (0, 2):
                parts[2] = str(int(name.split(".")[2]) + offset)
                grads.append(independent[".".join(parts)].grad)
            expected = sum(grads)
        else:
            expected = independent[name].grad
        torch.testing.assert_close(p.grad, expected, **tol)


def test_counts_flops_and_independence():
    loop, untied = make(), make(GPTLoopUntied)
    assert len(loop.transformer.h) == 2 and len(untied.transformer.h) == 4
    assert loop.estimate_flops() == untied.estimate_flops()
    assert loop.num_scaling_params()["total"] < untied.num_scaling_params()["total"]
    pointers = [p.data_ptr() for p in untied.parameters()]
    assert len(pointers) == len(set(pointers))
    with torch.device("meta"):
        d24 = GPTLoop(config(n_layer=24, num_loops=4, n_embd=1536, n_head=12,
                             n_kv_head=12, vocab_size=32768, sequence_len=2048))
    assert d24.num_scaling_params()["total"] == 270_532_633
    assert d24.estimate_flops() == 5_284_823_040


def test_execution_counts_and_gqa():
    model = nontrivial(make(cfg=config(n_embd=128, n_head=2, n_kv_head=1, num_loops=4)))
    counts = {"embedding": 0, "head": 0, "block": 0}
    handles = []
    def hook(name):
        def increment(*args):
            counts[name] += 1
        return increment
    for name, module in [("embedding", model.transformer.wte), ("head", model.lm_head),
                         ("block", model.transformer.h[0])]:
        handles.append(module.register_forward_hook(hook(name)))
    loss = model(tokens(), tokens())
    loss.backward()
    check_finite_gradients(model, loss)
    assert counts == {"embedding": 1, "head": 1, "block": 4}
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("chunks", [[12], [1] * 12, [3, 2, 7]])
def test_cache_and_causality(chunks):
    model = nontrivial(make()).eval()
    cfg = model.config
    cache = KVCache(2, cfg.n_kv_head, 32, 64, cfg.n_layer, DEVICE, COMPUTE_DTYPE)
    x = tokens()
    with torch.no_grad():
        expected = model(x)
        pos, outputs = 0, []
        for size in chunks:
            outputs.append(model(x[:, pos:pos+size], kv_cache=cache))
            pos += size
            assert cache.get_pos() == pos
        actual = torch.cat(outputs, dim=1)
        tol = dict(rtol=0.08, atol=0.003) if COMPUTE_DTYPE == torch.bfloat16 else dict(rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(actual, expected, **tol)
        changed = x.clone()
        changed[:, 7:] = (changed[:, 7:] + 1) % 64
        torch.testing.assert_close(model(changed)[:, :7], expected[:, :7], **tol)
    assert not torch.equal(cache.k_cache[0], cache.k_cache[2])
    cache.reset()
    assert cache.get_pos() == 0 and cache.prev_embedding is None
    bad = KVCache(2, 1, 32, 64, 2, DEVICE, COMPUTE_DTYPE)
    with pytest.raises(ValueError, match="effective layer"):
        model(x, kv_cache=bad)


def test_optimizer_checkpoint_and_nonfinite(tmp_path):
    model = nontrivial(make())
    opt = model.setup_optimizer()
    parameters = [p for group in opt.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters}) == len(list(model.parameters()))
    x, y = tokens(), tokens()
    before = model.transformer.h[0].attn.c_q.weight.detach().clone()
    loss = model(x, y)
    loss.backward()
    check_finite_gradients(model, loss)
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert not torch.equal(before, model.transformer.h[0].attn.c_q.weight)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "config": asdict(model.config)}, tmp_path / "checkpoint.pt")
    data = torch.load(tmp_path / "checkpoint.pt", map_location=DEVICE, weights_only=True)
    restored = make(cfg=GPTLoopConfig(**data["config"]))
    restored.load_state_dict(data["model"], assign=True)
    restored_opt = restored.setup_optimizer()
    restored_opt.load_state_dict(data["optimizer"])
    for m, optimizer in [(model, opt), (restored, restored_opt)]:
        m(x, y).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    model(x, y).backward()
    next(model.parameters()).grad.flatten()[0] = float("nan")
    with pytest.raises(FloatingPointError):
        check_finite_gradients(model, loss)
    save_training_state(tmp_path, 2, 0, x, y, {"doc_buffer": [[1, 2]], "doc_idx": 5})
    draw = torch.rand(4, device=DEVICE)
    rx, ry, state = load_training_state(tmp_path, 2, 0, DEVICE)
    torch.testing.assert_close(rx, x)
    torch.testing.assert_close(ry, y)
    torch.testing.assert_close(draw, torch.rand(4, device=DEVICE), rtol=0, atol=0)
    assert state == {"doc_buffer": [[1, 2]], "doc_idx": 5}


@pytest.mark.skipif(DEVICE != "cuda", reason="GPU compile smoke")
def test_compiled_forward_backward():
    model, reference = matched_pair()
    x, y = tokens(), tokens()
    compiled = torch.compile(model, fullgraph=True)
    loss = compiled(x, y)
    expected = reference(x, y)
    torch.testing.assert_close(loss, expected, rtol=0.002, atol=0.002)
    loss.backward()
    check_finite_gradients(model, loss)
