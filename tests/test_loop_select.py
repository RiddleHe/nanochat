"""Run: NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 python -m unittest tests.test_loop_select -v

Also run with LOOP_TEST_DEVICE=cuda NANOCHAT_DTYPE=bfloat16 on the allocation.
These tests need only torch and the stdlib; no tokenizer or data download.
Optimizer integration tests require the repository-pinned torch >= 2.10;
older local builds reject the existing Muon kernel's mixed-dtype scalar lerp.
"""

import ast
import copy
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.model.gpt_base import GPTBaseConfig
from nanochat.model.gpt_base_no_embed_norm import GPTBaseNoEmbedNorm
from nanochat.model.gpt_loop_select import (
    GPTLoopSelect, GPTLoopSelectConfig, fixed_count_sigmoid,
    topk_straight_through, _WholeBlockResidual,
)
from nanochat.model_registry import get_model


torch.set_num_threads(2)
DEVICE = os.environ.get("LOOP_TEST_DEVICE", "cpu")
BF16 = COMPUTE_DTYPE == torch.bfloat16
OPTIMIZER_RUNTIME = tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 10)
TOL = dict(rtol=0.08, atol=0.004) if BF16 else dict(rtol=3e-4, atol=3e-6)


def actual_kv_cache_class():
    # Use the production class verbatim without importing engine's unrelated
    # tokenizer/checkpoint dependencies (e.g. the optional rustbpe extension).
    path = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVCache")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["KVCache"]


KVCache = actual_kv_cache_class()


def config(**kwargs):
    return GPTLoopSelectConfig(**(dict(sequence_len=32, vocab_size=64, n_layer=4,
        num_loops=2, n_embd=64, n_head=1, n_kv_head=1, select_k=2) | kwargs))


def make(cls=GPTLoopSelect, cfg=None, seed=17):
    with torch.device("meta"):
        model = cls(cfg or config())
    model.to_empty(device=DEVICE)
    torch.manual_seed(seed)
    model.init_weights()
    return model


def nontrivial(model):
    # Zero initial residual projections would make forward parity vacuous.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.025)
            block.mlp.c_proj.weight.normal_(std=0.025)
    return model


def tokens(length=11):
    return torch.randint(0, 64, (2, length), device=DEVICE)


def unrolled_reference(loop):
    c = loop.config
    ref = make(GPTBaseNoEmbedNorm, GPTBaseConfig(sequence_len=c.sequence_len,
        vocab_size=c.vocab_size, n_layer=c.effective_depth, n_head=c.n_head,
        n_kv_head=c.n_kv_head, n_embd=c.n_embd, window_pattern=c.window_pattern))
    source = loop.state_dict()
    state = {}
    for key in ref.state_dict():
        source_key = key
        if key.startswith("transformer.h."):
            parts = key.split(".")
            parts[2] = str(int(parts[2]) % c.n_layer)
            source_key = ".".join(parts)
        state[key] = source[source_key]
    ref.load_state_dict(state)
    # Match a repeated final-full physical pattern even for tiny odd-depth tests.
    for i, block in enumerate(ref.transformer.h):
        block.attn.window_size = loop.window_sizes[i]
    ref.window_sizes = list(loop.window_sizes)
    return ref


class LoopSelectTests(unittest.TestCase):
    def test_registry_config_physical_depth_and_flops(self):
        self.assertEqual(get_model("gpt_loop_select"), (GPTLoopSelectConfig, GPTLoopSelect))
        with torch.device("meta"):
            full = GPTLoopSelect(GPTLoopSelectConfig())
            learned = GPTLoopSelect(GPTLoopSelectConfig(selector_mode="learned"))
        self.assertEqual(full.config.n_layer, 24)
        self.assertEqual(len(full.transformer.h), 24)
        self.assertEqual(full.cache_layers, 96)
        self.assertEqual(full.window_sizes, ([(512, 0)] * 3 + [(2048, 0)]) * 24)
        self.assertEqual(full.num_scaling_params()["total"], 780_140_544)
        self.assertEqual(learned.num_scaling_params()["selector"], 36_960)
        self.assertEqual(full.estimate_flops(), 18_194_890_752)
        self.assertEqual(learned.estimate_flops() - full.estimate_flops(), 884_736)
        self.assertEqual(learned.selector_weight.shape, (24, 1536))
        self.assertEqual(learned.selector_bias.shape, (4, 24))
        for kwargs in ({"n_layer": 0}, {"num_loops": 0}, {"select_k": 0},
                       {"selector_mode": "random"}, {"window_pattern": ""},
                       {"selector_mode": "learned", "select_k": 4}):
            with self.assertRaises(ValueError):
                config(**kwargs)

    def test_identical_initialization_rng_and_raw_embeddings(self):
        full = make()
        cpu_after_full = torch.get_rng_state().clone()
        device_after_full = torch.cuda.get_rng_state() if DEVICE == "cuda" else None
        learned = make(cfg=config(selector_mode="learned"))
        self.assertTrue(torch.equal(cpu_after_full, torch.get_rng_state()))
        if device_after_full is not None:
            self.assertTrue(torch.equal(device_after_full, torch.cuda.get_rng_state()))
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, learned.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 3, 64, device=DEVICE) * 3
        self.assertIs(full._prepare_embeddings(x), x)
        self.assertLess(abs(learned.selector_weight.std().item() - 0.02 / 8), 0.0005)
        self.assertTrue(torch.equal(learned.selector_bias, torch.zeros_like(learned.selector_bias)))

    def assert_external_unrolled_gradients(self):
        model = nontrivial(make())
        ref = unrolled_reference(model)
        x, y = tokens(), tokens()
        torch.testing.assert_close(model(x), ref(x), rtol=0, atol=0)
        model(x, y).backward()
        ref(x, y).backward()
        reference_params = dict(ref.named_parameters())
        physical_reference = make(GPTBaseNoEmbedNorm, GPTBaseConfig(sequence_len=32,
            vocab_size=64, n_layer=4, n_head=1, n_kv_head=1, n_embd=64))
        physical_reference.load_state_dict(model.state_dict())
        for name, p in model.named_parameters():
            if name.startswith("transformer.h."):
                parts = name.split(".")
                expected = []
                for loop in range(model.config.num_loops):
                    parts[2] = str(int(name.split(".")[2]) + loop * model.config.n_layer)
                    expected.append(reference_params[".".join(parts)].grad)
                grad = sum(expected)
            else:
                grad = reference_params[name].grad
            self.assertIsNotNone(p.grad)
            torch.testing.assert_close(p.grad, grad, **TOL)
            dict(physical_reference.named_parameters())[name].grad = grad.clone()
        return model, physical_reference

    def test_external_unrolled_outputs_and_gradients(self):
        self.assert_external_unrolled_gradients()

    @unittest.skipUnless(OPTIMIZER_RUNTIME, "Production optimizer requires repo-pinned torch >= 2.10")
    def test_external_unrolled_one_optimizer_update(self):
        model, physical_reference = self.assert_external_unrolled_gradients()
        # Ordinary base Muon/AdamW acting on independently unrolled summed
        # gradients is the reference for the shared model's complete update.
        model.setup_optimizer().step()
        physical_reference.setup_optimizer().step()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, dict(physical_reference.named_parameters())[name], **TOL)

    def test_fixed_count_jacobian_against_independent_double_finite_difference(self):
        scores = torch.tensor([[0.2, -0.7, 1.4, 0.9, -1.2, 0.1]],
                              device=DEVICE, requires_grad=True)
        v = torch.tensor([[0.6, -0.3, 0.2, 1.2, -0.8, 0.4]], device=DEVICE)
        soft = fixed_count_sigmoid(scores, 4)
        (soft * v).sum().backward()

        def scalar_objective(values):
            lo, hi = min(values) - 40, max(values) + 40
            for _ in range(100):
                threshold = (lo + hi) / 2
                probs = [1 / (1 + math.exp(threshold - z)) for z in values]
                if sum(probs) > 4:
                    lo = threshold
                else:
                    hi = threshold
            return sum(p * weight for p, weight in zip(probs, v[0].tolist()))

        values = scores.detach()[0].tolist()
        numerical = []
        for j in range(len(values)):
            plus, minus = list(values), list(values)
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numerical.append((scalar_objective(plus) - scalar_objective(minus)) / 2e-5)
        torch.testing.assert_close(scores.grad[0], torch.tensor(numerical, device=DEVICE), rtol=2e-5, atol=2e-7)
        self.assertLess(abs(scores.grad.sum().item()), 2e-7)
        self.assertLess(abs(soft.sum().item() - 4), 2e-6)

    def test_binary_topk_ties_soft_backward_and_numerical_failures(self):
        scores = torch.zeros(2, 24, device=DEVICE, requires_grad=True)
        gates = topk_straight_through(scores, 16)
        self.assertTrue(torch.equal(gates[:, :16], torch.ones_like(gates[:, :16])))
        self.assertTrue(torch.equal(gates[:, 16:], torch.zeros_like(gates[:, 16:])))
        (gates * torch.arange(24, device=DEVICE)).sum().backward()
        self.assertGreater(scores.grad.abs().sum().item(), 0)
        self.assertLess(scores.grad.sum(-1).abs().max().item(), 2e-5)
        # CUDA asynchronous asserts poison the context; numerical-failure tests
        # deliberately execute on CPU, where the failure is caught synchronously.
        for bad in (torch.tensor([[float("nan"), 0.0]]), torch.tensor([[-1e9, 1e9]])):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite|Saturated|did not converge"):
                fixed_count_sigmoid(bad, 1)

    def test_complete_block_residual_gate_exact_forward_and_derivative(self):
        h = torch.tensor([[[2., -3.], [0.5, 4.]]], device=DEVICE, requires_grad=True)
        candidate = torch.tensor([[[5., 1.], [-2., 8.]]], device=DEVICE, requires_grad=True)
        gates = torch.tensor([[[0.], [1.]]], device=DEVICE, requires_grad=True)
        out = _WholeBlockResidual.apply(h, candidate, gates)
        torch.testing.assert_close(out, torch.cat((h[:, :1], candidate[:, 1:]), 1), rtol=0, atol=0)
        out.sum().backward()
        torch.testing.assert_close(gates.grad, torch.tensor([[[7.], [1.5]]], device=DEVICE))
        torch.testing.assert_close(h.grad, torch.tensor([[[1., 1.], [0., 0.]]], device=DEVICE))
        torch.testing.assert_close(candidate.grad, 1 - h.grad)

    def test_once_per_loop_original_order_and_every_token_kv(self):
        model = nontrivial(make(cfg=config(selector_mode="learned")))
        # Force token-independent stable ties: exactly blocks 0 and 1 retained.
        with torch.no_grad():
            model.selector_weight.zero_()
            model.selector_bias.zero_()
        calls = []
        handles = [block.register_forward_hook(lambda m, a, o, j=j: calls.append(j))
                   for j, block in enumerate(model.transformer.h)]
        with patch.object(model, "selection_gates", wraps=model.selection_gates) as selection:
            cache = KVCache(2, 1, 32, 64, model.cache_layers, DEVICE, COMPUTE_DTYPE)
            with torch.no_grad():
                model(tokens(), kv_cache=cache)
            self.assertEqual(selection.call_count, 2)
        self.assertEqual(calls, list(range(4)) * 2)
        for handle in handles:
            handle.remove()
        # Both rejected blocks still wrote K/V for all 11 tokens, each round.
        for layer in (2, 3, 6, 7):
            self.assertTrue((cache.k_cache[layer, :, :11].abs().sum(dim=(-1, -2)) > 0).all())
        self.assertFalse(torch.equal(cache.k_cache[0], cache.k_cache[4]))

    def test_causal_cache_parity_for_full_and_learned_with_sliding_windows(self):
        for mode in ("full", "learned"):
            with self.subTest(mode=mode):
                model = nontrivial(make(cfg=config(selector_mode=mode, sequence_len=512))).eval()
                x = tokens(140)
                with torch.no_grad():
                    expected = model(x)
                    changed = x.clone()
                    changed[:, 129:] = (changed[:, 129:] + 1) % 64
                    torch.testing.assert_close(model(changed)[:, :129], expected[:, :129], **TOL)
                    for chunks in ([140], [67, 61, 12], [1] * 140):
                        cache = KVCache(2, 1, 512, 64, model.cache_layers, DEVICE, COMPUTE_DTYPE)
                        pos, outputs = 0, []
                        for size in chunks:
                            outputs.append(model(x[:, pos:pos + size], kv_cache=cache))
                            pos += size
                            self.assertEqual(cache.get_pos(), pos)
                        torch.testing.assert_close(torch.cat(outputs, 1), expected, **TOL)
                    bad = KVCache(2, 1, 512, 64, 4, DEVICE, COMPUTE_DTYPE)
                    with self.assertRaisesRegex(ValueError, "separate slots"):
                        model(x, kv_cache=bad)

    @unittest.skipUnless(OPTIMIZER_RUNTIME, "Production optimizer requires repo-pinned torch >= 2.10")
    def test_selector_learns_after_initial_zero_residual_update(self):
        model = make(cfg=config(selector_mode="learned"))
        optimizer = model.setup_optimizer()
        group = next(g for g in optimizer.param_groups if g.get("role") == "selector")
        self.assertEqual((group["kind"], group["lr"], group["betas"], group["eps"], group["weight_decay"]),
                         ("adamw", 3e-4, (0.9, 0.95), 1e-8, 0.0))
        ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        x, y = tokens(), tokens()
        model(x, y).backward()
        self.assertEqual(model.selector_weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(x, y).backward()
        self.assertGreater(model.selector_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.selector_bias.grad.abs().sum().item(), 0)
        before = model.selector_weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.selector_weight))

    def test_scorer_gradient_and_explicit_optimizer_partition(self):
        model = nontrivial(make(cfg=config(selector_mode="learned")))
        optimizer = model.setup_optimizer()
        scorer_ids = {id(model.selector_weight), id(model.selector_bias)}
        groups = [g for g in optimizer.param_groups if g.get("role") == "selector"]
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual({id(p) for p in group["params"]}, scorer_ids)
        self.assertEqual((group["kind"], group["lr"], group["betas"], group["eps"], group["weight_decay"]),
                         ("adamw", 3e-4, (0.9, 0.95), 1e-8, 0.0))
        self.assertFalse(any(id(p) in scorer_ids for g in optimizer.param_groups
                             if g["kind"] == "muon" for p in g["params"]))
        model(tokens(), tokens()).backward()
        for p in (model.selector_weight, model.selector_bias):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_activation_checkpointing_preserves_both_modes(self):
        for mode in ("full", "learned"):
            with self.subTest(mode=mode):
                ordinary = nontrivial(make(cfg=config(selector_mode=mode)))
                recomputed = copy.deepcopy(ordinary)
                recomputed.config.checkpoint_blocks = True
                x, y = tokens(), tokens()
                loss_a, loss_b = ordinary(x, y), recomputed(x, y)
                torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
                loss_a.backward()
                loss_b.backward()
                for a, b in zip(ordinary.parameters(), recomputed.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
