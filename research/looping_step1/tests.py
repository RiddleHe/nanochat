"""CPU algebra tests and real-checkpoint GPU preflight; never train the baseline."""
import os
import sys
os.environ["NANOCHAT_DTYPE"] = "float32" if sys.argv[-1] == "cpu" else "bfloat16"
import copy
import time
from types import MethodType
import numpy as np
import torch
from common import ROOT, SOURCE, GROUPS, VARIANTS, SETTINGS, now, sha, write_json, matrices, load_model, batch, data, parameter_digest, runtime_record
from interventions import VisitGradients, ScaleOutputs, attach_lora, LoopLoRA
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig


def tiny():
    torch.manual_seed(11)
    model = GPTLoop(GPTLoopConfig.from_block_depth(
        2, num_loops=4, n_embd=64, n_head=2, n_kv_head=2, vocab_size=128, sequence_len=32))
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0, 0.02)
            block.mlp.c_proj.weight.normal_(0, 0.02)
    return model


def cpu():
    torch.set_num_threads(2)
    model = tiny()
    x, y = torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))
    reference = model(x, y).detach()
    probe = VisitGradients(model)
    model(x, y).backward()
    rows, error = probe.statistics()
    probe.close()
    assert error < 2e-6, error
    checked = 1
    for kind in ("attn", "mlp"):
        for loop in range(4):
            hook = ScaleOutputs(model, kind, loop, 1.0)
            torch.testing.assert_close(model(x, y), reference, rtol=0, atol=0)
            hook.close()
            hook = ScaleOutputs(model, kind, loop, 0.9)
            changed = model(x, y)
            hook.close()
            assert not torch.equal(changed.detach(), reference)
            checked += 2
    for group in GROUPS:
        copies = {}
        for variant in VARIANTS:
            candidate = tiny()
            adapters, named = attach_lora(candidate, group, variant, small_rank=2)
            loss = candidate(x, y)
            torch.testing.assert_close(loss, reference, atol=0, rtol=0)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)
            copies[variant] = (adapters, named)
            checked += 2
        shared, independent = copies["shared_small"][0], copies["independent"][0]
        assert sum(p.numel() for _, p in copies["independent"][1]) == sum(
            p.numel() for _, p in copies["shared_large"][1])
        for s, d in zip(shared, independent):
            torch.testing.assert_close(s.B[0].grad, sum(b.grad for b in d.B), atol=2e-6, rtol=2e-5)
            for a in d.A:
                torch.testing.assert_close(a, s.A[0], atol=0, rtol=0)
        checked += 2
    write_json(ROOT / "tests_cpu.json", dict(passed=True, checks=checked,
                                             source_hashes={p.name: sha(p) for p in SOURCE.glob("*.py")},
                                             gradient_sum_error=error, time=now()))


def clear_adapters(model):
    for block in model.transformer.h:
        for parent in (block.attn, block.mlp):
            for name, module in list(parent.named_children()):
                if isinstance(module, LoopLoRA):
                    setattr(parent, name, module.base)
    model._forward_blocks = MethodType(GPTLoop._forward_blocks, model)
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)


def gpu():
    start = time.monotonic()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert json_cpu_passed()
    model, _, _ = load_model()
    frozen = list(model.named_parameters())
    digest = parameter_digest(frozen)
    x, y = batch(data("gradient"), slice(0, 2))
    model.requires_grad_(False)
    for _, m in matrices(model):
        m.weight.requires_grad_(True)
    probe = VisitGradients(model)
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    loss = model(x, y)
    loss.backward()
    rows, error = probe.statistics()
    probe.close()
    assert error < 0.02, error
    gradient_seconds = time.monotonic() - tick
    gradient_memory = torch.cuda.max_memory_allocated() / 2**20
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    del rows
    with torch.no_grad():
        reference = model(x, y, loss_reduction="none")
    sham = ScaleOutputs(model, "mlp", 3, 1.0)
    with torch.no_grad():
        torch.testing.assert_close(model(x, y, loss_reduction="none"), reference, atol=0, rtol=0)
    sham.close()
    variants = []
    for group in GROUPS:
        for variant in VARIANTS:
            clear_adapters(model)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            adapters, named = attach_lora(model, group, variant)
            parameters = [p for _, p in named]
            with torch.no_grad():
                torch.testing.assert_close(model(x, y, loss_reduction="none"), reference, atol=0, rtol=0)
            count = sum(p.numel() for p in parameters)
            assert count == (2949120 if variant == "shared_small" else 11796480)
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.95),
                                          eps=1e-8, weight_decay=0.0, fused=True)
            durations = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tick = time.monotonic()
                value = model(x, y)
                assert torch.isfinite(value)
                value.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                optimizer.step()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - tick)
            if group == "mlp" and variant == "independent":
                # Save after three updates; compare update four with a restored update four.
                saved = dict(adapters={n: p.detach().cpu().clone() for n, p in named},
                             optimizer=copy.deepcopy(optimizer.state_dict()))
                torch.save(saved, ROOT / "preflight_resume.pt")
                optimizer.zero_grad(set_to_none=True)
                model(x, y).backward()
                optimizer.step()
                expected = {n: p.detach().cpu().clone() for n, p in named}
                loaded = torch.load(ROOT / "preflight_resume.pt", weights_only=False)
                with torch.no_grad():
                    for n, p in named:
                        p.copy_(loaded["adapters"][n])
                optimizer.load_state_dict(loaded["optimizer"])
                optimizer.zero_grad(set_to_none=True)
                model(x, y).backward()
                optimizer.step()
                for n, p in named:
                    torch.testing.assert_close(p.cpu(), expected[n], atol=0, rtol=0)
            row = dict(group=group, variant=variant, trainable=count,
                       loss_after_test_updates=value.item(), seconds_per_microbatch=float(np.median(durations[1:])),
                       peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
            variants.append(row)
            print(row, flush=True)
            assert parameter_digest(frozen) == digest
            del optimizer, parameters, named, adapters
    clear_adapters(model)
    write_json(ROOT / "tests_gpu.json", dict(
        passed=True, time=now(), real_checkpoint=True, batch=2, length=2048,
        gradient_sum_error=error, gradient_batch_seconds=gradient_seconds,
        gradient_peak_memory_mib=gradient_memory, all_9_zero_adapter_parity=True,
        all_9_backward_finite=True, frozen_baseline_unchanged=True,
        exact_adapter_optimizer_resume_passed=True, variants=variants,
        source_hashes={p.name: sha(p) for p in SOURCE.glob("*.py")},
        runtime=runtime_record(), deterministic_backward=True,
        elapsed_seconds=time.monotonic() - start,
    ))


def json_cpu_passed():
    import json
    return json.loads((ROOT / "tests_cpu.json").read_text())["passed"]


if __name__ == "__main__":
    cpu() if sys.argv[-1] == "cpu" else gpu()
