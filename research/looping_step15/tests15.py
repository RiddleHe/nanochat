"""Step 1.5 checks. `python tests15.py cpu` (tiny model) / `python tests15.py gpu` (real checkpoint, 1 GPU)."""
import os
import sys
os.environ['NANOCHAT_DTYPE'] = 'float32' if sys.argv[-1] == 'cpu' else 'bfloat16'
import copy
import json
import time
from types import MethodType
import numpy as np
import torch
from shared15 import (ROOT, VARIANTS, EXPECTED_PARAMETERS, PROTOCOL, previous, probe, step1_int, now, sha,
                      sources, write_json, select, jobs)
from local15 import attach_local, LocalScale
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig


def detach(model, layer):
    mlp = model.transformer.h[layer].mlp
    if isinstance(mlp.c_fc, step1_int.LoopLoRA):
        mlp.c_fc = mlp.c_fc.base
    model._forward_blocks = MethodType(GPTLoop._forward_blocks, model)
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)


def record_visits(adapter):
    seen = []
    original = adapter.forward

    def forward(x):
        seen.append(adapter.visit)
        return original(x)
    adapter.forward = forward
    return seen


def tiny():
    torch.manual_seed(11)
    model = GPTLoop(GPTLoopConfig.from_block_depth(
        3, num_loops=4, n_embd=64, n_head=2, n_kv_head=2, vocab_size=128, sequence_len=32))
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0, 0.02)
            block.mlp.c_proj.weight.normal_(0, 0.02)
    return model


def cpu():
    torch.set_num_threads(2)
    checks = 0
    # Selection rule: ties by layer, deterministic random picks, undefined scores rejected.
    scores = {f'L{i + 1:02d}.up': float(i % 5) for i in range(24)}
    a, b = select(scores), select(dict(scores))
    assert a == b and len(set(a['locations'])) == 6
    assert a['high'] == ['L05.up', 'L10.up'] and a['low'] == ['L01.up', 'L06.up'], a
    bad = dict(scores, **{'L03.up': None})
    try:
        select(bad)
        raise AssertionError('None score accepted')
    except ValueError:
        pass
    assert len(jobs(a['locations'])) == 18
    checks += 4
    x, y = torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))
    model = tiny()
    reference = model(x, y).detach()
    # Gradient reconstruction with the Step 2 probe.
    mods = {f'L{i + 1:02d}.up': b.mlp.c_fc for i, b in enumerate(model.transformer.h)}
    model.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)
    hooks, acc = probe.Probe(mods), probe.Accumulator()
    model(x, y).backward()
    rows = acc.add(hooks.take(), {n: m.weight.grad for n, m in mods.items()})
    hooks.close()
    assert max(r['sum_relative_error'] for r in rows) < 1e-5
    detach(model, 0)
    checks += 1
    # Local scaling: sham exact; one round changed; other layers untouched.
    for loop in range(4):
        h = LocalScale(model, 1, loop, 1.0)
        torch.testing.assert_close(model(x, y), reference, rtol=0, atol=0)
        h.close()
        h = LocalScale(model, 1, loop, 0.9)
        assert not torch.equal(model(x, y).detach(), reference)
        h.close()
        checks += 2
    # Local LoRA: one location only, exact start, routing, gradient relation shared vs independent.
    grads = {}
    for variant in VARIANTS:
        m = tiny()
        adapter, named = attach_local(m, 1, variant, check_count=False)
        visits = record_visits(adapter)
        loss = m(x, y)
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        assert visits == [0, 1, 2, 3], visits
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)
        assert all(not isinstance(b.mlp.c_fc, step1_int.LoopLoRA) for i, b in enumerate(m.transformer.h) if i != 1)
        assert not any(p.requires_grad for n, p in m.named_parameters() if '.A.' not in n and '.B.' not in n)
        grads[variant] = adapter
        checks += 5
    s, d = grads['shared_small'], grads['independent']
    for a_ in d.A:
        torch.testing.assert_close(a_, s.A[0], atol=0, rtol=0)
    torch.testing.assert_close(s.B[0].grad, sum(b_.grad for b_ in d.B), atol=2e-6, rtol=2e-5)
    assert all(b_.grad.abs().sum() > 0 for b_ in d.B), 'Each round must route to its own adapter'
    rank, _ = VARIANTS['independent']
    assert EXPECTED_PARAMETERS['independent'] == 4 * rank * (1536 + 6144)
    assert EXPECTED_PARAMETERS['shared_large'] == 32 * (1536 + 6144)
    assert EXPECTED_PARAMETERS['shared_small'] == 8 * (1536 + 6144)
    checks += 6
    write_json(ROOT / 'tests_cpu.json', dict(passed=True, checks=checks, source_hashes=sources(), time=now()))
    print('CPU_TESTS_PASSED', checks, flush=True)


def gpu():
    started = time.monotonic()
    assert json.loads((ROOT / 'tests_cpu.json').read_text())['passed']
    model, _, _ = previous.load_model()
    model.requires_grad_(False)
    frozen = list(model.named_parameters())
    digest = previous.parameter_digest(frozen)
    dev = previous.data('development')
    x, y = previous.batch(dev, slice(0, 2))
    with torch.no_grad():
        reference = model(x, y, loss_reduction='none')
    report = dict(variants=[])
    # Gradient reconstruction on the real model (Step 2 probe, all 24 expansions).
    mods = {f'L{i + 1:02d}.up': b.mlp.c_fc for i, b in enumerate(model.transformer.h)}
    for m in mods.values():
        m.weight.requires_grad_(True)
    hooks, acc = probe.Probe(mods), probe.Accumulator()
    xd, yd = previous.batch(previous.data('gradient'), slice(0, 2))
    tick = time.monotonic()
    model(xd, yd).backward()
    rows = acc.add(hooks.take(), {n: m.weight.grad for n, m in mods.items()})
    hooks.close()
    report['gradient_sum_error'] = max(r['sum_relative_error'] for r in rows)
    report['gradient_batch_seconds'] = time.monotonic() - tick
    assert report['gradient_sum_error'] < PROTOCOL['gradient_sum_relative_tolerance'], report
    del acc, rows
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    torch.cuda.empty_cache()
    # Scale sham exact; one-layer scaling changes output.
    layer = 13
    h = LocalScale(model, layer, 2, 1.0)
    with torch.no_grad():
        torch.testing.assert_close(model(x, y, loss_reduction='none'), reference, atol=0, rtol=0)
    h.close()
    h = LocalScale(model, layer, 2, 0.9)
    with torch.no_grad():
        assert not torch.equal(model(x, y, loss_reduction='none'), reference)
    h.close()
    trn = previous.data('train')
    order = np.random.default_rng(42).permutation(len(trn))
    for variant in VARIANTS:
        detach(model, layer)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        adapter, named = attach_local(model, layer, variant)
        params = [p for _, p in named]
        visits = record_visits(adapter)
        with torch.no_grad():
            torch.testing.assert_close(model(x, y, loss_reduction='none'), reference, atol=0, rtol=0)
        assert visits == [0, 1, 2, 3], visits
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
        durations = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            xb, yb = previous.batch(trn, order[2 * step:2 * step + 2])
            torch.cuda.synchronize()
            t = time.monotonic()
            value = model(xb, yb)
            value.backward()
            if step == 0 and variant == 'independent':
                assert all(b.grad.abs().sum() > 0 for b in adapter.B), 'Round routing failed'
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            opt.step()
            torch.cuda.synchronize()
            durations.append(time.monotonic() - t)
        if variant == 'independent':
            saved = dict(adapters={n: p.detach().cpu().clone() for n, p in named},
                         optimizer=copy.deepcopy(opt.state_dict()))
            xb, yb = previous.batch(trn, order[6:8])
            opt.zero_grad(set_to_none=True)
            model(xb, yb).backward()
            opt.step()
            expected = {n: p.detach().cpu().clone() for n, p in named}
            with torch.no_grad():
                for n, p in named:
                    p.copy_(saved['adapters'][n])
            opt.load_state_dict(saved['optimizer'])
            opt.zero_grad(set_to_none=True)
            model(xb, yb).backward()
            opt.step()
            for n, p in named:
                torch.testing.assert_close(p.detach().cpu(), expected[n], atol=0, rtol=0)
            report['exact_resume_passed'] = True
        assert previous.parameter_digest(frozen) == digest, 'Base weights changed'
        sec = float(np.median(durations[1:]))
        row = dict(variant=variant, trainable=sum(p.numel() for p in params), loss=value.item(),
                   seconds_per_microbatch=sec, est_hours_per_run=sec * 128 * 200 / 3600 + 0.25,
                   peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
        report['variants'].append(row)
        print(row, flush=True)
        del opt, params, named, adapter
    detach(model, layer)
    write_json(ROOT / 'tests_gpu.json', dict(
        passed=True, time=now(), **report, zero_adapter_parity=True, scale_sham_exact=True,
        single_location_routing=True, frozen_baseline_unchanged=True, source_hashes=sources(),
        runtime=previous.runtime_record(), elapsed_seconds=time.monotonic() - started))
    print('GPU_PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    cpu() if sys.argv[-1] == 'cpu' else gpu()
