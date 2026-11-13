"""Checks for per-loop Muon.

  python tests_perloop.py cpu     tiny model, CPU, fp32 (math of the twin and the optimizer)
  python tests_perloop.py gpu     real baseline, 1 GPU: reproduces the baseline's first steps,
                                  smoke-tests v1/v2, checks exact resume (writes into --out)
"""
import os
import sys
os.environ["NANOCHAT_DTYPE"] = "float32" if sys.argv[1] == "cpu" else "bfloat16"
import copy
import json
import re
import subprocess

import torch
import torch._dynamo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig  # noqa: E402
from nanochat.optim import MuonAdamW  # noqa: E402
from perloop import PerLoopMuonAdamW, build_twin, sync  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def tiny(seed=11):
    torch.manual_seed(seed)
    m = GPTLoop(GPTLoopConfig.from_block_depth(3, num_loops=4, n_embd=64, n_head=2, n_kv_head=2,
                                               vocab_size=128, sequence_len=32))
    m.init_weights()
    with torch.no_grad():  # projections start at zero; make every matrix get a gradient
        for b in m.transformer.h:
            b.attn.c_proj.weight.normal_(0, 0.02)
            b.mlp.c_proj.weight.normal_(0, 0.02)
    return m


def batches(n, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 128, (2, 16), generator=g), torch.randint(0, 128, (2, 16), generator=g))
            for _ in range(n)]


def groups(model):
    opt = model.setup_optimizer(weight_decay=0.1)
    return [dict(g) for g in opt.param_groups]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def set_hparams(opt, step):
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"]
        if g["kind"] == "muon":
            g["momentum"] = 0.85 + 0.01 * step


def cpu():
    torch.set_num_threads(4)
    checks = 0
    data = batches(4)

    # 1. Twin = GPTLoop: identical loss, per-loop gradients sum to the shared gradient.
    master = tiny()
    reference = copy.deepcopy(master)
    twin, loops = build_twin(master)
    x, y = data[0]
    torch.testing.assert_close(twin(x, y), reference(x, y), rtol=0, atol=0)
    twin(x, y).backward()
    reference(x, y).backward()
    for (name, p), (_, q) in zip(master.named_parameters(), reference.named_parameters()):
        g = sum(c.grad for c in loops[p]) if p in loops else p.grad
        assert rel(g, q.grad) < 1e-5, (name, rel(g, q.grad))
        if p in loops:
            assert all(c.grad.abs().sum() > 0 for c in loops[p]), f"{name}: a loop got no gradient"
    assert all(p.grad is None for p in master.transformer.h.parameters()), "master blocks must not be used"
    assert master.transformer.wte.weight is twin.transformer.wte.weight
    checks += 4

    # 2. native mode == nanochat MuonAdamW, given the same gradients (3 steps, all params).
    #    Run nanochat's compiled kernels eagerly so the comparison is exact (bitwise).
    torch._dynamo.config.disable = True
    master, reference = tiny(), tiny()
    twin, loops = build_twin(master)
    mine = PerLoopMuonAdamW(groups(master), "native", loops)
    ref_opt = MuonAdamW(groups(reference))
    for g in ref_opt.param_groups:
        g["initial_lr"] = g["lr"]
    ref_params = dict(reference.named_parameters())
    for step, (x, y) in enumerate(data[:3]):
        set_hparams(mine, step)
        set_hparams(ref_opt, step)
        twin(x, y).backward()
        for name, p in master.named_parameters():
            ref_params[name].grad = (torch.stack([c.grad for c in loops[p]]).sum(0) if p in loops
                                     else p.grad).clone()
        mine.step()
        sync(loops)
        ref_opt.step()
        twin.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    for name, p in master.named_parameters():
        torch.testing.assert_close(p, ref_params[name], rtol=0, atol=0, msg=name)
    for p, copies in loops.items():
        assert all(torch.equal(c, p) for c in copies), "twin out of sync"
    torch._dynamo.config.disable = False
    checks += 2

    # 3. Variants on constructed gradients: identical loops / one active loop.
    def run(mode, loop_scale, steps=3):
        m = tiny()
        tw, lp = build_twin(m)
        opt = PerLoopMuonAdamW(groups(m), mode, lp)
        start = {n: p.detach().clone() for n, p in m.named_parameters()}
        stats = []
        for step, (x, y) in enumerate(data[:steps]):
            set_hparams(opt, step)
            tw(x, y).backward()
            with torch.no_grad():
                for p, copies in lp.items():
                    total = sum(c.grad for c in copies)
                    for k, c in enumerate(copies):
                        c.grad = total * loop_scale[k]
            opt.step()
            sync(lp)
            tw.zero_grad(set_to_none=True)
            stats.append(opt.stats)
        return {n: (p.detach() - start[n]) for n, p in m.named_parameters()}, stats

    # One active loop (grad of loop k = full gradient, others 0): per-loop momentum and PE see
    # exactly the native input, so v1 must reproduce native exactly, for every k.
    native, _ = run("native", [0.25] * 4)
    for k in range(4):
        only = [1.0 if j == k else 0.0 for j in range(4)]
        v1k, stats = run("v1", only)
        worst = max(rel(v1k[n], native[n]) for n in native)
        print(f"v1 vs native, only loop {k} active: max rel diff {worst:.2e}", flush=True)
        assert worst < 1e-5
        assert all(((r - 1).abs() < 1e-3).all() and (c > 0.9999).all() for _, r, c in stats[-1])
    checks += 4
    # Four identical loops: B = 4 x A up to bf16 noise. Not exact: PE amplifies weak directions,
    # so bf16 rounding of a rescaled input already moves cos(PE(x), PE(x/4)) to ~0.99 on this model.
    same = [0.25] * 4
    v1, stats = run("v1", same)
    for _, ratio, cos in stats[-1]:
        assert (cos > 0.95).all() and ((ratio - 4).abs() < 0.15).all(), (ratio, cos)
    blocks = [n for n in native if n.startswith("transformer.h.")]
    worst = max(rel(v1[n], native[n]) for n in blocks)
    print(f"v1 vs native, identical loops (bf16 noise only): max rel diff {worst:.3f}", flush=True)
    assert worst < 0.3
    checks += 2
    v2, stats = run("v2", same)
    for _, ratio, cos in stats[-1]:  # PE applied twice: same direction up to PE's own S' spread
        assert (cos > 0.9).all() and (ratio > 0.5).all() and (ratio < 2).all(), (ratio, cos)
    assert all(torch.isfinite(d).all() for d in v2.values())
    checks += 2

    # 4. Real loops disagree: v1 keeps |A| per matrix and differs from native; v2 differs.
    def real(mode):
        m = tiny()
        tw, lp = build_twin(m)
        opt = PerLoopMuonAdamW(groups(m), mode, lp)
        for step, (x, y) in enumerate(data[:3]):
            set_hparams(opt, step)
            tw(x, y).backward()
            opt.step()
            sync(lp)
            tw.zero_grad(set_to_none=True)
        return m, opt
    m1, o1 = real("v1")
    ratios = torch.cat([r for _, r, _ in o1.stats])
    coss = torch.cat([c for _, _, c in o1.stats])
    assert (coss < 0.9999).any() and torch.isfinite(ratios).all()
    checks += 1

    # 5. Optimizer state round-trip: resume gives the same next step exactly.
    m, opt = real("v1")
    tw, lp = build_twin(m)
    opt2 = PerLoopMuonAdamW(groups(m), "v1", lp)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    m_b = copy.deepcopy(m)
    tw_b, lp_b = build_twin(m_b)
    opt_b = PerLoopMuonAdamW(groups(m_b), "v1", lp_b)
    opt_b.load_state_dict(copy.deepcopy(opt.state_dict()))
    x, y = data[3]
    for model_, twin_, loops_, o in [(m, tw, lp, opt2), (m_b, tw_b, lp_b, opt_b)]:
        set_hparams(o, 3)
        twin_(x, y).backward()
        o.step()
        sync(loops_)
    for (n, p), (_, q) in zip(m.named_parameters(), m_b.named_parameters()):
        assert torch.equal(p, q), n
    assert set(opt.state[next(iter(lp))].keys()) == {
        "momentum_buffer", "second_momentum_buffer", "loop_momentum_buffer", "b_second_momentum_buffer"}
    checks += 2

    json.dump(dict(passed=True, checks=checks), open(os.path.join(HERE, "tests_cpu.json"), "w"))
    print("CPU_TESTS_PASSED", checks, flush=True)


# ---------------------------------------------------------------------------

def train(out, *extra):
    cmd = [sys.executable, "-u", os.path.join(HERE, "train_perloop.py"), "--out", out, *extra]
    print(" ".join(cmd), flush=True)
    log = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(out + ".log"), "w").write(log.stdout + log.stderr)
    assert log.returncode == 0, log.stderr[-3000:]
    return log.stdout


def losses(text):
    return {int(s): float(v) for s, v in re.findall(r"^step (\d+)/\d+ .*?raw_loss: ([\d.]+)", text, re.M)}


def gpu():
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    base = open("/hdd/mh3897/cc/looping-runs/b12_r4_10b_20261019/train.log").read()
    base_loss = losses(base)
    base_val0 = float(re.search(r"Step 00000 \| Validation bpb: ([\d.]+)", base).group(1))
    report = {}

    # a) native mode on 1 GPU reproduces the 2-GPU baseline: val at step 0, losses of steps 0-9.
    text = train(f"{out}/native", "--mode", "native", "--stop-after", "10", "--save-steps", "--eval-every", "10")
    mine = losses(text)
    val0 = float(re.search(r"Step 00000 \| Validation bpb: ([\d.]+)", text).group(1))
    report["native_val0"], report["baseline_val0"] = val0, base_val0
    report["native_loss_diff"] = {s: mine[s] - base_loss[s] for s in range(10)}
    print(json.dumps(report, indent=2), flush=True)
    assert abs(val0 - base_val0) < 2e-4, "Validation data/eval differs from the baseline"
    assert abs(mine[0] - base_loss[0]) < 2e-3, "First batch differs: data order not reproduced"
    assert max(abs(d) for d in report["native_loss_diff"].values()) < 2e-2

    # b) v1 / v2 run, stats are finite; v1 with a save at step 2 for the resume check.
    for mode in ("v1", "v2"):
        text = train(f"{out}/{mode}", "--mode", mode, "--stop-after", "4", "--save-steps", "2",
                     "--eval-every", "0")
        report[f"{mode}_loss"] = losses(text)
        report[f"{mode}_lines"] = [l for l in text.splitlines() if l.startswith("step ")]
    # c) exact resume: continue v1 from step 2 and compare steps 2-3 and the final weights.
    os.rename(f"{out}/v1/checkpoints/model_000004.pt", f"{out}/v1_model_000004_continuous.pt")
    text = train(f"{out}/v1", "--mode", "v1", "--resume", "2", "--stop-after", "4", "--eval-every", "0")
    resumed = losses(text)
    report["resume_loss_diff"] = {s: resumed[s] - report["v1_loss"][s] for s in (2, 3)}
    a = torch.load(f"{out}/v1_model_000004_continuous.pt", map_location="cpu")
    b = torch.load(f"{out}/v1/checkpoints/model_000004.pt", map_location="cpu")
    report["resume_max_weight_diff"] = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
    print(json.dumps(report, indent=2), flush=True)
    assert max(abs(d) for d in report["resume_loss_diff"].values()) < 1e-4
    report["passed"] = True
    json.dump(report, open(os.path.join(HERE, "tests_gpu.json"), "w"), indent=2)
    print("GPU_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    cpu() if sys.argv[1] == "cpu" else gpu()
