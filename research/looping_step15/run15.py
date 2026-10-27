"""Step 1.5 jobs. Base weights never update.

  gradient        A/B on D1 and D2 for all 24 MLP expansions; freezes selection.json
  L14_independent one-location adapter run (any selected location x variant)
  scaling         48 one-layer one-round output scales on a dev subset; chosen ones on confirmation
  baseline_eval   unchanged model on full development and new confirmation
"""
import argparse
import gc
import json
import math
import os
import time
import traceback
import numpy as np
import torch
from shared15 import (ROOT, PROTOCOL, VARIANTS, EXPECTED_PARAMETERS, previous, probe, now, sha, write_json,
                      verify, selection_data, confirmation, frozen_selection, select, layer_index)
from local15 import attach_local, LocalScale


def state(folder, phase, **extra):
    write_json(folder / 'status.json', dict(phase=phase, time=now(), pid=os.getpid(), **extra))


def append(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


def paired(a, b):
    """Paired sequence bootstrap of BPB difference a-b (ratio of sums), exploratory."""
    def bpb(rows, idx):
        bn = np.array([r['byte_nats'] for r in rows])[idx].sum(-1)
        nb = np.array([r['bytes'] for r in rows])[idx].sum(-1)
        return bn / (math.log(2) * nb)
    n = len(a)
    assert n == len(b) and all(x['sequence'] == y['sequence'] for x, y in zip(a, b))
    full = np.arange(n)
    idx = np.random.default_rng(42).integers(0, n, size=(2000, n))
    boots = bpb(a, idx) - bpb(b, idx)
    return dict(delta_bpb=float(bpb(a, full) - bpb(b, full)),
                ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
                uncertainty='paired sequence bootstrap (2000, seed 42); evaluation sampling only')


# ---------------------------------------------------------------- gradient + selection
def gradient(folder):
    state(folder, 'loading')
    model, _, _ = previous.load_model()
    model.requires_grad_(False)
    modules = {f'L{i + 1:02d}.up': b.mlp.c_fc for i, b in enumerate(model.transformer.h)}
    for m in modules.values():
        m.weight.requires_grad_(True)
    before = previous.parameter_digest(model.named_parameters())
    tol = PROTOCOL['gradient_sum_relative_tolerance']
    started = time.monotonic()
    reports = {}
    for name in PROTOCOL['selections']:
        data = selection_data(name)
        acc, hooks, losses = probe.Accumulator(), probe.Probe(modules), []
        for i in range(PROTOCOL['batches']):
            model.zero_grad(set_to_none=True)
            x, y = previous.batch(data, slice(2 * i, 2 * i + 2))
            loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic loss')
            losses.append(float(loss.detach()))
            loss.backward()
            rows = acc.add(hooks.take(), {n: m.weight.grad for n, m in modules.items()})
            worst = max(r['sum_relative_error'] for r in rows)
            if worst >= tol:
                raise RuntimeError(f'Gradient reconstruction failed {name} batch {i}: {worst}')
            append(folder / f'{name}_batches.jsonl', dict(batch=i + 1, loss=losses[-1], rows=rows))
            state(folder, 'measuring', selection=name, batch=i + 1, seconds=time.monotonic() - started)
            print(json.dumps(dict(selection=name, batch=i + 1, loss=losses[-1], max_error=worst)), flush=True)
        hooks.close()
        summary = acc.summary()
        for r in summary:
            assert r['max_batch_sum_relative_error'] < tol and r['aggregate_sum_relative_error'] < tol, r
        reports[name] = dict(mean_ce=sum(losses) / len(losses), rows=summary)
        del acc
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    assert previous.parameter_digest(model.named_parameters()) == before
    scores = {r['location']: r['conflict_B'] for r in reports['D1']['rows']}
    chosen = select(scores)
    d2 = {r['location']: r['conflict_B'] for r in reports['D2']['rows']}
    order1 = sorted(scores, key=lambda k: -scores[k])
    order2 = sorted(d2, key=lambda k: -d2[k])
    rank1 = {k: i for i, k in enumerate(order1)}
    rank2 = {k: i for i, k in enumerate(order2)}
    keys = list(scores)
    spearman = float(np.corrcoef([rank1[k] for k in keys], [rank2[k] for k in keys])[0, 1])
    stability = dict(spearman_C_D1_vs_D2=spearman, D2_top2=order2[:2], D2_bottom2=order2[-2:][::-1],
                     D1_top2=chosen['high'], D1_bottom2=chosen['low'])
    chosen['stability'] = stability
    write_json(ROOT / 'selection.json', chosen)
    write_json(folder / 'results.json', dict(
        experiment='gradient', completed_utc=now(), checkpoint=19074, reports=reports,
        selection=chosen, selection_sha256=sha(ROOT / 'selection.json'),
        frozen_weights_unchanged=True, parameter_digest=before,
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        note='Selection frozen here, before any adapter training. C larger = less aligned, not proof of harm.'))
    state(folder, 'complete', selection=chosen['locations'])
    print(json.dumps(dict(selected=chosen['locations'], stability=stability)), flush=True)


# ---------------------------------------------------------------- local LoRA training
def lr_at(step):
    p = PROTOCOL
    if step < p['warmup']:
        return p['learning_rate'] * (step + 1) / p['warmup']
    progress = (step - p['warmup']) / (p['updates'] - p['warmup'] - 1)
    return p['final_learning_rate'] + (p['learning_rate'] - p['final_learning_rate']) * 0.5 * (
        1 + math.cos(math.pi * progress))


def save_adapter(path, step, named, optimizer, experiment, selection_sha):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(step=step, adapters={n: p.detach().cpu() for n, p in named},
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), protocol=PROTOCOL, experiment=experiment,
                    manifest_sha256=sha(ROOT / 'manifest.json'), selection_sha256=selection_sha), tmp)
    tmp.replace(path)


def restore_adapter(path, named, optimizer, experiment, selection_sha):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['protocol'] == PROTOCOL and saved['experiment'] == experiment
    assert saved['manifest_sha256'] == sha(ROOT / 'manifest.json')
    assert saved['selection_sha256'] == selection_sha
    with torch.no_grad():
        for n, p in named:
            p.copy_(saved['adapters'][n])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    return saved['step']


def train(folder, location, variant, resume=False):
    experiment = f'{location}_{variant}'
    chosen, selection_sha = frozen_selection()
    assert location in chosen['locations'], f'{location} not in frozen selection'
    layer = layer_index(location)
    state(folder, 'loading', experiment=experiment)
    model, _, _ = previous.load_model()
    frozen = list(model.named_parameters())
    before = previous.parameter_digest(frozen)
    trn, dev, conf = previous.data('train'), previous.data('development'), confirmation()
    x0, y0 = previous.batch(dev, slice(0, 2))
    with torch.no_grad():
        reference = model(x0, y0, loss_reduction='none')
    adapter, named = attach_local(model, layer, variant)
    parameters = [p for _, p in named]
    trainable = sum(p.numel() for p in parameters)
    with torch.no_grad():
        torch.testing.assert_close(model(x0, y0, loss_reduction='none'), reference, atol=0, rtol=0)
    p = PROTOCOL
    optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], betas=tuple(p['adam_betas']),
                                  eps=p['adam_eps'], weight_decay=0.0, fused=True)
    order = np.random.default_rng(p['train_order_seed']).permutation(len(trn))
    start = 0
    if resume:
        ckpts = sorted(folder.glob('adapter_*.pt'))
        if not ckpts:
            raise RuntimeError('Resume requested but no checkpoint exists')
        start = restore_adapter(ckpts[-1], named, optimizer, experiment, selection_sha)
    elif list(folder.glob('adapter_*.pt')):
        raise RuntimeError('Refusing to overwrite existing checkpoints')
    write_json(folder / 'config.json', dict(
        protocol=PROTOCOL, location=location, layer_index=layer, variant=variant,
        rank=VARIANTS[variant][0], independent=VARIANTS[variant][1], trainable_parameters=trainable,
        selection_sha256=selection_sha, selection_group=[g for g in ('high', 'low', 'random')
                                                        if location in chosen[g]][0],
        manifest_sha256=sha(ROOT / 'manifest.json'), frozen_parameter_sha256=before,
        initialization_matches_baseline=True))
    timings, started = [], time.monotonic()
    micro = p['accumulation']
    for step in range(start, p['updates'] + 1):
        if step in p['eval_steps']:
            state(folder, 'evaluating', experiment=experiment, step=step)
            ev = previous.evaluate(model, dev)
            write_json(folder / f'development_{step:03d}.json', ev)
            print(json.dumps(dict(step=step, development=ev['summary'])), flush=True)
            save_adapter(folder / f'adapter_{step:03d}.pt', step, named, optimizer, experiment, selection_sha)
        if step == p['updates']:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g['lr'] = lr
        loss_sum = torch.zeros((), device='cuda')
        torch.cuda.synchronize()
        tick = time.monotonic()
        for m in range(micro):
            pos = step * micro * 2 + m * 2
            x, y = previous.batch(trn, order[pos:pos + 2])
            loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss step={step} micro={m}')
            (loss / micro).backward()
            loss_sum += loss.detach().float()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(q).all().item() for q in parameters):
            raise RuntimeError('Nonfinite adapter parameters')
        torch.cuda.synchronize()
        dt = time.monotonic() - tick
        timings.append(dt)
        row = dict(step=step + 1, tokens=(step + 1) * p['tokens_per_update'], train_ce=(loss_sum / micro).item(),
                   gradient_norm=grad_norm.item(), lr=lr, seconds=dt)
        append(folder / 'training.jsonl', row)
        med = float(np.median(timings[-10:]))
        state(folder, 'training', experiment=experiment, **row, eta_seconds=(p['updates'] - 1 - step) * med,
              peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
        print(json.dumps(row), flush=True)
    state(folder, 'confirmation', experiment=experiment, step=p['updates'])
    cf = previous.evaluate(model, conf)
    write_json(folder / 'confirmation_200.json', cf)
    assert previous.parameter_digest(frozen) == before, 'Base weights changed'
    write_json(folder / 'results.json', dict(
        experiment=experiment, location=location, variant=variant, completed_utc=now(),
        updates=p['updates'], train_tokens=p['train_tokens'], trainable_parameters=trainable,
        development=json.loads((folder / f'development_{p["updates"]:03d}.json').read_text())['summary'],
        confirmation=cf['summary'], frozen_weights_unchanged=True, selection_sha256=selection_sha,
        elapsed_seconds=time.monotonic() - started,
        median_step_seconds=float(np.median(timings)) if timings else None,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20))
    state(folder, 'complete', experiment=experiment)


# ---------------------------------------------------------------- scaling control
def scaling(folder):
    chosen, selection_sha = frozen_selection()
    state(folder, 'loading')
    model, _, _ = previous.load_model()
    model.requires_grad_(False)
    before = previous.parameter_digest(model.named_parameters())
    dev = previous.data('development')
    n = PROTOCOL['scaling']['dev_sequences']
    subset = dev[:n]
    started = time.monotonic()
    base = previous.evaluate(model, subset)
    first = chosen['locations'][0]
    sham = LocalScale(model, layer_index(first), 3, 1.0)
    same = previous.evaluate(model, subset[:2], reset=sham.reset)
    sham.close()
    assert same['rows'] == base['rows'][:2], 'Scale 1.0 sham not bitwise identical'
    conditions = []
    for loc in chosen['locations']:
        for loop in range(4):
            for scale in PROTOCOL['scaling']['scales']:
                hook = LocalScale(model, layer_index(loc), loop, scale)
                ev = previous.evaluate(model, subset, reset=hook.reset)
                hook.close()
                row = dict(location=loc, round=loop + 1, scale=scale, dev_subset=ev['summary'],
                           vs_unchanged=paired(ev['rows'], base['rows']))
                conditions.append(row)
                append(folder / 'conditions.jsonl', row)
                state(folder, 'screening', done=len(conditions), total=48, seconds=time.monotonic() - started)
                print(json.dumps({k: row[k] for k in ('location', 'round', 'scale')} |
                                 dict(delta_bpb=row['vs_unchanged']['delta_bpb'])), flush=True)
    picks = {}
    for loc in chosen['locations']:
        rows = [r for r in conditions if r['location'] == loc]
        best = min(rows, key=lambda r: (r['dev_subset']['bpb'], r['round'], r['scale']))
        picks[loc] = dict(round=best['round'], scale=best['scale'], dev_subset_bpb=best['dev_subset']['bpb'])
    write_json(folder / 'picks_frozen_before_confirmation.json', picks)
    conf = confirmation()
    conf_base = previous.evaluate(model, conf)
    confirmed = {}
    for loc, pick in picks.items():
        hook = LocalScale(model, layer_index(loc), pick['round'] - 1, pick['scale'])
        ev = previous.evaluate(model, conf, reset=hook.reset)
        hook.close()
        write_json(folder / f'confirmation_{loc}.json', ev)
        confirmed[loc] = dict(**pick, confirmation=ev['summary'], vs_unchanged=paired(ev['rows'], conf_base['rows']))
    assert previous.parameter_digest(model.named_parameters()) == before
    write_json(folder / 'results.json', dict(
        experiment='scaling', completed_utc=now(), dev_subset_sequences=n, dev_subset_unchanged=base['summary'],
        confirmation_unchanged=conf_base['summary'], conditions=conditions, picks=picks, confirmed=confirmed,
        selection_sha256=selection_sha, sham_passed=True, frozen_weights_unchanged=True,
        elapsed_seconds=time.monotonic() - started,
        limitations=['Not compute-matched to adapters; two scales do not rule out other scaling methods.']))
    state(folder, 'complete')


def baseline_eval(folder):
    state(folder, 'loading')
    model, _, _ = previous.load_model()
    model.requires_grad_(False)
    for name, arr in (('development', previous.data('development')), ('confirmation', confirmation())):
        write_json(folder / f'{name}.json', previous.evaluate(model, arr))
    write_json(folder / 'results.json', dict(experiment='baseline_eval', completed_utc=now()))
    state(folder, 'complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = verify()
    assert manifest is not None, 'Run prepare15.py --freeze first'
    assert sha(previous.CHECKPOINT / 'model_019074.pt') == manifest['checkpoint_sha256']
    folder = ROOT / 'results' / args.job
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'results.json').exists():
        raise RuntimeError('Job already completed')
    try:
        if args.job == 'gradient':
            assert not (ROOT / 'selection.json').exists(), 'Selection already frozen'
            gradient(folder)
        elif args.job == 'scaling':
            scaling(folder)
        elif args.job == 'baseline_eval':
            baseline_eval(folder)
        else:
            loc, variant = args.job.split('_', 1)
            assert variant in VARIANTS, args.job
            train(folder, f'{loc}.up', variant, resume=args.resume)
    except BaseException:
        state(folder, 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
