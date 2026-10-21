"""Three independent experiments using the same frozen checkpoint and inputs."""
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import (
    ROOT, SETTINGS, GROUPS, VARIANTS, CHECKPOINT, now, sha, verify_source, load_model, matrices,
    parameter_digest, write_json, append_json, data, batch, evaluate,
)
from interventions import VisitGradients, ScaleOutputs, attach_lora


def state(folder, phase, **extra):
    write_json(folder / "status.json", dict(phase=phase, time=now(), pid=os.getpid(), **extra))


def paired_interval(a, b):
    """Paired sequence-block bootstrap; descriptive, not multiple-test adjusted."""
    differences = np.array([x["nats"] / x["tokens"] - y["nats"] / y["tokens"]
                            for x, y in zip(a, b)])
    rng = np.random.default_rng(42)
    boots = differences[rng.integers(0, len(differences), size=(2000, len(differences)))].mean(1)
    return dict(delta_ce=float(differences.mean()),
                ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
                improved_fraction=float((differences < 0).mean()),
                uncertainty="paired packed-sequence bootstrap, exploratory/unadjusted")


def gradient(folder):
    state(folder, "loading")
    model, _, _ = load_model()
    original = list(model.named_parameters())
    before = parameter_digest(original)
    model.requires_grad_(False)
    for _, module in matrices(model):
        module.weight.requires_grad_(True)
    inputs = data("gradient")
    probe = VisitGradients(model)
    started = time.monotonic()
    results = []
    for i in range(32):
        model.zero_grad(set_to_none=True)
        probe.reset()
        x, y = batch(inputs, slice(2 * i, 2 * i + 2))
        loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite diagnostic loss")
        loss.backward()
        stats, error = probe.statistics()
        # BF16 parallel reduction differences have a measured short-test bound;
        # the FP32 unit test independently requires tight mathematical equality.
        if error > 0.02:
            raise RuntimeError(f"Gradient sum failed at batch {i}: {error}")
        row = dict(batch=i, ce=loss.item(), max_sum_relative_error=error, matrices=stats)
        results.append(row)
        append_json(folder / "batches.jsonl", row)
        elapsed = time.monotonic() - started
        state(folder, "measuring", batches_completed=i + 1, batches_total=32,
              elapsed_seconds=elapsed, eta_seconds=elapsed / (i + 1) * (31 - i),
              ce=loss.item(), max_sum_relative_error=error)
        print(json.dumps(dict(batch=i + 1, ce=loss.item(), sum_error=error,
                              seconds=elapsed)), flush=True)
    probe.close()
    assert parameter_digest(original) == before
    summary = []
    for j, (name, _) in enumerate(matrices(model)):
        cs = np.array([r["matrices"][j]["cosine"] for r in results])
        valid = np.array([r["matrices"][j]["valid_norms"] for r in results])
        pair_valid = valid[:, :, None] & valid[:, None, :]
        masked = np.where(pair_valid, cs, np.nan)
        # IMPORTANT: average per-batch cosines, NOT cosine of averaged gradients.
        # Each batch already contributes the gradient of its mean token loss.
        mean = np.nanmean(masked, axis=0)
        norms = np.array([r["matrices"][j]["norms"] for r in results])
        summary.append(dict(
            name=name,
            mean_cosine=[[float(v) if np.isfinite(v) else None for v in row] for row in mean],
            mean_norms=norms.mean(0).tolist(),
            norm_shares=(norms / norms.sum(1, keepdims=True).clip(1e-30)).mean(0).tolist(),
            mean_cancellation_ratio=float(np.mean([r["matrices"][j]["cancellation_ratio"] for r in results])),
        ))
    write_json(folder / "results.json", dict(
        experiment="gradient", completed_utc=now(), summary=summary,
        frozen_weights_unchanged=True, frozen_parameter_sha256=before,
        total_prediction_tokens=131072, final_loss_only=True,
        elapsed_seconds=time.monotonic() - started,
        max_sum_relative_error=max(r["max_sum_relative_error"] for r in results),
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        limitations=["Final checkpoint only; gradient disagreement is not proof that sharing harms quality.",
                     "32 fixed batches, one pretraining seed; no multiple-comparison significance claims."],
    ))
    state(folder, "complete", batches_completed=32)


def sensitivity(folder):
    state(folder, "loading")
    model, _, _ = load_model()
    model.requires_grad_(False)
    original = list(model.named_parameters())
    before = parameter_digest(original)
    inputs = data("sensitivity")
    started = time.monotonic()
    baseline = evaluate(model, inputs)
    write_json(folder / "baseline.json", baseline)
    # Sham intervention traverses identical hooks and must be bitwise equivalent.
    sham = ScaleOutputs(model, "mlp", 3, 1.0)
    unchanged = evaluate(model, inputs[:2], reset=sham.reset)
    sham.close()
    assert unchanged["rows"] == baseline["rows"][:2]
    results = []
    for loop in range(4):
        for kind in ("attn", "mlp"):
            for scale in (0.9, 1.1):
                hook = ScaleOutputs(model, kind, loop, scale)
                measured = evaluate(model, inputs, reset=hook.reset)
                hook.close()
                row = dict(loop=loop + 1, module=kind, scale=scale, **measured,
                           **paired_interval(measured["rows"], baseline["rows"]))
                results.append(row)
                write_json(folder / f"loop{loop + 1}_{kind}_{scale}.json", row)
                elapsed = time.monotonic() - started
                state(folder, "measuring", conditions_completed=len(results), conditions_total=16,
                      elapsed_seconds=elapsed, eta_seconds=elapsed / len(results) * (16 - len(results)))
                print(json.dumps({k: v for k, v in row.items() if k != "rows"}), flush=True)
    assert parameter_digest(original) == before
    write_json(folder / "results.json", dict(
        experiment="sensitivity", completed_utc=now(), baseline=baseline["summary"],
        conditions=[{k: v for k, v in r.items() if k != "rows"} for r in results],
        frozen_weights_unchanged=True, sham_passed=True,
        frozen_parameter_sha256=before,
        elapsed_seconds=time.monotonic() - started,
        limitations=["Scales one module class in all 24 blocks of a single loop.",
                     "Sensitivity is not proof of sharing limitation; not a layer-local intervention.",
                     "Intervals are descriptive and unadjusted for the 16 exploratory comparisons."],
    ))
    state(folder, "complete", conditions_completed=16)


def lr_at(step):
    if step < SETTINGS["warmup"]:
        return SETTINGS["learning_rate"] * (step + 1) / SETTINGS["warmup"]
    progress = (step - SETTINGS["warmup"]) / (SETTINGS["updates"] - SETTINGS["warmup"] - 1)
    return SETTINGS["final_learning_rate"] + (
        SETTINGS["learning_rate"] - SETTINGS["final_learning_rate"]
    ) * 0.5 * (1 + math.cos(math.pi * progress))


def save_adapter(path, step, named, optimizer, experiment):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(
        step=step, adapters={n: p.detach().cpu() for n, p in named},
        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(), settings=SETTINGS, experiment=experiment,
        source_manifest_sha256=sha(ROOT / "experiment_manifest.json"),
        data_manifest_sha256=sha(ROOT / "data/manifest.json"),
    ), tmp)
    tmp.replace(path)


def restore_adapter(path, named, optimizer, experiment):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["settings"] == SETTINGS and saved["experiment"] == experiment
    assert saved["source_manifest_sha256"] == sha(ROOT / "experiment_manifest.json")
    assert saved["data_manifest_sha256"] == sha(ROOT / "data/manifest.json")
    with torch.no_grad():
        for name, p in named:
            p.copy_(saved["adapters"][name])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    return saved["step"]


def lora(folder, group, variant, resume=False):
    experiment = f"{group}_{variant}"
    state(folder, "loading", experiment=experiment)
    model, _, _ = load_model()
    frozen = list(model.named_parameters())
    before = parameter_digest(frozen)
    train, dev = data("train"), data("development")
    x0, y0 = batch(dev, slice(0, 2))
    with torch.no_grad():
        reference = model(x0, y0, loss_reduction="none")
    adapters, named = attach_lora(model, group, variant)
    parameters = [p for _, p in named]
    trainable = sum(p.numel() for p in parameters)
    assert trainable == (2949120 if variant == "shared_small" else 11796480)
    with torch.no_grad():
        torch.testing.assert_close(model(x0, y0, loss_reduction="none"), reference, atol=0, rtol=0)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.0, fused=True)
    order = np.random.default_rng(42).permutation(len(train))
    start_step = 0
    if resume:
        checkpoints = sorted(folder.glob("adapter_*.pt"))
        if not checkpoints:
            raise RuntimeError("Resume requested but no checkpoint exists")
        start_step = restore_adapter(checkpoints[-1], named, optimizer, experiment)
    elif list(folder.glob("adapter_*.pt")):
        raise RuntimeError("Refusing to overwrite existing training checkpoints")
    write_json(folder / "config.json", dict(
        **SETTINGS, group=group, variant=variant, trainable_parameters=trainable,
        total_parameters=sum(p.numel() for p in model.parameters()),
        initialization_matches_baseline=True, frozen_parameter_sha256=before,
        baseline_checkpoint=str(CHECKPOINT / "model_019074.pt"),
        lora_forward_matmul_flops_per_token=2 * trainable * (1 if variant == "independent" else 4),
        flops_note="Extra LoRA forward matrix multiplies only, multiply+add=2. NOT full training FLOPs; same-parameter shared-large has 4x these adapter FLOPs.",
    ))
    timings = []
    started = time.monotonic()
    for step in range(start_step, 201):
        if step in SETTINGS["eval_steps"]:
            state(folder, "evaluating", experiment=experiment, step=step)
            evaluation = evaluate(model, dev)
            write_json(folder / f"development_{step:03d}.json", evaluation)
            print(json.dumps(dict(step=step, development=evaluation["summary"])), flush=True)
            save_adapter(folder / f"adapter_{step:03d}.pt", step, named, optimizer, experiment)
        if step == 200:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        loss_sum = torch.zeros((), device="cuda")
        torch.cuda.synchronize()
        tick = time.monotonic()
        for micro in range(128):
            pos = step * 256 + micro * 2
            x, y = batch(train, order[pos:pos + 2])
            loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step={step},micro={micro}")
            (loss / 128).backward()
            loss_sum += loss.detach().float()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in parameters):
            raise RuntimeError("Nonfinite adapter parameters")
        torch.cuda.synchronize()
        duration = time.monotonic() - tick
        timings.append(duration)
        row = dict(step=step + 1, tokens=(step + 1) * 524288,
                   train_ce=(loss_sum / 128).item(), gradient_norm=grad_norm.item(),
                   lr=lr, seconds=duration, tokens_per_second=524288 / duration)
        append_json(folder / "training.jsonl", row)
        median = float(np.median(timings[-10:]))
        state(folder, "training", experiment=experiment, **row,
              median_step_seconds=median, eta_training_seconds=(199 - step) * median,
              peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
        print(json.dumps(row), flush=True)
    state(folder, "confirmation", experiment=experiment, step=200)
    confirmation = evaluate(model, data("confirmation"))
    write_json(folder / "confirmation_200.json", confirmation)
    assert parameter_digest(frozen) == before
    write_json(folder / "results.json", dict(
        experiment=experiment, completed_utc=now(), updates=200,
        train_tokens=104857600, trainable_parameters=trainable,
        development=json.loads((folder / "development_200.json").read_text())["summary"],
        confirmation=confirmation["summary"], frozen_weights_unchanged=True,
        elapsed_seconds=time.monotonic() - started,
        median_step_seconds=float(np.median(timings)) if timings else None,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        limitations=["One adaptation seed and one pretraining seed; exploratory fixed-hyperparameter screen.",
                     "No claim of matched total training FLOPs; no early-training prediction tested.",
                     "All 24 layers targeted together; does not isolate individual bottleneck layers."],
    ))
    state(folder, "complete", experiment=experiment, step=200)


def baseline_eval(folder):
    state(folder, "loading")
    model, _, _ = load_model()
    model.requires_grad_(False)
    for split in ("development", "confirmation"):
        measured = evaluate(model, data(split))
        write_json(folder / f"{split}.json", measured)
    write_json(folder / "results.json", dict(completed_utc=now(), experiment="baseline_eval"))
    state(folder, "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["gradient", "sensitivity", "baseline_eval"] +
                        [f"{group}_{variant}" for group in GROUPS for variant in VARIANTS])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_source()
    frozen = json.loads((ROOT / "experiment_manifest.json").read_text())
    assert sha(ROOT / "data/manifest.json") == frozen["data_manifest_sha256"]
    assert sha(CHECKPOINT / "model_019074.pt") == frozen["checkpoint_sha256"]
    folder = ROOT / "results" / args.job
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "results.json").exists():
        raise RuntimeError("Job already completed")
    try:
        if args.job == "gradient":
            gradient(folder)
        elif args.job == "sensitivity":
            sensitivity(folder)
        elif args.job == "baseline_eval":
            baseline_eval(folder)
        else:
            group, variant = args.job.split("_", 1)
            lora(folder, group, variant, resume=args.resume)
    except BaseException as exc:
        state(folder, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
