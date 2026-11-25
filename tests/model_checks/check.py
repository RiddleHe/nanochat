#!/usr/bin/env python3
"""Registry-driven checks, executed in a saved run's environment.

CPU checks use a smaller width/vocabulary/context, preserving depth, head count,
architecture flags, and target layers. GPU checks add compiled/eager and FP32/
recipe-dtype comparisons and an optimizer update at the actual training shape.
Distributed checks compare a small global batch with the sharded optimizer, then
exercise a compiled update at the actual per-rank training shape. Reports state
the shapes tested; none of these checks use the training dataset.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from nanochat import flash_attention as attention
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.engine import KVCache
from nanochat.model_registry import get_model
from nanochat.optim import MuonAdamW


@contextmanager
def precision(dtype, use_fa3=False):
    """Model modules import COMPUTE_DTYPE by value; change all copies together."""
    saved = []
    for name, module in list(sys.modules.items()):
        if name.startswith("nanochat.") and hasattr(module, "COMPUTE_DTYPE"):
            saved.append((module, module.COMPUTE_DTYPE))
            module.COMPUTE_DTYPE = dtype
    old = attention.USE_FA3
    attention.USE_FA3 = use_fa3
    try:
        yield
    finally:
        attention.USE_FA3 = old
        for module, value in saved:
            module.COMPUTE_DTYPE = value


def small_config(config):
    cfg = dict(config)
    # Preserve depth and head layout, including serialized target-layer mappings.
    cfg.update(sequence_len=16, vocab_size=128, n_embd=cfg["n_head"] * 8)
    return cfg


def make_model(name, config, device, *, perturb=True):
    Config, Model = get_model(name)
    torch.manual_seed(42)
    with torch.device("meta"):
        model = Model(Config(**config))
    model.to_empty(device=device)
    # to_empty leaves arbitrary memory. Poison it so missing initialization is
    # detected reliably, instead of occasionally passing because memory was zero.
    with torch.no_grad():
        for value in list(model.parameters()) + list(model.buffers()):
            if value.is_floating_point():
                value.fill_(float("nan"))
    model.init_weights()
    missing = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
               if value.is_floating_point() and not torch.isfinite(value).all()]
    if missing:
        raise AssertionError(f"Uninitialized or nonfinite tensors after init_weights: {missing}")
    if perturb:
        # Zero residual projections can hide attention-gradient bugs on step one.
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim >= 2 and not torch.count_nonzero(p):
                    p.normal_(std=0.01)
    return model


def tokens(config, batch, device, seed=123):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(config["vocab_size"], (batch, config["sequence_len"]),
                         device=device, generator=generator)


def finite(value, label):
    if not torch.isfinite(value).all():
        raise AssertionError(f"Nonfinite {label}")


def backward(model, x, optimizer):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    finite(logits, "logits")
    loss = model(x, x.roll(-1, 1))
    finite(loss, "loss")
    loss.backward()
    # Some registered variants intentionally retain unused checkpoint parameters.
    # Every parameter actually assigned to the optimizer must have a finite grad.
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                raise AssertionError(f"Optimizer parameter has no gradient: {tuple(p.shape)}")
            finite(p.grad, "gradient")
    return logits.detach(), loss.detach()


def gradients(model):
    return {name: p.grad.detach().float().clone() for name, p in model.named_parameters() if p.grad is not None}


def compare(actual, expected, label, rtol=2e-4, atol=2e-6):
    # Normwise comparison avoids meaningless relative errors on almost-zero entries.
    actual, expected = actual.float(), expected.float()
    finite(actual, label)
    difference = torch.linalg.vector_norm(actual - expected).item()
    reference = torch.linalg.vector_norm(expected).item()
    bound = atol * expected.numel() ** 0.5 + rtol * reference
    if difference > bound:
        raise AssertionError(f"{label}: L2 error {difference} > {bound}")
    return dict(l2_error=difference, reference_l2=reference, bound=bound, rtol=rtol, atol=atol)


def compare_gradients(model, reference, **tolerances):
    actual = gradients(model)
    if actual.keys() != reference.keys():
        raise AssertionError("Gradient parameter names differ")
    errors = {name: compare(value, reference[name], f"gradient {name}", **tolerances)
              for name, value in actual.items()}
    return max((v["l2_error"] / max(v["bound"], 1e-30) for v in errors.values()), default=0)


def cache_check(model, config, x, dtype):
    model.eval()
    with torch.no_grad():
        full = model(x)
        cache = KVCache(batch_size=x.shape[0], num_heads=config["n_kv_head"],
                        seq_len=x.shape[1], head_dim=config["n_embd"] // config["n_head"],
                        num_layers=config["n_layer"], device=x.device, dtype=dtype)
        # Prefill then decode: exercise both cache paths rather than only single tokens.
        pieces = [model(x[:, :3], kv_cache=cache)]
        pieces.extend(model(x[:, i:i + 1], kv_cache=cache) for i in range(3, x.shape[1]))
        cached = torch.cat(pieces, dim=1)
        error = compare(cached, full, "cached logits", rtol=0.025 if dtype != torch.float32 else 2e-4,
                        atol=1e-4 if dtype != torch.float32 else 2e-6)
    model.train()
    return error


def checkpoint_check(name, config, model, optimizer, x):
    """Use the trainer's checkpoint format and prove optimizer state survives a next update."""
    with tempfile.TemporaryDirectory() as directory:
        save_checkpoint(directory, 1, model.state_dict(), optimizer.state_dict(),
                        dict(step=1, model_type=name, model_config=asdict(model.config)))
        weights, state, metadata = load_checkpoint(directory, 1, x.device, load_optimizer=True)
        other = make_model(name, config, x.device)
        other.load_state_dict(weights, strict=True)
        other_optimizer = other.setup_optimizer()
        other_optimizer.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)
        for candidate, opt in ((model, optimizer), (other, other_optimizer)):
            backward(candidate, x, opt)
            opt.step()
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        if metadata["model_config"] != json.loads(json.dumps(asdict(model.config))):
            raise AssertionError("Checkpoint configuration changed")
    return "passed: predictions and next optimizer update match exactly"


def basic_checks(name, config, device):
    with precision(torch.float32):
        model = make_model(name, config, device)
        optimizer = model.setup_optimizer()
        x = tokens(config, 2, device)
        logits, loss = backward(model, x, optimizer)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer.step()
        for p in model.parameters():
            finite(p, "updated parameter")
        if not any(not torch.equal(p, before[n]) for n, p in model.named_parameters()):
            raise AssertionError("Optimizer did not update any parameter")
        result = dict(loss=loss.item(), finite_outputs_gradients_update=True,
                      cache=cache_check(model, config, x, torch.float32),
                      checkpoint=checkpoint_check(name, config, model, optimizer, x))
    return result


def numerical_checks(name, config, device, dtype):
    with precision(torch.float32):
        reference = make_model(name, config, device)
        optimizer = reference.setup_optimizer()
        x = tokens(config, 2, device)
        ref_logits, ref_loss = backward(reference, x, optimizer)
        ref_grads = gradients(reference)
        weights = {n: p.detach().clone() for n, p in reference.state_dict().items()}
    with precision(dtype, attention.HAS_FA3 and dtype == torch.bfloat16):
        model = make_model(name, config, device)
        model.load_state_dict(weights)
        optimizer = model.setup_optimizer()
        logits, loss = backward(model, x, optimizer)
        result = dict(dtype=str(dtype), logits=compare(logits, ref_logits, "reduced precision logits", rtol=.08, atol=.001),
                      loss=compare(loss, ref_loss, "reduced precision loss", rtol=.02, atol=.001),
                      gradient_error_fraction=compare_gradients(model, ref_grads, rtol=.12, atol=2e-5))
        eager_grads = gradients(model)
        model.zero_grad(set_to_none=True)
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        compiled_logits, compiled_loss = backward(compiled, x, optimizer)
        result["compiled_logits"] = compare(compiled_logits, logits, "compiled logits", rtol=.025, atol=.0005)
        result["compiled_loss"] = compare(compiled_loss, loss, "compiled loss", rtol=.01, atol=.0005)
        result["compiled_gradient_error_fraction"] = compare_gradients(model, eager_grads, rtol=.08, atol=2e-5)
        result["cache"] = cache_check(model, config, x, dtype)
        optimizer.step()
        for p in model.parameters():
            finite(p, "compiled updated parameter")
    return result


def distributed_comparison(name, config, device, dtype, rank, world):
    """The real sharded optimizer must agree with one optimizer on the same global batch."""
    with precision(dtype, attention.HAS_FA3 and dtype == torch.bfloat16):
        model = make_model(name, config, device)
        reference = make_model(name, config, device)
        reference.load_state_dict(model.state_dict())
        optimizer = model.setup_optimizer()
        reference_optimizer = MuonAdamW(reference.setup_optimizer().param_groups)
        all_x = torch.cat([tokens(config, 2, device, seed=123 + i) for i in range(world)])
        backward(reference, all_x, reference_optimizer)
        backward(model, all_x[rank * 2:(rank + 1) * 2], optimizer)
        reference_optimizer.step()
        optimizer.step()
        worst = 0
        for (name, p), q in zip(model.named_parameters(), reference.parameters()):
            error = compare(p, q, f"global batch update {name}", rtol=.03, atol=.002)
            worst = max(worst, error["l2_error"] / max(error["bound"], 1e-30))
            broadcast = p.detach().clone()
            dist.broadcast(broadcast, src=0)
            torch.testing.assert_close(p, broadcast, rtol=0, atol=0)
    return dict(global_batch_error_fraction=worst, parameters_identical_across_ranks=True)


def full_shape_check(name, config, device, batch, dtype, distributed):
    # Reduced-shape reference/sharded checks specialize the same fused optimizer
    # functions. Start this independent shape with the fresh compiler cache that
    # the training process has, rather than exhausting its specialization limit.
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with precision(dtype, attention.HAS_FA3 and dtype == torch.bfloat16):
        model = make_model(name, config, device)
        optimizer = model.setup_optimizer()
        x = tokens(config, batch, device, seed=123 + int(os.environ.get("RANK", 0)))
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        # Full-size logits are only needed once here; a second forward would double peak memory.
        loss = compiled(x, x.roll(-1, 1))
        finite(loss, "full shape loss")
        loss.backward()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise AssertionError("Full shape optimizer parameter has no gradient")
                finite(p.grad, "full shape gradient")
        before = next(model.parameters()).detach().clone()
        optimizer.step()
        if torch.equal(before, next(model.parameters())):
            raise AssertionError("Full shape optimizer did not update embeddings")
        for p in model.parameters():
            finite(p, "full shape parameter")
            if distributed:
                reference = p.detach().clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(p, reference, rtol=0, atol=0)
        torch.cuda.synchronize()
        return dict(loss=loss.item(), config=asdict(model.config), batch_per_rank=batch,
                    compiled=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    parameters_identical_across_ranks=distributed)


def run_checks(manifest, mode):
    name = manifest.get("model_type") or next(
        a.split("=", 1)[1] for a in manifest["command"] if a.startswith("--model-type="))
    config = manifest["model_config"]
    small = small_config(config)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cpu" if mode == "cpu" else f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.set_num_threads(1)
    if mode != "cpu":
        torch.cuda.set_device(device)
        # Reference calculations use FP32 rather than TF32.
        torch.set_float32_matmul_precision("highest")
    if mode == "distributed":
        dist.init_process_group("nccl", device_id=device)
        if world != manifest["gpu_count"] or world < 2:
            raise ValueError("Distributed check must use the training world size")
    dtype = getattr(torch, manifest["env"]["NANOCHAT_DTYPE"])
    result = dict(status="passed", mode=mode, model_type=name, small_config=small,
                  training_started=False, rank=rank, world_size=world,
                  attention_backend="fa3" if attention.HAS_FA3 and dtype == torch.bfloat16 and mode != "cpu" else "sdpa")
    if mode == "distributed":
        result["global_batch"] = distributed_comparison(name, small, device, dtype, rank, world)
    else:
        result["basic"] = basic_checks(name, small, device)
        if mode == "gpu":
            result["numerics"] = numerical_checks(name, small, device, dtype)
    if mode != "cpu":
        result["full_shape"] = full_shape_check(name, config, device, manifest["device_batch_size"],
                                                dtype, mode == "distributed")
        result["gpu_name"] = torch.cuda.get_device_name(device)
    if mode == "distributed":
        ranks = [None] * world
        dist.all_gather_object(ranks, result.copy())
        result["ranks"] = ranks
        dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu", "gpu", "distributed"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = dict(status="failed", mode=args.mode, training_started=False)
    try:
        result = run_checks(json.loads(args.manifest.read_text()), args.mode)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if int(os.environ.get("RANK", 0)) == 0:
            args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
