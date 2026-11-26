"""Bounded CUDA/BF16/FA3 acceptance gate for both Weighted12 loss arms.

NANOCHAT_DTYPE=bfloat16 torchrun --standalone --nproc_per_node=2 \
    -m scripts.preflight_weighted_loop_ab --output /path/preflight.json

Uses the actual WeightedLoop at B4/T32/D64, four layers/four loops/k2. The
single-worker global-batch reference averages two independent B4 loss units;
it never pools balance across eight sequences. Optimizer acceptance is bitwise
against MuonAdamW with identical Muon batch shapes and Adam row slices. This
isolates communication from BF16 kernel-shape effects. No training is launched.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import traceback

import torch
import torch.distributed as dist

from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init
from nanochat.engine import KVCache
import nanochat.flash_attention as attention
from nanochat.optim import DistMuonAdamW, MuonAdamW
from scripts.train_weighted_loop_ab import WeightedLoop, WeightedLoopConfig


# Fixed before GPU execution; never automatically relaxed on failure. Relative
# vector error handles near-zero entries without a permissive elementwise atol.
# BF16 forward/backward accumulation and compiler fusion can differ slightly.
TOLERANCES = {"logits_relative_l2": .03, "logits_max_scaled": .08,
              "gradients_relative_l2": .04, "gradients_max_scaled": .12,
              "global_gradients_relative_l2": .02, "global_gradients_max_scaled": .04}


def digest(tensors):
    result = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def compare_tensors(actual, expected, relative_l2, max_scaled):
    """Compare every tensor, with finite checks and compact worst-case evidence."""
    if actual.keys() != expected.keys():
        raise AssertionError("Tensor names differ")
    report = {"passed": True, "worst_relative_l2": 0., "worst_max_scaled": 0., "failures": []}
    for name in actual:
        a, b = actual[name].detach().float(), expected[name].detach().float()
        if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            raise AssertionError(f"Invalid tensor {name}")
        delta = a - b
        rel = float(delta.norm() / b.norm().clamp_min(1e-12))
        scaled = float(delta.abs().max() / b.abs().max().clamp_min(1e-12))
        report["worst_relative_l2"] = max(report["worst_relative_l2"], rel)
        report["worst_max_scaled"] = max(report["worst_max_scaled"], scaled)
        if rel > relative_l2 or scaled > max_scaled:
            report["passed"] = False
            report["failures"].append({"name": name, "relative_l2": rel, "max_scaled": scaled})
    return report


def exact_state(actual, expected, path="state"):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: state keys differ")
        for key in actual:
            exact_state(actual[key], expected[key], f"{path}.{key}")
    elif torch.is_tensor(actual):
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise AssertionError(f"{path}: tensors are not bitwise equal")
    elif actual != expected:
        raise AssertionError(f"{path}: values differ")


def make_model(alpha, device, compiled):
    torch.manual_seed(731)
    torch.cuda.manual_seed(731)
    config = WeightedLoopConfig(sequence_len=32, vocab_size=64, n_layer=4,
                               n_embd=64, n_head=1, n_kv_head=1, num_loops=4,
                               select_k=2, checkpoint_blocks=True,
                               window_pattern="SSSL", balance_alpha=alpha)
    with torch.device("meta"):
        model = WeightedLoop(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.025)
            block.mlp.c_proj.weight.normal_(std=.025)
        # Nonzero residuals expose task gradients; margins separate numerical
        # BF16 parity from intentional discontinuities of top-k at ties. Rotate
        # support across loops so every physical layer participates, while
        # pooled usage stays nonuniform and the B auxiliary gradient is nonzero.
        bias = torch.tensor([.6, .2, -.2, -.6], device=device)
        for loop in range(4):
            model.selector_bias[loop].copy_(bias.roll((0, 0, 1, 2)[loop]))
    if compiled:
        for block in model.transformer.h:
            block.forward = torch.compile(block.forward, dynamic=False)
    return model


def batch(rank, device):
    generator = torch.Generator(device=device).manual_seed(9100 + rank)
    x = torch.randint(64, (4, 32), device=device, generator=generator)
    y = torch.randint(64, x.shape, device=device, generator=generator)
    return x, y


def gradients(model, x, y):
    model.train()
    model.zero_grad(set_to_none=True)
    ce, aux = model.training_loss(x, y)
    (ce + aux).backward()
    if not bool(torch.isfinite(ce + aux)):
        raise AssertionError("Nonfinite loss")
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"Missing/nonfinite gradient: {name}")
        result[name] = parameter.grad.detach().clone()
    for name in ("selector_weight", "selector_bias"):
        if not bool((result[name] != 0).any()):
            raise AssertionError(f"Vacuous scorer gradient: {name}")
    return result, {"ce": float(ce.detach()), "weighted_aux": float(aux.detach())}


def make_reference(groups, names, initial, averaged, world_size):
    """Replicate distributed kernel shapes, retaining views into full weights."""
    parameters = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
    for name, parameter in parameters.items():
        parameter.grad = averaged[name].clone()
    reference_groups, mappings = [], []
    for group in groups:
        options = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        keys = [names[id(parameter)] for parameter in group["params"]]
        if group["kind"] == "muon":
            chunk = (len(keys) + world_size - 1) // world_size
            heads = []
            for start in range(0, len(keys), chunk):
                selected = [parameters[key] for key in keys[start:start + chunk]]
                reference_groups.append({**options, "params": selected})
                heads.append(selected[0])
            mappings.append({"kind": "muon", "heads": heads})
        else:
            refs, by_name = [], {}
            for name in keys:
                parameter = parameters[name]
                if parameter.numel() >= 1024:
                    if parameter.shape[0] % world_size:
                        raise AssertionError("Nondivisible Adam row slices")
                    rows = parameter.shape[0] // world_size
                    shards = []
                    for rank in range(world_size):
                        selected = slice(rank * rows, (rank + 1) * rows)
                        shard = torch.nn.Parameter(parameter.detach()[selected])
                        shard.grad = averaged[name][selected].clone()
                        shards.append(shard)
                else:
                    shards = [parameter]
                refs.extend(shards)
                by_name[name] = shards
            reference_groups.append({**options, "params": refs})
            mappings.append({"kind": "adamw", "parameters": by_name})
    return parameters, MuonAdamW(reference_groups), mappings


def check_owned_state(optimizer, reference, mappings, names, rank, world_size):
    for index, (group, mapping) in enumerate(zip(optimizer.param_groups, mappings)):
        if group["kind"] == "muon":
            chunk = (len(group["params"]) + world_size - 1) // world_size
            owned = min(chunk, max(0, len(group["params"]) - rank * chunk))
            actual = optimizer.state[group["params"][0]]
            expected = {key: torch.zeros_like(value) for key, value in actual.items()}
            if owned:
                source = reference.state[mapping["heads"][rank]]
                for key in expected:
                    expected[key][:owned].copy_(source[key])
            exact_state(actual, expected, f"muon_group_{index}")
        else:
            for parameter in group["params"]:
                name = names[id(parameter)]
                shards = mapping["parameters"][name]
                selected = shards[rank] if len(shards) == world_size else shards[0]
                exact_state(optimizer.state[parameter], reference.state[selected], name)


def check_arm(alpha, rank, world_size, device):
    model = make_model(alpha, device, compiled=True)
    eager = make_model(alpha, device, compiled=False)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    exact_state(initial, dict(eager.named_parameters()), "eager_initialization")
    initial_hashes = gather(digest(initial))
    if len(set(initial_hashes)) != 1:
        raise AssertionError("Rank initializations differ")
    x, y = batch(rank, device)
    local, losses = gradients(model, x, y)
    eager_grad, eager_losses = gradients(eager, x, y)
    checks = {"compiled_eager_gradients": compare_tensors(
        local, eager_grad, TOLERANCES["gradients_relative_l2"], TOLERANCES["gradients_max_scaled"])}
    model.eval()
    eager.eval()
    with torch.no_grad():
        logits = model(x)
        expected = eager(x)
        checks["compiled_eager_logits"] = compare_tensors(
            {"logits": logits}, {"logits": expected},
            TOLERANCES["logits_relative_l2"], TOLERANCES["logits_max_scaled"])
        # Same shape with changed suffix: a strict causal check independent of
        # different-sequence-length kernel effects.
        changed = x.clone()
        changed[:, 16:] = (changed[:, 16:] + 1) % 64
        future = model(changed)
        exact_state(logits[:, :16], future[:, :16], "future_token_independence")
        prefix = model(x[:, :16])
        cache = KVCache(batch_size=4, num_heads=1, seq_len=32, head_dim=64,
                        num_layers=model.cache_layers, device=device, dtype=torch.bfloat16)
        cached = torch.cat([model(x[:, :16], kv_cache=cache),
                            model(x[:, 16:17], kv_cache=cache),
                            model(x[:, 17:], kv_cache=cache)], dim=1)
        if cache.get_pos() != 32:
            raise AssertionError("Incorrect KV cache position")
        checks["prefix_and_cached_logits"] = compare_tensors(
            {"prefix": prefix, "cache": cached}, {"prefix": logits[:, :16], "cache": logits},
            TOLERANCES["logits_relative_l2"], TOLERANCES["logits_max_scaled"])
    del eager
    local_hashes = gather(digest(local))
    if len(set(local_hashes)) != world_size:
        raise AssertionError("Rank-local gradients unexpectedly identical")
    averaged = {}
    for name, value in local.items():
        averaged[name] = value.clone()
        dist.all_reduce(averaged[name], op=dist.ReduceOp.AVG)
    # Compute the global objective independently on one worker, as two B4 units.
    # FP32 accumulation avoids BF16 gradient-buffer order effects; casting back
    # gives the dtype consumed by the real optimizer's AVG reductions.
    global_model = make_model(alpha, device, compiled=True)
    manual = {name: torch.zeros_like(value, dtype=torch.float32) for name, value in initial.items()}
    global_losses = []
    for source_rank in range(world_size):
        reference_grad, reference_losses = gradients(global_model, *batch(source_rank, device))
        global_losses.append(reference_losses)
        for name, value in reference_grad.items():
            manual[name].add_(value.float(), alpha=1. / world_size)
    manual = {name: value.to(averaged[name].dtype) for name, value in manual.items()}
    checks["global_batch_gradients"] = compare_tensors(
        averaged, manual, TOLERANCES["global_gradients_relative_l2"],
        TOLERANCES["global_gradients_max_scaled"])
    del global_model
    optimizer = model.setup_optimizer(weight_decay=.05)
    if not isinstance(optimizer, DistMuonAdamW):
        raise AssertionError("Expected inherited DistMuonAdamW")
    named = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in named.items()}
    refs, reference_optimizer, mappings = make_reference(optimizer.param_groups, names, initial, averaged, world_size)
    # Local gradients are untouched: DistMuonAdamW performs its own reduction.
    optimizer.step()
    reference_optimizer.step()
    torch.cuda.synchronize(device)
    exact_state(named, refs, "distributed_update")
    check_owned_state(optimizer, reference_optimizer, mappings, names, rank, world_size)
    final_hashes = gather(digest(named))
    if len(set(final_hashes)) != 1:
        raise AssertionError("Ranks disagree after update")
    changed_parameters = sum(not torch.equal(initial[name], value) for name, value in named.items())
    if changed_parameters != len(named):
        raise AssertionError("Some parameters did not update")
    return {"alpha": alpha, "passed": all(check["passed"] for check in checks.values()),
            "initial_sha256": initial_hashes[0], "final_sha256": final_hashes[0],
            "local_gradient_sha256": local_hashes[rank], "losses": losses,
            "eager_losses": eager_losses, "global_microbatch_losses": global_losses,
            "checks": checks, "exact_distributed_update_and_owned_state": True,
            "exact_future_token_independence": True, "distinct_rank_gradients": True,
            "changed_parameters": changed_parameters, "parameter_count": len(named)}


def write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report = {"passed": False, "rank": rank, "torch_version": torch.__version__,
              "tolerances": TOLERANCES, "reports": []}
    if rank == 0:
        write_json(args.output, {**report, "status": "incomplete"})
    try:
        if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 10):
            raise RuntimeError("Requires pinned torch 2.10 or newer")
        if os.environ.get("TORCH_COMPILE_DISABLE", "0") not in ("0", ""):
            raise RuntimeError("Compiled kernels must remain enabled")
        _, rank, _, world_size, device = compute_init("cuda")
        if world_size != 2 or COMPUTE_DTYPE != torch.bfloat16:
            raise RuntimeError("Requires two workers and NANOCHAT_DTYPE=bfloat16")
        if not attention.HAS_FA3 or not attention.USE_FA3:
            raise RuntimeError("Real Flash Attention 3 is required; fallback is not accepted")
        report.update(world_size=world_size, compute_dtype=str(COMPUTE_DTYPE),
                      device_name=torch.cuda.get_device_name(device), fa3=True,
                      compiled_blocks=True, harness_recompile_limit=64)
        with torch._dynamo.config.patch(recompile_limit=64, suppress_errors=False):
            for alpha in (0., .01):
                report["reports"].append(check_arm(alpha, rank, world_size, device))
        if report["reports"][0]["initial_sha256"] != report["reports"][1]["initial_sha256"]:
            raise AssertionError("A/B initialization differs")
        if report["reports"][0]["final_sha256"] == report["reports"][1]["final_sha256"]:
            raise AssertionError("A/B updates are identical; auxiliary test is vacuous")
        report["passed"] = all(arm["passed"] for arm in report["reports"])
        reports = gather(report)
        if rank == 0:
            write_json(args.output, {"passed": all(item["passed"] for item in reports),
                                     "ranks": reports})
        return 0 if all(item["passed"] for item in reports) else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if rank == 0:
            write_json(args.output, report)
        traceback.print_exc()
        return 1
    finally:
        write_json(args.output.with_name(args.output.stem + f".rank{rank}.json"), report)
        compute_cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
