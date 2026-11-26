"""Two-rank CUDA optimizer integration check, using real model gradients.

Run on the authorized pair with the pinned PyTorch 2.10 runtime:
    NANOCHAT_DTYPE=bfloat16 torchrun --standalone --nproc_per_node=2 \
        tests/test_loop_distributed.py

Both full/learned arms use nonzero residual projections, activation recomputation,
compiled blocks, and the unchanged compiled optimizer kernels. Rank-local input
batches differ. Manually averaged gradients feed the single-process MuonAdamW
reference. Acceptance is bitwise against reference groups split to the identical
Muon batch shapes / Adam row slices used by DistMuonAdamW. All owned optimizer
moments must also match exactly, and final model hashes must agree across ranks.

The ordinary unsplit MuonAdamW reference is also measured, but only diagnostic:
different batched GEMM shapes can select different BF16 accumulation kernels in
Polar Express. No tolerance is used to hide distributed communication errors.
The harness allows 64 compiled variants per function because three optimizer
layouts and both arms intentionally coexist; production retains its defaults.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

# Support torchrun's file entry point as well as module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init
from nanochat.model.gpt_loop_select import GPTLoopSelect, GPTLoopSelectConfig
from nanochat.optim import DistMuonAdamW, MuonAdamW
from scripts.verify_loop_resume import Comparison


def digest(parameters):
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_reference(groups, names, initial, averaged_gradients, world_size, matched_shapes):
    """Make a MuonAdamW reference; shard views share its full parameter storage."""
    parameters = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
    for name, parameter in parameters.items():
        parameter.grad = averaged_gradients[name].clone()
    reference_groups, state_mapping = [], []
    for group in groups:
        options = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        keys = [names[id(parameter)] for parameter in group["params"]]
        if group["kind"] == "muon":
            chunk = (len(keys) + world_size - 1) // world_size if matched_shapes else len(keys)
            chunk_first_parameters = []
            for start in range(0, len(keys), chunk):
                chunk_parameters = [parameters[key] for key in keys[start:start + chunk]]
                reference_groups.append({**options, "params": chunk_parameters})
                chunk_first_parameters.append(chunk_parameters[0])
            state_mapping.append({"kind": "muon", "chunks": chunk_first_parameters})
        else:
            refs, by_name = [], {}
            for name in keys:
                parameter = parameters[name]
                if matched_shapes and parameter.numel() >= 1024:
                    if parameter.shape[0] % world_size:
                        raise AssertionError("Reference Adam row slicing requires divisible first dimension")
                    rows = parameter.shape[0] // world_size
                    shards = []
                    for rank in range(world_size):
                        selected = slice(rank * rows, (rank + 1) * rows)
                        shard = torch.nn.Parameter(parameter.detach()[selected])
                        shard.grad = averaged_gradients[name][selected].clone()
                        shards.append(shard)
                    refs.extend(shards)
                    by_name[name] = shards
                else:
                    refs.append(parameter)
                    by_name[name] = [parameter]
            reference_groups.append({**options, "params": refs})
            state_mapping.append({"kind": "adamw", "parameters": by_name})
    return parameters, MuonAdamW(reference_groups), state_mapping


def optimizer_state_comparison(optimizer, reference, mapping, names, rank, world_size):
    comparison = Comparison("owned_optimizer_state", max_details=12)
    for group_index, (group, mapped) in enumerate(zip(optimizer.param_groups, mapping)):
        if group["kind"] == "muon":
            chunk = (len(group["params"]) + world_size - 1) // world_size
            owned = min(chunk, max(0, len(group["params"]) - rank * chunk))
            actual = optimizer.state[group["params"][0]]
            expected = {}
            for key, value in actual.items():
                if not torch.is_tensor(value):
                    raise AssertionError(f"Unexpected non-tensor Muon state {key}")
                expected[key] = torch.zeros_like(value)
                if owned:
                    source = reference.state[mapped["chunks"][rank]][key]
                    expected[key][:owned].copy_(source)
            comparison.compare(expected, actual, f"muon_group_{group_index}")
        else:
            for parameter in group["params"]:
                name = names[id(parameter)]
                candidates = mapped["parameters"][name]
                selected = candidates[rank] if len(candidates) == world_size else candidates[0]
                comparison.compare(reference.state[selected], optimizer.state[parameter], name)
    return comparison.result


def check_arm(mode, rank, world_size, device, compile_blocks):
    torch.manual_seed(731)
    torch.cuda.manual_seed(731)
    config = GPTLoopSelectConfig(sequence_len=16, vocab_size=64, n_layer=4,
                                n_embd=256, n_head=4, n_kv_head=4,
                                num_loops=2, select_k=2, selector_mode=mode,
                                checkpoint_blocks=True, window_pattern="SSSL")
    with torch.device("meta"):
        model = GPTLoopSelect(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.025)
            block.mlp.c_proj.weight.normal_(std=0.025)
    if compile_blocks:
        for block in model.transformer.h:
            block.forward = torch.compile(block.forward, dynamic=False)
    optimizer = model.setup_optimizer(weight_decay=0.05)
    if not isinstance(optimizer, DistMuonAdamW):
        raise AssertionError("Model did not select DistMuonAdamW")
    named = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in named.items()}
    initial = {name: parameter.detach().clone() for name, parameter in named.items()}
    initial_hashes = [None] * world_size
    dist.all_gather_object(initial_hashes, digest(initial))
    if len(set(initial_hashes)) != 1:
        raise AssertionError("Model initialization differs across ranks")
    generator = torch.Generator(device=device).manual_seed(9100 + rank)
    x = torch.randint(0, config.vocab_size, (2, config.sequence_len), device=device, generator=generator)
    y = torch.randint(0, config.vocab_size, x.shape, device=device, generator=generator)
    loss = model(x, y)
    loss.backward()
    missing = [name for name, parameter in named.items() if parameter.grad is None]
    if missing:
        raise AssertionError(f"Missing gradients: {missing}")
    finite = all(bool(torch.isfinite(parameter.grad).all().item()) for parameter in named.values())
    if not finite or not bool(torch.isfinite(loss).item()):
        raise AssertionError("Nonfinite loss/gradient")
    nonzero_names = [name for name, parameter in named.items() if bool((parameter.grad != 0).any().item())]
    if len(nonzero_names) != len(named):
        raise AssertionError(f"Vacuous zero-gradient parameters: {sorted(named.keys() - set(nonzero_names))}")
    averaged = {}
    for name, parameter in named.items():
        averaged[name] = parameter.grad.detach().clone()
        dist.all_reduce(averaged[name], op=dist.ReduceOp.AVG)
    # Prove ranks supplied different data/gradients, so the reduction is exercised.
    gradient_hashes = [None] * world_size
    dist.all_gather_object(gradient_hashes, digest({name: p.grad for name, p in named.items()}))
    if len(set(gradient_hashes)) != world_size:
        raise AssertionError("Rank-local gradients unexpectedly identical")
    matched_parameters, matched_optimizer, mapping = make_reference(
        optimizer.param_groups, names, initial, averaged, world_size, matched_shapes=True)
    ordinary_parameters, ordinary_optimizer, _ = make_reference(
        optimizer.param_groups, names, initial, averaged, world_size, matched_shapes=False)
    optimizer.step()
    matched_optimizer.step()
    ordinary_optimizer.step()
    torch.cuda.synchronize(device)
    exact = Comparison("distributed_vs_shape_matched_MuonAdamW", max_details=12)
    exact.compare(matched_parameters, named)
    state_check = optimizer_state_comparison(optimizer, matched_optimizer, mapping, names, rank, world_size)
    diagnostic = Comparison("ordinary_full_group_vs_shape_matched_diagnostic", max_details=8)
    diagnostic.compare(ordinary_parameters, matched_parameters)
    final_hashes = [None] * world_size
    dist.all_gather_object(final_hashes, digest(named))
    changed = sum(not torch.equal(initial[name], value) for name, value in named.items())
    local = {
        "rank": rank, "mode": mode, "loss": float(loss.detach().item()),
        "passed": exact.result["equal"] and state_check["equal"] and len(set(final_hashes)) == 1
                  and changed == len(named),
        "nonzero_gradient_parameters": len(nonzero_names), "parameters_changed": changed,
        "parameter_count": len(named), "ranks_have_distinct_gradients": True,
        "ranks_have_identical_final_parameters": len(set(final_hashes)) == 1,
        "parameters": exact.result, "optimizer_state": state_check,
        "full_group_diagnostic": diagnostic.result,
    }
    reports = [None] * world_size
    dist.all_gather_object(reports, local)
    return {"mode": mode, "passed": all(report["passed"] for report in reports), "ranks": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("full", "learned"), default=["full", "learned"])
    parser.add_argument("--no-compile-blocks", action="store_true")
    args = parser.parse_args()
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 10):
        raise RuntimeError("Use the repository-pinned torch 2.10 runtime")
    if os.environ.get("TORCH_COMPILE_DISABLE", "0") not in ("0", ""):
        raise RuntimeError("Compiled optimizer kernels must remain enabled")
    _, rank, _, world_size, device = compute_init("cuda")
    try:
        if world_size != 2 or COMPUTE_DTYPE != torch.bfloat16:
            raise RuntimeError("This acceptance check requires two ranks and NANOCHAT_DTYPE=bfloat16")
        # Three independent optimizer layouts exercise more shapes/strides than
        # a production run. The default eight variants fails before comparison.
        # Scope this bounded allowance to the harness; preserve all numerical
        # kernels, compiler settings, and bitwise acceptance checks unchanged.
        with torch._dynamo.config.patch(recompile_limit=64):
            reports = [check_arm(mode, rank, world_size, device, not args.no_compile_blocks)
                       for mode in args.modes]
        passed = all(report["passed"] for report in reports)
        if rank == 0:
            print(json.dumps({"passed": passed, "torch_version": torch.__version__,
                              "compute_dtype": str(COMPUTE_DTYPE), "world_size": world_size,
                              "compiled_blocks": not args.no_compile_blocks,
                              "harness_recompile_limit": 64,
                              "acceptance_tolerance": "bitwise against identical kernel shapes",
                              "reports": reports}, sort_keys=True, allow_nan=False), flush=True)
        return 0 if passed else 1
    finally:
        compute_cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
