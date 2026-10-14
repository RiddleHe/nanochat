"""Rank-local data/RNG state for exact continuation at an optimizer boundary."""
import os

import torch
import torch.distributed as dist


def save_training_state(directory, step, rank, x, y, data_state, scaler=None):
    state = {"x": x.cpu(), "y": y.cpu(), "data": data_state,
             "rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state() if x.is_cuda else None,
             "scaler": scaler.state_dict() if scaler is not None else None}
    path = os.path.join(directory, f"train_{step:06d}_rank{rank}.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_training_state(directory, step, rank, device, scaler=None):
    path = os.path.join(directory, f"train_{step:06d}_rank{rank}.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return state["x"].to(device), state["y"].to(device), state["data"]


def check_finite_gradients(model, loss):
    # All ranks must agree to stop before the distributed optimizer collectives.
    finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                          if p.grad is not None] + [torch.isfinite(loss).all()]).all().int()
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("Non-finite loss/gradient; optimizer step was not applied")
