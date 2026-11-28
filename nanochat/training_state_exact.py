"""CPU-serializable per-rank iterator and RNG state for exact training resume.

Save after an optimizer step, with no prefetched batch. Restore after constructing
and loading the model and optimizer, immediately before continuing training.
The caller saves model, optimizer, schedule/step, and any GradScaler separately.
Only the rank's selected CUDA device is touched; no other GPUs are initialized.
"""

import random

import numpy as np
import torch


def capture_rng_state(device="cpu"):
    """Capture global Python, NumPy, CPU Torch, and selected-device Torch RNG."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Exact RNG capture supports CPU and CUDA devices")
    numpy_state = np.random.get_state()
    return {
        "format_version": 1,
        "python": random.getstate(),
        # NumPy arrays/scalars are deliberately converted: their pickle globals
        # are not permitted by torch.load(weights_only=True).
        "numpy": {
            "generator": str(numpy_state[0]),
            "keys": torch.tensor(numpy_state[1].astype(np.int64), dtype=torch.int64),
            "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": (torch.cuda.get_rng_state(device).cpu().clone()
                       if device.type == "cuda" else None),
        "device_type": device.type,
    }


def restore_rng_state(state, device="cpu"):
    """Restore RNG after all initialization that might consume randomness."""
    device = torch.device(device)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported RNG checkpoint format")
    if state.get("device_type") != device.type:
        raise ValueError("RNG checkpoint device type differs from restore device")
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((
        numpy_state["generator"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"], numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError("Missing CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def capture_rank_training_state(loader, *, rank, world_size, device):
    """Capture one rank; collect/save one such record for every training rank."""
    if loader.rank != rank or loader.world_size != world_size:
        raise ValueError("Loader rank/world_size differs from training rank/world_size")
    return {
        "format_version": 1, "rank": int(rank), "world_size": int(world_size),
        "loader": loader.state_dict(), "rng": capture_rng_state(device),
    }


def restore_rank_training_state(state, loader, *, rank, world_size, device):
    """Validate rank topology, restore iterator, then restore this rank's RNG."""
    if state.get("format_version") != 1:
        raise ValueError("Unsupported rank training checkpoint format")
    if state.get("rank") != rank or state.get("world_size") != world_size:
        raise ValueError("Rank training checkpoint rank/world_size mismatch")
    if loader.rank != rank or loader.world_size != world_size:
        raise ValueError("Loader rank/world_size differs from training rank/world_size")
    loader.load_state_dict(state["loader"])
    restore_rng_state(state["rng"], device)
