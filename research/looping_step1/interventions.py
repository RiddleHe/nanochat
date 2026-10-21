"""Minimal measurement hooks and per-use LoRA; no changes to baseline files."""
import math
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from common import GROUPS, TARGETS, SETTINGS, matrices


class VisitGradients:
    def __init__(self, model):
        self.modules = dict(matrices(model))
        self.handles = []
        self.visits = {}
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def capture(module, args, output):
            slot = {"input": args[0].detach()}
            self.visits.setdefault(name, []).append(slot)

            def backward(grad):
                # Reproduce the dtype of Linear's actual weight-gradient matmul.
                # Accumulation/statistical reductions below are FP32.
                with torch.no_grad():
                    incoming = slot.pop("input")
                    slot["gradient"] = (
                        grad.reshape(-1, grad.shape[-1]).T
                        @ incoming.reshape(-1, incoming.shape[-1])
                    ).float()
            output.register_hook(backward)
        return capture

    def reset(self):
        self.visits.clear()

    def statistics(self, loops=4):
        rows = []
        max_error = 0.0
        for name, module in self.modules.items():
            visits = self.visits[name]
            assert len(visits) == loops, (name, len(visits))
            gradients = torch.stack([v.pop("gradient").flatten() for v in visits])
            if not torch.isfinite(gradients).all():
                raise RuntimeError(f"Nonfinite gradients: {name}")
            gram = gradients @ gradients.T
            norms = gram.diag().clamp_min(0).sqrt()
            valid = norms > 1e-20
            cosine = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
            cosine = cosine.clamp(-1, 1)
            total = gradients.sum(0)
            ordinary = module.weight.grad.flatten().float()
            error = ((total - ordinary).norm() / ordinary.norm().clamp_min(1e-20)).item()
            max_error = max(max_error, error)
            rows.append(dict(
                name=name, cosine=cosine.cpu().tolist(), norms=norms.cpu().tolist(),
                valid_norms=valid.cpu().tolist(),
                cancellation_ratio=(total.norm() / norms.sum().clamp_min(1e-20)).item(),
                sum_relative_error=error,
            ))
        self.reset()
        return rows, max_error

    def close(self):
        for h in self.handles:
            h.remove()


class ScaleOutputs:
    def __init__(self, model, kind, loop, scale):
        self.loop, self.scale = loop, scale
        self.counts, self.scaled = {}, 0
        self.handles = []
        for i, block in enumerate(model.transformer.h):
            self.handles.append(getattr(block, kind).register_forward_hook(self._hook(i)))

    def _hook(self, i):
        def hook(module, args, output):
            visit = self.counts.get(i, 0)
            self.counts[i] = visit + 1
            if visit == self.loop:
                self.scaled += 1
                return output * self.scale
            return output
        return hook

    def reset(self):
        if self.counts:
            assert all(v == 4 for v in self.counts.values())
            assert self.scaled == len(self.counts)
        self.counts = {}
        self.scaled = 0

    def close(self):
        self.reset()
        for h in self.handles:
            h.remove()


class LoopLoRA(nn.Module):
    def __init__(self, base, rank, independent, seed, loops=4):
        super().__init__()
        self.base = base
        self.independent = independent
        self.visit = 0
        self.rank = rank
        count = loops if independent else 1
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        first = torch.empty(rank, base.weight.shape[1], device=base.weight.device, dtype=torch.float32)
        first.uniform_(-1 / math.sqrt(first.shape[1]), 1 / math.sqrt(first.shape[1]), generator=generator)
        self.A = nn.ParameterList([nn.Parameter(first.clone()) for _ in range(count)])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(base.weight.shape[0], rank, device=base.weight.device))
            for _ in range(count)
        ])

    def forward(self, x):
        index = self.visit if self.independent else 0
        a, b = self.A[index].to(x.dtype), self.B[index].to(x.dtype)
        return self.base(x) + F.linear(F.linear(x, a), b)


def attach_lora(model, group, variant, small_rank=None):
    model.requires_grad_(False)
    rank = SETTINGS["rank_small"][group] if small_rank is None else small_rank
    if variant == "shared_large":
        rank *= model.config.num_loops
    adapters = []
    for i, block in enumerate(model.transformer.h):
        for short in GROUPS[group]:
            path = TARGETS[short]
            parent_name, name = path.split(".")
            parent = getattr(block, parent_name)
            adapter = LoopLoRA(
                getattr(parent, name), rank, variant == "independent",
                SETTINGS["seed"] + 100 * i + list(TARGETS).index(short),
                model.config.num_loops,
            )
            setattr(parent, name, adapter)
            adapters.append(adapter)

    def blocks(self, x, cos_sin, kv_cache):
        if kv_cache is not None:
            raise ValueError("These experiments use full-sequence forwards, not generation caches")
        for loop in range(self.config.num_loops):
            for adapter in adapters:
                adapter.visit = loop
            for i, block in enumerate(self.transformer.h):
                x = block(x, cos_sin, None, cache_layer_idx=loop * len(self.transformer.h) + i)
        return x

    model._forward_blocks = MethodType(blocks, model)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert all(".A." in n or ".B." in n for n, _ in parameters)
    return adapters, parameters
