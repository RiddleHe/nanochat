"""Dense candidate implementation of per-token, per-loop block selection.

``n_layer`` is PHYSICAL depth. Each loop reuses the same blocks, in their
original order, with a separate KV cache slot for every (loop, block). Every
candidate computes all tokens' K/V, including tokens that reject its residual.
There is no sparse compute saving in this implementation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.common import get_dist_info, print0
from nanochat.model.gpt_base import GPTBaseConfig
from nanochat.model.gpt_base_no_embed_norm import GPTBaseNoEmbedNorm
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class GPTLoopSelectConfig(GPTBaseConfig):
    n_layer: int = 24
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 1536
    num_loops: int = 4
    selector_mode: str = "full"
    select_k: int = 16
    checkpoint_blocks: bool = False

    @property
    def effective_depth(self):
        return self.n_layer * self.num_loops

    def __post_init__(self):
        if self.n_layer < 1 or self.num_loops < 1:
            raise ValueError("Physical n_layer and num_loops must be positive")
        if self.selector_mode not in ("full", "learned"):
            raise ValueError("selector_mode must be full or learned")
        if not 1 <= self.select_k <= self.n_layer:
            raise ValueError("select_k must be in [1, physical n_layer]")
        if self.selector_mode == "learned" and self.select_k == self.n_layer:
            raise ValueError("Use selector_mode=full for the all-on baseline")
        if not self.window_pattern or any(c not in "SL" for c in self.window_pattern):
            raise ValueError("window_pattern must be a nonempty string of S/L")


class _FixedCountSigmoid(torch.autograd.Function):
    """FP32 threshold solve with its implicit derivative, temperature fixed at 1.

    This is the derivative of the fixed-count *soft* relaxation, not discrete
    top-k. A saturated denominator is a numerical failure, never clamped away.
    Fixed iterations permit compilation without a data-dependent host loop.
    """

    @staticmethod
    def forward(ctx, scores, k):
        scores = scores.float()
        torch._assert_async(torch.isfinite(scores).all(), "Nonfinite selector scores")
        # Centering improves numerical invariance to a shared score offset.
        scores = scores - scores.mean(dim=-1, keepdim=True)
        lo = scores.amin(dim=-1, keepdim=True) - 32.0
        hi = scores.amax(dim=-1, keepdim=True) + 32.0
        for _ in range(40):
            threshold = (lo + hi) * 0.5
            too_many = torch.sigmoid(scores - threshold).sum(dim=-1, keepdim=True) > k
            lo = torch.where(too_many, threshold, lo)
            hi = torch.where(too_many, hi, threshold)
        soft = torch.sigmoid(scores - (lo + hi) * 0.5)
        a = soft * (1.0 - soft)
        denominator = a.sum(dim=-1, keepdim=True)
        torch._assert_async((denominator > 1e-12).all(), "Saturated selector threshold derivative")
        torch._assert_async(((soft.sum(dim=-1) - k).abs() < 1e-4).all(), "Selector threshold did not converge")
        ctx.save_for_backward(a, denominator)
        return soft

    @staticmethod
    def backward(ctx, grad_output):
        a, denominator = ctx.saved_tensors
        v = grad_output.float()
        centered = v - (a * v).sum(dim=-1, keepdim=True) / denominator
        return a * centered, None


def fixed_count_sigmoid(scores, k):
    """Differentiable fixed-count sigmoid; last-axis sum equals k (FP32)."""
    if not 0 < k < scores.shape[-1]:
        raise ValueError("The soft relaxation requires 0 < k < number of scores")
    return _FixedCountSigmoid.apply(scores, k)


class _HardForwardSoftBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hard, soft):
        # Direct return avoids roundoff in soft + stop_gradient(hard - soft).
        return hard.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


def topk_straight_through(scores, k):
    """Exact binary forward, lower-index tie breaking, biased soft backward."""
    soft = fixed_count_sigmoid(scores, k)
    order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
    hard = torch.zeros_like(soft).scatter_(-1, order[..., :k], 1.0)
    return _HardForwardSoftBackward.apply(hard, soft)


class _WholeBlockResidual(torch.autograd.Function):
    """h + gate * (candidate - h), with exact binary forward in BF16 too.

    The algebraic subtract/add implementation can round a retained candidate a
    second time in BF16. Selecting the exact branch avoids that numerical change;
    backward is the same residual-gate derivative, including the scorer path.
    """

    @staticmethod
    def forward(ctx, h, candidate, gate):
        ctx.save_for_backward(h, candidate, gate)
        return torch.where(gate.bool(), candidate, h)

    @staticmethod
    def backward(ctx, grad_output):
        h, candidate, gate = ctx.saved_tensors
        selected = gate.bool()
        grad_h = torch.where(selected, 0.0, grad_output)
        grad_candidate = torch.where(selected, grad_output, 0.0)
        grad_gate = (grad_output.float() * (candidate.float() - h.float())).sum(dim=-1, keepdim=True)
        return grad_h, grad_candidate, grad_gate


class GPTLoopSelect(GPTBaseNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        self.cache_layers = config.effective_depth
        self.window_sizes = self.window_sizes * config.num_loops
        # The full model has no unused scorer parameters (DistMuonAdamW expects
        # every optimized parameter to have a gradient, even when it is zero).
        if config.selector_mode == "learned":
            self.selector_weight = nn.Parameter(torch.empty(config.n_layer, config.n_embd))
            self.selector_bias = nn.Parameter(torch.empty(config.num_loops, config.n_layer))
        else:
            self.register_parameter("selector_weight", None)
            self.register_parameter("selector_bias", None)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        if self.selector_weight is not None:
            device = self.selector_weight.device
            devices = [device.index] if device.type == "cuda" else []
            # Neither backbone initialization nor the subsequent data/RNG stream
            # changes when the learned scorer is added to the full model.
            with torch.random.fork_rng(devices=devices):
                nn.init.normal_(self.selector_weight, std=0.02 / self.config.n_embd**0.5)
                nn.init.zeros_(self.selector_bias)

    def selection_gates(self, loop_entrance, loop):
        """Compute once per loop; useful for held-out selection diagnostics too."""
        if self.selector_weight is None:
            return torch.ones(*loop_entrance.shape[:-1], self.config.n_layer,
                              device=loop_entrance.device, dtype=torch.float32)
        # Scorer normalization does not alter the backbone residual stream.
        with torch.autocast(device_type=loop_entrance.device.type, enabled=False):
            x = F.rms_norm(loop_entrance.float(), (self.config.n_embd,))
            scores = F.linear(x, self.selector_weight.float(), self.selector_bias[loop].float())
            return topk_straight_through(scores, self.config.select_k)

    def _forward_blocks(self, x, cos_sin, kv_cache):
        if kv_cache is not None and kv_cache.n_layers != self.cache_layers:
            raise ValueError("KV cache needs num_loops * physical n_layer separate slots")
        for loop in range(self.config.num_loops):
            gates = None if self.selector_weight is None else self.selection_gates(x, loop)
            for j, block in enumerate(self.transformer.h):
                cache_idx = loop * self.config.n_layer + j
                if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    if kv_cache is not None:
                        raise ValueError("Activation recomputation cannot mutate a KV cache")
                    candidate = checkpoint(block, x, cos_sin, None, cache_idx,
                                           use_reentrant=False, preserve_rng_state=False)
                else:
                    candidate = block(x, cos_sin, kv_cache, cache_layer_idx=cache_idx)
                x = candidate if gates is None else _WholeBlockResidual.apply(x, candidate, gates[..., j:j+1])
        return x

    def num_scaling_params(self):
        counts = {
            "wte": sum(p.numel() for p in self.transformer.wte.parameters()),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters()),
            "transformer_matrices": sum(p.numel() for p in self.transformer.h.parameters()),
            "scalars": 0,
            "selector": 0 if self.selector_weight is None else self.selector_weight.numel() + self.selector_bias.numel(),
        }
        counts["total"] = sum(counts.values())
        assert counts["total"] == sum(p.numel() for p in self.parameters())
        return counts

    def flop_breakdown(self):
        """Muyu model-training convention: 6 per matmul parameter use + attention.

        All candidate matrices AND all candidate attention run on every loop.
        The shared scorer matrix is counted once per loop. Like GPTBase, this
        estimate excludes scalar/elementwise work (including threshold solving),
        activation recomputation, optimizer work and evaluation. Report actual
        runtime separately; selecting 16/24 does not reduce candidate FLOPs.
        """
        counts = self.num_scaling_params()
        t, d = self.config.sequence_len, self.config.n_embd
        return {
            "backbone_matrices": 6 * self.config.num_loops * counts["transformer_matrices"],
            "lm_head": 6 * counts["lm_head"],
            "attention": sum(12 * d * (t if w[0] < 0 else min(t, w[0])) for w in self.window_sizes),
            "selector_matrix": 0 if self.selector_weight is None else 6 * self.config.num_loops * self.selector_weight.numel(),
        }

    def estimate_flops(self):
        return sum(self.flop_breakdown().values())

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2,
                        matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        # Preserve GPTBase's exact backbone groups. The scorer never enters Muon
        # or dimension-based LR scaling; its schedule uses initial_lr like all groups.
        ddp, rank, local_rank, world_size = get_dist_info()
        scale = (self.config.n_embd / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW backbone parameters by {scale:.6f}")
        matrices = [p for p in self.transformer.h.parameters() if p.dim() >= 2]
        groups = [
            dict(kind="adamw", params=list(self.lm_head.parameters()), lr=unembedding_lr * scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=list(self.transformer.wte.parameters()), lr=embedding_lr * scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        ]
        for shape in sorted({p.shape for p in matrices}):
            groups.append(dict(kind="muon", params=[p for p in matrices if p.shape == shape],
                               lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9,
                               weight_decay=weight_decay))
        if self.selector_weight is not None:
            if ddp and self.selector_weight.numel() >= 1024 and self.config.n_layer % world_size:
                raise ValueError("Selector rows must divide world_size for distributed AdamW")
            groups.append(dict(kind="adamw", role="selector",
                               params=[self.selector_weight, self.selector_bias],
                               lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))
        grouped = [p for group in groups for p in group["params"]]
        assert len(grouped) == len({id(p) for p in grouped}) == len(list(self.parameters()))
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
