"""Weighted12 A/B entry: shared candidates, discrete support, ordinary gradients.

Run as ``python -m scripts.train_weighted_loop_ab --config ... --output ...``.
A/B differ only in balance_alpha (0 or .01). All candidates still compute.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import nanochat.model.gpt_base as base
from nanochat.model.gpt_loop_select import GPTLoopSelect, GPTLoopSelectConfig


@dataclass
class WeightedLoopConfig(GPTLoopSelectConfig):
    selector_mode: str = "learned"
    select_k: int = 12
    balance_alpha: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.selector_mode != "learned":
            raise ValueError("Weighted selection requires selector_mode=learned")
        if self.balance_alpha not in (0.0, 0.01):
            raise ValueError("The frozen A/B screen uses balance_alpha 0 or 0.01")


def weighted_topk(scores, k):
    """Stable lower-index ties; unselected scores have no direct task gradient."""
    if not 0 < k < scores.shape[-1]:
        raise ValueError("Expected 0 < k < number of scores")
    scores = scores.float()
    torch._assert_async(torch.isfinite(scores).all(), "Nonfinite selector scores")
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
    weights = k * scores.gather(-1, indices).softmax(dim=-1)
    torch._assert_async((weights > 0).all(), "Selected weight underflow")
    return torch.zeros_like(scores).scatter(-1, indices, weights)


def weighted_residual(h, candidate, gate):
    """FP32 residual arithmetic, then restore the backbone activation dtype (BF16)."""
    return (h.float() + gate.float() * (candidate.float() - h.float())).to(h.dtype)


def pooled_balance(selected_counts, probability_sums, positions, k):
    """Pool loops before multiplying f and P; counts never carry gradients."""
    decisions = positions * selected_counts.shape[0]
    frequencies = selected_counts.detach().sum(0) / (decisions * k)
    probabilities = probability_sums.sum(0) / decisions
    return selected_counts.shape[-1] * (frequencies * probabilities).sum()


class WeightedLoop(GPTLoopSelect):
    @staticmethod
    def validate_training_config(cfg):
        """Reject changes to the frozen balance unit before device initialization."""
        if cfg["device_batch_size"] != 4:
            raise ValueError("Weighted12 requires four sequences per microbatch")
        model = cfg["model"]
        expected = dict(n_layer=24, num_loops=4, select_k=12, sequence_len=2048,
                        n_embd=1536, n_head=12, n_kv_head=12, selector_mode="learned")
        for key, value in expected.items():
            if model.get(key) != value:
                raise ValueError(f"Frozen Weighted12 requires {key}={value}")
        if model.get("balance_alpha") not in (0.0, 0.01):
            raise ValueError("Expected balance_alpha 0 or 0.01")

    def selection_scores(self, loop_entrance, loop):
        with torch.autocast(device_type=loop_entrance.device.type, enabled=False):
            x = F.rms_norm(loop_entrance.float(), (self.config.n_embd,))
            return F.linear(x, self.selector_weight.float(), self.selector_bias[loop].float())

    def selection_gates(self, loop_entrance, loop):
        return weighted_topk(self.selection_scores(loop_entrance, loop), self.config.select_k)

    def _run_blocks(self, x, cos_sin, kv_cache, collect_stats):
        if kv_cache is not None and kv_cache.n_layers != self.cache_layers:
            raise ValueError("KV cache requires separate slots for every loop and block")
        counts, probabilities, masses = [], [], []
        positions = x.shape[0] * x.shape[1]
        for loop in range(self.config.num_loops):
            scores = self.selection_scores(x, loop)
            gates = weighted_topk(scores, self.config.select_k)
            if collect_stats:
                counts.append((gates.detach() > 0).float().sum((0, 1)))
                probabilities.append(scores.softmax(-1).sum((0, 1)))
                masses.append(gates.detach().sum((0, 1)))
            for j, block in enumerate(self.transformer.h):
                cache_idx = loop * self.config.n_layer + j
                if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    if kv_cache is not None:
                        raise ValueError("Activation recomputation cannot mutate a KV cache")
                    candidate = checkpoint(block, x, cos_sin, None, cache_idx,
                                           use_reentrant=False, preserve_rng_state=False)
                else:
                    candidate = block(x, cos_sin, kv_cache, cache_layer_idx=cache_idx)
                x = weighted_residual(x, candidate, gates[..., j:j+1])
        if not collect_stats:
            return x, None
        counts, probabilities = torch.stack(counts), torch.stack(probabilities)
        return x, dict(balance=pooled_balance(counts, probabilities, positions, self.config.select_k),
                       selected_counts=counts, gate_mass=torch.stack(masses))

    def _forward_blocks(self, x, cos_sin, kv_cache):
        return self._run_blocks(x, cos_sin, kv_cache, False)[0]

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", *, return_stats=False):
        # Ordinary evaluation is inherited pure CE; stats are explicit return values,
        # never attributes or hooks that activation recomputation could mutate.
        if not return_stats:
            return super().forward(idx, targets, kv_cache, loss_reduction)
        _, length = idx.shape
        start = 0 if kv_cache is None else kv_cache.get_pos()
        assert start + length <= self.cos.size(1), "Sequence exceeds rotary cache"
        assert idx.device == self.cos.device and self.cos.dtype == base.COMPUTE_DTYPE
        if kv_cache is not None:
            assert start + length <= kv_cache.max_seq_len, "Sequence exceeds KV cache"
        cos_sin = self.cos[:, start:start+length], self.sin[:, start:start+length]
        x = self._prepare_embeddings(self.transformer.wte(idx).to(base.COMPUTE_DTYPE))
        x, stats = self._run_blocks(x, cos_sin, kv_cache, True)
        logits = self.lm_head(base.norm(x))[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        result = logits if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-1, reduction=loss_reduction)
        return result, stats

    def training_loss(self, idx, targets, *, return_stats=False):
        if idx.shape[0] != 4:
            raise ValueError("Balance aggregation requires exactly four sequences")
        ce, stats = self(idx, targets, return_stats=True)
        losses = (ce, self.config.balance_alpha * stats["balance"])
        return (*losses, stats) if return_stats else losses


def main():
    from scripts.train_loop_select import main as train
    train(model_class=WeightedLoop, config_class=WeightedLoopConfig)


if __name__ == "__main__":
    main()
