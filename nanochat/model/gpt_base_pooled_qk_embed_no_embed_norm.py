"""Raw initial residual; target attention optionally normalizes Emb and sums heads.

Target attention is mean_h(softmax(Q_h K_h^T / sqrt(d))) @ Emb @ W_o.
There is no W_v at these layers. All other blocks retain ordinary attention.
"""

from dataclasses import dataclass, fields
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nanochat.model import gpt_base as base
from nanochat.model.gpt_base_no_embed_norm import GPTBaseNoEmbedNorm


D24_TARGET_LAYERS = (9, 10, 11, 12, 13, 14, 15, 18, 21, 22, 23)


def target_layers_for_depth(depth):
    if depth < 1:
        raise ValueError("Depth must be positive")
    # Integer half-up rounding, including collapsed targets at shallow depths.
    return tuple(sorted({(2 * i * (depth - 1) + 23) // 46 for i in D24_TARGET_LAYERS}))


@dataclass
class GPTBasePooledQKEmbedNoEmbedNormConfig(base.GPTBaseConfig):
    attention_chunk_size: int = 128
    pooled_qk_target_layers: tuple[int, ...] | None = None
    norm_value_input: bool = False
    sum_head_attention: bool = False

    def __post_init__(self):
        if type(self.norm_value_input) is not bool or type(self.sum_head_attention) is not bool:
            raise ValueError("norm_value_input and sum_head_attention must be booleans")
        if self.sum_head_attention and type(self) is not GPTBasePooledQKEmbedNoEmbedNormConfig:
            raise ValueError("sum_head_attention is supported only by pooled embedding attention")
        expected = target_layers_for_depth(self.n_layer)
        if self.pooled_qk_target_layers is not None and tuple(self.pooled_qk_target_layers) != expected:
            raise ValueError(f"Target layers must match the fixed depth mapping: {expected}")
        self.pooled_qk_target_layers = expected  # Serialized with checkpoint config.
        if self.attention_chunk_size < 1:
            raise ValueError("attention_chunk_size must be positive")
        if self.n_kv_head != self.n_head:
            raise ValueError("Pooled QK embedding attention currently requires equal Q/K head counts")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("Head dimension must be integral and even for RoPE")
        enabled = [f.name for f in fields(base.GPTBaseConfig) if getattr(self, f.name) is True]
        if enabled:
            raise ValueError(f"This architecture requires all BoV flags disabled: {enabled}")


class PooledQKEmbeddingAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = config.attention_chunk_size
        self.sum_head_attention = config.sum_head_attention
        self.c_q = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = base.Linear(config.n_embd, config.n_embd, bias=False)

    def _chunk(self, q, k, embeddings, query_positions, key_positions):
        scores = (q @ k.transpose(-2, -1)).float() * self.head_dim**-0.5
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = distance >= 0
        if self.window_size[0] >= 0:
            allowed = allowed & (distance <= self.window_size[0])
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        # Reduce AFTER each head's softmax; retain the full embedding width.
        pooled = (probabilities.sum(dim=1) if self.sum_head_attention
                  else probabilities.mean(dim=1)).to(embeddings.dtype)
        return pooled @ embeddings

    def forward(self, x, x0, ve, cos_sin, kv_cache):
        batch, length, width = x.shape
        q = self.c_q(x).view(batch, length, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, length, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = base.norm(base.apply_rotary_emb(q, cos, sin)) * 1.2
        k = base.norm(base.apply_rotary_emb(k, cos, sin)) * 1.2
        offset = 0
        embeddings = x0
        if kv_cache is not None:
            if torch.is_grad_enabled():
                raise ValueError("KV cache is inference-only; use no_grad or inference_mode")
            offset = kv_cache.get_pos()
            end = offset + length
            if end > kv_cache.max_seq_len:
                raise ValueError("Sequence exceeds KV cache capacity")
            key_cache, embedding_cache = kv_cache.get_layer_cache(self.layer_idx)
            key_cache[:, offset:end].copy_(k)
            # Reuse V-cache storage for prepared embedding values, without projection.
            # This preserves KVCache.prefill/reset and beam/batch broadcast behavior.
            embedding_cache[:, offset:end].copy_(x0.view(batch, length, self.n_head, self.head_dim))
            k = key_cache[:, :end]
            embeddings = embedding_cache[:, :end].reshape(batch, end, width)

        q, k = q.transpose(1, 2), k.transpose(1, 2)
        chunks = []
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            # Skip wholly masked key regions; mask within each rectangular chunk.
            key_start = max(0, offset + start - self.window_size[0]) if self.window_size[0] >= 0 else 0
            key_end = offset + end
            args = (q[:, :, start:end], k[:, :, key_start:key_end],
                    embeddings[:, key_start:key_end],
                    torch.arange(offset + start, offset + end, device=x.device),
                    torch.arange(key_start, key_end, device=x.device))
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self._chunk, *args, use_reentrant=False))
            else:
                chunks.append(self._chunk(*args))
        output = self.c_proj(torch.cat(chunks, dim=1))
        if kv_cache is not None and self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(length)
        return output, None


class GPTBasePooledQKEmbedNoEmbedNorm(GPTBaseNoEmbedNorm):
    def _prepare_embedding_values(self, embeddings):
        # Once per forward, across full hidden width before any head splitting.
        return base.norm(embeddings) if self.config.norm_value_input else embeddings

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            self.transformer.h[i].attn = PooledQKEmbeddingAttention(config, i, self.window_sizes[i])

    @torch.no_grad()
    def init_weights(self):
        # Same distributions as GPTBase; target layers have no c_v to initialize.
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        bound = math.sqrt(3 / self.config.n_embd)
        for block in self.transformer.h:
            for name in ("c_q", "c_k", "c_v"):
                if hasattr(block.attn, name):
                    nn.init.uniform_(getattr(block.attn, name).weight, -bound, bound)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * bound, 0.4 * bound)
            nn.init.zeros_(block.mlp.c_proj.weight)
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.config.n_embd // self.config.n_head)
        if base.COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=base.COMPUTE_DTYPE)

    def _pooling_flops_per_token(self):
        # Same rectangular effective-context convention as GPTBase. H-1 additions
        # and one division per pooled probability; backward divides H gradients.
        t = self.config.sequence_len
        reduction_work = self.config.n_head - 1 if self.config.sum_head_attention else self.config.n_head
        return sum(reduction_work * (t if self.window_sizes[i][0] < 0 else min(t, self.window_sizes[i][0]))
                   for i in self.config.pooled_qk_target_layers)

    def estimate_flops(self):
        """Training FLOPs/token: baseline convention, absent W_v, plus head reduction.

        Architecture estimate, NOT measured kernel work: excludes checkpoint
        recomputation, rectangular chunk padding, softmax and normalization, as
        the baseline excludes its attention-kernel overhead. Used by the trainer
        to derive the fixed-FLOP horizon. Full-width mixing costs the same as the
        baseline's concatenated head mixing. Mean adds 2*H*T; sum adds (H-1)*T.
        """
        # A sum broadcasts its gradient; a mean additionally scales gradients.
        factor = 1 if self.config.sum_head_attention else 2
        return super().estimate_flops() + factor * self._pooling_flops_per_token()

    def estimate_forward_flops(self):
        """Forward FLOPs/token using the same architecture-level convention."""
        return super().estimate_flops() // 3 + self._pooling_flops_per_token()
