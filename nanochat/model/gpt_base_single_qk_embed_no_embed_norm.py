"""One Q/K head mixes raw Emb, with a learned scalar per target attention output."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nanochat.model import gpt_base as base
from nanochat.model.gpt_base_pooled_qk_embed_no_embed_norm import (
    GPTBasePooledQKEmbedNoEmbedNormConfig,
    GPTBasePooledQKEmbedNoEmbedNorm,
)


@dataclass
class GPTBaseSingleQKEmbedNoEmbedNormConfig(GPTBasePooledQKEmbedNoEmbedNormConfig):
    """Global n_head still describes ordinary layers and determines Q/K width."""


class SingleQKEmbeddingAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = 1
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = config.attention_chunk_size
        self.c_q = base.Linear(config.n_embd, self.head_dim, bias=False)
        self.c_k = base.Linear(config.n_embd, self.head_dim, bias=False)
        self.c_proj = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.a_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def _chunk(self, q, k, embeddings, query_positions, key_positions):
        scores = (q @ k.transpose(-2, -1)).float() * self.head_dim**-0.5
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = distance >= 0
        if self.window_size[0] >= 0:
            allowed = allowed & (distance <= self.window_size[0])
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        return probabilities.squeeze(1).to(embeddings.dtype) @ embeddings

    def forward(self, x, x0, ve, cos_sin, kv_cache):
        batch, length, width = x.shape
        q = self.c_q(x).view(batch, length, 1, self.head_dim)
        k = self.c_k(x).view(batch, length, 1, self.head_dim)
        q = base.norm(base.apply_rotary_emb(q, *cos_sin)) * 1.2
        k = base.norm(base.apply_rotary_emb(k, *cos_sin)) * 1.2
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
            # Existing uniform cache allocation: use only its first key head.
            # Raw embeddings still occupy all D value channels, as in pooled QK.
            key_cache[:, offset:end, :1].copy_(k)
            embedding_cache[:, offset:end].copy_(x0.reshape(
                batch, length, width // self.head_dim, self.head_dim))
            k = key_cache[:, :end, :1]
            embeddings = embedding_cache[:, :end].reshape(batch, end, width)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        chunks = []
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
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
        output = (output.float() * self.a_scale).to(output.dtype)
        if kv_cache is not None and self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(length)
        return output, None


class GPTBaseSingleQKEmbedNoEmbedNorm(GPTBasePooledQKEmbedNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            self.transformer.h[i].attn = SingleQKEmbeddingAttention(config, i, self.window_sizes[i])

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        for i in self.config.pooled_qk_target_layers:
            self.transformer.h[i].attn.a_scale.fill_(1.0)

    def num_scaling_params(self):
        counts = super().num_scaling_params()
        scalars = len(self.config.pooled_qk_target_layers)
        counts["transformer_matrices"] -= scalars
        counts["scalars"] += scalars
        return counts

    def _qk_flops_saved(self):
        # GPTBase already counts actual (smaller) Q/K matrices and absent W_V.
        # Its attention term assumes D QK channels; retain only d at targets.
        cfg = self.config
        d = cfg.n_embd // cfg.n_head
        return sum(2 * (cfg.n_embd - d) * (
            cfg.sequence_len if self.window_sizes[i][0] < 0
            else min(cfg.sequence_len, self.window_sizes[i][0]))
            for i in cfg.pooled_qk_target_layers)

    def estimate_flops(self):
        """Training FLOPs/token including the per-layer output scalar.

        Same architecture convention as pooled QK: excludes checkpoint replay,
        masked chunk padding, softmax/norms, optimizer and evaluation work.
        """
        # D forward multiplies, D input-gradient multiplies, and ~2D for
        # the scalar-gradient multiply/reduction (leading-order convention).
        scale_work = 4 * self.config.n_embd * len(self.config.pooled_qk_target_layers)
        return base.GPTBase.estimate_flops(self) - 3 * self._qk_flops_saved() + scale_work

    def estimate_forward_flops(self):
        scale_work = self.config.n_embd * len(self.config.pooled_qk_target_layers)
        return base.GPTBase.estimate_flops(self) // 3 - self._qk_flops_saved() + scale_work
