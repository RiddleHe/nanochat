"""Mix disjoint raw embedding chunks, then project with head-specific W_O blocks.

At target layers: y_i = sum_h W_O[:, h*d:(h+1)*d] @ sum_j p_h(i,j) Emb[j,h].
One square output matrix holds the column blocks. No W_V or channel scales.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.model import gpt_base as base
from nanochat.model.gpt_base_pooled_qk_embed_no_embed_norm import (
    GPTBasePooledQKEmbedNoEmbedNormConfig,
    GPTBasePooledQKEmbedNoEmbedNorm,
)


@dataclass
class GPTBaseSplitEmbedHeadOutNoEmbedNormConfig(GPTBasePooledQKEmbedNoEmbedNormConfig):
    """Keep the fixed target mapping; attention_chunk_size is unused with FA3/SDPA."""


class SplitEmbeddingHeadOutputAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = base.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, x0, ve, cos_sin, kv_cache):
        batch, length, _ = x.shape
        q = self.c_q(x).view(batch, length, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, length, self.n_head, self.head_dim)
        q = base.norm(base.apply_rotary_emb(q, *cos_sin)) * 1.2
        k = base.norm(base.apply_rotary_emb(k, *cos_sin)) * 1.2
        # A reshape assigns contiguous, disjoint channels to heads. Gradients
        # remain live to the raw embedding table. Never expand values to H*D.
        v = x0.reshape(batch, length, self.n_head, self.head_dim)
        if kv_cache is None:
            mixed = base.flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=self.window_size)
        else:
            if torch.is_grad_enabled():
                raise ValueError("KV cache is inference-only; use no_grad or inference_mode")
            if kv_cache.get_pos() + length > kv_cache.max_seq_len:
                raise ValueError("Sequence exceeds KV cache capacity")
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            # Cache raw chunks, exactly as used in the non-cached path.
            mixed = base.flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=self.window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(length)
        # Concatenation + square projection equals the sum of per-head maps.
        return self.c_proj(mixed.reshape(batch, length, -1)), None


class GPTBaseSplitEmbedHeadOutNoEmbedNorm(GPTBasePooledQKEmbedNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            self.transformer.h[i].attn = SplitEmbeddingHeadOutputAttention(
                config, i, self.window_sizes[i])

    def estimate_flops(self):
        """Actual matrix sizes and ordinary attention work need no correction."""
        return base.GPTBase.estimate_flops(self)

    def estimate_forward_flops(self):
        return base.GPTBase.estimate_flops(self) // 3
