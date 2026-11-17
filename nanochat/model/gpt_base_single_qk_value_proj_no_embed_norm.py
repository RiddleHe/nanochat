"""Single-head attention over projected embeddings, with a learned output scalar.

At targets: a_scale * W_O * Attention(Q, K, W_V * prepared_Emb).
W_V is d-by-D and W_O is D-by-d; ordinary layers retain all H heads.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.model import gpt_base as base
from nanochat.model.gpt_base_single_qk_embed_no_embed_norm import (
    GPTBaseSingleQKEmbedNoEmbedNormConfig,
    GPTBaseSingleQKEmbedNoEmbedNorm,
)


@dataclass
class GPTBaseSingleQKValueProjNoEmbedNormConfig(GPTBaseSingleQKEmbedNoEmbedNormConfig):
    """norm_value_input defaults to False; the normalized recipe enables it."""


class SingleQKProjectedEmbeddingAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = 1
        self.head_dim = config.n_embd // config.n_head
        self.c_q = base.Linear(config.n_embd, self.head_dim, bias=False)
        self.c_k = base.Linear(config.n_embd, self.head_dim, bias=False)
        self.c_v = base.Linear(config.n_embd, self.head_dim, bias=False)
        self.c_proj = base.Linear(self.head_dim, config.n_embd, bias=False)
        self.a_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x, x0, ve, cos_sin, kv_cache):
        batch, length, _ = x.shape
        q = self.c_q(x).view(batch, length, 1, self.head_dim)
        k = self.c_k(x).view(batch, length, 1, self.head_dim)
        v = self.c_v(x0).view(batch, length, 1, self.head_dim)
        q = base.norm(base.apply_rotary_emb(q, *cos_sin)) * 1.2
        k = base.norm(base.apply_rotary_emb(k, *cos_sin)) * 1.2
        if kv_cache is None:
            mixed = base.flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=self.window_size)
        else:
            if torch.is_grad_enabled():
                raise ValueError("KV cache is inference-only; use no_grad or inference_mode")
            if kv_cache.get_pos() + length > kv_cache.max_seq_len:
                raise ValueError("Sequence exceeds KV cache capacity")
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            # The shared cache allocates H heads. Only the first slot is live
            # here, for both keys and projected values; allocation is unchanged.
            mixed = base.flash_attn.flash_attn_with_kvcache(
                q, k_cache[:, :, :1], v_cache[:, :, :1], k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=self.window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(length)
        output = self.c_proj(mixed.reshape(batch, length, self.head_dim))
        return (output.float() * self.a_scale).to(output.dtype), None


class GPTBaseSingleQKValueProjNoEmbedNorm(GPTBaseSingleQKEmbedNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            self.transformer.h[i].attn = SingleQKProjectedEmbeddingAttention(
                config, i, self.window_sizes[i])

    def _attention_flops_saved(self):
        # Base already counts actual rectangular Q/K/V/O matrices. Both score
        # products AND value mixing now use d channels instead of D.
        return 2 * self._qk_flops_saved()

    def estimate_flops(self):
        """Architecture training FLOPs/token; excludes norms/softmax/replay."""
        scale_work = 4 * self.config.n_embd * len(self.config.pooled_qk_target_layers)
        return base.GPTBase.estimate_flops(self) - 3 * self._attention_flops_saved() + scale_work

    def estimate_forward_flops(self):
        scale_work = self.config.n_embd * len(self.config.pooled_qk_target_layers)
        return base.GPTBase.estimate_flops(self) // 3 - self._attention_flops_saved() + scale_work
