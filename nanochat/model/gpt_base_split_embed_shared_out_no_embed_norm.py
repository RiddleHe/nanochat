"""Split raw Emb across heads, scale head results, sum, then apply shared W_O.

At target layers: y_i = W_O sum_h s_h * sum_j p_h(i,j) Emb[j,h].
Each head owns a disjoint D/H-channel embedding chunk; W_O is D-by-(D/H)
and is shared across heads within a layer. There is no W_V or head average.
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
class GPTBaseSplitEmbedSharedOutNoEmbedNormConfig(GPTBasePooledQKEmbedNoEmbedNormConfig):
    """Keep the fixed target mapping; attention_chunk_size is unused with FA3/SDPA."""


class SplitEmbeddingSharedOutputAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = base.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = base.Linear(self.head_dim, config.n_embd, bias=False)
        self.head_scale = nn.Parameter(torch.empty(self.n_head, self.head_dim,
                                                   dtype=torch.float32))

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
            # Cache raw chunks; scaling happens after attention in both paths.
            mixed = base.flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=self.window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(length)
        # Sum (not mean) in FP32, then do ONE small output projection.
        combined = (mixed.float() * self.head_scale[None, None]).sum(dim=2)
        return self.c_proj(combined.to(mixed.dtype)), None


class GPTBaseSplitEmbedSharedOutNoEmbedNorm(GPTBasePooledQKEmbedNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            self.transformer.h[i].attn = SplitEmbeddingSharedOutputAttention(
                config, i, self.window_sizes[i])

    @torch.no_grad()
    def init_weights(self):
        # Inherited initialization handles rectangular c_proj and missing c_v.
        super().init_weights()
        for i in self.config.pooled_qk_target_layers:
            self.transformer.h[i].attn.head_scale.fill_(1.0)

    def num_scaling_params(self):
        counts = super().num_scaling_params()
        scales = sum(self.transformer.h[i].attn.head_scale.numel()
                     for i in self.config.pooled_qk_target_layers)
        counts["transformer_matrices"] -= scales
        counts["scalars"] += scales
        return counts

    def setup_optimizer(self, **kwargs):
        ddp, _, _, world_size = base.get_dist_info()
        if ddp:
            for i in self.config.pooled_qk_target_layers:
                scale = self.transformer.h[i].attn.head_scale
                if scale.numel() >= 1024 and scale.shape[0] % world_size:
                    raise ValueError("Head-scale AdamW sharding requires n_head divisible by world_size")
        return super().setup_optimizer(**kwargs)

    def _extra_flops(self, training):
        cfg = self.config
        d = cfg.n_embd // cfg.n_head
        # Scale: D multiplies forward; backward D input multiplies and ~2D
        # multiply/reduce work for scale gradients. Head sum: (H-1)*d forward
        # additions, with a gradient broadcast (no arithmetic) backward.
        per_target = (4 if training else 1) * cfg.n_embd + (cfg.n_head - 1) * d
        return len(cfg.pooled_qk_target_layers) * per_target

    def estimate_flops(self):
        """Training FLOPs/token under the shared architecture convention.

        Base counts actual smaller matrices, excludes head_scale, and correctly
        counts D total value channels. No pooled-probability or full-width-per-
        head correction applies. Excludes softmax/norms, kernel overhead,
        optimizer, evaluation and checkpoint recomputation.
        """
        return base.GPTBase.estimate_flops(self) + self._extra_flops(training=True)

    def estimate_forward_flops(self):
        return base.GPTBase.estimate_flops(self) // 3 + self._extra_flops(training=False)
