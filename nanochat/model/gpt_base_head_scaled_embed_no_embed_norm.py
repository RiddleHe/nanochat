"""Head-specific diagonal embedding maps at the fixed pooled-QK target layers.

Output = W_o mean_h(g_h * (softmax(Q_h K_h^T / sqrt(d)) @ Emb)).
Each g_h spans the FULL embedding width; there is no target-layer W_v.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.model import gpt_base as base
from nanochat.model.gpt_base_pooled_qk_embed_no_embed_norm import (
    GPTBasePooledQKEmbedNoEmbedNormConfig,
    GPTBasePooledQKEmbedNoEmbedNorm,
    PooledQKEmbeddingAttention,
)


@dataclass
class GPTBaseHeadScaledEmbedNoEmbedNormConfig(GPTBasePooledQKEmbedNoEmbedNormConfig):
    """Retain the fixed depth mapping, masking and chunk configuration."""


class HeadScaledEmbeddingAttention(PooledQKEmbeddingAttention):
    def __init__(self, config, layer_idx, window_size):
        super().__init__(config, layer_idx, window_size)
        self.head_scale = nn.Parameter(torch.empty(config.n_head, config.n_embd,
                                                   dtype=torch.float32))

    def _chunk(self, q, k, embeddings, query_positions, key_positions):
        scores = (q @ k.transpose(-2, -1)).float() * self.head_dim**-0.5
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = distance >= 0
        if self.window_size[0] >= 0:
            allowed = allowed & (distance <= self.window_size[0])
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        # [B,H,Q,D], not [B,H,Q,D/H]. Never pool probabilities before scaling.
        mixed = probabilities.to(embeddings.dtype) @ embeddings.unsqueeze(1)
        # Applying a token-independent diagonal map after mixing is equivalent
        # to scaling each token first. Keep scale multiplication/reduction FP32.
        return (mixed.float() * self.head_scale[None, :, None, :]).mean(1).to(embeddings.dtype)


class GPTBaseHeadScaledEmbedNoEmbedNorm(GPTBasePooledQKEmbedNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            self.transformer.h[i].attn = HeadScaledEmbeddingAttention(
                config, i, self.window_sizes[i])

    @torch.no_grad()
    def init_weights(self):
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
        """Extra work over GPTBase's attention term with the current matrices.

        Baseline AV uses D total value channels; here each of H heads uses D.
        Forward extra AV: 2*(H-1)*D*T. Forward scale and head mean: 2*H*D.
        Training extra AV: 6*(H-1)*D*T. Scale/mean forward+backward: 6*H*D,
        plus (H-1)*D to combine the shared embedding's per-head gradients.
        Reduction costs use the usual leading-order per-token approximation
        (N rather than N-1 for coefficient gradients over batch/query tokens).
        """
        cfg = self.config
        total = 0
        for i in cfg.pooled_qk_target_layers:
            window = self.window_sizes[i][0]
            context = cfg.sequence_len if window < 0 else min(window, cfg.sequence_len)
            extra_av = (cfg.n_head - 1) * cfg.n_embd * context
            if training:
                total += 6 * extra_av + (7 * cfg.n_head - 1) * cfg.n_embd
            else:
                total += 2 * extra_av + 2 * cfg.n_head * cfg.n_embd
        return total

    def estimate_flops(self):
        """Architecture training FLOPs/token, including head scaling/mixing.

        GPTBase excludes head_scale from dense matrix parameters and sees the
        absent target W_v. Do not add the pooled model's probability-mean cost:
        this model averages vectors instead. As in that model, estimates omit
        checkpoint recomputation, masked chunk padding, softmax and norms.
        """
        return base.GPTBase.estimate_flops(self) + self._extra_flops(training=True)

    def estimate_forward_flops(self):
        """Forward-only FLOPs/token under the same effective-context convention."""
        return base.GPTBase.estimate_flops(self) // 3 + self._extra_flops(training=False)
