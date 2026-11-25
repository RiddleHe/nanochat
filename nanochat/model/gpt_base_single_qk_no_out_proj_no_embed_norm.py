"""Proposal 9: single-head attention over normalized Emb, without W_v or W_o.

At fixed target layers, y = a_scale * Attention(Q, K, RMSNorm(Emb)).
The initial residual remains raw. A zero-initialized scalar preserves the
baseline's zero initial attention contribution without an output matrix.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.model.gpt_base_single_qk_embed_no_embed_norm import (
    GPTBaseSingleQKEmbedNoEmbedNormConfig,
    GPTBaseSingleQKEmbedNoEmbedNorm,
)


@dataclass
class GPTBaseSingleQKNoOutProjNoEmbedNormConfig(GPTBaseSingleQKEmbedNoEmbedNormConfig):
    norm_value_input: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.norm_value_input:
            raise ValueError("Proposal 9 requires norm_value_input=True")


class GPTBaseSingleQKNoOutProjNoEmbedNorm(GPTBaseSingleQKEmbedNoEmbedNorm):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for i in config.pooled_qk_target_layers:
            # Retain the shared attention/cache path; Identity has no parameters
            # and performs no projection. No dormant W_o remains in checkpoints.
            self.transformer.h[i].attn.c_proj = nn.Identity()

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        for i in self.config.pooled_qk_target_layers:
            self.transformer.h[i].attn.a_scale.zero_()

    # Inherited FLOP accounting counts actual matrix sizes, so removing W_o
    # automatically saves 6*D^2 training FLOPs/token per target (2*D^2 forward).
    # It also retains single-head QK correction and 4*D scalar training work:
    # F_train = 6*P_mat + sum_ordinary(12*D*T_l)
    #           + sum_target(6*(d+D)*T_l + 4*D).
    # P_mat includes the LM head, excludes embedding lookups and scalars.
    # Norms, softmax, optimizer, evaluation and checkpoint replay are excluded.
