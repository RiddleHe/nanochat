"""
GPT with per-Q/K/V Attention Residuals (AttnRes-QKV).

This is a small probe variant of GPTAttnRes (see gpt_attn_res.py). In the base
AttnRes model, a single learned pseudo-query attends over all prior sublayer
outputs to build ONE input h, from which q, k, and v are all projected:

    h   = attn_res(query, prior_outputs)
    q, k, v = c_q(norm(h)), c_k(norm(h)), c_v(norm(h))

Here we instead give q, k, and v EACH their own pseudo-query, so each can
independently choose which previous layers (depths) to read from:

    h_q = attn_res(query_q, prior_outputs)
    h_k = attn_res(query_k, prior_outputs)
    h_v = attn_res(query_v, prior_outputs)
    q, k, v = c_q(norm(h_q)), c_k(norm(h_k)), c_v(norm(h_v))

The learned depth-attention weights of query_q / query_k / query_v (visualized
as a heatmap) reveal what information each of q, k, v needs at each layer:
concentration on recent layers => accumulated context; concentration on the
embedding / early layers => original token identity.

Everything else (smear, value embeddings, sliding window, the pre-MLP and final
AttnRes queries, optimizer, init) is inherited unchanged from GPTAttnRes.
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn
from nanochat.model.gpt import norm, apply_rotary_emb, CausalSelfAttention
from nanochat.model.gpt_attn_res import GPTAttnResConfig, GPTAttnRes, Block


class GPTAttnResQKVConfig(GPTAttnResConfig):
    """Identical fields to GPTAttnResConfig; separate type for the registry."""
    pass


class CausalSelfAttentionQKV(CausalSelfAttention):
    """
    Same as CausalSelfAttention, but accepts three separate inputs (x_q, x_k, x_v)
    for the q/k/v projections instead of a single shared input x. Parameter set is
    identical to the parent (c_q, c_k, c_v, c_proj, ve_gate), so init and the
    optimizer treat it exactly like the base attention.
    """

    def forward(self, x_q, x_k, x_v, ve, cos_sin, window_size, kv_cache):
        B, T, C = x_q.size()

        # Project from three separate (already-normed) inputs.
        q = self.c_q(x_q).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x_k).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x_v).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): gate is part of the value path, so it reads x_v.
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x_v[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Rotary + QK norm (identical to the base attention).
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class GPTAttnResQKV(GPTAttnRes):
    """
    AttnRes with separate q/k/v depth-queries. Reuses all of GPTAttnRes
    (_attn_res, init_weights, setup_optimizer, estimate_flops, generate, ...);
    only the attention module (QKV-split) and the query count differ.
    """

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)

        # Swap each block's attention for the QKV-split version. Parameter shapes
        # are identical, so init_weights() (inherited) initializes it unchanged.
        for layer_idx in range(config.n_layer):
            self.transformer.h[layer_idx].attn = CausalSelfAttentionQKV(config, layer_idx)

        # Queries per layer: q, k, v before attention (3) + 1 before MLP, plus 1 final.
        self.n_queries = 4 * config.n_layer + 1
        n_queries_padded = ((self.n_queries + 7) // 8) * 8
        self.attn_res_queries = nn.Parameter(torch.zeros(n_queries_padded, config.n_embd))

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        import torch.nn.functional as F
        B, T = idx.size()

        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear (identical to GPTAttnRes).
        if kv_cache is None:
            assert T > 1
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # --- Per-Q/K/V Attention Residuals ---
        v_list = [x]  # v_0 = token embedding
        qi = 0

        for i, block in enumerate(self.transformer.h):
            # Separate depth-attention for q, k, v.
            h_q = self._attn_res(self.attn_res_queries[qi], v_list); qi += 1
            h_k = self._attn_res(self.attn_res_queries[qi], v_list); qi += 1
            h_v = self._attn_res(self.attn_res_queries[qi], v_list); qi += 1

            ve = self.value_embeds[str(i)](idx).to(h_q.dtype) if str(i) in self.value_embeds else None
            attn_out = block.attn(norm(h_q), norm(h_k), norm(h_v), ve, cos_sin, self.window_sizes[i], kv_cache)
            v_list.append(attn_out)

            # Shared depth-attention before the MLP (unchanged).
            h = self._attn_res(self.attn_res_queries[qi], v_list); qi += 1
            mlp_out = block.mlp(norm(h))
            v_list.append(mlp_out)

        # Final aggregation over all sublayer outputs.
        x = self._attn_res(self.attn_res_queries[qi], v_list)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits
