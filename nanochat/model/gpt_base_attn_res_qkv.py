"""
GPTBase with full attention residuals and independent Q/K/V depth routing.

At layer l, the source bank is [x0, attn_0, mlp_0, ..., attn_{l-1}, mlp_{l-1}].
Three learned D-vectors select separate mixtures for Wq, Wk, Wv. Routing uses
RMS-normalized source keys and raw source values, with softmax over depth.
A fourth query routes into the MLP after appending the attention output;
one final query aggregates the completed bank for the language-model head.
These are raw sublayer outputs, not cumulative residual states.

Keeps GPTBase's smear, RoPE, QK norm/scaling, windows, and ReLU-squared MLP.
No value embeddings, value gates, residual lambdas, or BoV mixing flags.
Query layout: [Q_0, K_0, V_0, MLP_0, ..., Q_{L-1}, K_{L-1}, V_{L-1},
MLP_{L-1}, final, padding...]. Zero initialization gives uniform routing.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn


# Reuse only the plain GPTBase building blocks.
from nanochat.model.gpt_base import norm, Linear, apply_rotary_emb, MLP


@dataclass
class GPTBaseAttnResQKVConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


class CausalSelfAttention(nn.Module):
    """GPTBase attention with separate, already-normalized projection inputs."""

    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension"
        assert 0 < self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, xq, xk, xv, cos_sin, kv_cache=None):
        B, T, _ = xq.shape
        q = self.c_q(xq).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(xk).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(xv).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=self.window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=self.window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        return self.c_proj(y.contiguous().view(B, T, self.n_embd))


class Block(nn.Module):
    """Weight container; the model owns source-bank routing and execution order."""

    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, window_size)
        self.mlp = MLP(config)


class GPTBaseAttnResQKV(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        assert config.n_layer > 0 and config.sequence_len > 0
        assert config.n_embd >= 24, "Smear reads the first 24 embedding channels"
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, self.window_sizes[layer_idx]) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # AttnRes pseudo-queries: Q, K, V, MLP per layer + one final output query.
        # Initialized to zero so initial attention weights are uniform (critical for training stability).
        # Padded to multiple of 8 so distributed optimizer can shard across up to 8 GPUs.
        self.n_queries = 4 * config.n_layer + 1
        n_queries_padded = ((self.n_queries + 7) // 8) * 8
        self.attn_res_queries = nn.Parameter(torch.zeros(n_queries_padded, config.n_embd))

        # Smear: mix previous token's embedding into current token (cheap bigram info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))


        # Rotary embeddings (over-computed 10X, same as base GPT)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _attn_res(queries, layer_outputs):
        """Route R queries [R,D] over N sources [B,T,D], returning [R,B,T,D].

        Stack and normalize the bank once for all three Q/K/V queries. The
        scores are content-dependent at each token; no token-axis mixing
        occurs here. As in the original AttnRes, no sqrt(D) scaling is used.
        """
        values = torch.stack(layer_outputs, dim=0)
        keys = norm(values)
        scores = torch.einsum('rd,nbtd->rnbt', queries.to(keys.dtype), keys)
        weights = scores.float().softmax(dim=1).to(values.dtype)
        return torch.einsum('rnbt,nbtd->rbtd', weights, values)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # AttnRes pseudo-queries: zero init (uniform initial attention weights)
        torch.nn.init.zeros_(self.attn_res_queries)

        # Explicit initialization is essential after meta -> to_empty.
        self.smear_gate.reset_parameters()
        torch.nn.init.zeros_(self.smear_lambda)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert pattern and all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = (self.transformer.wte.weight.numel() +
                          self.attn_res_queries.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        # Per query/source: score dot product + weighted sum = 4*D forward
        # FLOPs, approximately 12*D including backward. For layer l (0-based),
        # Q/K/V each read 2*l+1 sources; MLP reads 2*l+2; final reads 2*L+1.
        # Sum = 4*L**2 + 3*L + 1. Count routing explicitly, excluding padded
        # query rows. Like the base estimate, omit norm/softmax elementwise ops.
        L, D = self.config.n_layer, self.config.n_embd
        routing_flops = 12 * D * (4 * L * L + 3 * L + 1)
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops + routing_flops
        return num_flops_per_token

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.attn_res_queries.numel() +
                   self.smear_gate.weight.numel() + self.smear_lambda.numel())
        total = wte + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': 0,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        attn_res_params = [self.attn_res_queries]
        smear_params = [self.smear_gate.weight, self.smear_lambda]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(attn_res_params) + len(smear_params))

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=attn_res_params, lr=0.001, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        n_layer = self.config.n_layer

        # Rotary embeddings
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        assert T > 0 and T0 + T <= self.cos.size(1), "Rotary cache exhausted"
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Preserve smear across prefill, single-token decode, and chunked decode.
        previous = None if kv_cache is None else kv_cache.prev_embedding
        if kv_cache is not None:
            kv_cache.prev_embedding = x[:, -1:, :]
        first = x[:, :1]
        if previous is not None:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(first[..., :24]))
            first = first + gate * previous
        if T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([first, x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x = first

        # The bank contains post-smear x0 and raw (not residual-added) outputs.
        sources = [x]
        for i, block in enumerate(self.transformer.h):
            queries = self.attn_res_queries[4*i:4*i+4]
            hq, hk, hv = self._attn_res(queries[:3], sources).unbind(0)
            attn_out = block.attn(norm(hq), norm(hk), norm(hv), cos_sin, kv_cache)
            sources.append(attn_out)

            hm = self._attn_res(queries[3:4], sources)[0]
            sources.append(block.mlp(norm(hm)))

        final_query = self.attn_res_queries[4*n_layer:4*n_layer+1]
        x = norm(self._attn_res(final_query, sources)[0])

        # Logits
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

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
