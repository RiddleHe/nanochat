"""
GPT with Gated Full Attention Residuals.
Builds on AttnRes (gpt_attn_res.py) by adding a sigmoid bottleneck gate
after each depth-wise attention residual aggregation:

    h = _attn_res(query, layer_outputs)
    h = h * sigmoid(W_up(W_down(h)))

where W_down: d -> d//4, W_up: d//4 -> d.

Motivation: the AttnRes softmax over layers sums to 1, creating the same
"attention sink" pressure as sequence-level softmax attention. The gate
adds input-dependent sparsity and non-linearity, following the findings of
"Gated Attention for LLMs" (Qiu et al., 2025).

W_up is zero-initialized so gates start at sigmoid(0) = 0.5 (uniform scaling).
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn

from nanochat.gpt import norm, Linear, has_ve, apply_rotary_emb, CausalSelfAttention, MLP


@dataclass
class GPTGatedAttnResConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


class AttnResGate(nn.Module):
    """Bottleneck sigmoid gate: y * sigmoid(W_up(W_down(y)))."""
    def __init__(self, d, r=4):
        super().__init__()
        self.down = Linear(d, d // r, bias=False)
        self.up = Linear(d // r, d, bias=False)

    def forward(self, y):
        return y * torch.sigmoid(self.up(self.down(y)))


class Block(nn.Module):
    """Weight container for attention + MLP. Forward is called by the model directly."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)


class GPTGatedAttnRes(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # AttnRes pseudo-queries: 2 per transformer layer (before attn, before mlp) + 1 for final output.
        # Padded to multiple of 8 so distributed optimizer can shard across up to 8 GPUs.
        self.n_queries = 2 * config.n_layer + 1
        n_queries_padded = ((self.n_queries + 7) // 8) * 8
        self.attn_res_queries = nn.Parameter(torch.zeros(n_queries_padded, config.n_embd))

        # Gated AttnRes: one bottleneck sigmoid gate per AttnRes call
        self.attn_res_gates = nn.ModuleList([AttnResGate(config.n_embd) for _ in range(self.n_queries)])

        # Smear
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Backout
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _attn_res(self, query, layer_outputs, gate):
        """
        Compute gated attention residual: softmax-weighted sum over prior layer outputs,
        then apply sigmoid bottleneck gate.
        """
        V = torch.stack(layer_outputs, dim=0)  # (N, B, T, D)
        K = norm(V)
        logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
        weights = logits.softmax(dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)
        h = gate(h)
        return h

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

        # AttnRes pseudo-queries: zero init
        torch.nn.init.zeros_(self.attn_res_queries)

        # AttnRes gates: zero-init the up projection so gate starts at sigmoid(0) = 0.5
        for gate in self.attn_res_gates:
            torch.nn.init.uniform_(gate.down.weight, -s, s)
            torch.nn.init.zeros_(gate.up.weight)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

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
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
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
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.attn_res_queries.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        gate_matrices = sum(p.numel() for p in self.attn_res_gates.parameters())
        scalars = (self.attn_res_queries.numel() +
                   self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + gate_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'gate_matrices': gate_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        gate_params = list(self.attn_res_gates.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        attn_res_params = [self.attn_res_queries]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == (len(matrix_params) + len(gate_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(attn_res_params) + len(smear_params))

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=attn_res_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Gate matrices and transformer matrices both go to Muon, grouped by shape
        all_matrix_params = matrix_params + gate_params
        for shape in sorted({p.shape for p in all_matrix_params}):
            group_params = [p for p in all_matrix_params if p.shape == shape]
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
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear
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

        # --- Gated Full Attention Residuals ---
        v_list = [x]  # v_0 = token embedding
        backout_layer = n_layer // 2
        h_backout = None
        qi = 0  # query/gate index

        for i, block in enumerate(self.transformer.h):
            # Gated AttnRes before attention sublayer
            h = self._attn_res(self.attn_res_queries[qi], v_list, self.attn_res_gates[qi])
            qi += 1

            ve = self.value_embeds[str(i)](idx).to(h.dtype) if str(i) in self.value_embeds else None
            attn_out = block.attn(norm(h), ve, cos_sin, self.window_sizes[i], kv_cache)
            v_list.append(attn_out)

            # Gated AttnRes before MLP sublayer
            h = self._attn_res(self.attn_res_queries[qi], v_list, self.attn_res_gates[qi])
            qi += 1

            mlp_out = block.mlp(norm(h))
            v_list.append(mlp_out)

            if i == backout_layer:
                h_backout = h

        # Final gated AttnRes aggregation
        x = self._attn_res(self.attn_res_queries[qi], v_list, self.attn_res_gates[qi])

        # Backout
        if h_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * h_backout
        x = norm(x)

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
