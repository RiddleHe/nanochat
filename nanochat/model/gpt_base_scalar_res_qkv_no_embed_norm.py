"""ScalarResQKVNoEmbedNorm: scalar sums over raw embeddings and sublayer outputs.

The bank is [Emb, attn_0, mlp_0, ...]. Q/K/V each have a vector of
2*l+1 coefficients; MLP has 2*l+2; final readout has 2*L+1. Coefficients
are shared across tokens/heads and positive: g(z) = z + sqrt(1 + z^2).
Logits start at zero, giving unit coefficients. A training-only recovery loss
penalizes logits below -2; positive logits are unpenalized.
Emb is the raw token embedding, with no initial embedding RMSNorm.
There is no depth attention or softmax. RMSNorm follows each complete sum.
Ordinary causal token attention is inherited from the AttnResQKV blocks.
Scalar optimizer settings match GPTBase's gamma_v (AdamW, high LR, no decay).
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


# Reuse the same token-attention and MLP building blocks as AttnResQKV.
from nanochat.model.gpt_base import norm, Linear
from nanochat.model.gpt_base_attn_res_qkv import Block


@dataclass
class GPTBaseScalarResQKVNoEmbedNormConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # Persist the storage layout independently of the runtime world size.
    scalar_res_pad_to: int = 4
    scalar_res_gate: str = "sqrt"
    scalar_res_recovery_weight: float = 0.01
    scalar_res_recovery_margin: float = 2.0


class GPTBaseScalarResQKVNoEmbedNorm(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        assert config.n_layer > 0 and config.sequence_len > 0
        if config.scalar_res_gate != "sqrt":
            raise ValueError("scalar_res_gate must be 'sqrt'")
        for name in ('scalar_res_recovery_weight', 'scalar_res_recovery_margin'):
            value = getattr(config, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, self.window_sizes[layer_idx]) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # One flat parameter gives compiled AdamW a single coefficient shape.
        # Receiver views are created in forward, never cached across device moves.
        sizes = []
        for layer in range(config.n_layer):
            sizes.extend([2 * layer + 1] * 3)
            sizes.append(2 * layer + 2)
        sizes.append(2 * config.n_layer + 1)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        self.scalar_res_offsets = tuple(offsets)
        self.scalar_res_active = offsets[-1]
        pad_to = config.scalar_res_pad_to
        if type(pad_to) is not int or pad_to <= 0:
            raise ValueError("scalar_res_pad_to must be a positive integer")
        storage_size = ((self.scalar_res_active + pad_to - 1) // pad_to) * pad_to
        self.scalar_res_logits_flat = nn.Parameter(torch.empty(storage_size, dtype=torch.float32))

        # Rotary embeddings (over-computed 10X, same as base GPT)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _receiver_logits(self, receiver):
        start = self.scalar_res_offsets[receiver]
        end = self.scalar_res_offsets[receiver + 1]
        return self.scalar_res_logits_flat[start:end]

    def _receiver_coefficients(self, receiver, coefficients=None):
        if coefficients is None:
            return self._sqrt_gate(self._receiver_logits(receiver))
        start = self.scalar_res_offsets[receiver]
        end = self.scalar_res_offsets[receiver + 1]
        return coefficients[start:end]

    @staticmethod
    def _sqrt_gate(logits):
        """FP32 positive gate; rationalize the negative tail to avoid cancellation."""
        z = logits.float()
        root = torch.sqrt(1.0 + z.square())
        # Clamp only the inactive reciprocal branch at positive z: otherwise
        # root - z can round to zero and poison torch.where's backward with NaNs.
        negative = (root - z.clamp_max(0.0)).reciprocal()
        return torch.where(z >= 0.0, root + z, negative)

    def _compute_scalar_recovery(self):
        """Mean over active raw logits; no padding or token-dependent activations."""
        z = self.scalar_res_logits_flat[:self.scalar_res_active].float()
        tail = F.relu(-z - self.config.scalar_res_recovery_margin)
        return self.config.scalar_res_recovery_weight * tail.square().mean()

    @staticmethod
    def _scalar_res(coefficients, layer_outputs):
        """Sum R coefficient vectors [R,N] over N raw sources [B,T,D]."""
        values = torch.stack(layer_outputs, dim=0)
        # FP32 master coefficients; activation dtype for the forward contraction.
        return torch.einsum('rn,nbtd->rbtd', coefficients.to(values.dtype), values)

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

        # g(0) = 1: unit coefficients recover ordinary residual sums.
        self.scalar_res_logits_flat.zero_()

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
                          self.scalar_res_logits_flat.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        # Weighted sum: ~2*D forward FLOPs per source/receiver pair,
        # ~6*D including backward. Q/K/V read 2*l+1; MLP reads 2*l+2;
        # final reads 2*L+1. Exclude coefficients from the matrix term above
        # to avoid double counting. Omit elementwise norms, gates, and recovery.
        L, D = self.config.n_layer, self.config.n_embd
        mixing_flops = 6 * D * (4 * L * L + 3 * L + 1)
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops + mixing_flops
        return num_flops_per_token

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.scalar_res_active
        scalar_padding = self.scalar_res_logits_flat.numel() - scalars
        total = wte + lm_head + transformer_matrices + scalars + scalar_padding
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': 0,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'scalar_padding': scalar_padding,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        scalar_params = [self.scalar_res_logits_flat]
        # DistMuonAdamW shards parameters with >=1024 entries along dimension 0.
        if ddp and self.scalar_res_logits_flat.numel() >= 1024:
            if self.scalar_res_logits_flat.numel() % world_size != 0:
                raise ValueError(
                    f"Flat scalar parameter length ({self.scalar_res_logits_flat.numel()}) "
                    f"must be divisible by world_size ({world_size}); choose a "
                    "compatible scalar_res_pad_to in the model config."
                )
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(scalar_params))

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=scalar_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
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
        # The bank contains raw token embeddings and raw sublayer outputs.
        sources = [x]
        # Compute all active gates once, then use views at the receiving sites.
        all_coefficients = self._sqrt_gate(self.scalar_res_logits_flat[:self.scalar_res_active])
        for i, block in enumerate(self.transformer.h):
            start = self.scalar_res_offsets[4*i]
            end = self.scalar_res_offsets[4*i+3]
            coefficients = all_coefficients[start:end].view(3, 2*i+1)
            hq, hk, hv = self._scalar_res(coefficients, sources).unbind(0)
            attn_out = block.attn(norm(hq), norm(hk), norm(hv), cos_sin, kv_cache)
            sources.append(attn_out)

            hm = self._scalar_res(self._receiver_coefficients(4*i+3, all_coefficients)[None, :], sources)[0]
            sources.append(block.mlp(norm(hm)))

        final_coefficients = self._receiver_coefficients(4*n_layer, all_coefficients)[None, :]
        x = norm(self._scalar_res(final_coefficients, sources)[0])

        # Logits
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if self.training and self.config.scalar_res_recovery_weight > 0.0:
                recovery = self._compute_scalar_recovery()
                if loss_reduction == 'none':
                    recovery = torch.where(targets.reshape(-1) != -1, recovery, 0.0)
                elif loss_reduction == 'sum':
                    recovery = recovery * (targets != -1).sum()
                # For mean CE, every valid token shares this same scalar penalty.
                loss = loss + recovery
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
