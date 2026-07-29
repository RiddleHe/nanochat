"""
GPT base model: vanilla GPT-2 style architecture (no value embeddings, no per-layer
resid/x0 lambdas), with the smear gate retained.

Notable features (vs nanochat.model.gpt):
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
- Smear gate (cheap bigram-like info from previous token's embedding)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTBaseConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"

    # --- Chunk deep-KV (research/chunk-deep-kv) --------------------------------
    # The sequence is cut into chunks of `chunk_size`. EARLY layers (the first
    # `chunk_kv_frac` of layers) get an extra gated attention branch whose K/V
    # are projected -- through the layer's own c_k/c_v, no new matrices -- from
    # states of STRICTLY EARLIER chunks. Two source choices:
    #   chunk_deep_kv: the final block's output   (already-processed content)
    #   chunk_same_kv: the same layer's input     (visibility-only control)
    # The control isolates "processed content" from "just seeing further".
    # Sources are detached (Transformer-XL style truncation), so no gradient
    # flows across chunks and training parallelism is unchanged. Per-head gates
    # init at 0 => the model starts bit-exactly at the baseline computation.
    chunk_deep_kv: bool = False
    chunk_same_kv: bool = False
    chunk_size: int = 256
    chunk_kv_frac: float = 0.3334  # first frac of layers get the branch
    # v1 (chunk_recurrent=False) obtains the sources from a no-grad pass-1
    # forward over the whole sequence, costing ~+41% training FLOPs.
    # v2 (chunk_recurrent=True) is a single-pass chunk loop: chunks run
    # sequentially through ALL layers, normal layers attend to
    # [previous chunks' same-layer KV (with grad) + own chunk] via FA3's
    # bottom-right causal alignment => mathematically equivalent to standard
    # training, and the branch reads the cached (detached) final states of
    # already-finished chunks. No pass-1 tax; residual cost ~+8% is the branch.
    chunk_recurrent: bool = False


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Chunk deep-KV: per-head gate (init 0) on the extra cross-chunk branch.
        # Only the first `chunk_kv_frac` of layers carry one.
        self.is_chunk_early = ((config.chunk_deep_kv or config.chunk_same_kv)
                               and layer_idx < max(1, int(config.n_layer * config.chunk_kv_frac)))
        if self.is_chunk_early:
            self.chunk_gate = nn.Parameter(torch.empty(self.n_head))

    def _chunk_branch(self, q, bk, bv, y):
        """Attend q over branch keys/values (bk, bv) and merge via the per-head
        gate. Shapes: q (B,T,H,D); bk/bv (B,S,Hkv,D); y (B,T,H,D)."""
        group = self.n_head // self.n_kv_head
        bk_t, bv_t = bk.transpose(1, 2), bv.transpose(1, 2)
        if group > 1:
            bk_t = bk_t.repeat_interleave(group, dim=1)
            bv_t = bv_t.repeat_interleave(group, dim=1)
        y2 = F.scaled_dot_product_attention(q.transpose(1, 2), bk_t, bv_t)
        gate = self.chunk_gate.view(1, 1, self.n_head, 1).to(y.dtype)
        return y + gate * y2.transpose(1, 2)

    def forward(self, x, cos_sin, kv_cache, chunk_src=None, chunk_ctx=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if chunk_ctx is not None and kv_cache is None:
            # v2 chunk-recurrent: x is ONE chunk. Prepend previous chunks'
            # same-layer K/V (kept WITH grad => exact equivalence to standard
            # attention) and let FA3's bottom-right causal alignment give the
            # suffix queries the right mask. Then bank this chunk's K/V.
            pk, pv = chunk_ctx["kv"].get(self.layer_idx, (None, None))
            k_cat = k if pk is None else torch.cat([pk, k], dim=1)
            v_cat = v if pv is None else torch.cat([pv, v], dim=1)
            chunk_ctx["kv"][self.layer_idx] = (k_cat, v_cat)
            y = flash_attn.flash_attn_func(q, k_cat, v_cat, causal=True, window_size=self.window_size)
            # Early-layer branch over finished chunks' final states (detached).
            # Sources are strictly earlier chunks, so every query in this chunk
            # may see all of them -> no mask needed.
            bk, bv = chunk_ctx["branch"].get(self.layer_idx, (None, None))
            if self.is_chunk_early and bk is not None:
                y = self._chunk_branch(q, bk, bv, y)
        elif kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=self.window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=self.window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # v1 chunk deep-KV branch: extra attention over K/V projected from the
        # detached pass-1 states of strictly earlier chunks. Queries reuse q
        # (already rotary'd, QK-normed and scaled).
        if chunk_src is not None and self.is_chunk_early and kv_cache is None:
            Cs = self.config.chunk_size
            if T > Cs:
                cos, sin = cos_sin
                bk = self.c_k(chunk_src).view(B, T, self.n_kv_head, self.head_dim)
                bv = self.c_v(chunk_src).view(B, T, self.n_kv_head, self.head_dim)
                bk = norm(apply_rotary_emb(bk, cos, sin)) * 1.2
                group = self.n_head // self.n_kv_head
                bk_t, bv_t = bk.transpose(1, 2), bv.transpose(1, 2)
                if group > 1:
                    bk_t = bk_t.repeat_interleave(group, dim=1)
                    bv_t = bv_t.repeat_interleave(group, dim=1)
                pos = torch.arange(T, device=x.device)
                vis = (pos[None, :] // Cs) < (pos[:, None] // Cs)  # (Tq, Tk) strictly-earlier-chunk
                q_t = q.transpose(1, 2)                            # (B, H, T, D)
                # chunk-0 queries see nothing (all-masked rows -> NaN), so slice them off
                y2 = F.scaled_dot_product_attention(
                    q_t[:, :, Cs:, :], bk_t, bv_t, attn_mask=vis[Cs:, :])
                y_extra = torch.zeros_like(q_t)
                y_extra[:, :, Cs:, :] = y2
                gate = self.chunk_gate.view(1, 1, self.n_head, 1).to(y.dtype)
                y = y + gate * y_extra.transpose(1, 2)

        # Re-assemble the heads and project back to residual stream.
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx, window_size)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, chunk_src=None, chunk_ctx=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, chunk_src=chunk_src, chunk_ctx=chunk_ctx)
        x = x + self.mlp(norm(x))
        return x


class GPTBase(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                Block(config, layer_idx, self.window_sizes[layer_idx])
                for layer_idx in range(config.n_layer)
            ]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # chunk deep-KV gates start at 0 => the branch contributes nothing and
            # the model is bit-exactly the baseline at step 0.
            if hasattr(block.attn, 'chunk_gate'):
                torch.nn.init.zeros_(block.attn.chunk_gate)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and smear scalars
        nparams_exclude = (self.transformer.wte.weight.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        # Chunk deep-KV honest accounting. Three extra costs:
        #   (a) v1 only: the no-grad pass-1 forward = 1/3 of a plain fwd+bwd;
        #   (b) the branch's extra c_k/c_v calls at early layers (6/param);
        #   (c) the branch attention itself: each query sees ~(t-chunk)/2 keys
        #       on average, at each early layer (12*h*q*eff for fwd+bwd).
        # Measured ratios vs baseline at d12 shape: v1 1.414x, v2 1.081x.
        if self.config.chunk_deep_kv or self.config.chunk_same_kv:
            plain = num_flops_per_token
            n_early = sum(1 for b in self.transformer.h
                          if getattr(b.attn, 'is_chunk_early', False))
            proj_numel = sum(b.attn.c_k.weight.numel() + b.attn.c_v.weight.numel()
                             for b in self.transformer.h
                             if getattr(b.attn, 'is_chunk_early', False))
            eff = max(0, (t - self.config.chunk_size) / 2)
            tax = 0 if self.config.chunk_recurrent else plain / 3
            num_flops_per_token = plain + tax + 6 * proj_numel + n_early * 12 * h * q * eff
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.smear_gate.weight.numel() + self.smear_lambda.numel()
        # chunk gates already live inside transformer.h, so they are counted in
        # transformer_matrices above; nothing to add here (kept as a note so the
        # assert below stays understandable).
        total = wte + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = [p for p in self.transformer.h.parameters() if p.dim() >= 2]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        smear_params = [self.smear_gate.weight, self.smear_lambda]
        # chunk deep-KV per-head gates (init 0): 1-D, so they are NOT picked up by
        # the dim>=2 matrix filter. Give them the scalar/AdamW treatment with no
        # weight decay so a useful branch is free to grow away from 0.
        chunk_gate_params = [b.attn.chunk_gate for b in self.transformer.h
                             if hasattr(b.attn, 'chunk_gate')]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params)
                                                + len(lm_head_params) + len(smear_params)
                                                + len(chunk_gate_params))

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if chunk_gate_params:
            param_groups.append(dict(kind='adamw', params=chunk_gate_params,
                                     lr=scalar_lr * dmodel_lr_scale, betas=(0.8, 0.95),
                                     eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
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

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer.
        cfg = self.config
        chunk_early = [i for i, b in enumerate(self.transformer.h)
                       if getattr(b.attn, 'is_chunk_early', False)]

        if cfg.chunk_recurrent and kv_cache is None:
            # ---- v2: single-pass chunk-recurrent trunk -----------------------
            # Chunks run sequentially through ALL layers. Normal layers keep full
            # gradient flow over previous chunks' same-layer KV, so this is
            # mathematically equivalent to the standard trunk (verified bit-exact
            # at gate=0 in tests/test_chunk_deep_kv.py). Only the branch sources
            # are detached. Rotary is sliced per chunk to preserve absolute pos.
            Cs = cfg.chunk_size
            cos_full, sin_full = cos_sin
            ctx = {"kv": {}, "branch": {}}
            outs = []
            for c0 in range(0, T, Cs):
                c1 = min(c0 + Cs, T)
                cs_chunk = (cos_full[:, c0:c1], sin_full[:, c0:c1])
                xc = x[:, c0:c1]
                for block in self.transformer.h:
                    xc = block(xc, cs_chunk, kv_cache, chunk_ctx=ctx)
                outs.append(xc)
                # This chunk is finished: bank its (detached) final states as
                # branch K/V for every early layer, rotary'd at absolute positions.
                if cfg.chunk_deep_kv and chunk_early:
                    srcn = norm(xc.detach())
                    for i in chunk_early:
                        attn = self.transformer.h[i].attn
                        bk = attn.c_k(srcn).view(B, c1 - c0, attn.n_kv_head, attn.head_dim)
                        bv = attn.c_v(srcn).view(B, c1 - c0, attn.n_kv_head, attn.head_dim)
                        bk = norm(apply_rotary_emb(bk, cs_chunk[0], cs_chunk[1])) * 1.2
                        pbk, pbv = ctx["branch"].get(i, (None, None))
                        ctx["branch"][i] = (bk if pbk is None else torch.cat([pbk, bk], dim=1),
                                            bv if pbv is None else torch.cat([pbv, bv], dim=1))
            x = torch.cat(outs, dim=1)
        elif (cfg.chunk_deep_kv or cfg.chunk_same_kv) and kv_cache is None:
            # ---- v1: two-pass trunk (pass-1 is no-grad; costs ~+41% FLOPs) ----
            def run_trunk(xin, chunk_srcs=None, collect=None):
                collected = {}
                xc = xin
                for i, block in enumerate(self.transformer.h):
                    if collect is not None and i in collect:
                        collected[i] = xc
                    src = None if chunk_srcs is None else chunk_srcs.get(i)
                    xc = block(xc, cos_sin, kv_cache, chunk_src=src)
                return xc, collected

            with torch.no_grad():
                xf, inputs = run_trunk(x.detach(),
                                       collect=set(chunk_early) if cfg.chunk_same_kv else None)
            if cfg.chunk_deep_kv:
                # deep sources: the final block's output (already-processed content)
                src_all = norm(xf)
                chunk_srcs = {i: src_all for i in chunk_early}
            else:
                # same-layer control: each early layer reads its OWN input depth
                chunk_srcs = {i: norm(inputs[i]) for i in chunk_early}
            x, _ = run_trunk(x, chunk_srcs)
        else:
            for block in self.transformer.h:
                x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
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
