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
    # If True, blend the initial normalized embedding x0 back into the residual at the
    # last 1/3 of blocks: x = 0.5 * x + 0.5 * x0 (applied before each late block).
    add_init_res: bool = False
    # Mix x0 at one pre-norm input site with learnable (α, β) coeffs, leaving the
    # residual stream untouched. attn_only sets attn_in = norm(α_a x + β_a x0) and
    # leaves mlp_in = norm(x); mlp_only is the dual: attn_in = norm(x) and
    # mlp_in = norm(α_m (x + attn_out) + β_m x0). Used to localize which sub-layer's
    # x0 input is responsible for any pre-norm gain. Both require learn_init_coeffs
    # and are mutually exclusive with each other and with add_init_res (all target
    # pre-norm input sites). Compatible with v-writers (add_init_qkv, add_init_v,
    # add_init_res_v), which touch q/k/v rather than the pre-norm input.
    add_init_pre_norm_attn_only: bool = False
    add_init_pre_norm_mlp_only: bool = False
    # If True, capture v from layer 0's attention (after c_v projection) and mix it into
    # the last 1/3 of layers' v: v_i = 0.5 * v_i + 0.5 * v_layer0. Lightweight analogue
    # of gpt.py's value-residual without per-layer learned VE tables. Shared-mode only:
    # for the per-layer-c_v variant, see add_init_res_v.
    add_init_v: bool = False
    # If True, at the last 1/3 of layers, blend v with this SAME layer's projection of
    # norm(x0) through c_v: v = 0.5*v + 0.5*c_v(norm(x0)). Disentangles "Wv_layer0 is
    # special" from "x0 input is special" relative to add_init_v. Mutually exclusive
    # with add_init_v and add_init_qkv (all three write to v).
    add_init_res_v: bool = False
    # If True, detach x0 before the block loop so every x0-using path (add_init_res,
    # add_init_pre_norm_{attn,mlp}_only, add_init_qkv, add_init_res_v) consumes a detached x0.
    # Forward values are unchanged; only the gradient skip-connection back to the
    # embedding through these paths is killed. Used to isolate "is the win the backward
    # gradient skip to the embedding?" from forward semantic leakage. No effect on
    # add_init_v, which routes through layer-0's v rather than x0.
    detach_init_value: bool = False
    # If True, at the last 1/3 of layers, blend q, k, v with projections of x0 through
    # this SAME layer's c_q, c_k, c_v: q = 0.5*q + 0.5*c_q(norm(x0)), and same for k, v.
    # Apples-to-apples to add_init_res for "what does the qkv skip buy?" without touching
    # the residual stream that the MLP reads. Mutually exclusive with add_init_v and
    # add_init_res_v.
    add_init_qkv: bool = False
    # Same forward shape as add_init_qkv, but with a SINGLE (alpha, beta) pair shared
    # across q/k/v at each late layer instead of three independent pairs. Used to
    # disentangle "norm-then-add" (vs add_init_pre_norm_attn_only's add-then-norm) from
    # "independent per-projection coeffs" as explanations for why add_init_qkv beats
    # add_init_pre_norm_attn_only. Requires learn_init_coeffs (no fixed-coeff form).
    add_init_qkv_shared: bool = False
    # If True, at the last 1/3 of layers, compute v directly from x0 instead of x:
    # v = c_v(norm(x0)). Replaces (rather than blends with) the standard v = c_v(x)
    # path, isolating "what if v has no current-stream content at all?" from the
    # blended add_init_res_v variant. Mutually exclusive with add_init_v /
    # add_init_res_v / add_init_qkv (all four write to v).
    # When learn_init_coeffs=True, an additional per-late-layer learnable scalar
    # gamma_v (init 1.0) scales the result: v = gamma_v * c_v(norm(x0)). At step
    # 0 this matches the no-learn variant; the optimizer can push gamma_v above
    # 1.0 to amplify the x0 contribution, effectively breaking the softmax sum-
    # to-1 constraint on the attention-output magnitude.
    v_from_x0: bool = False
    # If True, at the last 1/3 of layers, apply Exclusive Self-Attention (XSA;
    # Zhai 2026, https://arxiv.org/abs/2603.09078): remove the component of the
    # attention output that lies along v via vector rejection, before c_proj:
    #   v̂_i = v_i / ||v_i||
    #   y_i ← y_i - (y_i · v̂_i) v̂_i
    # The result is guaranteed orthogonal to v_i. v_i still flows through the
    # residual stream, so the token's own value content is not lost — only the
    # parallel-to-v component of the attention path is removed, encouraging
    # attention to provide purely contextual (non-self) information. No learned
    # coeffs. Independent of the x0 / v_writer flags; modifies y, not v.
    v_exclude_self: bool = False
    # If True, at the last 1/3 of layers, scale the standard v = c_v(x) projection
    # by the per-late-layer learnable gamma_v (init 1.0). Apples-to-apples control
    # for v_from_x0 + learn_init_coeffs: same scaling structure, but the projection
    # input is x (current stream) rather than norm(x0) (initial embedding), so any
    # bpb gain is attributable to "having an unbounded coefficient on v" rather
    # than "x0 is the right input to project". Requires learn_init_coeffs (no
    # fixed-coeff form), mutually exclusive with the other v-writers.
    v_scale_learn: bool = False
    # If True, at the last 1/3 of layers, replace v entirely with the cached
    # layer-0 v (v1): v = ve. Like add_init_v in capturing layer-0's v, but like
    # v_from_x0 in doing identity replacement (no 0.5/0.5 blend with the current
    # layer's c_v(x)). When learn_init_coeffs=True, gamma_v (per-late-layer scalar,
    # init 1.0) scales the result: v = gamma_v * ve. Apples-to-apples ablation
    # against v_from_x0: same identity-replacement structure, but the source is
    # layer-0's v projection (Wv_0(x_0)) rather than current layer's c_v(norm(x0)).
    # Mutually exclusive with the other v-writers.
    v_from_v1: bool = False
    # If True, at the last 1/3 of layers, replace y = c_proj(y) with
    #   y = alpha_o * c_proj(y) + beta_o * c_proj(norm(x0))
    # i.e. route x0 through the SAME c_proj used for the attention output, with
    # per-late-layer learnable scalars (alpha_o init 1.0, beta_o init 0.0). The
    # x0 path is parallel to v_from_x0 / add_init_res_v but lives at the output
    # projection instead of v — it tests whether mixing in a c_proj(norm(x0))
    # contribution at the residual write site (rather than at v) explains the
    # x0-skip gain. Requires learn_init_coeffs (no fixed-coeff form). Independent
    # of v-writers (touches y/c_proj, not v) and of pre-norm writers.
    add_init_proj: bool = False
    # Switches the layer set that all gpt_base routing flags apply to. The set
    # is exposed on each Block/CausalSelfAttention as `is_target` (see
    # `_is_target_layer`). Default (False): "last 1/3" (for d12 → {8,9,10,11},
    # 4 of 12). True: gpt.py's `has_ve(layer_idx, n_layer)` pattern,
    # `layer_idx % 2 == (n_layer - 1) % 2` — alternating, last layer always
    # included (for d12 → {1,3,5,7,9,11}, 6 of 12). Affects ALL routing:
    # add_init_res, add_init_v, add_init_res_v, add_init_qkv{,_shared},
    # add_init_pre_norm_{attn,mlp}_only, v_from_x0, v_from_v1, v_from_value_emb,
    # add_init_value_emb{,_nanogpt}, add_init_proj, v_scale_learn, v_exclude_self,
    # plus VE-table layer set + dead-c_v exclusion + extra-matmul accounting.
    on_every_two_layers: bool = False
    # If True, at the last 1/3 of layers, apply gpt.py's value-embed scheme
    # verbatim: v = v + gate * VE_l[idx] with a per-late-layer input-dependent
    # gate `gate = 3 * sigmoid(ve_gate(x[..., :12]))` (per kv-head, range (0, 3)).
    # Coefficient on v is fixed at 1.0 (no learnable alpha). Init follows gpt.py:
    # VE table uniform[-s, s] with s = sqrt(3)/sqrt(n_embd) (NOT the aligned init
    # used by v_from_value_emb / add_init_value_emb), ve_gate.weight uniform[0,
    # 0.02] so the initial gate ≈ 1.5. ve_gate.weight is shape (n_kv_head, 12)
    # → routed to Muon naturally via the dim>=2 matrix_params filter, while the
    # VE table goes to its own AdamW group at embedding_lr * 0.5 (same as for
    # v_from_value_emb / add_init_value_emb). Mutually exclusive with the other
    # v-writers. FLOPs/token effectively equals gpt_base (ve_gate is ~3.5K
    # FLOPs/tok, 5e-6 of total — included implicitly via 6*nparams).
    add_init_value_emb_nanogpt: bool = False
    # If True, at the last 1/3 of layers, blend the per-late-layer learned
    # value embedding table into v: v = 0.5*v + 0.5*VE_l[idx] (no-learn) or
    # v = alpha_v*v + beta_v*VE_l[idx] (learn_init_coeffs, init 1.0/0.0).
    # Apples-to-apples baseline for v_from_value_emb: same VE table, same
    # aligned init (VE_l[t] = c_v_l_init(norm(wte_init[t]))), but ADD to c_v(x)
    # rather than REPLACE it — mirrors add_init_res_v's relationship to
    # v_from_x0. At step 0 with smear_lambda=0, v matches gpt_base (no-learn:
    # 0.5*c_v(x) + 0.5*c_v(norm(wte)) ≈ c_v(x); learn: beta=0 so v=c_v(x)
    # exactly). Late-layer c_v stays live (still computed for the blend), so
    # NO dead-cv FLOPs adjustment — FLOPs/token equals gpt_base; only the VE
    # table is excluded from the matmul-FLOPs term (it's a lookup, not a
    # matmul). Mutually exclusive with the other v-writers.
    add_init_value_emb: bool = False
    # If True, at the last 1/3 of layers, replace v with a per-late-layer
    # learned value embedding table indexed by token id: v = VE_l[idx]. Mirrors
    # gpt.py's value_embeds (per-layer Embedding(vocab, n_kv_head*head_dim)) but
    # as identity replacement rather than gpt.py's additive blend with c_v(x).
    # Strictly more expressive than v_from_x0: same "v depends only on token
    # ids" hypothesis without the rank-n_embd bottleneck. Init: each row is set
    # to c_v_l_init(norm(wte_init[t])), so at step 0 (smear_lambda=0) v at late
    # layers exactly matches v_from_x0 — clean apples-to-apples for the rank
    # lift. Late-layer c_v is dead (forward skipped, no grad path to loss) and
    # excluded from FLOPs / optimizer, mirroring v_from_v1. When
    # learn_init_coeffs is set, gamma_v scales the result: v = gamma_v * VE_l[idx]
    # (per-late-layer scalar, init 1.0). Does NOT consume x0. Mutually exclusive
    # with the other v-writers.
    v_from_value_emb: bool = False
    # If True, replace the hardcoded 0.5/0.5 blends in add_init_res / add_init_v /
    # add_init_res_v / add_init_qkv with per-late-layer learnable scalars (alpha, beta),
    # init to (1.0, 0.0). At step 0 the model is identical to a no-skip vanilla baseline;
    # the optimizer learns how much x0 (or layer-0 v) contribution each late layer wants.
    # alpha multiplies the "main" branch, beta multiplies the x0/ve "skip" branch.
    # Required by add_init_pre_norm_{attn,mlp}_only (no fixed-coeff form is supported).
    learn_init_coeffs: bool = False
    # If True (and v_from_v1=True), add a per-late-layer per-head scaling vector
    # lambda_v of shape (n_kv_head,), init 1.0, that scales the cached layer-0 v
    # head-wise: v = lambda_v.view(1,1,-1,1) * ve. Apples-to-apples vs. v_from_v1_learn
    # (single scalar gamma_v): per-head lambda lets each KV-head pick its own gain
    # on the v_0 signal — recovers some of the per-layer expressivity that
    # v_from_x0 has via independent c_v's, at minimal extra params (n_kv_head per
    # late layer). Mutually exclusive with learn_init_coeffs; requires v_from_v1.
    learn_init_scaling_vec: bool = False


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok


def _is_target_layer(layer_idx, config):
    """Single source of truth for which layers receive gpt_base routing flags
    (all add_init_*, v_from_*, value-emb variants, etc.). Default: layer in the
    last 1/3 of the trunk. With on_every_two_layers=True: gpt.py's has_ve pattern
    — alternating, last layer always included."""
    if config.on_every_two_layers:
        return layer_idx % 2 == (config.n_layer - 1) % 2
    return layer_idx >= (2 * config.n_layer) // 3

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
        self.is_target = _is_target_layer(layer_idx, config)
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
        # Learnable blend coeffs (init 1.0/0.0 in init_weights) for late-layer x0 skips,
        # gated on cfg.learn_init_coeffs and which v-writing flag is active.
        v_writer = config.add_init_v or config.add_init_res_v or config.add_init_qkv or config.add_init_value_emb
        if self.is_target and config.learn_init_coeffs and v_writer:
            self.alpha_v = nn.Parameter(torch.empty(1))
            self.beta_v = nn.Parameter(torch.empty(1))
        if self.is_target and config.learn_init_coeffs and config.add_init_qkv:
            self.alpha_q = nn.Parameter(torch.empty(1))
            self.beta_q = nn.Parameter(torch.empty(1))
            self.alpha_k = nn.Parameter(torch.empty(1))
            self.beta_k = nn.Parameter(torch.empty(1))
        # Single shared (alpha, beta) pair for add_init_qkv_shared (one pair per
        # late layer, used for q, k, v all together).
        if self.is_target and config.add_init_qkv_shared:
            self.alpha_qkv = nn.Parameter(torch.empty(1))
            self.beta_qkv = nn.Parameter(torch.empty(1))
        # gamma_v: per-late-layer scalar (init 1.0) that multiplies the v projection.
        # Shared by v_from_x0 (scales c_v(norm(x0))), v_scale_learn (scales c_v(x)),
        # v_from_v1 (scales the cached layer-0 v), and v_from_value_emb (scales the
        # per-layer VE table lookup) — flags are mutually exclusive, so only one
        # path activates the same param.
        if self.is_target and config.learn_init_coeffs and (config.v_from_x0 or config.v_scale_learn or config.v_from_v1 or config.v_from_value_emb):
            self.gamma_v = nn.Parameter(torch.empty(1))
        # add_init_proj: per-late-layer (alpha_o, beta_o) for blending the attention
        # output projection with c_proj(norm(x0)). Gated on learn_init_coeffs
        # (required by add_init_proj — no fixed-coeff form).
        if self.is_target and config.learn_init_coeffs and config.add_init_proj:
            self.alpha_o = nn.Parameter(torch.empty(1))
            self.beta_o = nn.Parameter(torch.empty(1))
        # lambda_v: per-late-layer per-head scale (init 1.0) on the cached
        # layer-0 v under v_from_v1.
        if self.is_target and config.learn_init_scaling_vec and config.v_from_v1:
            self.lambda_v = nn.Parameter(torch.empty(self.n_kv_head))
        # ve_gate (add_init_value_emb_nanogpt): input-dependent gate matching
        # gpt.py's ResFormer-style value-embed gating exactly. Linear(12,
        # n_kv_head, bias=False). Treated as a regular matrix param (Muon group,
        # via the dim>=2 filter). Hardcoded ve_gate_channels=12 to match gpt.py.
        if self.is_target and config.add_init_value_emb_nanogpt:
            self.ve_gate_channels = 12
            self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def forward(self, x, x0, ve, cos_sin, kv_cache):
        B, T, C = x.size()
        cfg = self.config

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        # Skip c_v(x) at late layers under v_from_v1 / v_from_x0 / v_from_value_emb:
        # its output would be overwritten by the elif chain below (with ve for
        # v_from_v1, with c_v(x0n) for v_from_x0, with VE_l[idx] passed in via ve
        # for v_from_value_emb), so the forward call is wasted. For v_from_v1 and
        # v_from_value_emb c_v.weight also has no gradient path to the loss — kept
        # out of the optimizer (handled in setup_optimizer). For v_from_x0 c_v.weight
        # still trains via the c_v(x0n) call. Layer 0 always computes c_v normally
        # (is_target=False).
        if self.is_target and (cfg.v_from_v1 or cfg.v_from_x0 or cfg.v_from_value_emb):
            v = None
            v_init = None
        else:
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
            v_init = v  # captured for shared-mode add_init_v (always returned, ignored if unused)

        # Compute normalized x0 once if any late-layer x0->qkv path needs it.
        # x0 is already detached at GPTBase.forward entry if cfg.detach_init_value.
        need_x0n = self.is_target and (cfg.add_init_qkv or cfg.add_init_res_v or cfg.v_from_x0 or cfg.add_init_qkv_shared or cfg.add_init_proj)
        x0n = norm(x0) if need_x0n else None

        # add_init_qkv: blend each of q, k, v with this layer's projection of norm(x0).
        if self.is_target and cfg.add_init_qkv:
            cqx = self.c_q(x0n).view(B, T, self.n_head, self.head_dim)
            ckx = self.c_k(x0n).view(B, T, self.n_kv_head, self.head_dim)
            cvx = self.c_v(x0n).view(B, T, self.n_kv_head, self.head_dim)
            if cfg.learn_init_coeffs:
                # Cast scalar params to activation dtype to keep q/k/v in bf16 for FA3.
                aq, bq = self.alpha_q.to(q.dtype), self.beta_q.to(q.dtype)
                ak, bk = self.alpha_k.to(k.dtype), self.beta_k.to(k.dtype)
                av, bv = self.alpha_v.to(v.dtype), self.beta_v.to(v.dtype)
                q = aq * q + bq * cqx
                k = ak * k + bk * ckx
                v = av * v + bv * cvx
            else:
                q = 0.5 * q + 0.5 * cqx
                k = 0.5 * k + 0.5 * ckx
                v = 0.5 * v + 0.5 * cvx

        # add_init_qkv_shared: same as add_init_qkv but with one (alpha, beta) pair
        # shared across q, k, v. Tests whether per-projection coefficient independence
        # is the load-bearing piece behind add_init_qkv > add_init_pre_norm_attn_only.
        elif self.is_target and cfg.add_init_qkv_shared:
            cqx = self.c_q(x0n).view(B, T, self.n_head, self.head_dim)
            ckx = self.c_k(x0n).view(B, T, self.n_kv_head, self.head_dim)
            cvx = self.c_v(x0n).view(B, T, self.n_kv_head, self.head_dim)
            a, b = self.alpha_qkv.to(q.dtype), self.beta_qkv.to(q.dtype)
            q = a * q + b * cqx
            k = a * k + b * ckx
            v = a * v + b * cvx

        # add_init_v: blend layer-0's captured v into v (shared mode).
        elif self.is_target and cfg.add_init_v:
            if cfg.learn_init_coeffs:
                av, bv = self.alpha_v.to(v.dtype), self.beta_v.to(v.dtype)
                v = av * v + bv * ve
            else:
                v = 0.5 * v + 0.5 * ve

        # add_init_value_emb: blend the per-late-layer VE table lookup into v.
        # `ve` here is VE_l[idx] (shape (B, T, n_kv_head*head_dim)) prepared by
        # GPTBase.forward. c_v(x) stays as the main path (not skipped).
        elif self.is_target and cfg.add_init_value_emb:
            ve_v = ve.view(B, T, self.n_kv_head, self.head_dim)
            if cfg.learn_init_coeffs:
                av, bv = self.alpha_v.to(v.dtype), self.beta_v.to(v.dtype)
                v = av * v + bv * ve_v
            else:
                v = 0.5 * v + 0.5 * ve_v

        # add_init_value_emb_nanogpt: gpt.py's value-embed scheme verbatim.
        # `ve` is VE_l[idx] from GPTBase.forward; gate is an input-dependent
        # per-head scalar in (0, 3). v keeps its full c_v(x) magnitude (coef 1).
        elif self.is_target and cfg.add_init_value_emb_nanogpt:
            ve_v = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * ve_v

        # v_from_v1: replace v entirely with the cached layer-0 v. Optional scalings
        # (mutually exclusive): gamma_v (single scalar, learn_init_coeffs) or
        # lambda_v (per-head vector, learn_init_scaling_vec), both init 1.0.
        elif self.is_target and cfg.v_from_v1:
            v = ve
            if cfg.learn_init_coeffs:
                v = self.gamma_v.to(v.dtype) * v
            elif cfg.learn_init_scaling_vec:
                v = self.lambda_v.view(1, 1, -1, 1).to(v.dtype) * v

        # add_init_res_v: blend this layer's own c_v(norm(x0)) (self-wv mode).
        elif self.is_target and cfg.add_init_res_v:
            cvx = self.c_v(x0n).view(B, T, self.n_kv_head, self.head_dim)
            if cfg.learn_init_coeffs:
                av, bv = self.alpha_v.to(v.dtype), self.beta_v.to(v.dtype)
                v = av * v + bv * cvx
            else:
                v = 0.5 * v + 0.5 * cvx

        # v_from_x0: replace v entirely with this layer's c_v(norm(x0)). When
        # learn_init_coeffs is set, gamma_v scales the result (init 1.0).
        elif self.is_target and cfg.v_from_x0:
            v = self.c_v(x0n).view(B, T, self.n_kv_head, self.head_dim)
            if cfg.learn_init_coeffs:
                v = self.gamma_v.to(v.dtype) * v

        # v_from_value_emb: replace v entirely with the per-late-layer learned
        # value embedding table looked up by token id. `ve` here is the table
        # lookup result (shape (B, T, n_kv_head*head_dim)) prepared by
        # GPTBase.forward. When learn_init_coeffs is set, gamma_v scales the
        # result (init 1.0).
        elif self.is_target and cfg.v_from_value_emb:
            v = ve.view(B, T, self.n_kv_head, self.head_dim)
            if cfg.learn_init_coeffs:
                v = self.gamma_v.to(v.dtype) * v

        # v_scale_learn: control variant — scale the standard c_v(x) by gamma_v.
        elif self.is_target and cfg.v_scale_learn:
            v = self.gamma_v.to(v.dtype) * v

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
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

        # XSA (cfg.v_exclude_self): vector-reject y onto v, i.e. remove the
        # component of y along v before c_proj. With GQA, broadcast v across
        # query-head groups so each query head rejects onto its own KV-group's v.
        if self.is_target and cfg.v_exclude_self:
            group = self.n_head // self.n_kv_head
            v_q = v if group == 1 else v.repeat_interleave(group, dim=2)
            v_hat = F.normalize(v_q, dim=-1)
            y = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat

        # Re-assemble the heads and project back to residual stream.
        # add_init_proj (late layers): y = alpha_o * c_proj(y) + beta_o * c_proj(norm(x0)),
        # routing x0 through the SAME output projection used for attention output.
        y = y.contiguous().view(B, T, -1)
        if self.is_target and cfg.add_init_proj:
            proj_y = self.c_proj(y)
            proj_x0 = self.c_proj(x0n)
            a_o, b_o = self.alpha_o.to(proj_y.dtype), self.beta_o.to(proj_y.dtype)
            y = a_o * proj_y + b_o * proj_x0
        else:
            y = self.c_proj(y)
        return y, v_init


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
        self.is_target = _is_target_layer(layer_idx, config)
        self.attn = CausalSelfAttention(config, layer_idx, window_size)
        self.mlp = MLP(config)
        # Learnable blend coeffs (init 1.0/0.0 in init_weights) for add_init_res.
        if self.is_target and config.learn_init_coeffs and config.add_init_res:
            self.alpha_res = nn.Parameter(torch.empty(1))
            self.beta_res = nn.Parameter(torch.empty(1))
        # Learnable blend coeffs for add_init_pre_norm_{attn,mlp}_only: a single
        # (alpha, beta) pair at the one pre-norm input mixing site that flag turns on.
        if self.is_target and config.add_init_pre_norm_attn_only:
            self.alpha_attn_in = nn.Parameter(torch.empty(1))
            self.beta_attn_in = nn.Parameter(torch.empty(1))
        if self.is_target and config.add_init_pre_norm_mlp_only:
            self.alpha_mlp_in = nn.Parameter(torch.empty(1))
            self.beta_mlp_in = nn.Parameter(torch.empty(1))

    def forward(self, x, x0, ve, cos_sin, kv_cache):
        cfg = self.config
        if self.is_target and cfg.add_init_res:
            if cfg.learn_init_coeffs:
                # Cast scalar params to activation dtype: master weights stay fp32 for
                # optimizer precision, but the multiply must run in bf16 to keep FA3 happy.
                a, b = self.alpha_res.to(x.dtype), self.beta_res.to(x.dtype)
                x = a * x + b * x0
            else:
                x = 0.5 * x + 0.5 * x0
        if self.is_target and cfg.add_init_pre_norm_attn_only:
            a_a, b_a = self.alpha_attn_in.to(x.dtype), self.beta_attn_in.to(x.dtype)
            attn_in = norm(a_a * x + b_a * x0)
        else:
            attn_in = norm(x)
        attn_out, v_init = self.attn(attn_in, x0, ve, cos_sin, kv_cache)
        x = x + attn_out
        if self.is_target and cfg.add_init_pre_norm_mlp_only:
            a_m, b_m = self.alpha_mlp_in.to(x.dtype), self.beta_mlp_in.to(x.dtype)
            mlp_in = norm(a_m * x + b_m * x0)
        else:
            mlp_in = norm(x)
        x = x + self.mlp(mlp_in)
        return x, v_init


class GPTBase(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        x0_users = [config.add_init_res,
                    config.add_init_pre_norm_attn_only, config.add_init_pre_norm_mlp_only,
                    config.add_init_qkv, config.add_init_qkv_shared,
                    config.add_init_res_v, config.v_from_x0, config.add_init_proj]
        assert not (config.detach_init_value and not any(x0_users)), \
            "detach_init_value requires at least one x0-using flag enabled"
        v_writers = [config.add_init_qkv, config.add_init_qkv_shared,
                     config.add_init_v, config.add_init_res_v,
                     config.v_from_x0, config.v_from_v1, config.v_scale_learn,
                     config.v_from_value_emb, config.add_init_value_emb,
                     config.add_init_value_emb_nanogpt]
        assert sum(v_writers) <= 1, \
            "add_init_qkv, add_init_qkv_shared, add_init_v, add_init_res_v, v_from_x0, v_from_v1, v_scale_learn, v_from_value_emb, add_init_value_emb, add_init_value_emb_nanogpt all write to v — at most one"
        pre_norm_writers = [config.add_init_res,
                            config.add_init_pre_norm_attn_only,
                            config.add_init_pre_norm_mlp_only]
        assert sum(pre_norm_writers) <= 1, \
            "add_init_res / add_init_pre_norm_{attn,mlp}_only all target pre-norm input — at most one"
        coeff_users = [config.add_init_res,
                       config.add_init_pre_norm_attn_only, config.add_init_pre_norm_mlp_only,
                       config.add_init_v, config.add_init_res_v, config.add_init_qkv,
                       config.add_init_qkv_shared,
                       config.v_from_x0, config.v_from_v1, config.v_scale_learn,
                       config.v_from_value_emb, config.add_init_value_emb,
                       config.add_init_proj]
        assert not (config.learn_init_coeffs and not any(coeff_users)), \
            "learn_init_coeffs requires at least one of add_init_res/pre_norm_{attn,mlp}_only/v/res_v/qkv{,_shared}/v_from_x0/v_from_v1/v_scale_learn/v_from_value_emb/add_init_proj"
        pre_norm_any = (config.add_init_pre_norm_attn_only
                        or config.add_init_pre_norm_mlp_only)
        assert not (pre_norm_any and not config.learn_init_coeffs), \
            "add_init_pre_norm_{attn,mlp}_only requires learn_init_coeffs (no fixed-coeff form)"
        assert not (config.add_init_qkv_shared and not config.learn_init_coeffs), \
            "add_init_qkv_shared requires learn_init_coeffs (no fixed-coeff form)"
        assert not (config.v_scale_learn and not config.learn_init_coeffs), \
            "v_scale_learn requires learn_init_coeffs (no fixed-coeff form — at gamma_v=1 this is gpt_base)"
        assert not (config.add_init_proj and not config.learn_init_coeffs), \
            "add_init_proj requires learn_init_coeffs (no fixed-coeff form — at alpha_o=1,beta_o=0 this is gpt_base)"
        assert not (config.learn_init_coeffs and config.learn_init_scaling_vec), \
            "learn_init_coeffs and learn_init_scaling_vec are mutually exclusive (both scale v under v_from_v1)"
        assert not (config.learn_init_scaling_vec and not config.v_from_v1), \
            "learn_init_scaling_vec requires v_from_v1 (the only path that consumes lambda_v)"
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
        # Per-late-layer learned value embedding tables for v_from_value_emb (replace),
        # add_init_value_emb (aligned-init blend), and add_init_value_emb_nanogpt
        # (gpt.py-style uniform-init + input-dependent gate). Mirrors gpt.py's
        # value_embeds (ModuleDict keyed by layer_idx as str). Only late layers get
        # a table; early layers compute v normally. Gated on the flags so old
        # gpt_base checkpoints (without these keys) still load strict=True.
        if config.v_from_value_emb or config.add_init_value_emb or config.add_init_value_emb_nanogpt:
            head_dim = config.n_embd // config.n_head
            kv_dim = config.n_kv_head * head_dim
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer) if _is_target_layer(i, config)
            })
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
            # Learnable blend coeffs: alpha (main) -> 1.0, beta (skip) -> 0.0.
            # At step 0, the model is identical to a no-skip vanilla baseline.
            if hasattr(block, 'alpha_res'):
                block.alpha_res.fill_(1.0); block.beta_res.fill_(0.0)
            if hasattr(block, 'alpha_attn_in'):
                block.alpha_attn_in.fill_(1.0); block.beta_attn_in.fill_(0.0)
            if hasattr(block, 'alpha_mlp_in'):
                block.alpha_mlp_in.fill_(1.0); block.beta_mlp_in.fill_(0.0)
            attn = block.attn
            if hasattr(attn, 'alpha_v'):
                attn.alpha_v.fill_(1.0); attn.beta_v.fill_(0.0)
            if hasattr(attn, 'alpha_q'):
                attn.alpha_q.fill_(1.0); attn.beta_q.fill_(0.0)
                attn.alpha_k.fill_(1.0); attn.beta_k.fill_(0.0)
            if hasattr(attn, 'alpha_qkv'):
                attn.alpha_qkv.fill_(1.0); attn.beta_qkv.fill_(0.0)
            # gamma_v (v_from_x0 / v_scale_learn + learn_init_coeffs): init 1.0 —
            # at step 0 the model matches the corresponding no-learn variant.
            if hasattr(attn, 'gamma_v'):
                attn.gamma_v.fill_(1.0)
            # add_init_proj: alpha_o (main, c_proj(y)) -> 1.0, beta_o (skip,
            # c_proj(norm(x0))) -> 0.0. At step 0 the output projection is identical
            # to the standard y = c_proj(y).
            if hasattr(attn, 'alpha_o'):
                attn.alpha_o.fill_(1.0); attn.beta_o.fill_(0.0)
            # lambda_v (v_from_v1 + learn_init_scaling_vec): per-head scale init 1.0;
            # at step 0 matches v_from_v1 (no-learn).
            if hasattr(attn, 'lambda_v'):
                attn.lambda_v.fill_(1.0)

        # Aligned init for the VE tables (v_from_value_emb / add_init_value_emb):
        # VE_l[t] = c_v_l(norm(wte[t])). With smear_lambda=0 at init, v at late
        # layers matches v_from_x0 at step 0 for v_from_value_emb, and the
        # add_init_value_emb blend (0.5/0.5 or beta=0) reduces to gpt_base.
        # Computed in fp32 here; cast to COMPUTE_DTYPE below alongside wte.
        if self.config.v_from_value_emb or self.config.add_init_value_emb:
            wte_normed = F.rms_norm(self.transformer.wte.weight.float(), (n_embd,))
            for blk in self.transformer.h:
                if not blk.is_target:
                    continue
                ve_init = F.linear(wte_normed, blk.attn.c_v.weight.float())
                self.value_embeds[str(blk.layer_idx)].weight.copy_(ve_init)

        # gpt.py-style init for add_init_value_emb_nanogpt: VE table uniform[-s, s]
        # (same scale as c_v); ve_gate.weight uniform[0, 0.02] so initial gate
        # 3*sigmoid(~0+) ≈ 1.5 (live from step 0 so VE rows get gradient flow).
        if self.config.add_init_value_emb_nanogpt:
            for blk in self.transformer.h:
                if not blk.is_target:
                    continue
                torch.nn.init.uniform_(self.value_embeds[str(blk.layer_idx)].weight, -s, s)
                torch.nn.init.uniform_(blk.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            if self.config.v_from_value_emb or self.config.add_init_value_emb or self.config.add_init_value_emb_nanogpt:
                for ve in self.value_embeds.values():
                    ve.to(dtype=COMPUTE_DTYPE)

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
        # Exclude non-matmul params: embeddings, smear scalars, learnable blend coeffs
        coeff_numel = sum(
            getattr(m, n).numel()
            for m in [b for b in self.transformer.h] + [b.attn for b in self.transformer.h]
            for n in ('alpha_res', 'beta_res',
                      'alpha_attn_in', 'beta_attn_in', 'alpha_mlp_in', 'beta_mlp_in',
                      'alpha_q', 'beta_q', 'alpha_k', 'beta_k', 'alpha_v', 'beta_v',
                      'alpha_qkv', 'beta_qkv', 'gamma_v', 'alpha_o', 'beta_o',
                      'lambda_v')
            if hasattr(m, n)
        )
        # Dead c_v under v_from_v1 / v_from_value_emb: at late layers, c_v's forward
        # call is skipped and its output isn't on the loss path — neither forward
        # nor backward FLOPs are spent on it. Exclude from accounting so target_flops
        # auto num_iterations bumps up (~+1.9% for d12), matching the actual compute
        # saved.
        dead_cv_numel = 0
        if self.config.v_from_v1 or self.config.v_from_value_emb:
            for blk in self.transformer.h:
                if blk.is_target:
                    dead_cv_numel += blk.attn.c_v.weight.numel()
        # Value embedding tables (v_from_value_emb / add_init_value_emb /
        # add_init_value_emb_nanogpt): embedding lookups, not matmuls — exclude
        # from the 6*nparams matmul-FLOPs term, like wte. Note: only
        # v_from_value_emb marks c_v as dead; add_init_value_emb* keep c_v live
        # (used for the v=c_v(x)+... path), so dead_cv_numel stays 0 there and
        # FLOPs/token equals gpt_base. The ve_gate matmul under nanogpt is left
        # implicit in the 6*nparams term (~3.5K FLOPs/tok, 5e-6 of total).
        value_embeds_numel = 0
        if self.config.v_from_value_emb or self.config.add_init_value_emb or self.config.add_init_value_emb_nanogpt:
            value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() +
                          value_embeds_numel +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() +
                          coeff_numel + dead_cv_numel)

        # Extra (live) forward calls of c_q/c_k/c_v/c_proj at late layers — i.e. the
        # weight matrix appears in TWO matmul nodes per forward whose outputs both
        # reach the loss. Each extra call adds another 6 FLOPs/param/token (one
        # forward + one dW + one dx). (add_init_v does NOT belong here: its layer-0
        # c_v reuse goes through ONE forward matmul; autograd accumulates downstream
        # gradients into a single dW.)
        extra_matmul_numel = 0
        cfg = self.config
        for blk in self.transformer.h:
            if not blk.is_target:
                continue
            attn = blk.attn
            if cfg.add_init_res_v:
                extra_matmul_numel += attn.c_v.weight.numel()
            if cfg.add_init_qkv or cfg.add_init_qkv_shared:
                extra_matmul_numel += attn.c_q.weight.numel()
                extra_matmul_numel += attn.c_k.weight.numel()
                extra_matmul_numel += attn.c_v.weight.numel()
            if cfg.add_init_proj:
                extra_matmul_numel += attn.c_proj.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude + extra_matmul_numel) + attn_flops
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
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters()) if (self.config.v_from_value_emb or self.config.add_init_value_emb or self.config.add_init_value_emb_nanogpt) else 0
        total = wte + lm_head + transformer_matrices + scalars + value_embeds
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'value_embeds': value_embeds,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect learnable blend coeffs (alpha = main path, beta = skip path) into
        # named lists so they can go into the right AdamW groups (mirrors gpt.py's
        # resid_lambdas/x0_lambdas treatment).
        alpha_params, beta_params = [], []
        for block in self.transformer.h:
            for n in ('alpha_res', 'alpha_attn_in', 'alpha_mlp_in'):
                if hasattr(block, n): alpha_params.append(getattr(block, n))
            for n in ('beta_res', 'beta_attn_in', 'beta_mlp_in'):
                if hasattr(block, n): beta_params.append(getattr(block, n))
            for n in ('alpha_q', 'alpha_k', 'alpha_v', 'alpha_qkv', 'alpha_o'):
                if hasattr(block.attn, n): alpha_params.append(getattr(block.attn, n))
            for n in ('beta_q', 'beta_k', 'beta_v', 'beta_qkv', 'beta_o'):
                if hasattr(block.attn, n): beta_params.append(getattr(block.attn, n))
            # gamma_v (v_from_x0 + learn_init_coeffs, init 1.0): grouped with the
            # betas so it gets the high-LR / no-decay treatment, matching the user
            # intent that this scalar should be free to grow above 1.0 if helpful.
            if hasattr(block.attn, 'gamma_v'): beta_params.append(block.attn.gamma_v)
        coeff_params = alpha_params + beta_params

        # Dead params under v_from_v1 / v_from_value_emb: late-layer c_v.weight
        # is unused (forward skipped, no grad path to loss). Exclude from the
        # optimizer so Muon's grad-stack doesn't fail on None gradients. They
        # stay in the state_dict (init values, never updated) — preserves
        # load_state_dict compatibility with other gpt_base variants.
        dead_param_ids = set()
        if self.config.v_from_v1 or self.config.v_from_value_emb:
            for blk in self.transformer.h:
                if blk.is_target:
                    dead_param_ids.add(id(blk.attn.c_v.weight))

        # lambda_v (per-head scaling vector under v_from_v1 + learn_init_scaling_vec):
        # its own AdamW group at a small LR — init at 1.0, evolves slowly.
        lambda_v_params = []
        for blk in self.transformer.h:
            if hasattr(blk.attn, 'lambda_v'):
                lambda_v_params.append(blk.attn.lambda_v)
        lambda_v_ids = {id(p) for p in lambda_v_params}

        # Separate out all parameters into groups
        all_block_params = [p for p in self.transformer.h.parameters()
                            if id(p) not in dead_param_ids and id(p) not in lambda_v_ids]
        matrix_params = [p for p in all_block_params if p.dim() >= 2]  # exclude scalar coeffs
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        smear_params = [self.smear_gate.weight, self.smear_lambda]
        # Value embeddings (v_from_value_emb / add_init_value_emb /
        # add_init_value_emb_nanogpt): own AdamW group, like gpt.py.
        value_embeds_params = list(self.value_embeds.parameters()) if (self.config.v_from_value_emb or self.config.add_init_value_emb or self.config.add_init_value_emb_nanogpt) else []
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(smear_params) + len(coeff_params) + len(lambda_v_params) + len(dead_param_ids) + len(value_embeds_params)

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
        # Learnable blend coeffs (only present when learn_init_coeffs is set).
        # alpha (main, init 1.0): low LR + weight decay, like gpt.py's resid_lambdas.
        # beta  (skip, init 0.0): high LR no decay,        like gpt.py's x0_lambdas.
        if alpha_params:
            param_groups.append(dict(kind='adamw', params=alpha_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05))
        if beta_params:
            param_groups.append(dict(kind='adamw', params=beta_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        # lambda_v: per-head scale init at 1.0; small LR keeps it close to identity
        # but lets each head pick its own gain on the cached layer-0 v.
        if lambda_v_params:
            param_groups.append(dict(kind='adamw', params=lambda_v_params, lr=0.005, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Value embedding tables (v_from_value_emb): own AdamW group at embedding_lr
        # * 0.5, mirroring gpt.py. Aligned init starts them on the v_from_x0
        # manifold; halved LR lets them drift off slowly rather than thrash.
        if value_embeds_params:
            param_groups.append(dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01))
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
        # x0 = post-smear residual; used by all add_init_* paths inside Block/Attention.
        # detach_init_value: detach once here (idempotent — same as detaching per-block).
        # x is passed un-detached, so add_init_v's layer-0 v capture is unaffected.
        # ve = layer-0 v capture, only consumed by add_init_v (shared-mode value residual).
        x0 = x
        if self.config.detach_init_value:
            x0 = x0.detach()
        ve = None
        capture_v1 = self.config.add_init_v or self.config.v_from_v1
        for i, block in enumerate(self.transformer.h):
            # ve fed to block: per-late-layer VE table lookup under
            # v_from_value_emb / add_init_value_emb / add_init_value_emb_nanogpt,
            # otherwise the captured layer-0 v (for add_init_v / v_from_v1) or None.
            if (self.config.v_from_value_emb or self.config.add_init_value_emb or self.config.add_init_value_emb_nanogpt) and str(i) in self.value_embeds:
                ve_in = self.value_embeds[str(i)](idx).to(x.dtype)
            else:
                ve_in = ve
            x, v_init = block(x, x0, ve_in, cos_sin, kv_cache)
            if capture_v1 and i == 0:
                ve = v_init
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
