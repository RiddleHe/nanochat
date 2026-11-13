"""Per-loop Muon for the looped transformer (12 shared blocks x 4 loops).

A shared matrix W is used once per loop, so its gradient is a sum G = g1 + g2 + g3 + g4.
Native Muon (the baseline) treats G as one gradient. Here every loop keeps its own momentum
and gets its own Polar Express before the loops are combined:

  native  momentum(G) -> PE -> NorMuon                                  (= nanochat MuonAdamW)
  v1      momentum(g_k) -> PE, per loop -> sum -> NorMuon -> rescale to ||A||
  v2      momentum(g_k) -> PE, per loop -> sum -> PE -> NorMuon

A is the update the native optimizer would make at this step (a shadow native path with its
own momentum and NorMuon state, run every step). v1 rescales each matrix's update to ||A||,
so v1 changes only the direction of the update, never its size; v2 is not rescaled.
lr scaling and cautious weight decay are unchanged and applied after this.

Per-loop gradients come from an unrolled twin: a GPTBase with 48 blocks whose block weights are
4 value-identical copies of the 12 shared blocks. Its forward is the same computation as GPTLoop,
so FLOPs are unchanged; after every optimizer step the copies are re-synced from the 12 masters.
Embeddings, lm_head and smear gate are the SAME Parameter objects in both models.
"""
import torch
from torch import Tensor

from nanochat.model.gpt_base import GPTBase, GPTBaseConfig
from nanochat.optim import MuonAdamW, polar_express_coeffs

MODES = ("native", "v1", "v2")


# ---------------------------------------------------------------------------
# Unrolled twin (per-loop gradients without hooks; compile-friendly)

@torch.no_grad()
def build_twin(master):
    """Return (twin, loops). loops maps each master block Parameter to its 4 per-loop copies."""
    cfg = master.config
    num_loops, num_blocks = cfg.num_loops, cfg.block_depth
    twin_cfg = GPTBaseConfig(sequence_len=cfg.sequence_len, vocab_size=cfg.vocab_size, n_layer=cfg.n_layer,
                             n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd,
                             window_pattern=cfg.window_pattern)
    with torch.device("meta"):
        twin = GPTBase(twin_cfg)
    twin.transformer.h.to_empty(device=master.get_device())
    # Shared (non-looped) parameters are the same objects: one gradient, one optimizer state.
    twin.transformer.wte = master.transformer.wte
    twin.lm_head = master.lm_head
    twin.smear_gate = master.smear_gate
    twin.smear_lambda = master.smear_lambda
    twin.cos, twin.sin = master.cos, master.sin
    assert twin.window_sizes == master.window_sizes
    loops = {}
    for i, block in enumerate(master.transformer.h):
        for name, p in block.named_parameters():
            copies = [twin.transformer.h[k * num_blocks + i].get_parameter(name) for k in range(num_loops)]
            assert all(q.shape == p.shape and q.dtype == p.dtype for q in copies)
            loops[p] = copies
    n_twin = sum(p.numel() for p in twin.transformer.h.parameters())
    assert n_twin == num_loops * sum(p.numel() for p in master.transformer.h.parameters())
    sync(loops)
    return twin, loops


@torch.no_grad()
def sync(loops):
    """Copy the 12 master blocks into all 4 loop copies."""
    dst, src = [], []
    for p, copies in loops.items():
        for q in copies:
            dst.append(q)
            src.append(p)
    torch._foreach_copy_(dst, src)


# ---------------------------------------------------------------------------
# The stages of nanochat's muon_step_fused, unchanged, split so they can be recombined.

def nesterov(grad: Tensor, buf: Tensor, momentum_t: Tensor) -> Tensor:
    momentum = momentum_t.to(grad.dtype)
    buf.lerp_(grad, 1 - momentum)
    return grad.lerp(buf, momentum)


def polar_express(g: Tensor, ns_steps: int) -> Tensor:
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def normuon(g: Tensor, buf: Tensor, beta2_t: Tensor, red_dim: int) -> Tensor:
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    buf.lerp_(v_mean.to(dtype=buf.dtype), 1 - beta2)
    step_size = buf.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    return g * final_scale.to(g.dtype)


def cautious_update(g: Tensor, params: Tensor, lr_t: Tensor, wd_t: Tensor) -> None:
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * params) >= 0
    params.sub_(lr * g + lr * wd * params * mask)


# ---------------------------------------------------------------------------

class PerLoopMuonAdamW(MuonAdamW):
    """MuonAdamW whose Muon groups read per-loop gradients from the twin.

    AdamW groups (embeddings, lm_head, smear) are untouched. Muon state keys:
      momentum_buffer, second_momentum_buffer      native / shadow-native path (A)
      loop_momentum_buffer (L, n, *shape)          per-loop momentum (v1, v2)
      b_second_momentum_buffer                     NorMuon state of the combined update (v1, v2)
    After step(), self.stats holds (params, ||B||/||A|| before rescaling, cos(A, B)) per group.
    """

    def __init__(self, param_groups, mode, loops):
        assert mode in MODES, mode
        super().__init__(param_groups)
        self.mode = mode
        self.loops = loops
        self.stats = []
        self.finite = []
        for group in self.param_groups:
            if group["kind"] == "muon":
                assert all(p in loops for p in group["params"]), "Muon params must be looped block matrices"

    def _step_muon(self, group: dict) -> None:
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        n, shape, device, dtype = len(params), p0.shape, p0.device, p0.dtype
        num_loops = len(self.loops[p0])
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        second_shape = (n, shape[-2], 1) if shape[-2] >= shape[-1] else (n, 1, shape[-1])

        loop_grads = torch.stack([torch.stack([q.grad for q in self.loops[p]]) for p in params], dim=1)
        native = loop_grads.sum(0)  # = the gradient GPTLoop would have produced
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        ns = group["ns_steps"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(n, *shape, dtype=dtype, device=device)
            state["second_momentum_buffer"] = torch.zeros(second_shape, dtype=dtype, device=device)
        # A: the native update (always computed: it is the update in native mode, the norm target
        # in v1, and a logged reference in v2).
        a = nesterov(native, state["momentum_buffer"], self._muon_momentum_t)
        a = polar_express(a, ns)
        a = normuon(a, state["second_momentum_buffer"], self._muon_beta2_t, red_dim)

        if self.mode == "native":
            update = a
            self.finite.append(torch.isfinite(native).all())
        else:
            if "loop_momentum_buffer" not in state:
                state["loop_momentum_buffer"] = torch.zeros(num_loops, n, *shape, dtype=dtype, device=device)
                state["b_second_momentum_buffer"] = torch.zeros(second_shape, dtype=dtype, device=device)
            b = nesterov(loop_grads, state["loop_momentum_buffer"], self._muon_momentum_t)
            b = polar_express(b, ns).float().sum(0).bfloat16()  # one PE per loop, then add
            if self.mode == "v2":
                b = polar_express(b, ns)
            b = normuon(b, state["b_second_momentum_buffer"], self._muon_beta2_t, red_dim)
            a32, b32 = a.float(), b.float()
            norm_a, norm_b = a32.norm(dim=(-2, -1)), b32.norm(dim=(-2, -1))
            cos = (a32 * b32).sum(dim=(-2, -1)) / (norm_a * norm_b).clamp_min(1e-30)
            ratio = norm_b / norm_a.clamp_min(1e-30)
            self.stats.append((params, ratio, cos))
            if self.mode == "v1":
                scale = norm_a / norm_b.clamp_min(1e-30)
                b = (b32 * scale[:, None, None]).to(b.dtype)
            update = b
            self.finite.append(torch.isfinite(native).all() & torch.isfinite(ratio).all())

        cautious_update(update, stacked_params, self._muon_lr_t, self._muon_wd_t)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        self.stats, self.finite = [], []
        super().step()
