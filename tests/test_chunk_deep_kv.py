"""
Correctness tests for the chunk deep-KV variants.

Three properties must hold, and each one caught a real bug during development:

1. PARITY   -- with the branch gates at 0 the model must be bit-exactly the
               baseline. For v2 this is a stronger claim: the chunk-recurrent
               trunk must also equal the ordinary non-chunked trunk, which is
               what makes the "no pass-1 tax" version a fair comparison.
2. GRADIENT -- the gates must actually receive gradient, otherwise the branch
               can never turn on and the whole experiment silently measures
               nothing. (This is how the c_proj=0 init blind spot was found:
               untrained models zero the attention output, so gate grads were
               identically 0 and the smoke test "passed" while testing nothing.)
3. CAUSALITY-- the branch may only read STRICTLY EARLIER chunks. Positions in
               chunk 0 must be unaffected by the branch; later positions must be
               affected. A leak here would be train/test contamination.

Run:  python -m pytest tests/test_chunk_deep_kv.py -v
"""

import copy

import pytest
import torch

from nanochat.model_registry import get_model

CHUNK = 64  # small chunk so a short test sequence still spans several chunks
SEQ = 256   # 4 chunks


def _build(model_type, seed=0, **cfg_over):
    ConfigCls, ModelCls = get_model(model_type)
    cfg = ConfigCls(
        sequence_len=SEQ, vocab_size=256, n_layer=4, n_head=4, n_kv_head=2,
        n_embd=128, **cfg_over,
    )
    if hasattr(cfg, "chunk_size"):
        cfg.chunk_size = CHUNK
    torch.manual_seed(seed)
    model = ModelCls(cfg)
    model.init_weights()
    return model


def _randomize_c_proj(model, seed=1234):
    """init_weights zeroes attn.c_proj and mlp.c_proj, so an untrained model
    writes nothing into the residual stream and every branch effect is masked.
    Give them real values so the tests exercise a live network."""
    g = torch.Generator().manual_seed(seed)  # CPU generator => device-independent values
    for block in model.transformer.h:
        for w in (block.attn.c_proj.weight, block.mlp.c_proj.weight):
            vals = torch.empty(w.shape, dtype=torch.float32).normal_(0, 0.02, generator=g)
            w.data.copy_(vals.to(w.device, w.dtype))


def _copy_shared_weights(src, dst):
    """Copy every parameter that exists in both models (i.e. everything except
    the chunk gates), so a baseline and a chunk model differ ONLY by the branch."""
    sd = src.state_dict()
    missing = dst.load_state_dict({k: v for k, v in sd.items() if k in dst.state_dict()},
                                  strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    # anything not copied must be a chunk gate
    for k in missing.missing_keys:
        assert "chunk_gate" in k, f"unexpected uninitialized param: {k}"


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("chunk deep-KV paths need CUDA (flash-attn)")
    return torch.device("cuda")


@pytest.mark.parametrize("model_type", [
    "gpt_base_chunk_deep_kv",
    "gpt_base_chunk_same_kv",
    "gpt_base_chunk_deep_kv_v2",
    "gpt_base_chunk_deep_kv_v2_slim",
])
def test_parity_at_zero_gate(device, model_type):
    """With gates at 0 (their init), every variant must equal plain gpt_base
    bit-for-bit. For the v2 variants this additionally proves the chunk-recurrent
    trunk is mathematically equivalent to the standard trunk."""
    base = _build("gpt_base").to(device)
    _randomize_c_proj(base)
    chunk = _build(model_type).to(device)
    _copy_shared_weights(base, chunk)
    _randomize_c_proj(chunk)  # same seed => same values as base
    for b in chunk.transformer.h:
        if hasattr(b.attn, "chunk_gate"):
            assert torch.all(b.attn.chunk_gate == 0), "gates must init to 0"

    idx = torch.randint(0, 256, (2, SEQ), device=device)
    with torch.no_grad():
        want = base(idx)
        got = chunk(idx)
    diff = (want - got).abs().max().item()
    assert diff == 0.0, f"{model_type}: gate=0 must be bit-exact, got max|diff|={diff:.3e}"


@pytest.mark.parametrize("model_type", [
    "gpt_base_chunk_deep_kv",
    "gpt_base_chunk_same_kv",
    "gpt_base_chunk_deep_kv_v2",
])
def test_gate_receives_gradient(device, model_type):
    """The gates must get non-zero gradient, else the branch can never turn on."""
    model = _build(model_type).to(device)
    _randomize_c_proj(model)
    idx = torch.randint(0, 256, (2, SEQ), device=device)
    targets = torch.randint(0, 256, (2, SEQ), device=device)
    model(idx, targets=targets).backward()

    gates = [(i, b.attn.chunk_gate) for i, b in enumerate(model.transformer.h)
             if hasattr(b.attn, "chunk_gate")]
    assert gates, "no chunk gates were created"
    for i, g in gates:
        assert g.grad is not None, f"layer {i}: gate grad is None"
        assert g.grad.abs().max().item() > 0, f"layer {i}: gate grad is identically 0"


@pytest.mark.parametrize("model_type", [
    "gpt_base_chunk_deep_kv",
    "gpt_base_chunk_deep_kv_v2",
])
def test_branch_reads_only_earlier_chunks(device, model_type):
    """Opening the gates must leave chunk-0 outputs EXACTLY unchanged (nothing
    earlier exists to read) while changing later chunks. Any movement in chunk 0
    means the branch is leaking same-chunk or future information."""
    model = _build(model_type).to(device)
    _randomize_c_proj(model)
    idx = torch.randint(0, 256, (2, SEQ), device=device)

    with torch.no_grad():
        closed = model(idx)
        for b in model.transformer.h:
            if hasattr(b.attn, "chunk_gate"):
                b.attn.chunk_gate.fill_(1.0)
        opened = model(idx)

    d = (opened - closed).abs()
    d_first = d[:, :CHUNK].max().item()
    d_rest = d[:, CHUNK:].max().item()
    assert d_first == 0.0, f"branch leaked into chunk 0: max|diff|={d_first:.3e}"
    assert d_rest > 0.0, "opening the gates changed nothing after chunk 0"


def test_flops_accounting_is_honest(device):
    """The FLOPs estimate must charge for the branch, and v1 must be charged
    substantially more than v2 (that difference is the whole point of v2).
    Equal-FLOPs comparisons are only fair if this is right."""
    base = _build("gpt_base").to(device)
    v1 = _build("gpt_base_chunk_deep_kv").to(device)
    v2 = _build("gpt_base_chunk_deep_kv_v2").to(device)

    f_base, f_v1, f_v2 = (m.estimate_flops() for m in (base, v1, v2))
    assert f_v1 > f_v2 > f_base, f"expected base < v2 < v1, got {f_base}, {f_v2}, {f_v1}"
    # v1 carries the pass-1 no-grad forward (~1/3 of a fwd+bwd) on top of v2's branch
    assert f_v1 / f_base > 1.3, f"v1 should cost >1.3x baseline, got {f_v1 / f_base:.3f}"
