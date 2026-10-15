"""CPU regression tests; no dataset, tokenizer download, or training run needed."""
from dataclasses import asdict

import pytest
import torch

import nanochat.flash_attention as fa
import nanochat.model.gpt_base_attn_res_qkv as model_module
from nanochat.model.gpt_base import GPTBaseConfig, CausalSelfAttention as BaseAttention
from nanochat.model_registry import get_model
from nanochat.engine import KVCache


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setattr(model_module, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(fa, 'USE_FA3', False)
    torch.manual_seed(123)


def make_model(n_kv_head=2, nonzero=False, **kwargs):
    Config, Model = get_model('gpt_base_attn_res_qkv')
    config = Config(sequence_len=32, vocab_size=61, n_layer=3,
                    n_head=4, n_kv_head=n_kv_head, n_embd=32, **kwargs)
    with torch.device('meta'):
        model = Model(config)
    model.to_empty(device='cpu')
    # Poison all allocated tensors: init_weights must initialize every parameter.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float('nan'))
    model.init_weights()
    if nonzero:
        # Exercise routing and attention; zero output projections hide mistakes.
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.1)
                block.mlp.c_proj.weight.normal_(std=0.1)
            model.attn_res_queries[:model.n_queries].normal_(std=0.2)
    return model


def test_meta_initialization_and_accounting():
    model = make_model()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.count_nonzero(model.attn_res_queries)
    assert not any('smear' in name for name in model.state_dict())
    assert model.n_queries == 13
    assert model.attn_res_queries.shape == (16, 32)
    assert not any('value_embed' in k or 've_gate' in k or 'resid_lambda' in k
                   for k in model.state_dict())
    counts = model.num_scaling_params()
    assert counts['value_embeds'] == 0
    assert counts['total'] == sum(p.numel() for p in model.parameters())
    # Verify FLOPs against actual routing calls, not the closed-form polynomial.
    calls = []
    route = model._attn_res
    model._attn_res = lambda q, sources: (calls.append((len(q), len(sources))) or route(q, sources))
    model(torch.randint(61, (1, 4)))
    routed_pairs = sum(nq * ns for nq, ns in calls)
    matrix_params = counts['transformer_matrices'] + counts['lm_head']
    attn_flops = sum(12 * 32 * min(w[0], 32) for w in model.window_sizes)
    assert model.estimate_flops() == 6 * matrix_params + attn_flops + 12 * 32 * routed_pairs


def test_uniform_and_independent_depth_routing():
    sources = [torch.randn(2, 5, 32) * scale for scale in (0.1, 2, 4)]
    route = model_module.GPTBaseAttnResQKV._attn_res
    queries = torch.zeros(3, 32)
    expected = torch.stack(sources).mean(0)
    torch.testing.assert_close(route(queries, sources), expected.expand(3, -1, -1, -1))
    queries[1].normal_()
    routed = route(queries, sources)
    torch.testing.assert_close(routed[0], expected)
    torch.testing.assert_close(routed[2], expected)
    assert not torch.allclose(routed[1], expected)


@pytest.mark.parametrize('projection', [0, 1, 2])
def test_query_changes_only_its_projection_input(projection):
    model = make_model(nonzero=True)
    idx = torch.randint(61, (2, 7))
    captured = {}
    hooks = []
    for name in ('c_q', 'c_k', 'c_v'):
        def capture(module, args, name=name):
            captured[name] = args[0].detach().clone()
        hooks.append(getattr(model.transformer.h[1].attn, name).register_forward_pre_hook(capture))
    model(idx)
    before = dict(captured)
    with torch.no_grad():
        model.attn_res_queries[4 + projection].add_(torch.randn(32))
    model(idx)
    for i, name in enumerate(('c_q', 'c_k', 'c_v')):
        if i == projection:
            assert not torch.allclose(before[name], captured[name])
        else:
            torch.testing.assert_close(before[name], captured[name], atol=0, rtol=0)
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize('n_kv_head', [2, 4])
def test_attention_matches_base_with_shared_input(n_kv_head):
    model = make_model(n_kv_head=n_kv_head, nonzero=True)
    base = BaseAttention(GPTBaseConfig(**asdict(model.config)), 0, (32, 0))
    attn = model.transformer.h[0].attn
    base.load_state_dict(attn.state_dict(), strict=True)
    x = torch.randn(2, 7, 32)
    rope = model.cos[:, :7], model.sin[:, :7]
    expected, _ = base(x, x, None, rope, None)
    torch.testing.assert_close(attn(x, x, x, rope), expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_gradients_and_optimizer_step(monkeypatch, dtype):
    monkeypatch.setattr(model_module, 'COMPUTE_DTYPE', dtype)
    model = make_model(nonzero=True)
    idx = torch.randint(61, (2, 8))
    targets = torch.randint(61, (2, 8))
    targets[0, 0] = -1
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # First layer Q/K/V have only x0, hence necessarily zero routing gradient.
    qgrad = model.attn_res_queries.grad
    assert not torch.count_nonzero(qgrad[:3])
    assert (qgrad[4:7].abs().sum(-1) > 0).all()
    assert not torch.count_nonzero(qgrad[model.n_queries:])
    opt = model.setup_optimizer()
    query_group = next(g for g in opt.param_groups
                       if any(p is model.attn_res_queries for p in g['params']))
    assert query_group['kind'] == 'adamw'
    assert query_group['lr'] == 0.001 and query_group['weight_decay'] == 0
    params = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(params) == len(set(params)) == len(list(model.parameters()))
    before = model.attn_res_queries.detach().clone()
    # Exercise optimizer math without compiling CPU optimizer kernels in tests.
    with torch.compiler.set_stance('force_eager'):
        opt.step()
    assert not torch.equal(before[4:7], model.attn_res_queries[4:7])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert torch.isfinite(model(idx, targets))


@pytest.mark.parametrize('n_kv_head', [2, 4])
@pytest.mark.parametrize('chunks', [(5, 1, 1, 1, 1), (3, 2, 4)])
@torch.no_grad()
def test_causality_and_cached_decode(n_kv_head, chunks):
    model = make_model(n_kv_head=n_kv_head, nonzero=True).eval()
    # Small explicit window ensures this test crosses the window boundary.
    model.transformer.h[0].attn.window_size = (2, 0)
    idx = torch.randint(61, (2, 9))
    full = model(idx)
    changed = idx.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 61
    torch.testing.assert_close(model(changed)[:, :5], full[:, :5], atol=1e-6, rtol=1e-5)
    cache = KVCache(2, n_kv_head, 9, 8, 3, 'cpu', torch.float32)
    pieces = []
    offset = 0
    for size in chunks:
        pieces.append(model(idx[:, offset:offset+size], kv_cache=cache))
        offset += size
        assert cache.get_pos() == offset
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-6, rtol=1e-5)
    # Single-token uncached input is supported, including naive generation.
    torch.testing.assert_close(model(idx[:, :1]), full[:, :1], atol=1e-6, rtol=1e-5)
    assert len(list(model.generate([1], 2, temperature=0))) == 2


def test_checkpoint_loader_roundtrip(tmp_path, monkeypatch):
    import nanochat.checkpoint_manager as cm
    model = make_model(nonzero=True)
    cm.save_checkpoint(tmp_path, 7, model.state_dict(), None, {
        'model_type': 'gpt_base_attn_res_qkv',
        'model_config': asdict(model.config), 'step': 7,
    })
    class TokenizerStub:
        def get_vocab_size(self):
            return 61
    monkeypatch.setattr(cm, 'get_tokenizer', TokenizerStub)
    restored, _, meta = cm.build_model(tmp_path, 7, torch.device('cpu'), 'eval')
    idx = torch.randint(61, (2, 7))
    torch.testing.assert_close(restored(idx), model(idx), atol=0, rtol=0)
    assert meta['step'] == 7


def test_compile_fullgraph_forward_backward():
    model = make_model(nonzero=True)
    idx = torch.randint(61, (2, 5))
    targets = torch.randint(61, (2, 5))
    eager = model(idx, targets)
    eager.backward()
    expected_grad = model.attn_res_queries.grad.clone()
    model.zero_grad(set_to_none=True)
    compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
    actual = compiled(idx, targets)
    actual.backward()
    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(model.attn_res_queries.grad, expected_grad)
