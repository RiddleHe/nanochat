"""CPU regression tests; no dataset, tokenizer download, or training run needed."""
from dataclasses import asdict

import pytest
import torch

import nanochat.flash_attention as fa
import nanochat.model.gpt_base_scalar_res_qkv_no_embed_norm as model_module
from nanochat.model.gpt_base import GPTBaseConfig, CausalSelfAttention as BaseAttention
from nanochat.model_registry import get_model
from nanochat.engine import KVCache


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setattr(model_module, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(fa, 'USE_FA3', False)
    torch.manual_seed(123)


def receiver_coefficients(model):
    return [model._receiver_coefficients(i) for i in range(4 * model.config.n_layer + 1)]


def base_config(model):
    config = asdict(model.config)
    for name in tuple(config):
        if name.startswith('scalar_res_'):
            config.pop(name)
    return GPTBaseConfig(**config)


def make_model(n_kv_head=2, nonzero=False, **kwargs):
    Config, Model = get_model('gpt_base_scalar_res_qkv_no_embed_norm')
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
            model.scalar_res_logits_flat[:model.scalar_res_active].normal_(mean=0.0, std=0.2)
    return model


def test_meta_initialization_and_accounting():
    model = make_model()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert torch.count_nonzero(model.scalar_res_logits_flat) == 0
    assert all(torch.equal(p, torch.ones_like(p)) for p in receiver_coefficients(model))
    assert not any('smear' in name for name in model.state_dict())
    assert [p.numel() for p in receiver_coefficients(model)] == [1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7]
    assert not any("attn_res_queries" in name for name in model.state_dict())
    assert not any('value_embed' in k or 've_gate' in k or 'resid_lambda' in k
                   for k in model.state_dict())
    counts = model.num_scaling_params()
    assert counts['value_embeds'] == 0
    assert counts['scalars'] == 46 and counts['scalar_padding'] == 2
    assert model.scalar_res_logits_flat.shape == (48,)
    assert torch.count_nonzero(model.scalar_res_logits_flat[model.scalar_res_active:]) == 0
    assert all(p.untyped_storage().data_ptr() == model.scalar_res_logits_flat.untyped_storage().data_ptr()
               for p in [model._receiver_logits(i) for i in range(4 * model.config.n_layer + 1)])
    assert counts['total'] == sum(p.numel() for p in model.parameters())
    # Verify FLOPs against actual routing calls, not the closed-form polynomial.
    calls = []
    route = model._scalar_res
    model._scalar_res = lambda q, sources: (calls.append((len(q), len(sources))) or route(q, sources))
    model(torch.randint(61, (1, 4)))
    routed_pairs = sum(nq * ns for nq, ns in calls)
    matrix_params = counts['transformer_matrices'] + counts['lm_head']
    attn_flops = sum(12 * 32 * min(w[0], 32) for w in model.window_sizes)
    assert model.estimate_flops() == 6 * matrix_params + attn_flops + 6 * 32 * routed_pairs


def test_signed_scalar_sums_and_gradients():
    sources = [torch.randn(2, 5, 32, requires_grad=True) for _ in range(3)]
    coeff = torch.tensor([[1., 1., 1.], [0., -2., 3.], [4., 0., 0.]], requires_grad=True)
    actual = model_module.GPTBaseScalarResQKVNoEmbedNorm._scalar_res(coeff, sources)
    expected = torch.stack([sum(coeff[r, j] * sources[j] for j in range(3)) for r in range(3)])
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (coeff, *sources), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (coeff, *sources), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_raw_embedding_source_and_normalized_receiver(monkeypatch):
    model = make_model(nonzero=True)
    idx = torch.randint(61, (2, 7))
    captured_sources = []
    route = model._scalar_res

    def capture_sources(coefficients, sources):
        captured_sources.append(sources[0].detach().clone())
        return route(coefficients, sources)

    monkeypatch.setattr(model, '_scalar_res', capture_sources)
    captured_q = []
    hook = model.transformer.h[0].attn.c_q.register_forward_pre_hook(
        lambda module, args: captured_q.append(args[0].detach().clone())
    )
    with torch.no_grad():
        model._receiver_logits(0).zero_()
    try:
        model(idx)
    finally:
        hook.remove()
    raw = model.transformer.wte(idx)
    assert len(captured_sources) == 2 * model.config.n_layer + 1
    for source in captured_sources:
        torch.testing.assert_close(source, raw, atol=0, rtol=0)
    torch.testing.assert_close(captured_q[0], model_module.norm(raw), atol=0, rtol=0)
    assert not torch.allclose(raw, model_module.norm(raw))


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
        model._receiver_logits(4 + projection).add_(torch.randn(3))
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
    base = BaseAttention(base_config(model), 0, (32, 0))
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
    assert all(model.scalar_res_logits_flat.grad[model.scalar_res_offsets[i]:model.scalar_res_offsets[i+1]].abs().sum() > 0
               for i in range(4, 7))
    assert torch.count_nonzero(model.scalar_res_logits_flat.grad[model.scalar_res_active:]) == 0
    opt = model.setup_optimizer(scalar_lr=0.5 * 2**0.5)
    scalar_group = next(g for g in opt.param_groups
                        if any(p is model.scalar_res_logits_flat for p in g['params']))
    assert scalar_group['kind'] == 'adamw'
    assert scalar_group['lr'] == 0.5 * 2**0.5
    assert scalar_group['betas'] == (0.96, 0.95)
    assert scalar_group['eps'] == 1e-10 and scalar_group['weight_decay'] == 0
    assert scalar_group['initial_lr'] == scalar_group['lr']
    assert all(p.dtype == torch.float32 for p in receiver_coefficients(model))
    params = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(params) == len(set(params)) == len(list(model.parameters()))
    before = model._receiver_coefficients(4).detach().clone()
    # Exercise optimizer math without compiling CPU optimizer kernels in tests.
    with torch.compiler.set_stance('force_eager'):
        opt.step()
    assert not torch.equal(before, model._receiver_coefficients(4))
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


@pytest.mark.parametrize('pad_to', [4, 32])
def test_checkpoint_loader_roundtrip(tmp_path, monkeypatch, pad_to):
    import nanochat.checkpoint_manager as cm
    model = make_model(nonzero=True, scalar_res_pad_to=pad_to,
                       scalar_res_recovery_weight=0.03, scalar_res_recovery_margin=1.5)
    cm.save_checkpoint(tmp_path, 7, model.state_dict(), None, {
        'model_type': 'gpt_base_scalar_res_qkv_no_embed_norm',
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
    assert restored.config.scalar_res_pad_to == pad_to
    assert restored.config.scalar_res_gate == 'sqrt'
    assert restored.config.scalar_res_recovery_weight == model.config.scalar_res_recovery_weight
    assert restored.config.scalar_res_recovery_margin == model.config.scalar_res_recovery_margin
    torch.testing.assert_close(restored.scalar_res_logits_flat, model.scalar_res_logits_flat, atol=0, rtol=0)


def test_compile_fullgraph_forward_backward():
    model = make_model(nonzero=True)
    with torch.no_grad():
        model.scalar_res_logits_flat[4] = -4
    idx = torch.randint(61, (2, 5))
    targets = torch.randint(61, (2, 5))
    eager = model(idx, targets)
    eager.backward()
    expected_grad = model.scalar_res_logits_flat.grad.clone()
    model.zero_grad(set_to_none=True)
    compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
    actual = compiled(idx, targets)
    actual.backward()
    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(model.scalar_res_logits_flat.grad, expected_grad)


@torch.no_grad()
def test_unit_coefficients_match_plain_residual_network(monkeypatch):
    import nanochat.model.gpt_base_no_embed_norm as base_module
    monkeypatch.setattr(base_module, 'COMPUTE_DTYPE', torch.float32)
    model = make_model(nonzero=True)
    model.scalar_res_logits_flat.zero_()
    base = base_module.GPTBaseNoEmbedNorm(base_config(model))
    base.init_weights()
    shared = {k: v for k, v in model.state_dict().items() if k != 'scalar_res_logits_flat'}
    base.load_state_dict(shared, strict=True)
    idx = torch.randint(61, (2, 9))
    torch.testing.assert_close(model(idx), base(idx), atol=1e-6, rtol=1e-5)


def test_d24_flops_and_distributed_scalar_layout():
    Config, Model = get_model('gpt_base_scalar_res_qkv_no_embed_norm')
    with torch.device('meta'):
        model = Model(Config(n_layer=24, n_embd=1536, n_head=12, n_kv_head=12))
    assert model.num_scaling_params()['scalars'] == 2377
    assert model.estimate_flops() == 4797121536
    assert round(3.91e19 / (model.estimate_flops() * 1048576)) == 7773
    assert model.num_scaling_params()['scalar_padding'] == 3
    assert model.scalar_res_logits_flat.shape == (2380,)
    assert model.scalar_res_logits_flat.numel() // 4 == 595
    assert len(receiver_coefficients(model)) == 97


@pytest.mark.parametrize('pad_to', [0, -4, 1.5, True])
def test_invalid_scalar_padding(pad_to):
    with pytest.raises(ValueError, match='positive integer'):
        make_model(scalar_res_pad_to=pad_to)


def test_distributed_padding_guard(monkeypatch):
    Config, Model = get_model('gpt_base_scalar_res_qkv_no_embed_norm')
    with torch.device('meta'):
        model = Model(Config(n_layer=24))
    monkeypatch.setattr(model_module, 'get_dist_info', lambda: (True, 0, 0, 8))
    with pytest.raises(ValueError, match='divisible by world_size'):
        model.setup_optimizer()
    monkeypatch.setattr(model_module, 'get_dist_info', lambda: (True, 0, 0, 4))
    opt = model.setup_optimizer()
    group = next(g for g in opt.param_groups if any(p is model.scalar_res_logits_flat for p in g['params']))
    assert len(group['params']) == 1


def test_flat_forward_matches_separate_receiver_parameters():
    # Compare a flat gated parameter against independent per-receiver logits.
    model = make_model(nonzero=True)
    logits = [torch.nn.Parameter(model._receiver_logits(i).detach().clone())
              for i in range(4 * model.config.n_layer + 1)]
    coeffs = [model._sqrt_gate(z) for z in logits]
    idx = torch.randint(61, (2, 7))
    actual = model(idx)
    norm = model_module.norm
    sources = [model.transformer.wte(idx)]
    rope = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
    for i, block in enumerate(model.transformer.h):
        qkv = torch.stack(coeffs[4*i:4*i+3])
        hq, hk, hv = model._scalar_res(qkv, sources).unbind(0)
        sources.append(block.attn(norm(hq), norm(hk), norm(hv), rope, None))
        hm = model._scalar_res(coeffs[4*i+3][None, :], sources)[0]
        sources.append(block.mlp(norm(hm)))
    x = norm(model._scalar_res(coeffs[-1][None, :], sources)[0])
    expected = model.lm_head(x)[..., :model.config.vocab_size].float()
    expected = 15 * torch.tanh(expected / 15)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    flat_grad, = torch.autograd.grad(actual, model.scalar_res_logits_flat, upstream)
    separate_grads = torch.autograd.grad(expected, logits, upstream)
    torch.testing.assert_close(flat_grad[:model.scalar_res_active], torch.cat(separate_grads))
    assert torch.count_nonzero(flat_grad[model.scalar_res_active:]) == 0
    # Padding must have no forward effect, even if its stored values change.
    with torch.no_grad():
        model.scalar_res_logits_flat[model.scalar_res_active:].fill_(123)
    torch.testing.assert_close(model(idx), actual, atol=0, rtol=0)


@pytest.mark.parametrize('world_size', [1, 4])
def test_compiled_flat_adamw_matches_separate_parameters(monkeypatch, world_size):
    """Real CPU compilation; the four-shard case stubs communication only."""
    import nanochat.optim as optim_module

    Config, Model = get_model('gpt_base_scalar_res_qkv_no_embed_norm')
    with torch.device('meta'):
        model = Model(Config(n_layer=24))
    offsets = model.scalar_res_offsets
    active = model.scalar_res_active
    initial = torch.zeros(model.scalar_res_logits_flat.numel())
    initial[:active].normal_(mean=0, std=0.2)
    separate = [torch.nn.Parameter(initial[a:b].clone()) for a, b in zip(offsets, offsets[1:])]
    settings = dict(lr=0.5, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0)
    reference = torch.optim.AdamW(separate, **settings, foreach=False)
    # Each rank has a full parameter replica, but owns only its state slice.
    replicas = [torch.nn.Parameter(initial.clone()) for _ in range(world_size)]
    factory = optim_module.MuonAdamW if world_size == 1 else optim_module.DistMuonAdamW
    optimizers = [factory([dict(kind='adamw', params=[p], **settings)]) for p in replicas]
    chunk = initial.numel() // world_size

    class Completed:
        def wait(self):
            pass

        def get_future(self):
            return self

    gathers = []

    def gather(output, local, async_op):
        assert async_op and local.numel() == chunk
        gathers.append((output, local.detach().clone()))
        return Completed()

    if world_size > 1:
        monkeypatch.setattr(optim_module.dist, 'all_gather_into_tensor', gather)

    # Exercise the actual compiled kernel, including changing step and LR tensors.
    torch._dynamo.reset()
    try:
        for step in range(3):
            gradient = torch.randn_like(initial)
            gradient[active:] = 0
            lr = settings['lr'] / (step + 1)
            reference.param_groups[0]['lr'] = lr
            for p, a, b in zip(separate, offsets, offsets[1:]):
                p.grad = gradient[a:b].clone()
            reference.step()
            gathers.clear()
            for rank, (p, opt) in enumerate(zip(replicas, optimizers)):
                group = opt.param_groups[0]
                group['lr'] = lr
                p.grad = gradient.clone()
                if world_size == 1:
                    opt.step()
                else:
                    a, b = rank * chunk, (rank + 1) * chunk
                    # Supply the already-averaged gradient returned by reduction.
                    info = dict(param_infos={p: dict(
                        future=Completed(), grad_slice=gradient[a:b], is_small=False,
                    )})
                    with torch.no_grad():
                        opt._compute_adamw(group, info, [], rank, world_size)
            if world_size > 1:
                gathered = torch.cat([local for _, local in gathers])
                with torch.no_grad():
                    for output, _ in gathers:
                        output.copy_(gathered)
            expected = torch.cat([p.detach() for p in separate])
            for p in replicas:
                torch.testing.assert_close(p[:active], expected, atol=2e-6, rtol=2e-5)
                assert torch.count_nonzero(p[active:]) == 0
            for key in ('exp_avg', 'exp_avg_sq'):
                actual_state = torch.cat([opt.state[p][key] for p, opt in zip(replicas, optimizers)])
                expected_state = torch.cat([reference.state[p][key] for p in separate])
                torch.testing.assert_close(actual_state[:active], expected_state)
                assert torch.count_nonzero(actual_state[active:]) == 0
    finally:
        torch._dynamo.reset()


def test_sqrt_gate_values_and_gradients():
    gate = model_module.GPTBaseScalarResQKVNoEmbedNorm._sqrt_gate
    z = torch.tensor([-1e6, -1e4, -4., -2., -1e-4, 0., 1e-4, 2., 1e4, 1e6], requires_grad=True)
    actual = gate(z)
    reference = torch.exp(torch.asinh(z.detach().double()))
    torch.testing.assert_close(actual.double(), reference, atol=1e-7, rtol=2e-6)
    assert torch.all(actual > 0)
    actual.sum().backward()
    reference_grad = reference / torch.sqrt(1 + z.detach().double().square())
    torch.testing.assert_close(z.grad.double(), reference_grad, atol=1e-8, rtol=2e-6)
    assert z.grad[5] == 1 and torch.all(z.grad > 0)
    assert gate(z.detach().bfloat16()).dtype == torch.float32


def test_recovery_threshold_gradient_and_padding():
    model = make_model()
    z = model.scalar_res_logits_flat
    with torch.no_grad():
        z[:7].copy_(torch.tensor([-4., -3., -2., -1., 0., 3., 1e4]))
        z[model.scalar_res_active:] = -1000
    penalty = model._compute_scalar_recovery()
    torch.testing.assert_close(penalty, torch.tensor(0.01 * 5 / model.scalar_res_active))
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(penalty, parameters, allow_unused=True)
    expected = torch.zeros_like(z)
    expected[:2] = torch.tensor([-0.04, -0.02]) / model.scalar_res_active
    for parameter, gradient in zip(parameters, gradients):
        if parameter is z:
            torch.testing.assert_close(gradient, expected)
        else:
            assert gradient is None


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_recovery_matches_valid_token_loss_reduction(reduction):
    model = make_model(nonzero=True, scalar_res_recovery_weight=0.2,
                       scalar_res_recovery_margin=1.5)
    idx = torch.randint(61, (2, 5))
    targets = idx.roll(-1, dims=1)
    targets[0, :2] = -1
    with torch.no_grad():
        model.scalar_res_logits_flat[0] = -4
    logits = model(idx)
    lm_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 61), targets.reshape(-1),
                                                ignore_index=-1, reduction=reduction)
    penalty = torch.tensor(0.2 * (4 - 1.5)**2 / model.scalar_res_active)
    valid = targets.reshape(-1) != -1
    extra = penalty if reduction == 'mean' else (penalty * valid.sum() if reduction == 'sum' else penalty * valid)
    torch.testing.assert_close(model(idx, targets, loss_reduction=reduction), lm_loss + extra)
    model.eval()
    torch.testing.assert_close(model(idx, targets, loss_reduction=reduction), lm_loss)
    torch.testing.assert_close(model(idx), logits)
    model.train()
    model.config.scalar_res_recovery_weight = 0
    torch.testing.assert_close(model(idx, targets, loss_reduction=reduction), lm_loss)


@pytest.mark.parametrize('reduction', ['sum', 'none'])
def test_ignored_targets_receive_no_recovery(reduction):
    model = make_model()
    with torch.no_grad():
        model.scalar_res_logits_flat[0] = -4
    idx = torch.randint(61, (2, 5))
    actual = model(idx, torch.full_like(idx, -1), loss_reduction=reduction)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize('kwargs', [
    {'scalar_res_gate': 'silu'},
    {'scalar_res_recovery_weight': -0.01},
    {'scalar_res_recovery_weight': float('nan')},
    {'scalar_res_recovery_margin': -2},
    {'scalar_res_recovery_margin': float('inf')},
])
def test_invalid_gate_config(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)


def test_direct_coefficient_checkpoint_is_not_silently_loaded():
    model = make_model()
    state = model.state_dict()
    state['scalar_res_flat'] = state.pop('scalar_res_logits_flat')
    with pytest.raises(RuntimeError, match='scalar_res_logits_flat'):
        model.load_state_dict(state)
