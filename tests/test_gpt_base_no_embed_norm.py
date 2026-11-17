"""Verify the embedding-boundary ablation without training or GPU execution."""

import torch

import nanochat.flash_attention as attention
import nanochat.model.gpt_base as base_module
from nanochat.model_registry import get_model


def test_raw_embedding_boundary_and_baseline_equivalence(monkeypatch):
    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(base_module, "COMPUTE_DTYPE", torch.float32)
    Config, Model = get_model("gpt_base_no_embed_norm")
    assert Config is base_module.GPTBaseConfig
    assert issubclass(Model, base_module.GPTBase)
    config = Config(sequence_len=16, vocab_size=31, n_layer=3,
                    n_head=4, n_kv_head=2, n_embd=32, window_pattern="L")
    models = []
    for cls in (base_module.GPTBase, Model):
        torch.manual_seed(42)
        with torch.device("meta"):
            model = cls(config)
        model.to_empty(device="cpu")
        model.init_weights()
        models.append(model)
    baseline, variant = models
    assert baseline.state_dict().keys() == variant.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, variant.state_dict()[name], atol=0, rtol=0)
    assert baseline.num_scaling_params() == variant.num_scaling_params()
    assert baseline.estimate_flops() == variant.estimate_flops()

    # Nonzero sublayer outputs expose changes to residual-stream magnitude.
    with torch.no_grad():
        for block in baseline.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
    variant.load_state_dict(baseline.state_dict())
    tokens = torch.randint(config.vocab_size, (2, 6))
    captured = {}

    def capture_block(module, args):
        captured["x"] = args[0].detach().clone()

    def capture_projection(module, args):
        captured["q_input"] = args[0].detach().clone()

    hooks = [variant.transformer.h[0].register_forward_pre_hook(capture_block),
             variant.transformer.h[0].attn.c_q.register_forward_pre_hook(capture_projection)]
    raw_logits = variant(tokens)
    for hook in hooks:
        hook.remove()
    raw_embeddings = variant.transformer.wte(tokens)
    torch.testing.assert_close(captured["x"], raw_embeddings, atol=0, rtol=0)
    torch.testing.assert_close(captured["q_input"], base_module.norm(raw_embeddings), atol=0, rtol=0)
    assert not torch.allclose(raw_logits, baseline(tokens))

    # Restoring only the removed boundary operation must recover the baseline,
    # including embedding gradients, final norm, logits, and masked loss.
    def restore_embedding_norm(module, args):
        x = base_module.norm(args[0])
        return (x, *args[1:])

    hook = variant.transformer.h[0].register_forward_pre_hook(restore_embedding_norm)
    targets = tokens.roll(-1, dims=1)
    targets[0, -1] = -1
    try:
        torch.testing.assert_close(variant(tokens), baseline(tokens), atol=0, rtol=0)
        for reduction in ("none", "sum", "mean"):
            torch.testing.assert_close(
                variant(tokens, targets, loss_reduction=reduction),
                baseline(tokens, targets, loss_reduction=reduction), atol=0, rtol=0)
        baseline(tokens, targets).backward()
        variant(tokens, targets).backward()
        for (name, parameter), (_, other) in zip(baseline.named_parameters(), variant.named_parameters()):
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0)
    finally:
        hook.remove()


def test_raw_embedding_cached_logits(monkeypatch):
    from nanochat.engine import KVCache

    monkeypatch.setattr(attention, "USE_FA3", False)
    monkeypatch.setattr(base_module, "COMPUTE_DTYPE", torch.float32)
    Config, Model = get_model("gpt_base_no_embed_norm")
    config = Config(sequence_len=16, vocab_size=31, n_layer=2,
                    n_head=2, n_kv_head=1, n_embd=32, window_pattern="L")
    torch.manual_seed(42)
    model = Model(config)
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
    tokens = torch.randint(config.vocab_size, (2, 6))
    cache = KVCache(batch_size=2, num_heads=config.n_kv_head,
                    seq_len=config.sequence_len, head_dim=config.n_embd // config.n_head,
                    num_layers=config.n_layer, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        expected = model(tokens)
        actual = torch.cat([model(tokens[:, :3], kv_cache=cache),
                            model(tokens[:, 3:4], kv_cache=cache),
                            model(tokens[:, 4:], kv_cache=cache)], dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert cache.get_pos() == tokens.size(1)
