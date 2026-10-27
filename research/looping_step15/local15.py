"""Layer-local interventions: one LoRA-wrapped MLP expansion, or one scaled MLP branch in one round."""
from types import MethodType
from shared15 import VARIANTS, EXPECTED_PARAMETERS, PROTOCOL, step1_int


def attach_local(model, layer, variant, check_count=True):
    """Wrap only block[layer].mlp.c_fc. Base stays frozen; only adapter A/B train."""
    model.requires_grad_(False)
    rank, independent = VARIANTS[variant]
    mlp = model.transformer.h[layer].mlp
    assert not isinstance(mlp.c_fc, step1_int.LoopLoRA), 'Already wrapped'
    loops = model.config.num_loops
    adapter = step1_int.LoopLoRA(mlp.c_fc, rank, independent,
                                 PROTOCOL['adapter_seed'] + 100 * layer + 4, loops)
    mlp.c_fc = adapter
    adapter.calls = []

    def blocks(self, x, cos_sin, kv_cache):
        if kv_cache is not None:
            raise ValueError('Full-sequence forwards only')
        for loop in range(self.config.num_loops):
            adapter.visit = loop
            for i, block in enumerate(self.transformer.h):
                x = block(x, cos_sin, None, cache_layer_idx=loop * len(self.transformer.h) + i)
        return x

    model._forward_blocks = MethodType(blocks, model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert all(f'transformer.h.{layer}.mlp.c_fc.' in n and ('.A.' in n or '.B.' in n) for n, _ in named), named
    if check_count:
        assert sum(p.numel() for _, p in named) == EXPECTED_PARAMETERS[variant]
    wrapped = [m for m in model.modules() if isinstance(m, step1_int.LoopLoRA)]
    assert wrapped == [adapter], 'Exactly one location may be adapted'
    return adapter, named


class LocalScale:
    """Multiply block[layer].mlp output (before residual add) by `scale` in one round only."""

    def __init__(self, model, layer, loop, scale):
        self.loop, self.scale, self.count, self.scaled = loop, scale, 0, 0
        self.handle = model.transformer.h[layer].mlp.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        visit = self.count
        self.count += 1
        if visit == self.loop:
            self.scaled += 1
            return output * self.scale
        return output

    def reset(self):
        if self.count:
            assert self.count == 4 and self.scaled == 1, (self.count, self.scaled)
        self.count = self.scaled = 0

    def close(self):
        self.reset()
        self.handle.remove()
