"""Fixed-depth recurrence over GPTBase blocks; no exit gate or intermediate loss.

n_layer is the EFFECTIVE depth. For example, n_layer=24, num_loops=4 owns
six blocks and applies them four times. Embeddings and LM head run only once.
"""
from dataclasses import asdict, dataclass

import torch

from nanochat.model.gpt_base import GPTBase, GPTBaseConfig


@dataclass
class GPTLoopConfig(GPTBaseConfig):
    num_loops: int = 4
    window_pattern: str = "L"

    def __post_init__(self):
        if self.num_loops < 1 or self.n_layer < 1 or self.n_layer % self.num_loops:
            raise ValueError("n_layer must be positive and divisible by num_loops >= 1")
        if self.window_pattern != "L":
            raise ValueError("The fixed-loop baseline uses full attention: window_pattern='L'")


class GPTLoop(GPTBase):
    def __init__(self, config, pad_vocab_size_to=64):
        physical = asdict(config)
        physical.pop("num_loops")
        physical["n_layer"] //= config.num_loops
        super().__init__(GPTBaseConfig(**physical), pad_vocab_size_to)
        self.config = config
        self.window_sizes *= config.num_loops

    def _forward_blocks(self, x, cos_sin, kv_cache):
        if kv_cache is not None and kv_cache.n_layers != self.config.n_layer:
            raise ValueError("KV cache needs one slot per effective layer, not per shared block")
        for loop in range(self.config.num_loops):
            for i, block in enumerate(self.transformer.h):
                # We share parameters, not activations or inference caches.
                layer_idx = loop * len(self.transformer.h) + i
                x = block(x, cos_sin, kv_cache, cache_layer_idx=layer_idx)
        return x

    def estimate_flops(self):
        matrices = self.num_scaling_params()["transformer_matrices"]
        return super().estimate_flops() + 6 * (self.config.num_loops - 1) * matrices


class GPTLoopUntied(GPTBase):
    """Independent layers, initialized to the same function as GPTLoop.

Use the same seed/config. Copies are independent parameters after initialization;
ordinary random-initialized GPTBase remains a separate performance reference.
"""
    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        period = self.config.n_layer // self.config.num_loops
        for i in range(period, self.config.n_layer):
            self.transformer.h[i].load_state_dict(self.transformer.h[i % period].state_dict())
