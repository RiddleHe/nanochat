"""
Model registry for selecting between different model architectures.

Usage:
    from nanochat.model_registry import get_model
    ConfigClass, ModelClass = get_model("gpt")        # base GPT
    ConfigClass, ModelClass = get_model("gpt_base")   # vanilla GPT-2 + smear

All variants are registered eagerly at module-import time. Configs that only
flip flags on a base config are subclassed inline and grouped by their model
family below.
"""

from dataclasses import dataclass

# -----------------------------------------------------------------------------
# gpt.py family: full nanochat GPT (value embeds, lambdas, smear, ...)
# -----------------------------------------------------------------------------
from nanochat.model.gpt import GPTConfig, GPT

@dataclass
class GPTNoLambdaConfig(GPTConfig):
    use_lambdas: bool = False

# -----------------------------------------------------------------------------
# gpt_base.py family: vanilla GPT-2 + smear (no value embeds, no lambdas)
# -----------------------------------------------------------------------------
from nanochat.model.gpt_base import GPTBaseConfig, GPTBase

# -----------------------------------------------------------------------------
# Chunk deep-KV family (research/chunk-deep-kv). See research_chunk_deep_kv.md.
# All four share the same idea: early layers get an extra gated attention branch
# over STRICTLY EARLIER chunks. They differ in what the branch reads and how the
# sources are produced.
# -----------------------------------------------------------------------------

@dataclass
class GPTBaseChunkDeepKVConfig(GPTBaseConfig):
    """v1: branch reads previous chunks' FINAL-layer states (processed content).
    Sources come from a no-grad pass-1 forward => ~+41% training FLOPs."""
    chunk_deep_kv: bool = True

@dataclass
class GPTBaseChunkSameKVConfig(GPTBaseConfig):
    """v1 CONTROL: identical machinery, but the branch reads previous chunks'
    SAME-LAYER states. Isolates 'processed content' from 'more visibility'."""
    chunk_same_kv: bool = True

@dataclass
class GPTBaseChunkDeepKVv2Config(GPTBaseConfig):
    """v2: same branch as v1, but the trunk is a single-pass chunk-recurrent
    loop, so the pass-1 tax disappears (~+41% -> ~+8% FLOPs)."""
    chunk_deep_kv: bool = True
    chunk_recurrent: bool = True

@dataclass
class GPTBaseChunkDeepKVv2SlimConfig(GPTBaseConfig):
    """v2-slim: v2 with half the branch layers (2 instead of 4 at d12),
    halving the remaining branch cost."""
    chunk_deep_kv: bool = True
    chunk_recurrent: bool = True
    chunk_kv_frac: float = 0.17  # 2 branch layers at d12

# -----------------------------------------------------------------------------
# Registry: model_type string -> (ConfigClass, ModelClass)
# -----------------------------------------------------------------------------
MODELS = {
    # gpt.py family
    "gpt":              (GPTConfig,             GPT),
    "gpt_nolambda":     (GPTNoLambdaConfig,     GPT),
    # gpt_base.py family
    "gpt_base":         (GPTBaseConfig,         GPTBase),
    # chunk deep-KV family
    "gpt_base_chunk_deep_kv":         (GPTBaseChunkDeepKVConfig,       GPTBase),
    "gpt_base_chunk_same_kv":         (GPTBaseChunkSameKVConfig,       GPTBase),
    "gpt_base_chunk_deep_kv_v2":      (GPTBaseChunkDeepKVv2Config,     GPTBase),
    "gpt_base_chunk_deep_kv_v2_slim": (GPTBaseChunkDeepKVv2SlimConfig, GPTBase),
}


def register(name, config_cls, model_cls):
    """Register a new model variant (e.g. for tests or experiments)."""
    MODELS[name] = (config_cls, model_cls)


def get_model(name="gpt"):
    """Get (ConfigClass, ModelClass) by name."""
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model type '{name}'. Available: {available}")
    return MODELS[name]
