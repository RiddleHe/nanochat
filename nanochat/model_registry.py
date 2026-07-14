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

@dataclass
class SkipAheadDenseConfig(GPTBaseConfig):
    skip_ahead_mode: str = "dense"
    skip_gate_source: str = "current"

@dataclass
class SkipAheadSparseConfig(GPTBaseConfig):
    skip_ahead_mode: str = "sparse"
    skip_gate_source: str = "current"

@dataclass
class SkipAheadDenseX0Config(GPTBaseConfig):
    skip_ahead_mode: str = "dense"
    skip_gate_source: str = "x0"

@dataclass
class SkipAheadSparseX0Config(GPTBaseConfig):
    skip_ahead_mode: str = "sparse"
    skip_gate_source: str = "x0"

# -----------------------------------------------------------------------------
# Registry: model_type string -> (ConfigClass, ModelClass)
# -----------------------------------------------------------------------------
MODELS = {
    # gpt.py family
    "gpt":              (GPTConfig,             GPT),
    "gpt_nolambda":     (GPTNoLambdaConfig,     GPT),
    # gpt_base.py family
    "gpt_base":              (GPTBaseConfig,              GPTBase),
    "skip_ahead_dense":      (SkipAheadDenseConfig,       GPTBase),
    "skip_ahead_sparse":     (SkipAheadSparseConfig,      GPTBase),
    "skip_ahead_dense_x0":   (SkipAheadDenseX0Config,     GPTBase),
    "skip_ahead_sparse_x0":  (SkipAheadSparseX0Config,    GPTBase),
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
