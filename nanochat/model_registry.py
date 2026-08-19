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
class SkipAheadConfig(GPTBaseConfig):
    skip_gate_enabled: bool = True
    skip_gate_source: str = "current"
    skip_gate_type: str = "sigmoid"

@dataclass
class SkipAheadX0Config(GPTBaseConfig):
    skip_gate_enabled: bool = True
    skip_gate_source: str = "x0"
    skip_gate_type: str = "sigmoid"

@dataclass
class SkipAheadTanhConfig(GPTBaseConfig):
    skip_gate_enabled: bool = True
    skip_gate_source: str = "current"

@dataclass
class SkipAheadX0TanhConfig(GPTBaseConfig):
    skip_gate_enabled: bool = True
    skip_gate_source: str = "x0"

@dataclass
class SkipAheadSqrtConfig(GPTBaseConfig):
    skip_gate_enabled: bool = True
    skip_gate_source: str = "current"
    skip_gate_type: str = "sqrt"
    skip_gate_l2_weight: float = 0.0
    skip_gate_recovery_weight: float = 0.01
    skip_gate_recovery_margin: float = 3.0

# -----------------------------------------------------------------------------
# Registry: model_type string -> (ConfigClass, ModelClass)
# -----------------------------------------------------------------------------
MODELS = {
    # gpt.py family
    "gpt":              (GPTConfig,             GPT),
    "gpt_nolambda":     (GPTNoLambdaConfig,     GPT),
    # gpt_base.py family
    "gpt_base":              (GPTBaseConfig,        GPTBase),
    "skip_ahead":            (SkipAheadConfig,       GPTBase),
    "skip_ahead_x0":         (SkipAheadX0Config,     GPTBase),
    "skip_ahead_tanh":       (SkipAheadTanhConfig,   GPTBase),
    "skip_ahead_x0_tanh":    (SkipAheadX0TanhConfig, GPTBase),
    "skip_ahead_sqrt":       (SkipAheadSqrtConfig,   GPTBase),
    # Historical names retained only so existing continuous-gate checkpoints load.
    "skip_ahead_dense":         (SkipAheadConfig,       GPTBase),
    "skip_ahead_dense_x0":      (SkipAheadX0Config,     GPTBase),
    "skip_ahead_dense_tanh":    (SkipAheadTanhConfig,   GPTBase),
    "skip_ahead_dense_x0_tanh": (SkipAheadX0TanhConfig, GPTBase),
    "skip_ahead_dense_sqrt":    (SkipAheadSqrtConfig,   GPTBase),
}

REMOVED_MODEL_TYPES = {"skip_ahead_sparse", "skip_ahead_sparse_x0"}


def register(name, config_cls, model_cls):
    """Register a new model variant (e.g. for tests or experiments)."""
    MODELS[name] = (config_cls, model_cls)


def get_model(name="gpt"):
    """Get (ConfigClass, ModelClass) by name."""
    if name in REMOVED_MODEL_TYPES:
        raise ValueError(
            f"Model type '{name}' was removed because hard-threshold sparse routing "
            "used a straight-through surrogate gradient"
        )
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model type '{name}'. Available: {available}")
    return MODELS[name]
