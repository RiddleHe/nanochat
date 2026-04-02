"""
Model registry for selecting between different model architectures.

Usage:
    from nanochat.model_registry import get_model
    ConfigClass, ModelClass = get_model("gpt")          # base GPT
    ConfigClass, ModelClass = get_model("linear_attn")   # variant
"""

from dataclasses import dataclass
from nanochat.gpt import GPTConfig, GPT

@dataclass
class GPTNoLambdaConfig(GPTConfig):
    use_lambdas: bool = False

@dataclass
class GPTXsaPreVeConfig(GPTConfig):
    xsa_mode: str = "pre_ve"

@dataclass
class GPTXsaPostVeConfig(GPTConfig):
    xsa_mode: str = "post_ve"

# Maps model_type string -> (ConfigClass, ModelClass)
MODELS = {
    "gpt": (GPTConfig, GPT),
    "gpt_nolambda": (GPTNoLambdaConfig, GPT),
    "gpt_xsa_pre_ve": (GPTXsaPreVeConfig, GPT),
    "gpt_xsa_post_ve": (GPTXsaPostVeConfig, GPT),
}

def register(name, config_cls, model_cls):
    """Register a new model variant."""
    MODELS[name] = (config_cls, model_cls)

def get_model(name="gpt"):
    """Get (ConfigClass, ModelClass) by name. Lazily imports non-default variants."""
    if name not in MODELS:
        _register_variants()
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model type '{name}'. Available: {available}")
    return MODELS[name]

def _register_variants():
    """Lazy import of variant model files so we don't load them unless needed."""
    from nanochat.gpt_attn_res import GPTAttnResConfig, GPTAttnRes
    register("attn_res", GPTAttnResConfig, GPTAttnRes)
    from nanochat.gpt_attn_res_input_query import GPTAttnResInputQueryConfig, GPTAttnResInputQuery
    register("attn_res_input_query", GPTAttnResInputQueryConfig, GPTAttnResInputQuery)
    from nanochat.gpt_gated_attn_res import GPTGatedAttnResConfig, GPTGatedAttnRes
    register("gated_attn_res", GPTGatedAttnResConfig, GPTGatedAttnRes)
    from nanochat.gpt_depth_v import GPTDepthVConfig, GPTDepthV
    register("depth_v", GPTDepthVConfig, GPTDepthV)
