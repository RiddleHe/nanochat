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
class GPTBaseAddInitResConfig(GPTBaseConfig):
    add_init_res: bool = True

@dataclass
class GPTBaseAddInitResDetachConfig(GPTBaseConfig):
    add_init_res: bool = True
    detach_init_value: bool = True

@dataclass
class GPTBaseAddInitVConfig(GPTBaseConfig):
    add_init_v: bool = True

@dataclass
class GPTBaseAddInitResVConfig(GPTBaseConfig):
    add_init_res_v: bool = True

@dataclass
class GPTBaseAddInitResVDetachConfig(GPTBaseConfig):
    add_init_res_v: bool = True
    detach_init_value: bool = True

@dataclass
class GPTBaseAddInitQkvConfig(GPTBaseConfig):
    add_init_qkv: bool = True

@dataclass
class GPTBaseVFromX0Config(GPTBaseConfig):
    v_from_x0: bool = True

@dataclass
class GPTBaseVFromV1Config(GPTBaseConfig):
    v_from_v1: bool = True

@dataclass
class GPTBaseVFromValueEmbConfig(GPTBaseConfig):
    v_from_value_emb: bool = True

@dataclass
class GPTBaseAddInitValueEmbConfig(GPTBaseConfig):
    add_init_value_emb: bool = True

@dataclass
class GPTBaseAddInitValueEmbNanogptConfig(GPTBaseConfig):
    add_init_value_emb_nanogpt: bool = True

@dataclass
class GPTBaseAddInitValueEmbNanogptEvery2Config(GPTBaseConfig):
    add_init_value_emb_nanogpt: bool = True
    on_every_two_layers: bool = True

@dataclass
class GPTBaseVExcludeSelfConfig(GPTBaseConfig):
    v_exclude_self: bool = True

@dataclass
class GPTBaseVFromX0ExcludeSelfConfig(GPTBaseConfig):
    v_from_x0: bool = True
    v_exclude_self: bool = True

# Learnable-coefficient variants: (alpha, beta) per late layer init to (1.0, 0.0).
# At step 0 the model is identical to vanilla; the optimizer learns how much x0 /
# layer-0 v contribution each late layer wants.
@dataclass
class GPTBaseAddInitResLearnConfig(GPTBaseConfig):
    add_init_res: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitVLearnConfig(GPTBaseConfig):
    add_init_v: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitResVLearnConfig(GPTBaseConfig):
    add_init_res_v: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitQkvLearnConfig(GPTBaseConfig):
    add_init_qkv: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitQkvSharedLearnConfig(GPTBaseConfig):
    add_init_qkv_shared: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseVFromX0LearnConfig(GPTBaseConfig):
    v_from_x0: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseVFromV1LearnConfig(GPTBaseConfig):
    v_from_v1: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseVFromValueEmbLearnConfig(GPTBaseConfig):
    v_from_value_emb: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitValueEmbLearnConfig(GPTBaseConfig):
    add_init_value_emb: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseVFromV1ScalingVecConfig(GPTBaseConfig):
    v_from_v1: bool = True
    learn_init_scaling_vec: bool = True

@dataclass
class GPTBaseVScaleLearnConfig(GPTBaseConfig):
    v_scale_learn: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitProjLearnConfig(GPTBaseConfig):
    add_init_proj: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitPreNormAttnOnlyLearnConfig(GPTBaseConfig):
    add_init_pre_norm_attn_only: bool = True
    learn_init_coeffs: bool = True

@dataclass
class GPTBaseAddInitPreNormMlpOnlyLearnConfig(GPTBaseConfig):
    add_init_pre_norm_mlp_only: bool = True
    learn_init_coeffs: bool = True

# -----------------------------------------------------------------------------
# Standalone variants (each has its own model class in its own file)
# -----------------------------------------------------------------------------
from nanochat.model.gpt_attn_res import GPTAttnResConfig, GPTAttnRes
from nanochat.model.gpt_attn_res_sink import GPTAttnResSinkConfig, GPTAttnResSink

# -----------------------------------------------------------------------------
# Registry: model_type string -> (ConfigClass, ModelClass)
# -----------------------------------------------------------------------------
MODELS = {
    # gpt.py family
    "gpt":              (GPTConfig,             GPT),
    "gpt_nolambda":     (GPTNoLambdaConfig,     GPT),
    # gpt_base.py family
    "gpt_base":                (GPTBaseConfig,             GPTBase),
    "gpt_base_add_init_res":         (GPTBaseAddInitResConfig,         GPTBase),
    "gpt_base_add_init_res_detach":  (GPTBaseAddInitResDetachConfig,   GPTBase),
    "gpt_base_add_init_v":            (GPTBaseAddInitVConfig,           GPTBase),
    "gpt_base_add_init_res_v":        (GPTBaseAddInitResVConfig,        GPTBase),
    "gpt_base_add_init_res_v_detach": (GPTBaseAddInitResVDetachConfig,  GPTBase),
    "gpt_base_add_init_qkv":          (GPTBaseAddInitQkvConfig,         GPTBase),
    "gpt_base_v_from_x0":             (GPTBaseVFromX0Config,            GPTBase),
    "gpt_base_v_from_x0_learn":       (GPTBaseVFromX0LearnConfig,       GPTBase),
    "gpt_base_v_from_v1":             (GPTBaseVFromV1Config,            GPTBase),
    "gpt_base_v_from_v1_learn":       (GPTBaseVFromV1LearnConfig,       GPTBase),
    "gpt_base_v_from_value_emb":      (GPTBaseVFromValueEmbConfig,      GPTBase),
    "gpt_base_v_from_value_emb_learn":(GPTBaseVFromValueEmbLearnConfig, GPTBase),
    "gpt_base_add_init_value_emb":    (GPTBaseAddInitValueEmbConfig,    GPTBase),
    "gpt_base_add_init_value_emb_learn":(GPTBaseAddInitValueEmbLearnConfig, GPTBase),
    "gpt_base_add_init_value_emb_nanogpt":(GPTBaseAddInitValueEmbNanogptConfig, GPTBase),
    "gpt_base_add_init_value_emb_nanogpt_every2":(GPTBaseAddInitValueEmbNanogptEvery2Config, GPTBase),
    "gpt_base_v_from_v1_scaling_vec": (GPTBaseVFromV1ScalingVecConfig,  GPTBase),
    "gpt_base_v_scale_learn":         (GPTBaseVScaleLearnConfig,        GPTBase),
    "gpt_base_v_exclude_self":        (GPTBaseVExcludeSelfConfig,       GPTBase),
    "gpt_base_v_from_x0_exclude_self": (GPTBaseVFromX0ExcludeSelfConfig, GPTBase),
    # learnable-coefficient variants
    "gpt_base_add_init_res_learn":    (GPTBaseAddInitResLearnConfig,    GPTBase),
    "gpt_base_add_init_v_learn":      (GPTBaseAddInitVLearnConfig,      GPTBase),
    "gpt_base_add_init_res_v_learn":  (GPTBaseAddInitResVLearnConfig,   GPTBase),
    "gpt_base_add_init_qkv_learn":    (GPTBaseAddInitQkvLearnConfig,    GPTBase),
    "gpt_base_add_init_qkv_shared_learn": (GPTBaseAddInitQkvSharedLearnConfig, GPTBase),
    "gpt_base_add_init_proj_learn":   (GPTBaseAddInitProjLearnConfig,   GPTBase),
    "gpt_base_add_init_pre_norm_attn_only_learn": (GPTBaseAddInitPreNormAttnOnlyLearnConfig, GPTBase),
    "gpt_base_add_init_pre_norm_mlp_only_learn":  (GPTBaseAddInitPreNormMlpOnlyLearnConfig,  GPTBase),
    # standalone variants
    "attn_res":      (GPTAttnResConfig,     GPTAttnRes),
    "attn_res_sink": (GPTAttnResSinkConfig, GPTAttnResSink),
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
