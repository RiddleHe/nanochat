"""Profile final-prompt-token attention in ordinary and bottleneck controls.

For direct-fact and visitor-register prompts, run all selected single-token
entities under two conditions:

1. ordinary: native attention and values everywhere;
2. bottleneck_control: intermediate post-entity prompt queries and generated
   queries lose only the entity token's value contribution in every layer,
   while the final prompt query remains unrestricted.

Native fused SDPA remains the model's inference path. Because fused SDPA does
not expose attention probabilities, the profiler separately recomputes the
final prompt query's QK softmax for observation. It also records each prompt
token's weighted value contribution after the attention output projection,
allowing a paired ordinary-versus-bottleneck contribution-vector comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from qwen_entity_attention_ablation import (
    ENTITY_SETS,
    MODEL,
    TEMPLATES,
    EncodedPrompt,
    encode_prompt,
    normalize_text,
    parse_ints,
    select,
)


ATTENTION_IMPLEMENTATION = "entity_prompt_profile"
CONDITIONS = ("ordinary", "bottleneck_control")
DEFAULT_TEMPLATE_IDS = (0, 3)


@dataclass
class ProfileState:
    condition: str | None = None
    entity_position: int | None = None
    readout_position: int | None = None
    prompt_length: int | None = None
    attention_by_layer: dict[int, torch.Tensor] | None = None
    contribution_by_layer: dict[int, torch.Tensor] | None = None
    intermediate_applications: int = 0
    generated_applications: int = 0


PROFILE = ProfileState()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch, key_value_heads * n_rep, seq_len, head_dim
    )


def explicit_profile_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Native SDPA inference plus final-query attention observation."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if PROFILE.condition is not None:
        if (
            PROFILE.entity_position is None
            or PROFILE.readout_position is None
            or PROFILE.prompt_length is None
            or PROFILE.attention_by_layer is None
            or PROFILE.contribution_by_layer is None
        ):
            raise RuntimeError("active profile is missing prompt boundaries")

        query_length = query.shape[-2]
        key_length = key.shape[-2]
        query_start = key_length - query_length
        query_positions = list(range(query_start, query_start + query_length))

        if (
            query_length == PROFILE.prompt_length
            and key_length == PROFILE.prompt_length
        ):
            if module.layer_idx in PROFILE.attention_by_layer:
                raise RuntimeError(
                    f"layer {module.layer_idx} prompt profile captured twice"
                )
            readout_local = PROFILE.readout_position - query_start
            if readout_local != query_length - 1:
                raise RuntimeError("readout is not the final prefill query")

            # Fused SDPA does not expose its probabilities. Recompute only the
            # final query's QK softmax for observation; the actual model output
            # below still comes from native SDPA.
            key_states = _repeat_kv(key, module.num_key_value_groups)
            value_states = _repeat_kv(value, module.num_key_value_groups)
            final_query = query[:, :, readout_local : readout_local + 1, :]
            final_scores = torch.matmul(
                final_query, key_states.transpose(2, 3)
            ) * scaling
            if attention_mask is not None:
                final_mask = attention_mask[
                    :, :, readout_local : readout_local + 1, : key_length
                ]
                final_scores = final_scores + final_mask
            final_weights = F.softmax(
                final_scores, dim=-1, dtype=torch.float32
            ).to(query.dtype)[:, :, 0, :]
            PROFILE.attention_by_layer[module.layer_idx] = (
                final_weights[0].detach().float().cpu()
            )

            # Per-token contribution to the final attention output. The output
            # projection is linear, so applying it to every token contribution
            # separately produces vectors whose sum is the projected output.
            weighted_values = final_weights.unsqueeze(-1) * value_states
            token_contributions = weighted_values.permute(0, 2, 1, 3)
            token_contributions = token_contributions.reshape(
                token_contributions.shape[0],
                token_contributions.shape[1],
                -1,
            )
            projected = F.linear(
                token_contributions,
                module.o_proj.weight,
                bias=None,
            )
            PROFILE.contribution_by_layer[module.layer_idx] = (
                projected[0].detach().float().cpu()
            )

        affected_queries: list[int] = []
        if PROFILE.condition == "bottleneck_control":
            intermediate_queries = [
                local_index
                for local_index, position in enumerate(query_positions)
                if PROFILE.entity_position
                < position
                < PROFILE.readout_position
            ]
            generated_queries = [
                local_index
                for local_index, position in enumerate(query_positions)
                if position >= PROFILE.prompt_length
            ]
            affected_queries = sorted(
                set(intermediate_queries + generated_queries)
            )
            if affected_queries:
                zero_value = value.clone()
                zero_value[:, :, PROFILE.entity_position, :] = 0
                zero_output, _ = sdpa_attention_forward(
                    module,
                    query,
                    key,
                    zero_value,
                    attention_mask,
                    scaling=scaling,
                    dropout=dropout,
                    **kwargs,
                )
                PROFILE.intermediate_applications += len(intermediate_queries)
                PROFILE.generated_applications += len(generated_queries)
                if len(affected_queries) == query_length:
                    return zero_output, None

                native_output, _ = sdpa_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling=scaling,
                    dropout=dropout,
                    **kwargs,
                )
                native_output = native_output.clone()
                native_output[:, affected_queries, :, :] = zero_output[
                    :, affected_queries, :, :
                ]
                return native_output, None
        elif PROFILE.condition != "ordinary":
            raise RuntimeError(f"unknown profile condition {PROFILE.condition!r}")

    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        **kwargs,
    )


@contextmanager
def profile_condition(condition: str, encoded: EncodedPrompt):
    if PROFILE.condition is not None:
        raise RuntimeError("nested attention profiles are unsupported")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    prompt_length = len(encoded.ids)
    readout_position = prompt_length - 1
    if not 0 <= encoded.entity_position < readout_position:
        raise ValueError("expected entity position before the final prompt token")

    PROFILE.condition = condition
    PROFILE.entity_position = encoded.entity_position
    PROFILE.readout_position = readout_position
    PROFILE.prompt_length = prompt_length
    PROFILE.attention_by_layer = {}
    PROFILE.contribution_by_layer = {}
    PROFILE.intermediate_applications = 0
    PROFILE.generated_applications = 0
    try:
        yield PROFILE
    finally:
        PROFILE.condition = None
        PROFILE.entity_position = None
        PROFILE.readout_position = None
        PROFILE.prompt_length = None
        PROFILE.attention_by_layer = None
        PROFILE.contribution_by_layer = None
        PROFILE.intermediate_applications = 0
        PROFILE.generated_applications = 0


def load_model(model_name: str, device: torch.device):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register(
        ATTENTION_IMPLEMENTATION, explicit_profile_attention_forward
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(device).eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(f"cannot locate transformer blocks for {model_name}")
    return model, tokenizer, len(model.model.layers)


def visible_completion(text: str) -> str:
    return normalize_text(text)


@torch.inference_mode()
def generate_and_profile(
    model,
    tokenizer,
    encoded: EncodedPrompt,
    condition: str,
    device: torch.device,
    n_layers: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with profile_condition(condition, encoded) as state:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_token_id,
        )
        generated_ids = (
            generated.sequences[0, len(encoded.ids):].detach().cpu().tolist()
        )
        if not generated_ids:
            raise RuntimeError("generation produced no continuation")

        expected_layers = set(range(n_layers))
        if set(state.attention_by_layer or {}) != expected_layers:
            raise RuntimeError("did not capture attention for exactly every layer")
        if set(state.contribution_by_layer or {}) != expected_layers:
            raise RuntimeError("did not capture contributions for every layer")

        expected_intermediate = 0
        expected_generated = 0
        if condition == "bottleneck_control":
            expected_intermediate = n_layers * (
                len(encoded.ids) - encoded.entity_position - 2
            )
            expected_generated = n_layers * max(len(generated_ids) - 1, 0)
        observed = (
            state.intermediate_applications,
            state.generated_applications,
        )
        expected = (expected_intermediate, expected_generated)
        if observed != expected:
            raise RuntimeError(
                "entity policy application mismatch: "
                f"observed={observed}, expected={expected}"
            )

        attention = torch.stack(
            [state.attention_by_layer[layer] for layer in range(n_layers)]
        )
        contributions = torch.stack(
            [state.contribution_by_layer[layer] for layer in range(n_layers)]
        )

    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return {
        "generated_token_ids": generated_ids,
        "completion": completion,
        "normalized_completion": visible_completion(completion),
        "attention_by_head": attention,
        "token_contributions": contributions,
        "intermediate_applications": expected_intermediate,
        "generated_applications": expected_generated,
    }


def token_label(tokenizer, token_id: int, is_entity: bool) -> str:
    if is_entity:
        return "<ENTITY>"
    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if text == "\n":
        return r"\n"
    if text == "\t":
        return r"\t"
    if text == " ":
        return "<space>"
    return text.replace("\n", r"\n").replace("\t", r"\t")


def validate_alignment(cases: list[EncodedPrompt]) -> None:
    if not cases:
        raise ValueError("cannot align an empty case list")
    reference = cases[0]
    for encoded in cases[1:]:
        if len(encoded.ids) != len(reference.ids):
            raise RuntimeError(
                f"{reference.template_name}: prompt lengths vary across entities"
            )
        if encoded.entity_position != reference.entity_position:
            raise RuntimeError(
                f"{reference.template_name}: entity positions vary"
            )
        for position, (left, right) in enumerate(
            zip(reference.ids, encoded.ids)
        ):
            if position == reference.entity_position:
                continue
            if left != right:
                raise RuntimeError(
                    f"{reference.template_name}: token mismatch at position "
                    f"{position} outside the entity"
                )


def heatmap(
    axis,
    values: torch.Tensor,
    labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
):
    image = axis.imshow(
        values.numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    axis.set_title(title)
    axis.set_xlabel("Prompt token")
    axis.set_ylabel("Layer")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axis.set_yticks(range(0, values.shape[0], 4))
    return image


def plot_attention_comparison(
    output_path: Path,
    template_name: str,
    token_labels: list[str],
    ordinary: torch.Tensor,
    bottleneck: torch.Tensor,
) -> None:
    difference = bottleneck - ordinary
    shared_max = float(torch.maximum(ordinary.max(), bottleneck.max()))
    diff_max = max(float(difference.abs().max()), 1e-8)
    width = max(18.0, 0.48 * len(token_labels) * 3)
    figure, axes = plt.subplots(1, 3, figsize=(width, 8), constrained_layout=True)
    first = heatmap(
        axes[0], ordinary, token_labels, "Ordinary", 0.0, shared_max, "viridis"
    )
    heatmap(
        axes[1], bottleneck, token_labels, "Final-token bottleneck", 0.0,
        shared_max, "viridis"
    )
    third = heatmap(
        axes[2], difference, token_labels, "Bottleneck − ordinary",
        -diff_max, diff_max, "coolwarm"
    )
    figure.colorbar(first, ax=axes[:2], shrink=0.85, label="Mean max-head attention")
    figure.colorbar(third, ax=axes[2], shrink=0.85, label="Attention difference")
    figure.suptitle(
        f"Final prompt token attention — {template_name.replace('_', ' ')}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_contribution_comparison(
    output_path: Path,
    template_name: str,
    token_labels: list[str],
    ordinary_norm: torch.Tensor,
    bottleneck_norm: torch.Tensor,
    delta_vector_norm: torch.Tensor,
) -> None:
    shared_max = float(torch.maximum(ordinary_norm.max(), bottleneck_norm.max()))
    delta_max = max(float(delta_vector_norm.max()), 1e-8)
    width = max(18.0, 0.48 * len(token_labels) * 3)
    figure, axes = plt.subplots(1, 3, figsize=(width, 8), constrained_layout=True)
    first = heatmap(
        axes[0], ordinary_norm, token_labels, "Ordinary", 0.0,
        shared_max, "magma"
    )
    heatmap(
        axes[1], bottleneck_norm, token_labels, "Final-token bottleneck",
        0.0, shared_max, "magma"
    )
    third = heatmap(
        axes[2], delta_vector_norm, token_labels,
        "‖Bottleneck contribution − ordinary contribution‖₂",
        0.0, delta_max, "inferno"
    )
    figure.colorbar(first, ax=axes[:2], shrink=0.85, label="Contribution-vector norm")
    figure.colorbar(third, ax=axes[2], shrink=0.85, label="Paired vector-difference norm")
    figure.suptitle(
        f"Weighted value contributions — {template_name.replace('_', ' ')}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--entity-set", choices=sorted(ENTITY_SETS), default="diverse100")
    parser.add_argument("--entity-ids", type=parse_ints)
    parser.add_argument("--template-ids", type=parse_ints, default=list(DEFAULT_TEMPLATE_IDS))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--out-dir")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    from types import SimpleNamespace
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    torch.manual_seed(1234)
    batch, query_heads, kv_heads, seq_len, head_dim = 1, 4, 2, 5, 3
    module = SimpleNamespace(
        num_key_value_groups=query_heads // kv_heads,
        layer_idx=0,
        training=False,
        o_proj=torch.nn.Linear(query_heads * head_dim, query_heads * head_dim, bias=False),
    )
    query = torch.randn(batch, query_heads, seq_len, head_dim)
    key = torch.randn(batch, kv_heads, seq_len, head_dim)
    value = torch.randn(batch, kv_heads, seq_len, head_dim)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    scaling = head_dim ** -0.5

    eager_output, expected_weights = eager_attention_forward(
        module, query, key, value, mask, scaling=scaling, dropout=0.0
    )
    expected_output, _ = sdpa_attention_forward(
        module, query, key, value, mask, scaling=scaling, dropout=0.0
    )
    observed_output, _ = explicit_profile_attention_forward(
        module, query, key, value, mask, scaling=scaling, dropout=0.0
    )
    torch.testing.assert_close(observed_output, expected_output)
    torch.testing.assert_close(observed_output, eager_output)

    encoded = EncodedPrompt(
        template_id=0,
        template_name="test",
        entity_id=0,
        entity="entity",
        entity_group="test",
        prompt="test",
        ids=list(range(seq_len)),
        tokens=[str(index) for index in range(seq_len)],
        entity_position=1,
    )
    with profile_condition("ordinary", encoded) as state:
        profiled_output, _ = explicit_profile_attention_forward(
            module, query, key, value, mask, scaling=scaling, dropout=0.0
        )
        torch.testing.assert_close(profiled_output, expected_output)
        torch.testing.assert_close(
            state.attention_by_layer[0], expected_weights[0, :, -1, :]
        )

    zero_value = value.clone()
    zero_value[:, :, 1, :] = 0
    zero_output, _ = sdpa_attention_forward(
        module, query, key, zero_value, mask, scaling=scaling, dropout=0.0
    )
    expected_bottleneck = expected_output.clone()
    affected = [2, 3]
    expected_bottleneck[:, affected, :, :] = zero_output[:, affected, :, :]
    with profile_condition("bottleneck_control", encoded) as state:
        bottleneck_output, _ = explicit_profile_attention_forward(
            module, query, key, value, mask, scaling=scaling, dropout=0.0
        )
        torch.testing.assert_close(bottleneck_output, expected_bottleneck)
        if state.intermediate_applications != len(affected):
            raise AssertionError("bottleneck self-test application count mismatch")
    print("self-test passed", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    selected_templates = select(TEMPLATES, args.template_ids, "template")
    selected_entities = select(
        ENTITY_SETS[args.entity_set], args.entity_ids, "entity"
    )
    if {template.template_id for _, template in selected_templates} != {0, 3}:
        raise ValueError("this profiler requires direct_fact and visitor_register")
    if len(selected_entities) != 100:
        raise ValueError("this experiment requires exactly 100 entities")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.out_dir or f"results/qwen_prompt_attention_profile_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    model, tokenizer, n_layers = load_model(args.model, device)
    if n_layers != 36:
        raise RuntimeError(f"expected 36 layers, found {n_layers}")

    template_cases: dict[str, list[EncodedPrompt]] = {}
    for _, template in selected_templates:
        cases = [
            encode_prompt(
                tokenizer, template, entity_id, entity, entity_group
            )
            for entity_id, (entity, entity_group) in selected_entities
        ]
        validate_alignment(cases)
        template_cases[template.name] = cases

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "model": args.model,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "n_layers": n_layers,
        "conditions": list(CONDITIONS),
        "templates": [
            {"template_id": template.template_id, "name": template.name, "text": template.text}
            for _, template in selected_templates
        ],
        "entity_set": args.entity_set,
        "n_entities": len(selected_entities),
        "max_new_tokens": args.max_new_tokens,
        "attention_statistic": (
            "post-softmax final-prompt-query attention; max over heads per "
            "entity, then arithmetic mean over all 100 entities"
        ),
        "attention_difference": "bottleneck_control minus ordinary",
        "value_statistic": (
            "per-token final-query weighted value contribution after o_proj; "
            "paired L2 vector difference before averaging over entities"
        ),
        "ordinary_policy": "native attention and values for every query",
        "bottleneck_policy": (
            "intermediate post-entity prompt queries and generated queries "
            "lose only attention_weight(query, entity) * value(entity) in "
            "every layer; final prompt query remains unrestricted; weights "
            "are not renormalized"
        ),
        "accuracy_json": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    all_profiles: dict[str, Any] = {}
    generation_path = output_dir / "generations.jsonl"
    with generation_path.open("x") as generation_file:
        for template_name, cases in template_cases.items():
            print(f"template {template_name}: {len(cases)} entities", flush=True)
            attention_lists = {condition: [] for condition in CONDITIONS}
            contribution_norm_lists = {condition: [] for condition in CONDITIONS}
            contribution_delta_lists = []
            entity_ids = []

            reference = cases[0]
            labels = [
                token_label(
                    tokenizer,
                    token_id,
                    position == reference.entity_position,
                )
                for position, token_id in enumerate(reference.ids)
            ]

            for case_index, encoded in enumerate(cases, start=1):
                paired_contributions = {}
                visible_by_condition = {}
                for condition in CONDITIONS:
                    result = generate_and_profile(
                        model,
                        tokenizer,
                        encoded,
                        condition,
                        device,
                        n_layers,
                        args.max_new_tokens,
                    )
                    attention = result.pop("attention_by_head")
                    contributions = result.pop("token_contributions")
                    attention_lists[condition].append(attention)
                    contribution_norm_lists[condition].append(
                        torch.linalg.vector_norm(contributions, dim=-1)
                    )
                    paired_contributions[condition] = contributions
                    visible_by_condition[condition] = result[
                        "normalized_completion"
                    ]

                    row = {
                        "condition": condition,
                        "template_id": encoded.template_id,
                        "template_name": encoded.template_name,
                        "entity_id": encoded.entity_id,
                        "entity": encoded.entity,
                        "entity_group": encoded.entity_group,
                        "prompt": encoded.prompt,
                        "prompt_token_ids": encoded.ids,
                        "prompt_tokens": encoded.tokens,
                        "entity_position": encoded.entity_position,
                        "readout_position": len(encoded.ids) - 1,
                        "prompt_length": len(encoded.ids),
                        "intermediate_attention_ablation_applications": result.pop(
                            "intermediate_applications"
                        ),
                        "generated_attention_ablation_applications": result.pop(
                            "generated_applications"
                        ),
                        **result,
                    }
                    generation_file.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    generation_file.flush()

                contribution_delta_lists.append(
                    torch.linalg.vector_norm(
                        paired_contributions["bottleneck_control"]
                        - paired_contributions["ordinary"],
                        dim=-1,
                    )
                )
                entity_ids.append(encoded.entity_id)
                print(
                    f"  {case_index:03d}/{len(cases)} {encoded.entity}: "
                    f"ordinary={visible_by_condition['ordinary']!r}; "
                    "bottleneck="
                    f"{visible_by_condition['bottleneck_control']!r}",
                    flush=True,
                )

            condition_profiles = {}
            for condition in CONDITIONS:
                attention_by_head = torch.stack(attention_lists[condition])
                if (
                    attention_by_head.ndim != 4
                    or attention_by_head.shape[0] != len(cases)
                    or attention_by_head.shape[1] != n_layers
                    or attention_by_head.shape[-1] != len(reference.ids)
                ):
                    raise RuntimeError("unexpected attention profile shape")
                head_sums = attention_by_head.sum(dim=-1)
                torch.testing.assert_close(
                    head_sums,
                    torch.ones_like(head_sums),
                    atol=2e-3,
                    rtol=2e-3,
                )
                if not torch.isfinite(attention_by_head).all():
                    raise RuntimeError("non-finite attention probability")
                if attention_by_head.min() < 0 or attention_by_head.max() > 1:
                    raise RuntimeError("attention probability outside [0, 1]")
                condition_profiles[condition] = {
                    "attention_by_head": attention_by_head,
                    "attention_head_max": attention_by_head.max(dim=2).values,
                    "contribution_norm": torch.stack(
                        contribution_norm_lists[condition]
                    ),
                }

            contribution_delta = torch.stack(contribution_delta_lists)
            all_profiles[template_name] = {
                "entity_ids": torch.tensor(entity_ids),
                "token_ids_reference": torch.tensor(reference.ids),
                "token_labels": labels,
                "entity_position": reference.entity_position,
                "readout_position": len(reference.ids) - 1,
                "conditions": condition_profiles,
                "contribution_delta_norm": contribution_delta,
            }

    torch.save(all_profiles, output_dir / "attention_profiles.pt")

    summary: dict[str, Any] = {}
    for template_name, profile in all_profiles.items():
        ordinary = profile["conditions"]["ordinary"]["attention_head_max"].mean(dim=0)
        bottleneck = profile["conditions"]["bottleneck_control"]["attention_head_max"].mean(dim=0)
        ordinary_contribution = profile["conditions"]["ordinary"]["contribution_norm"].mean(dim=0)
        bottleneck_contribution = profile["conditions"]["bottleneck_control"]["contribution_norm"].mean(dim=0)
        contribution_delta = profile["contribution_delta_norm"].mean(dim=0)
        labels = profile["token_labels"]

        plot_attention_comparison(
            output_dir / f"{template_name}_attention_comparison.png",
            template_name,
            labels,
            ordinary,
            bottleneck,
        )
        plot_contribution_comparison(
            output_dir / f"{template_name}_value_contribution_comparison.png",
            template_name,
            labels,
            ordinary_contribution,
            bottleneck_contribution,
            contribution_delta,
        )
        summary[template_name] = {
            "token_labels": labels,
            "entity_position": profile["entity_position"],
            "readout_position": profile["readout_position"],
            "ordinary_mean_max_head_attention": ordinary.tolist(),
            "bottleneck_mean_max_head_attention": bottleneck.tolist(),
            "bottleneck_minus_ordinary_attention": (bottleneck - ordinary).tolist(),
            "ordinary_mean_contribution_norm": ordinary_contribution.tolist(),
            "bottleneck_mean_contribution_norm": bottleneck_contribution.tolist(),
            "mean_paired_contribution_delta_norm": contribution_delta.tolist(),
        }
    (output_dir / "attention_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    print(f"saved {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
