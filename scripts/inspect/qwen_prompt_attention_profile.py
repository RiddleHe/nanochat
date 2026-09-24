"""Profile final-prompt-token attention in ordinary and bottleneck controls.

Run the selected templates and complete entity-name spans under two conditions:

1. ordinary: native attention and values everywhere;
2. bottleneck_control: intermediate post-entity prompt queries and generated
   queries lose every entity token's value contribution in every layer,
   while the final prompt query remains unrestricted.

Native fused SDPA remains the model's inference path. Because fused SDPA does
not expose attention probabilities, the profiler separately recomputes the
final prompt query's QK softmax for observation. It also records each prompt
token's weighted value contribution after the attention output projection,
allowing a paired ordinary-versus-bottleneck contribution-vector comparison.
Plots display a configurable number of name-token slots (default: four). Raw attention retains every
token; absent display slots are missing observations, never zero attention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import hashlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from qwen_entity_attention_ablation import (
    ENTITY_SETS,
    MODEL,
    TEMPLATES,
    EncodedPrompt,
    encode_prompt,
    base_row,
    load_entities,
    normalize_text,
    parse_ints,
    select,
)


ATTENTION_IMPLEMENTATION = "entity_prompt_profile"
CONDITIONS = ("ordinary", "bottleneck_control")
DEFAULT_TEMPLATE_IDS = (0, 1, 2, 3, 7)
DISPLAY_NAME_SLOTS = 4


@dataclass
class ProfileState:
    condition: str | None = None
    entity_position: int | None = None
    entity_positions: tuple[int, ...] = ()
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
            or not PROFILE.entity_positions
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
        if PROFILE.entity_positions[-1] >= key_length:
            raise RuntimeError("entity span extends beyond available keys")

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
            observed_weights = F.softmax(
                final_scores, dim=-1, dtype=torch.float32
            )[:, :, 0, :]
            PROFILE.attention_by_layer[module.layer_idx] = (
                observed_weights[0].detach().cpu()
            )
            final_weights = observed_weights.to(query.dtype)

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
                if PROFILE.entity_positions[-1]
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
                zero_value[:, :, list(PROFILE.entity_positions), :] = 0
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
    if not 0 <= encoded.entity_positions[0] <= encoded.entity_end_position < readout_position:
        raise ValueError("expected complete entity span before the final prompt token")

    PROFILE.condition = condition
    PROFILE.entity_position = encoded.entity_position
    PROFILE.entity_positions = encoded.entity_positions
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
        PROFILE.entity_positions = ()
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
                len(encoded.ids) - encoded.entity_end_position - 2
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
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    ended_with_eos = generated_ids[-1] in (eos_ids or [])
    return {
        "generated_token_ids": generated_ids,
        "completion": completion,
        "normalized_completion": visible_completion(completion),
        "ended_with_eos": ended_with_eos,
        "stop_reason": "eos" if ended_with_eos else ("max_new_tokens" if len(generated_ids) == max_new_tokens else "other"),
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


def build_alignment(tokenizer, cases: list[EncodedPrompt], template_text: str, display_name_slots: int = DISPLAY_NAME_SLOTS) -> dict[str, Any]:
    """Align context by template-relative character spans; display the requested number of name slots.

    A token overlapping the name belongs to the name, even if it also contains
    punctuation. It is never duplicated into the suffix. Omitted name tokens
    still remain in the real prompt and in the full raw attention tensors.
    """
    if display_name_slots < 1:
        raise ValueError("display_name_slots must be positive")
    if not cases or template_text.count("{entity}") != 1:
        raise ValueError("alignment requires cases and one entity placeholder")
    prefix, suffix = template_text.split("{entity}")
    per_case = []
    key_tokens = {}
    for encoded in cases:
        if encoded.prompt != prefix + encoded.entity + suffix:
            raise ValueError("prompt differs from the selected bare-name template")
        tokenized = tokenizer(encoded.prompt, add_special_tokens=True, return_offsets_mapping=True)
        if list(tokenized['input_ids']) != encoded.ids:
            raise RuntimeError("alignment tokenization differs from inference tokenization")
        offsets = tokenized['offset_mapping']
        name_end = len(prefix) + len(encoded.entity)
        positions = set(encoded.entity_positions)
        expected = {i for i, (left, right) in enumerate(offsets)
                    if (left, right) != (0, 0) and right > len(prefix) and left < name_end}
        if positions != expected:
            raise RuntimeError("alignment entity span differs from intervention span")
        mapped = {}
        for position, ((left, right), token_id) in enumerate(zip(offsets, encoded.ids)):
            if position in positions:
                index = position - encoded.entity_position
                if index >= display_name_slots:
                    continue
                key = ('name', index)
            elif position < encoded.entity_position:
                key = ('prefix', left, right, token_id)
            else:
                key = ('suffix', left - name_end, right - name_end, token_id)
            if key in mapped:
                raise RuntimeError(f"two tokens map to the same display column: {key}")
            mapped[key] = position
            if key[0] != 'name':
                key_tokens[key] = token_id
        per_case.append(mapped)
    keys = (sorted(k for k in key_tokens if k[0] == 'prefix')
            + [('name', i) for i in range(display_name_slots)]
            + sorted(k for k in key_tokens if k[0] == 'suffix'))
    indices = [[mapping.get(key, -1) for key in keys] for mapping in per_case]
    counts = [sum(row[col] >= 0 for row in indices) for col in range(len(keys))]
    labels = []
    for key, count in zip(keys, counts):
        label = f'NAME_{key[1]+1}' if key[0] == 'name' else token_label(tokenizer, key_tokens[key], False)
        if key[0] == 'name' or count != len(cases):
            label += f' [n={count}]'
        labels.append(label)
    return {'schema_version': 1, 'display_name_slots': display_name_slots,
            'column_keys': [list(k) for k in keys], 'token_labels': labels,
            'entity_ids': [c.entity_id for c in cases], 'source_token_indices': indices,
            'valid_mask': [[p >= 0 for p in row] for row in indices],
            'contributing_name_counts': counts,
            'entity_token_counts': [len(c.entity_positions) for c in cases],
            'omitted_name_tokens_per_case': [max(0, len(c.entity_positions)-display_name_slots) for c in cases],
            'raw_attention_includes_omitted_name_tokens': True,
            'padding_affects_model_input': False}


def align_tokens(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Map the last tensor dimension into display columns; missing means NaN."""
    result = torch.full((*values.shape[:-1], len(indices)), float('nan'), dtype=values.dtype)
    for column, position in enumerate(indices):
        if position >= 0:
            result[..., column] = values[..., position]
    return result


def masked_mean(values: torch.Tensor) -> torch.Tensor:
    """Mean over examples that have the given token slot, without zero filling."""
    return torch.nanmean(values, dim=0)


def json_matrix(values: torch.Tensor):
    """Emit missing cells as JSON null, never nonstandard NaN literals."""
    return [[float(v) if torch.isfinite(v) else None for v in row] for row in values]


def finite_max(values: torch.Tensor, absolute: bool = False) -> float:
    values = values[torch.isfinite(values)]
    if not values.numel():
        return 1e-8
    return max(float((values.abs() if absolute else values).max()), 1e-8)


def heatmap(
    axis,
    values: torch.Tensor,
    labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
):
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#dddddd")
    image = axis.imshow(
        np.ma.masked_invalid(values.numpy()),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=palette,
    )
    axis.set_title(title)
    axis.set_xlabel("Aligned prompt token (first four name slots; n = contributing names)")
    axis.set_ylabel("Layer")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axis.set_yticks(range(0, values.shape[0], 4))
    for tick, label in zip(axis.get_xticklabels(), labels):
        if label.startswith("NAME_"):
            tick.set_color("#a02020")
            tick.set_fontweight("bold")
    return image


def plot_attention_comparison(
    output_path: Path,
    template_name: str,
    token_labels: list[str],
    ordinary: torch.Tensor,
    bottleneck: torch.Tensor,
) -> None:
    difference = bottleneck - ordinary
    shared_max = max(finite_max(ordinary), finite_max(bottleneck))
    diff_max = finite_max(difference, absolute=True)
    width = max(14.0, 0.36 * len(token_labels))
    figure, axes = plt.subplots(3, 1, figsize=(width, 17), constrained_layout=True)
    first = heatmap(
        axes[0], ordinary, token_labels, "Ordinary", 0.0, shared_max, "viridis"
    )
    heatmap(
        axes[1], bottleneck, token_labels, "Blocked (final prompt query unrestricted)", 0.0,
        shared_max, "viridis"
    )
    third = heatmap(
        axes[2], difference, token_labels, "Blocked − ordinary",
        -diff_max, diff_max, "coolwarm"
    )
    figure.colorbar(first, ax=axes[:2], shrink=0.85, label="Mean max-head attention")
    figure.colorbar(third, ax=axes[2], shrink=0.85, label="Attention difference")
    figure.suptitle(
        f"Final prompt token attention — {template_name.replace('_', ' ')}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def plot_contribution_comparison(
    output_path: Path,
    template_name: str,
    token_labels: list[str],
    ordinary_norm: torch.Tensor,
    bottleneck_norm: torch.Tensor,
    delta_vector_norm: torch.Tensor,
) -> None:
    shared_max = max(finite_max(ordinary_norm), finite_max(bottleneck_norm))
    delta_max = finite_max(delta_vector_norm)
    width = max(14.0, 0.36 * len(token_labels))
    figure, axes = plt.subplots(3, 1, figsize=(width, 17), constrained_layout=True)
    first = heatmap(
        axes[0], ordinary_norm, token_labels, "Ordinary", 0.0,
        shared_max, "magma"
    )
    heatmap(
        axes[1], bottleneck_norm, token_labels, "Blocked (final prompt query unrestricted)",
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
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--entity-set", choices=sorted(ENTITY_SETS), default="diverse100")
    source.add_argument("--entities-json", type=Path, help="Same ordered full-name manifest as the main script")
    parser.add_argument("--entity-ids", type=parse_ints)
    parser.add_argument("--template-ids", type=parse_ints, default=list(DEFAULT_TEMPLATE_IDS))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--display-name-slots", type=int, default=DISPLAY_NAME_SLOTS, help="Number of entity-token columns in the displayed alignment; raw attention always retains every token")
    parser.add_argument("--out-dir")
    parser.add_argument("--reference-dir", type=Path, help="Consolidated ordinary/blocked controls; require identical prompts and generated token IDs")
    parser.add_argument("--plot-contributions", action="store_true", help="Also render the existing per-token value-contribution plots")
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
    if args.display_name_slots < 1:
        raise ValueError('--display-name-slots must be positive')
    if args.max_new_tokens < 1:
        raise ValueError('--max-new-tokens must be positive')
    selected_templates = select(TEMPLATES, args.template_ids, 'template')
    if args.entities_json:
        entity_pool, entity_sha = load_entities(args.entities_json)
    else:
        entity_pool, entity_sha = ENTITY_SETS[args.entity_set], None
    selected_entities = select(entity_pool, args.entity_ids, 'entity')
    reference_rows = {}
    reference_hashes = {}
    if args.reference_dir:
        for _, template in selected_templates:
            for condition, directory in [('ordinary', 'ordinary'), ('bottleneck_control', 'blocked')]:
                run_dir = args.reference_dir / 'runs' / f'{template.template_id:02d}_{template.name}' / directory
                ref_meta = json.loads((run_dir / 'metadata.json').read_text())
                if ref_meta['model'] != args.model or (ref_meta['max_new_tokens'] if 'max_new_tokens' in ref_meta else ref_meta['generation']['max_new_tokens']) != args.max_new_tokens:
                    raise ValueError('reference model or decoding limit differs')
                if entity_sha and (ref_meta['entities_sha256'] if 'entities_sha256' in ref_meta else ref_meta['entity_manifest_sha256']) != entity_sha:
                    raise ValueError('reference entity manifest differs')
                expected_block = condition == 'bottleneck_control'
                if (ref_meta['intermediate_prompt_entity_value_blocked_all_layers'] != expected_block
                    or ref_meta['generated_entity_value_blocked_all_layers'] != expected_block
                    or (ref_meta['final_prompt_entity_value_blocked_in_selected_span'] if 'final_prompt_entity_value_blocked_in_selected_span' in ref_meta else ref_meta['final_prompt_entity_value_blocked']) or (ref_meta['final_prompt_entity_attention_restore'] if 'final_prompt_entity_attention_restore' in ref_meta else ref_meta['restoration_enabled'])):
                    raise ValueError('reference intervention differs')
                raw = (run_dir / 'generations.jsonl').read_bytes()
                reference_hashes[f'{template.name}/{directory}'] = hashlib.sha256(raw).hexdigest()
                for line in raw.splitlines():
                    row = json.loads(line)
                    key = (template.template_id, row['entity_id'], condition)
                    if key in reference_rows:
                        raise ValueError('duplicate reference case')
                    reference_rows[key] = row
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.out_dir or f'results/qwen_prompt_attention_profile_{stamp}')
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model, tokenizer, n_layers = load_model(args.model, device)
    template_cases = {}
    alignments = {}
    for _, template in selected_templates:
        cases = [encode_prompt(tokenizer, template, i, entity, group)
                 for i, (entity, group) in selected_entities]
        template_cases[template.name] = cases
        alignments[template.name] = build_alignment(tokenizer, cases, template.text, args.display_name_slots)
        if args.reference_dir:
            for encoded in cases:
                for condition in CONDITIONS:
                    ref = reference_rows[(encoded.template_id, encoded.entity_id, condition)]
                    if (encoded.entity != ref['entity'] or encoded.prompt != ref['prompt']
                        or encoded.ids != ref['prompt_token_ids']
                        or list(encoded.entity_positions) != ref['entity_positions']):
                        raise RuntimeError('profile input differs from the saved control')
    metadata = {
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'command': [sys.executable] + sys.argv, 'model': args.model,
        'device': str(device), 'dtype': str(next(model.parameters()).dtype), 'n_layers': n_layers,
        'conditions': list(CONDITIONS),
        'templates': [dict(template_id=t.template_id, name=t.name, text=t.text) for _, t in selected_templates],
        'entity_set': 'custom' if args.entities_json else args.entity_set,
        'entities_json': str(args.entities_json.resolve()) if args.entities_json else None,
        'entities_sha256': entity_sha, 'n_entities': len(selected_entities),
        'entities': [dict(entity_id=i, entity=e, entity_group=g) for i, (e, g) in selected_entities],
        'max_new_tokens': args.max_new_tokens, 'do_sample': False,
        'display_name_slots': args.display_name_slots,
        'attention_statistic': 'final-prompt-query post-softmax attention; max over heads within each example and token, then mean over examples with that display slot; no sum across entity tokens',
        'attention_observation_dtype': 'float32 softmax of model-dtype QK logits; inference remains native SDPA',
        'attention_difference': 'paired blocked minus ordinary, after per-example per-token max over heads',
        'missing_slots': 'NaN in tensors, null in JSON; excluded from means; not zero; same eligibility in both conditions',
        'alignment': f'{args.display_name_slots} name-token ordinal slots, prefix/suffix aligned by template-relative character offsets and token IDs; boundary-crossing tokens belong only to the name',
        'omitted_name_tokens': f'name tokens after index {args.display_name_slots} are omitted only from displayed summaries; still in model inputs, intervention, and complete per-head raw attention',
        'ordinary_policy': 'native attention and values for all queries',
        'bottleneck_policy': 'all entity-token values blocked for intermediate queries after the name end and generated queries in every layer; final prompt query unrestricted; no renormalization',
        'value_statistic': 'per-token projected contribution norms; paired contribution-vector difference norm before averaging',
        'reference_dir': str(args.reference_dir.resolve()) if args.reference_dir else None,
        'reference_sha256': reference_hashes,
        'profiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'main_script_sha256': hashlib.sha256(Path(__file__).with_name('qwen_entity_attention_ablation.py').read_bytes()).hexdigest(),
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    summary = {}
    matches = []
    with (output_dir / 'generations.jsonl').open('x') as generation_file:
        for template_name, cases in template_cases.items():
            template_dir = output_dir / template_name
            template_dir.mkdir()
            alignment = alignments[template_name]
            (template_dir / 'alignment.json').write_text(json.dumps(alignment, ensure_ascii=False, indent=2)+'\n')
            raw_cases = []
            aligned_attention = {c: [] for c in CONDITIONS}
            aligned_norms = {c: [] for c in CONDITIONS}
            aligned_delta_norms = []
            print(f'template {template_name}: {len(cases)} entities', flush=True)
            for case_index, encoded in enumerate(cases):
                indices = alignment['source_token_indices'][case_index]
                stored = {**base_row(encoded), 'conditions': {}}
                paired_contributions = {}
                for condition in CONDITIONS:
                    result = generate_and_profile(model, tokenizer, encoded, condition, device, n_layers, args.max_new_tokens)
                    attention = result.pop('attention_by_head')
                    contributions = result.pop('token_contributions')
                    if attention.ndim != 3 or attention.shape[0] != n_layers or attention.shape[-1] != len(encoded.ids):
                        raise RuntimeError('unexpected raw attention shape')
                    if not torch.isfinite(attention).all() or attention.min() < 0 or attention.max() > 1:
                        raise RuntimeError('invalid attention coefficients')
                    torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=2e-6, rtol=2e-6)
                    norms = torch.linalg.vector_norm(contributions, dim=-1)
                    stored['conditions'][condition] = {'attention_by_head': attention, 'contribution_norm': norms}
                    aligned_attention[condition].append(align_tokens(attention.max(dim=1).values, indices))
                    aligned_norms[condition].append(align_tokens(norms, indices))
                    paired_contributions[condition] = contributions
                    row = {'condition': condition, **base_row(encoded),
                           'intermediate_attention_ablation_applications': result.pop('intermediate_applications'),
                           'generated_attention_ablation_applications': result.pop('generated_applications'), **result}
                    if args.reference_dir:
                        ref = reference_rows[(encoded.template_id, encoded.entity_id, condition)]
                        equal = all(row[key] == ref[key] for key in (
                            'generated_token_ids', 'completion', 'stop_reason',
                            'intermediate_attention_ablation_applications', 'generated_attention_ablation_applications'))
                        row['reference_match'] = equal
                        row['reference_record_id'] = f"t{encoded.template_id}:e{encoded.entity_id:03d}:{'ordinary' if condition == 'ordinary' else 'blocked'}"
                        matches.append({'record_id': row['reference_record_id'], 'match': equal})
                        if not equal:
                            (output_dir/'reference_mismatch.json').write_text(json.dumps({'profile':row, 'reference':ref},ensure_ascii=False,indent=2))
                            raise RuntimeError('profile generation differs from saved control; see reference_mismatch.json')
                    generation_file.write(json.dumps(row, ensure_ascii=False)+'\n')
                    generation_file.flush()
                torch.testing.assert_close(stored['conditions']['ordinary']['attention_by_head'][0],
                                           stored['conditions']['bottleneck_control']['attention_by_head'][0], atol=0, rtol=0)
                delta_norm = torch.linalg.vector_norm(paired_contributions['bottleneck_control']-paired_contributions['ordinary'],dim=-1)
                stored['paired_contribution_delta_norm'] = delta_norm
                raw_cases.append(stored)
                aligned_delta_norms.append(align_tokens(delta_norm, indices))
                print(f'  {case_index+1:03d}/{len(cases)} {encoded.entity}: profile saved; reference={bool(args.reference_dir)}', flush=True)
            torch.save({'schema_version':2,'template_name':template_name,'alignment':alignment,'cases':raw_cases},template_dir/'attention_profiles.pt')
            aligned = {c: torch.stack(aligned_attention[c]) for c in CONDITIONS}
            means = {c: masked_mean(aligned[c]) for c in CONDITIONS}
            paired_delta = aligned['bottleneck_control']-aligned['ordinary']
            difference = masked_mean(paired_delta)
            torch.testing.assert_close(difference,means['bottleneck_control']-means['ordinary'],atol=2e-7,rtol=2e-5,equal_nan=True)
            contribution_means = {c: masked_mean(torch.stack(aligned_norms[c])) for c in CONDITIONS}
            contribution_delta = masked_mean(torch.stack(aligned_delta_norms))
            torch.save({'alignment':alignment,'attention_head_max_by_case':aligned,
                        'paired_head_max_difference_by_case':paired_delta,
                        'ordinary_mean_max_head_attention':means['ordinary'],
                        'blocked_mean_max_head_attention':means['bottleneck_control'],
                        'blocked_minus_ordinary_attention':difference},template_dir/'aligned_attention.pt')
            labels = alignment['token_labels']
            plot_attention_comparison(template_dir/'attention_comparison.png',template_name,labels,means['ordinary'],means['bottleneck_control'])
            if args.plot_contributions:
                plot_contribution_comparison(template_dir/'value_contribution_comparison.png',template_name,labels,
                                             contribution_means['ordinary'],contribution_means['bottleneck_control'],contribution_delta)
            summary[template_name] = {'alignment':alignment,
                'ordinary_mean_max_head_attention':json_matrix(means['ordinary']),
                'blocked_mean_max_head_attention':json_matrix(means['bottleneck_control']),
                'blocked_minus_ordinary_attention':json_matrix(difference),
                'ordinary_mean_contribution_norm':json_matrix(contribution_means['ordinary']),
                'blocked_mean_contribution_norm':json_matrix(contribution_means['bottleneck_control']),
                'mean_paired_contribution_delta_norm':json_matrix(contribution_delta)}
            (output_dir/'attention_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    parity = {'reference_checked':bool(args.reference_dir),'expected_generations':2*len(selected_entities)*len(selected_templates),
              'compared_generations':len(matches),'all_match':all(m['match'] for m in matches) if matches else None,
              'cases':matches}
    (output_dir/'generation_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    print(f'saved {output_dir}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
