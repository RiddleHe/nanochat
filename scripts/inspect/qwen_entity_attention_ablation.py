"""Ablate or supplement direct attention to an entity in Qwen layers.

Each prompt contains one celebrity. Generated-token queries always lose the
entity value contribution in every layer. Optionally, prompt queries strictly
between the entity and final prompt token do too. For every width/start-layer
span, the final prompt query additionally loses the entity value contribution
in the selected layers. Native SDPA attention weights and normalization are
unchanged. Raw continuations are saved for manual semantic scoring.

The optional rescue condition records the ordinary final-prompt-query
attention to the entity, then runs the bottleneck while adding any positive
ordinary-minus-live entity-attention deficit after softmax. Other attention
coefficients are not renormalized, so the effective coefficient sum can
exceed one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MODEL = "Qwen/Qwen3-8B-Base"
ATTENTION_IMPLEMENTATION = "sdpa"


@dataclass(frozen=True)
class Template:
    template_id: int
    name: str
    text: str


TEMPLATES = [
    Template(
        0,
        "direct_fact",
        "Remember this fact: the person's name is {entity}.\n"
        "Question: What is the person's name?\n"
        "Respond with only the name.\n"
        "Answer:",
    ),
    Template(
        1,
        "person_in_list",
        "Among apple, mouse, {entity}, and flute, exactly one item is a person's name.\n"
        "Respond with only that person's name.\n"
        "Answer:",
    ),
    Template(
        2,
        "friend",
        "Here is a fact: {entity} is my friend.\n"
        "Question: What is my friend's name?\n"
        "Respond with only the name.\n"
        "Answer:",
    ),
    Template(
        3,
        "visitor_register",
        "The visitor signed the register with the name {entity}.\n"
        "Question: What name did the visitor write?\n"
        "Respond with only the name.\n"
        "Answer:",
    ),
    Template(
        4,
        "dialogue",
        "Speaker A: I met {entity} yesterday.\n"
        "Speaker B: Who did you meet?\n"
        "Answer with only the person's name:",
    ),
    Template(
        5,
        "occupation",
        "Remember this fact: the person's name is {entity}.\n"
        "Question: What is the person's occupation?\n"
        "Respond with only one short occupation.\n"
        "Answer:",
    ),
    Template(
        6,
        "gender",
        "Remember this fact: the person's name is {entity}.\n"
        "Question: What is the person's gender?\n"
        "Respond with only one short answer.\n"
        "Answer:",
    ),
]

ORIGINAL_ENTITIES = [
    "Einstein",
    "Darwin",
    "Mozart",
    "Shakespeare",
    "Tolkien",
    "Messi",
    "Elvis",
    "Madonna",
    "Gandhi",
    "Lincoln",
]

DIVERSE_ENTITY_GROUPS = {
    "science_math": [
        "Einstein", "Newton", "Darwin", "Tesla", "Turing",
        "Euler", "Gauss", "Kepler", "Maxwell", "Freud",
    ],
    "literature": [
        "Shakespeare", "Dickens", "Tolkien", "Orwell", "Kafka",
        "Joyce", "Homer", "Dante", "Morrison", "Christie",
    ],
    "music": [
        "Mozart", "Bach", "Wagner", "Elvis", "Madonna",
        "Rihanna", "Drake", "Cher", "Sinatra", "Lennon",
    ],
    "film_media": [
        "Monroe", "Cruise", "Pitt", "Freeman", "Nicholson",
        "Ledger", "Spielberg", "Cameron", "Oprah", "Roberts",
    ],
    "historical_leaders": [
        "Lincoln", "Washington", "Jefferson", "Roosevelt", "Churchill",
        "Gandhi", "Mandela", "Napoleon", "Caesar", "Victoria",
    ],
    "modern_leaders": [
        "Thatcher", "Obama", "Merkel", "Macron", "Lenin",
        "Stalin", "Castro", "Mao", "Erdogan", "Netanyahu",
    ],
    "sports": [
        "Messi", "Ronaldo", "Jordan", "Kobe", "Serena",
        "Bolt", "Phelps", "Ali", "Tyson", "LeBron",
    ],
    "technology_innovation": [
        "Edison", "Nobel", "Jobs", "Gates", "Musk",
        "Zuckerberg", "Watson", "Shannon", "Fleming", "Jenner",
    ],
    "global_figures": [
        "Yao", "Naomi", "Venus", "Jackie", "Bruce",
        "Chan", "Lee", "Che", "Chavez", "Modi",
    ],
    "women_across_fields": [
        "Swift", "Whitney", "Teresa", "Elizabeth", "Catherine",
        "Rosa", "Amelia", "Marie", "Ada", "Maya",
    ],
}

ENTITY_SETS = {
    "original10": [(entity, "original10") for entity in ORIGINAL_ENTITIES],
    "diverse100": [
        (entity, group)
        for group, entities in DIVERSE_ENTITY_GROUPS.items()
        for entity in entities
    ],
}


@dataclass(frozen=True)
class EncodedPrompt:
    template_id: int
    template_name: str
    entity_id: int
    entity: str
    entity_group: str
    prompt: str
    ids: list[int]
    tokens: list[str]
    entity_position: int
    entity_phrase: str | None = None
    entity_description: str | None = None
    description_positions: tuple[int, ...] = ()
    description_tail_position: int | None = None


@dataclass
class AblationState:
    disabled_layers: frozenset[int] = frozenset()
    entity_position: int | None = None
    readout_position: int | None = None
    prompt_length: int | None = None
    block_intermediate_prompt: bool = False
    capture_ordinary_entity_attention: bool = False
    restore_layers: frozenset[int] = frozenset()
    ordinary_entity_attention: dict[int, torch.Tensor] | None = None
    restore_trace: dict[int, dict[str, list[float]]] | None = None
    applications: int = 0
    intermediate_applications: int = 0
    generated_applications: int = 0
    restore_applications: int = 0


ABLATION = AblationState()


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


def _final_entity_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    readout_local: int,
    entity_position: int,
) -> torch.Tensor:
    """Return the final prompt query's entity coefficient per query head."""
    key_states = _repeat_kv(key, module.num_key_value_groups)
    key_length = key_states.shape[-2]
    final_query = query[:, :, readout_local : readout_local + 1, :]
    final_scores = torch.matmul(
        final_query, key_states.transpose(2, 3)
    ) * scaling
    if attention_mask is not None:
        final_mask = attention_mask[
            :, :, readout_local : readout_local + 1, :key_length
        ]
        final_scores = final_scores + final_mask
    final_weights = F.softmax(
        final_scores, dim=-1, dtype=torch.float32
    )
    return final_weights[:, :, 0, entity_position]


def entity_zero_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if ABLATION.entity_position is not None:
        if ABLATION.readout_position is None or ABLATION.prompt_length is None:
            raise RuntimeError("active entity policy is missing prompt boundaries")
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        if not 0 <= ABLATION.entity_position < key_length:
            raise RuntimeError(
                f"entity position {ABLATION.entity_position} outside key length {key_length}"
            )
        query_start = key_length - query_length
        query_positions = list(range(query_start, query_start + query_length))
        readout_queries = [
            index
            for index, position in enumerate(query_positions)
            if position == ABLATION.readout_position
        ]

        if (
            ABLATION.capture_ordinary_entity_attention
            and module.layer_idx in ABLATION.restore_layers
        ):
            if ABLATION.ordinary_entity_attention is None:
                raise RuntimeError("ordinary entity-attention capture has no store")
            if len(readout_queries) != 1:
                raise RuntimeError(
                    "ordinary entity-attention capture expected one readout query"
                )
            if module.layer_idx in ABLATION.ordinary_entity_attention:
                raise RuntimeError(
                    f"layer {module.layer_idx} ordinary entity attention captured twice"
                )
            ordinary_attention = _final_entity_attention(
                module,
                query,
                key,
                attention_mask,
                scaling,
                readout_queries[0],
                ABLATION.entity_position,
            )
            ABLATION.ordinary_entity_attention[module.layer_idx] = (
                ordinary_attention.detach().float().clone()
            )
            return sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **_kwargs,
            )

        generated_queries = [
            index
            for index, position in enumerate(query_positions)
            if position >= ABLATION.prompt_length
        ]
        intermediate_queries = []
        if ABLATION.block_intermediate_prompt:
            intermediate_queries = [
                index
                for index, position in enumerate(query_positions)
                if ABLATION.entity_position
                < position
                < ABLATION.readout_position
            ]
        ablated_readout_queries = []
        if module.layer_idx in ABLATION.disabled_layers:
            ablated_readout_queries = readout_queries

        affected_queries = sorted(
            set(
                generated_queries
                + intermediate_queries
                + ablated_readout_queries
            )
        )
        restore_readout = (
            module.layer_idx in ABLATION.restore_layers
            and len(readout_queries) == 1
        )
        if restore_readout and module.layer_idx in ABLATION.disabled_layers:
            raise RuntimeError(
                "readout entity attention cannot be ablated and restored together"
            )

        zero_output = None
        if affected_queries:
            zero_value = value.clone()
            zero_value[:, :, ABLATION.entity_position, :] = 0
            zero_output, _ = sdpa_attention_forward(
                module,
                query,
                key,
                zero_value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **_kwargs,
            )
            ABLATION.applications += len(ablated_readout_queries)
            ABLATION.intermediate_applications += len(intermediate_queries)
            ABLATION.generated_applications += len(generated_queries)
            if len(affected_queries) == query_length and not restore_readout:
                return zero_output, None

        if affected_queries or restore_readout:
            attention_output, _ = sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **_kwargs,
            )
            attention_output = attention_output.clone()
            if affected_queries:
                if zero_output is None:
                    raise RuntimeError("affected queries have no zero-value output")
                attention_output[:, affected_queries, :, :] = zero_output[
                    :, affected_queries, :, :
                ]

            if restore_readout:
                if (
                    ABLATION.ordinary_entity_attention is None
                    or ABLATION.restore_trace is None
                ):
                    raise RuntimeError("entity-attention restore has no reference")
                if module.layer_idx not in ABLATION.ordinary_entity_attention:
                    raise RuntimeError(
                        f"layer {module.layer_idx} has no ordinary attention reference"
                    )
                if module.layer_idx in ABLATION.restore_trace:
                    raise RuntimeError(
                        f"layer {module.layer_idx} entity attention restored twice"
                    )
                readout_local = readout_queries[0]
                live_attention = _final_entity_attention(
                    module,
                    query,
                    key,
                    attention_mask,
                    scaling,
                    readout_local,
                    ABLATION.entity_position,
                ).float()
                ordinary_attention = ABLATION.ordinary_entity_attention[
                    module.layer_idx
                ].to(device=live_attention.device, dtype=torch.float32)
                if ordinary_attention.shape != live_attention.shape:
                    raise RuntimeError(
                        "ordinary and live entity-attention shapes do not match"
                    )
                deficit = (ordinary_attention - live_attention).clamp_min(0)
                value_states = _repeat_kv(
                    value, module.num_key_value_groups
                )
                entity_value = value_states[
                    :, :, ABLATION.entity_position, :
                ]
                supplement = (
                    deficit.to(entity_value.dtype).unsqueeze(-1)
                    * entity_value
                )
                attention_output[:, readout_local, :, :] += supplement
                effective_entity_attention = live_attention + deficit
                ABLATION.restore_trace[module.layer_idx] = {
                    "ordinary_entity_attention": ordinary_attention[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "live_entity_attention_before_restore": live_attention[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "added_entity_attention_deficit": deficit[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "effective_entity_attention": effective_entity_attention[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "effective_attention_sum": (1.0 + deficit[0])
                    .detach()
                    .cpu()
                    .tolist(),
                }
                ABLATION.restore_applications += 1
            return attention_output, None

    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        **_kwargs,
    )


@contextmanager
def capture_ordinary_entity_attention(
    layers: range,
    entity_position: int,
    readout_position: int,
    prompt_length: int,
):
    if ABLATION.entity_position is not None:
        raise RuntimeError("nested entity-attention policies are not supported")
    if not 0 <= entity_position < readout_position == prompt_length - 1:
        raise ValueError(
            "expected entity_position < readout_position == prompt_length - 1"
        )
    ABLATION.entity_position = entity_position
    ABLATION.readout_position = readout_position
    ABLATION.prompt_length = prompt_length
    ABLATION.capture_ordinary_entity_attention = True
    ABLATION.restore_layers = frozenset(layers)
    ABLATION.ordinary_entity_attention = {}
    try:
        yield ABLATION
    finally:
        ABLATION.entity_position = None
        ABLATION.readout_position = None
        ABLATION.prompt_length = None
        ABLATION.capture_ordinary_entity_attention = False
        ABLATION.restore_layers = frozenset()
        ABLATION.ordinary_entity_attention = None


@contextmanager
def disable_entity_attention(
    layers: range,
    entity_position: int,
    readout_position: int,
    prompt_length: int,
    block_intermediate_prompt: bool,
    restore_layers: range = range(0),
    ordinary_entity_attention: dict[int, torch.Tensor] | None = None,
):
    if ABLATION.entity_position is not None:
        raise RuntimeError("nested entity-attention policies are not supported")
    if not 0 <= entity_position < readout_position == prompt_length - 1:
        raise ValueError(
            "expected entity_position < readout_position == prompt_length - 1"
        )
    ABLATION.disabled_layers = frozenset(layers)
    ABLATION.entity_position = entity_position
    ABLATION.readout_position = readout_position
    ABLATION.prompt_length = prompt_length
    ABLATION.block_intermediate_prompt = block_intermediate_prompt
    ABLATION.restore_layers = frozenset(restore_layers)
    ABLATION.ordinary_entity_attention = ordinary_entity_attention
    ABLATION.restore_trace = {}
    ABLATION.applications = 0
    ABLATION.intermediate_applications = 0
    ABLATION.generated_applications = 0
    ABLATION.restore_applications = 0
    try:
        yield ABLATION
    finally:
        ABLATION.disabled_layers = frozenset()
        ABLATION.entity_position = None
        ABLATION.readout_position = None
        ABLATION.prompt_length = None
        ABLATION.block_intermediate_prompt = False
        ABLATION.restore_layers = frozenset()
        ABLATION.ordinary_entity_attention = None
        ABLATION.restore_trace = None
        ABLATION.applications = 0
        ABLATION.intermediate_applications = 0
        ABLATION.generated_applications = 0
        ABLATION.restore_applications = 0


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def load_model(model_name: str, device: torch.device):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register(
        ATTENTION_IMPLEMENTATION, entity_zero_attention_forward
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
    return model, tokenizer, list(model.model.layers)


def encode_prompt(
    tokenizer,
    template: Template,
    entity_id: int,
    entity: str,
    entity_group: str,
    description: str | None = None,
) -> EncodedPrompt:
    if template.text.count("{entity}") != 1:
        raise ValueError("expected exactly one entity placeholder")
    prefix, suffix = template.text.split("{entity}")
    entity_phrase = entity if description is None else f"{entity} ({description})"
    prompt = prefix + entity_phrase + suffix
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    start = len(prefix)
    end = start + len(entity)
    positions = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if (left, right) != (0, 0) and right > start and left < end
    ]
    if len(positions) != 1:
        raise RuntimeError(
            f"{template.name}/{entity}: entity must be exactly one token; got {positions}"
        )
    description_positions = ()
    if description is not None:
        description_start = end + len(" (")
        description_end = description_start + len(description)
        description_positions = tuple(
            index
            for index, (left, right) in enumerate(encoded["offset_mapping"])
            if (left, right) != (0, 0)
            and right > description_start and left < description_end
        )
        if not description_positions or not (
            positions[0] < min(description_positions)
            <= max(description_positions) < len(encoded["input_ids"]) - 1
        ):
            raise RuntimeError("description must lie between entity and readout")
    ids = list(encoded["input_ids"])
    return EncodedPrompt(
        template_id=template.template_id,
        template_name=template.name,
        entity_id=entity_id,
        entity=entity,
        entity_group=entity_group,
        prompt=prompt,
        ids=ids,
        tokens=list(tokenizer.convert_ids_to_tokens(ids)),
        entity_position=positions[0],
        entity_phrase=entity_phrase,
        entity_description=description,
        description_positions=description_positions,
        description_tail_position=(
            description_positions[-1] if description_positions else None
        ),
    )


@torch.inference_mode()
def greedy_completion(
    model,
    tokenizer,
    encoded: EncodedPrompt,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
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
    generated_ids = generated.sequences[0, len(encoded.ids):].detach().cpu().tolist()
    if not generated_ids or not generated.scores:
        raise RuntimeError("generation produced no next token")
    next_token_id = int(generated_ids[0])
    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    ended_with_eos = generated_ids[-1] in (eos_token_ids or [])
    return {
        "next_token_id": next_token_id,
        "next_token_text": tokenizer.decode(
            [next_token_id], skip_special_tokens=False
        ),
        "next_token_score": float(
            generated.scores[0][0, next_token_id].float().cpu()
        ),
        "generated_token_ids": generated_ids,
        "completion": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "ended_with_eos": ended_with_eos,
        "stop_reason": (
            "eos" if ended_with_eos else (
                "max_new_tokens" if len(generated_ids) == max_new_tokens
                else "other"
            )
        ),
    }


@torch.inference_mode()
def ordinary_entity_attention_reference(
    model,
    encoded: EncodedPrompt,
    device: torch.device,
    layers: range,
) -> dict[int, torch.Tensor]:
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    expected_layers = set(layers)
    with capture_ordinary_entity_attention(
        range(min(expected_layers), max(expected_layers) + 1),
        encoded.entity_position,
        len(encoded.ids) - 1,
        len(encoded.ids),
    ) as state:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        observed_layers = set(state.ordinary_entity_attention or {})
        if observed_layers != expected_layers:
            raise RuntimeError(
                "ordinary entity-attention reference mismatch: "
                f"observed={sorted(observed_layers)}, "
                f"expected={sorted(expected_layers)}"
            )
        return {
            layer: attention.detach().clone()
            for layer, attention in (state.ordinary_entity_attention or {}).items()
        }


def normalize_text(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return unicodedata.normalize("NFKC", text).casefold().strip()


def contains_entity(text: str, entity: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(normalize_text(entity))}(?!\w)",
        normalize_text(text),
    ) is not None


def completion_fields(completion: dict[str, Any], entity: str) -> dict[str, Any]:
    return {
        **completion,
        "normalized_completion": normalize_text(completion["completion"]),
        "entity_in_completion": contains_entity(completion["completion"], entity),
    }


def write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--widths", type=parse_ints, default=parse_ints("1,2,4"))
    parser.add_argument(
        "--start-layers",
        type=parse_ints,
        help="optional comma-separated subset; default is every valid start layer",
    )
    parser.add_argument("--entity-ids", type=parse_ints)
    parser.add_argument(
        "--entity-set",
        choices=sorted(ENTITY_SETS),
        default="original10",
        help="entity pool; original10 preserves the previous default",
    )
    parser.add_argument("--template-ids", type=parse_ints)
    parser.add_argument(
        "--entity-descriptions-json",
        type=Path,
        help="optional name-to-description JSON; render NAME (DESCRIPTION) while ablating only NAME",
    )
    parser.add_argument(
        "--block-intermediate-prompt-entity",
        action="store_true",
        help=(
            "remove the entity value contribution in every layer for prompt "
            "queries strictly between the entity and final prompt token; "
            "generated queries are always blocked"
        ),
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--ordinary-baseline-only",
        action="store_true",
        help=(
            "write fully unablated greedy generations only; no prompt, "
            "generated, or final-query entity-value blocking"
        ),
    )
    parser.add_argument(
        "--restore-entity-attention-start-layer",
        type=int,
        help=(
            "run only the bottleneck entity-attention rescue: from this layer "
            "through the final layer, add the positive per-head ordinary-minus-"
            "live entity-attention deficit to the final prompt query after "
            "native attention, without renormalizing other coefficients"
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="write intervention rows only; reuse a separately verified baseline run",
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def select(values, selected: list[int] | None, label: str):
    indices = selected if selected is not None else list(range(len(values)))
    if any(index < 0 or index >= len(values) for index in indices):
        raise ValueError(f"{label} indices must be in [0, {len(values) - 1}]")
    return [(index, values[index]) for index in indices]


def base_row(encoded: EncodedPrompt) -> dict[str, Any]:
    return {
        "template_id": encoded.template_id,
        "template_name": encoded.template_name,
        "entity_id": encoded.entity_id,
        "entity": encoded.entity,
        "entity_group": encoded.entity_group,
        "entity_phrase": encoded.entity_phrase or encoded.entity,
        "entity_description": encoded.entity_description,
        "description_positions": list(encoded.description_positions),
        "description_tail_position": encoded.description_tail_position,
        "prompt_token_ids": encoded.ids,
        "prompt_tokens": encoded.tokens,
        "prompt": encoded.prompt,
        "entity_position": encoded.entity_position,
        "entity_token": encoded.tokens[encoded.entity_position],
        "readout_position": len(encoded.ids) - 1,
        "prompt_length": len(encoded.ids),
    }


def load_entity_descriptions(path: Path | None, entity_pool):
    if path is None:
        return {}, None
    raw = path.read_bytes()
    descriptions = json.loads(raw)
    expected = {entity for entity, _ in entity_pool}
    if not isinstance(descriptions, dict) or set(descriptions) != expected:
        raise ValueError("description manifest must exactly cover the entity pool")
    for entity, description in descriptions.items():
        if (
            not isinstance(description, str)
            or description != description.strip()
            or not description.startswith(("a ", "an "))
            or len(description.split()) < 2
            or "\n" in description or "\r" in description
        ):
            raise ValueError(f"{entity}: expected an indefinite description")
    return descriptions, hashlib.sha256(raw).hexdigest()


def validate_policy_counts(
    state: AblationState,
    encoded: EncodedPrompt,
    n_layers: int,
    generated_token_count: int,
    expected_readout_applications: int,
    block_intermediate_prompt: bool,
    expected_restore_applications: int = 0,
) -> None:
    expected_intermediate = 0
    if block_intermediate_prompt:
        expected_intermediate = n_layers * (
            len(encoded.ids) - encoded.entity_position - 2
        )
    expected_generated = n_layers * max(generated_token_count - 1, 0)
    observed = (
        state.applications,
        state.intermediate_applications,
        state.generated_applications,
        state.restore_applications,
    )
    expected = (
        expected_readout_applications,
        expected_intermediate,
        expected_generated,
        expected_restore_applications,
    )
    if observed != expected:
        raise RuntimeError(
            "entity policy application mismatch: "
            "observed readout/intermediate/generated/restore="
            f"{observed}, expected={expected}"
        )


def main() -> int:
    args = parse_args()
    restore_only = args.restore_entity_attention_start_layer is not None
    control_only = (
        args.baseline_only or args.ordinary_baseline_only or restore_only
    )
    if sum((args.baseline_only, args.ordinary_baseline_only, restore_only)) > 1:
        raise ValueError(
            "baseline-only, ordinary-baseline-only, and entity-attention "
            "restore modes are mutually exclusive"
        )
    if control_only and args.skip_baselines:
        raise ValueError(
            "control-only modes and --skip-baselines are mutually exclusive"
        )
    if args.ordinary_baseline_only and args.block_intermediate_prompt_entity:
        raise ValueError(
            "--ordinary-baseline-only cannot block intermediate prompt queries"
        )
    if restore_only and not args.block_intermediate_prompt_entity:
        raise ValueError(
            "entity-attention restore requires "
            "--block-intermediate-prompt-entity"
        )
    if any(width < 1 for width in args.widths):
        raise ValueError("all widths must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    entity_pool = ENTITY_SETS[args.entity_set]
    if len({entity for entity, _ in entity_pool}) != len(entity_pool):
        raise RuntimeError(f"{args.entity_set} contains duplicate entities")
    selected_templates = select(TEMPLATES, args.template_ids, "template")
    selected_entities = select(entity_pool, args.entity_ids, "entity")
    descriptions, descriptions_sha256 = load_entity_descriptions(
        args.entity_descriptions_json, entity_pool
    )
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir or f"results/qwen_entity_attention_ablation_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    results_path = out_dir / "generations.jsonl"

    device = torch.device(args.device)
    model, tokenizer, layers = load_model(args.model, device)
    n_layers = len(layers)
    if any(width > n_layers for width in args.widths):
        raise ValueError(f"widths cannot exceed the model's {n_layers} layers")
    if restore_only and not (
        0 <= args.restore_entity_attention_start_layer < n_layers
    ):
        raise ValueError(
            "restore start layer must be in "
            f"[0, {n_layers - 1}]"
        )

    cases = [
        encode_prompt(
            tokenizer, template, entity_id, entity, entity_group,
            description=descriptions.get(entity),
        )
        for _, template in selected_templates
        for entity_id, (entity, entity_group) in selected_entities
    ]
    spans = []
    for width in args.widths:
        starts = list(range(n_layers - width + 1))
        if args.start_layers is not None:
            starts = [start for start in starts if start in args.start_layers]
        spans.extend((width, start, start + width) for start in starts)
    if not spans and not control_only:
        raise ValueError("no valid layer spans selected")
    if control_only:
        spans = []

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "model": args.model,
        "device": str(device),
        "n_layers": n_layers,
        "layer_definition": "zero-based self-attention module index",
        "span_interval": "start_layer <= layer < end_layer_exclusive",
        "intervention": (
            "none; ordinary greedy generation with native SDPA"
            if args.ordinary_baseline_only
            else (
                "bottleneck plus an add-only final-prompt entity-attention "
                "rescue: generated and intermediate post-entity prompt queries "
                "lose the entity value in every layer; from the requested start "
                "layer through the final layer, the positive per-head ordinary-"
                "minus-live entity-attention deficit times the live entity value "
                "is added after native attention; other coefficients are unchanged "
                "and the effective coefficient sum may exceed one"
                if restore_only
                else (
                    "native SDPA with only the entity value zeroed before the "
                    "weighted sum; generated queries are blocked in every layer, "
                    "intermediate post-entity prompt queries are blocked in every "
                    "layer only when requested, and the final prompt query is "
                    "blocked only in the selected span; attention weights unchanged "
                    "and no renormalization"
                )
            )
        ),
        "generated_entity_value_blocked_all_layers": (
            not args.ordinary_baseline_only
        ),
        "intermediate_prompt_entity_value_blocked_all_layers": (
            args.block_intermediate_prompt_entity
            and not args.ordinary_baseline_only
        ),
        "final_prompt_entity_value_blocked_in_selected_span": (
            not control_only
        ),
        "final_prompt_entity_attention_restore": restore_only,
        "restore_policy": (
            "max(ordinary_entity_attention - live_entity_attention, 0) "
            "added times entity value after native attention, independently "
            "per query head; no renormalization"
            if restore_only
            else None
        ),
        "restore_start_layer": (
            args.restore_entity_attention_start_layer if restore_only else None
        ),
        "restore_end_layer_exclusive": n_layers if restore_only else None,
        "templates": [asdict(template) for _, template in selected_templates],
        "entity_set": args.entity_set,
        "entity_descriptions_json": (
            str(args.entity_descriptions_json.resolve())
            if args.entity_descriptions_json else None
        ),
        "entity_descriptions_sha256": descriptions_sha256,
        "entity_descriptions": descriptions,
        "entity_phrase_format": "{entity} ({description})" if descriptions else "{entity}",
        "ablation_and_answer_target": "original single-token entity name",
        "entities": [
            {
                "entity_id": entity_id,
                "entity": entity,
                "entity_group": entity_group,
            }
            for entity_id, (entity, entity_group) in selected_entities
        ],
        "widths": [] if control_only else args.widths,
        "start_layers": None if control_only else args.start_layers,
        "max_new_tokens": args.max_new_tokens,
        "baseline_only": args.baseline_only,
        "ordinary_baseline_only": args.ordinary_baseline_only,
        "skip_baselines": args.skip_baselines,
        "n_baseline_trials": (
            0 if args.skip_baselines or restore_only else len(cases)
        ),
        "n_ordinary_reference_prefills": len(cases) if restore_only else 0,
        "n_restore_trials": len(cases) if restore_only else 0,
        "n_trials_per_span": 0 if control_only else len(cases),
        "scoring_note": (
            "entity_in_completion is a whole-word name diagnostic only; for "
            "occupation and gender, inspect the complete baseline/control "
            "outputs and freeze their normalized words/subwords before comparing "
            "later intervention outputs"
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    if restore_only:
        restore_start = args.restore_entity_attention_start_layer
        restore_layers = range(restore_start, n_layers)
        with results_path.open("x") as output:
            print(
                f"entity-attention rescue: {len(cases)} cases, "
                f"layers [{restore_start}, {n_layers})",
                flush=True,
            )
            for case_index, encoded in enumerate(cases, start=1):
                ordinary_reference = ordinary_entity_attention_reference(
                    model,
                    encoded,
                    device,
                    restore_layers,
                )
                with disable_entity_attention(
                    range(0),
                    encoded.entity_position,
                    len(encoded.ids) - 1,
                    len(encoded.ids),
                    True,
                    restore_layers=restore_layers,
                    ordinary_entity_attention=ordinary_reference,
                ) as state:
                    completion = completion_fields(
                        greedy_completion(
                            model,
                            tokenizer,
                            encoded,
                            device,
                            args.max_new_tokens,
                        ),
                        encoded.entity,
                    )
                    validate_policy_counts(
                        state,
                        encoded,
                        n_layers,
                        len(completion["generated_token_ids"]),
                        0,
                        True,
                        expected_restore_applications=n_layers - restore_start,
                    )
                    restore_trace = {
                        str(layer): trace
                        for layer, trace in sorted(
                            (state.restore_trace or {}).items()
                        )
                    }
                    intermediate_applications = state.intermediate_applications
                    generated_applications = state.generated_applications
                    restore_applications = state.restore_applications
                positive_deficit_heads = sum(
                    deficit > 0.0
                    for trace in restore_trace.values()
                    for deficit in trace["added_entity_attention_deficit"]
                )
                total_deficit = sum(
                    deficit
                    for trace in restore_trace.values()
                    for deficit in trace["added_entity_attention_deficit"]
                )
                write_jsonl(output, {
                    "condition": "last_token_bottleneck_entity_attention_restore",
                    **base_row(encoded),
                    "width": n_layers - restore_start,
                    "start_layer": restore_start,
                    "end_layer_exclusive": n_layers,
                    "disabled_layers": [],
                    "restore_layers": list(restore_layers),
                    "block_intermediate_prompt_entity": True,
                    "attention_ablation_applications": 0,
                    "readout_attention_ablation_applications": 0,
                    "intermediate_attention_ablation_applications": (
                        intermediate_applications
                    ),
                    "generated_attention_ablation_applications": (
                        generated_applications
                    ),
                    "readout_entity_attention_restore_applications": (
                        restore_applications
                    ),
                    "positive_deficit_head_layers": positive_deficit_heads,
                    "total_added_entity_attention_deficit": total_deficit,
                    "entity_attention_restore_trace": restore_trace,
                    **completion,
                })
                print(
                    f"  restore {case_index:03d}/{len(cases)} "
                    f"{encoded.template_name}/{encoded.entity}: "
                    f"{completion['completion']!r}",
                    flush=True,
                )
        print(f"saved {results_path}", flush=True)
        return 0

    with results_path.open("x") as output:
        if not args.skip_baselines:
            print(f"baselines: {len(cases)}", flush=True)
            for case_index, encoded in enumerate(cases, start=1):
                if args.ordinary_baseline_only:
                    completion = completion_fields(
                        greedy_completion(
                            model, tokenizer, encoded, device, args.max_new_tokens
                        ),
                        encoded.entity,
                    )
                    intermediate_applications = 0
                    generated_applications = 0
                else:
                    with disable_entity_attention(
                        range(0),
                        encoded.entity_position,
                        len(encoded.ids) - 1,
                        len(encoded.ids),
                        args.block_intermediate_prompt_entity,
                    ) as state:
                        completion = completion_fields(
                            greedy_completion(
                                model,
                                tokenizer,
                                encoded,
                                device,
                                args.max_new_tokens,
                            ),
                            encoded.entity,
                        )
                        validate_policy_counts(
                            state,
                            encoded,
                            n_layers,
                            len(completion["generated_token_ids"]),
                            0,
                            args.block_intermediate_prompt_entity,
                        )
                        intermediate_applications = state.intermediate_applications
                        generated_applications = state.generated_applications
                write_jsonl(output, {
                    "condition": (
                        "ordinary_baseline"
                        if args.ordinary_baseline_only
                        else (
                            "last_token_bottleneck_control"
                            if args.block_intermediate_prompt_entity
                            else "generated_blocked_control"
                        )
                    ),
                    **base_row(encoded),
                    "width": None,
                    "start_layer": None,
                    "end_layer_exclusive": None,
                    "disabled_layers": [],
                    "block_intermediate_prompt_entity": (
                        args.block_intermediate_prompt_entity
                    ),
                    "readout_attention_ablation_applications": 0,
                    "intermediate_attention_ablation_applications": (
                        intermediate_applications
                    ),
                    "generated_attention_ablation_applications": (
                        generated_applications
                    ),
                    **completion,
                })
                print(
                    f"  baseline {case_index:02d}/{len(cases)} "
                    f"{encoded.template_name}/{encoded.entity}: "
                    f"{completion['completion']!r}",
                    flush=True,
                )

        for span_index, (width, start, end) in enumerate(spans, start=1):
            print(
                f"span {span_index:03d}/{len(spans)}: "
                f"layers [{start}, {end}), width={width}",
                flush=True,
            )
            for encoded in cases:
                with disable_entity_attention(
                    range(start, end),
                    encoded.entity_position,
                    len(encoded.ids) - 1,
                    len(encoded.ids),
                    args.block_intermediate_prompt_entity,
                ) as state:
                    completion = completion_fields(
                        greedy_completion(
                            model, tokenizer, encoded, device, args.max_new_tokens
                        ),
                        encoded.entity,
                    )
                    applications = state.applications
                    validate_policy_counts(
                        state,
                        encoded,
                        n_layers,
                        len(completion["generated_token_ids"]),
                        width,
                        args.block_intermediate_prompt_entity,
                    )
                    intermediate_applications = state.intermediate_applications
                    generated_applications = state.generated_applications
                write_jsonl(output, {
                    "condition": (
                        "last_token_bottleneck_ablation"
                        if args.block_intermediate_prompt_entity
                        else "last_token_readout_ablation"
                    ),
                    **base_row(encoded),
                    "width": width,
                    "start_layer": start,
                    "end_layer_exclusive": end,
                    "disabled_layers": list(range(start, end)),
                    "block_intermediate_prompt_entity": (
                        args.block_intermediate_prompt_entity
                    ),
                    "attention_ablation_applications": applications,
                    "readout_attention_ablation_applications": applications,
                    "intermediate_attention_ablation_applications": (
                        intermediate_applications
                    ),
                    "generated_attention_ablation_applications": (
                        generated_applications
                    ),
                    **completion,
                })

    print(f"saved {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
