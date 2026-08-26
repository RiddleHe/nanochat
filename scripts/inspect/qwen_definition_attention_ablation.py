"""Ablate final-readout attention to a rotating entity in Qwen.

Ten one-token entity/definition pairs are rotated cyclically. In each prompt,
nine pairs are complete demonstrations and the held-out entity is placed last
with only a colon, inviting a definition. For every four-layer span, only the
final colon query loses the held-out entity's value contribution. Native SDPA,
all attention weights, and every other query/value contribution are unchanged.
Raw greedy continuations are saved for manual semantic inspection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


MODEL = "Qwen/Qwen3-8B-Base"
ATTENTION_IMPLEMENTATION = "sdpa"
WIDTH = 4
PROMPT_PREFIX = "Definitions: "


@dataclass(frozen=True)
class EntityDefinition:
    entity: str
    definition: str


ENTITY_DEFINITIONS = (
    EntityDefinition("Syria", "Country in the Middle East"),
    EntityDefinition("Samsung", "South Korean electronics company"),
    EntityDefinition("Tokyo", "Capital of Japan"),
    EntityDefinition("Mozart", "Austrian composer"),
    EntityDefinition("Python", "General-purpose programming language"),
    EntityDefinition("Toyota", "Japanese automobile manufacturer"),
    EntityDefinition("Nile", "Major river in Africa"),
    EntityDefinition("Shakespeare", "English playwright"),
    EntityDefinition("Mars", "Fourth planet from the Sun"),
    EntityDefinition("Picasso", "Spanish painter"),
)


@dataclass(frozen=True)
class RenderedPrompt:
    rotation_id: int
    target_id: int
    target: EntityDefinition
    demonstrations: tuple[EntityDefinition, ...]
    prompt: str
    label_spans: tuple[tuple[str, int, int], ...]
    target_span: tuple[int, int]


@dataclass(frozen=True)
class EncodedPrompt:
    rotation_id: int
    target_id: int
    target_entity: str
    target_definition: str
    demonstration_order: tuple[str, ...]
    prompt: str
    ids: list[int]
    tokens: list[str]
    demonstration_positions: tuple[int, ...]
    target_position: int
    readout_position: int


@dataclass
class AblationState:
    disabled_layers: frozenset[int] = frozenset()
    target_position: int | None = None
    readout_position: int | None = None
    applications: int = 0


ABLATION = AblationState()


def target_zero_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
):
    """Remove only the final prompt query's target-value contribution."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    active = (
        ABLATION.target_position is not None
        and ABLATION.readout_position is not None
        and module.layer_idx in ABLATION.disabled_layers
    )
    if active:
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        if not 0 <= ABLATION.target_position < key_length:
            raise RuntimeError(
                f"target position {ABLATION.target_position} outside key length "
                f"{key_length}"
            )

        query_start = key_length - query_length
        local_readout = ABLATION.readout_position - query_start
        if 0 <= local_readout < query_length:
            zero_value = value.clone()
            zero_value[:, :, ABLATION.target_position, :] = 0
            ablated_output, _ = sdpa_attention_forward(
                module,
                query,
                key,
                zero_value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **_kwargs,
            )
            normal_output, _ = sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **_kwargs,
            )
            normal_output = normal_output.clone()
            normal_output[:, local_readout : local_readout + 1, :, :] = (
                ablated_output[:, local_readout : local_readout + 1, :, :]
            )
            ABLATION.applications += 1
            return normal_output, None

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
def disable_target_attention(
    layers: range,
    target_position: int,
    readout_position: int,
):
    if ABLATION.target_position is not None:
        raise RuntimeError("nested target-attention ablations are not supported")
    ABLATION.disabled_layers = frozenset(layers)
    ABLATION.target_position = target_position
    ABLATION.readout_position = readout_position
    ABLATION.applications = 0
    try:
        yield ABLATION
    finally:
        ABLATION.disabled_layers = frozenset()
        ABLATION.target_position = None
        ABLATION.readout_position = None
        ABLATION.applications = 0


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def render_prompt(rotation_id: int) -> RenderedPrompt:
    if not 0 <= rotation_id < len(ENTITY_DEFINITIONS):
        raise ValueError(f"rotation id {rotation_id} is out of range")

    target = ENTITY_DEFINITIONS[rotation_id]
    demonstrations = (
        ENTITY_DEFINITIONS[rotation_id + 1 :] + ENTITY_DEFINITIONS[:rotation_id]
    )
    chunks = [PROMPT_PREFIX]
    label_spans: list[tuple[str, int, int]] = []
    cursor = len(PROMPT_PREFIX)

    for index, item in enumerate(demonstrations):
        if index:
            chunks.append("; ")
            cursor += 2
        start = cursor
        chunks.append(item.entity)
        cursor += len(item.entity)
        label_spans.append((item.entity, start, cursor))
        suffix = f": {item.definition}"
        chunks.append(suffix)
        cursor += len(suffix)

    chunks.append("; ")
    cursor += 2
    target_start = cursor
    chunks.append(target.entity)
    cursor += len(target.entity)
    target_span = (target_start, cursor)
    label_spans.append((target.entity, *target_span))
    chunks.append(":")
    prompt = "".join(chunks)

    if not prompt.endswith(":") or prompt.endswith(": "):
        raise AssertionError("prompt must end at the target colon")
    return RenderedPrompt(
        rotation_id=rotation_id,
        target_id=rotation_id,
        target=target,
        demonstrations=demonstrations,
        prompt=prompt,
        label_spans=tuple(label_spans),
        target_span=target_span,
    )


def overlapping_positions(offsets, start: int, end: int) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if (left, right) != (0, 0) and right > start and left < end
    ]


def encode_prompt(tokenizer, rotation_id: int) -> EncodedPrompt:
    rendered = render_prompt(rotation_id)
    encoded = tokenizer(
        rendered.prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    label_positions: list[int] = []
    for entity, start, end in rendered.label_spans:
        positions = overlapping_positions(offsets, start, end)
        if len(positions) != 1:
            raise RuntimeError(
                f"rotation {rotation_id}/{entity}: entity label must be exactly "
                f"one token; got positions {positions}"
            )
        label_positions.append(positions[0])

    target_position = label_positions[-1]
    readout_positions = overlapping_positions(
        offsets,
        len(rendered.prompt) - 1,
        len(rendered.prompt),
    )
    if len(readout_positions) != 1:
        raise RuntimeError(
            f"rotation {rotation_id}: final colon must be exactly one token; "
            f"got positions {readout_positions}"
        )

    ids = list(encoded["input_ids"])
    tokens = list(tokenizer.convert_ids_to_tokens(ids))
    readout_position = readout_positions[0]
    if readout_position != len(ids) - 1 or tokens[readout_position] != ":":
        raise RuntimeError(
            f"rotation {rotation_id}: expected final token ':', got "
            f"position {readout_position}/{len(ids) - 1} token "
            f"{tokens[readout_position]!r}"
        )

    return EncodedPrompt(
        rotation_id=rotation_id,
        target_id=rendered.target_id,
        target_entity=rendered.target.entity,
        target_definition=rendered.target.definition,
        demonstration_order=tuple(item.entity for item in rendered.demonstrations),
        prompt=rendered.prompt,
        ids=ids,
        tokens=tokens,
        demonstration_positions=tuple(label_positions[:-1]),
        target_position=target_position,
        readout_position=readout_position,
    )


def load_model(model_name: str, device: torch.device):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register(
        ATTENTION_IMPLEMENTATION,
        target_zero_attention_forward,
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
    generated_ids = generated.sequences[0, len(encoded.ids) :].detach().cpu().tolist()
    if not generated_ids or not generated.scores:
        raise RuntimeError("generation produced no next token")
    next_token_id = int(generated_ids[0])
    return {
        "next_token_id": next_token_id,
        "next_token_text": tokenizer.decode(
            [next_token_id],
            skip_special_tokens=False,
        ),
        "next_token_score": float(
            generated.scores[0][0, next_token_id].float().cpu()
        ),
        "generated_token_ids": generated_ids,
        "completion": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--start-layers",
        type=parse_ints,
        help="optional comma-separated subset; default is every valid start layer",
    )
    parser.add_argument(
        "--rotation-ids",
        type=parse_ints,
        help="optional comma-separated subset; default is all ten rotations",
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def selected_rotations(rotation_ids: list[int] | None) -> list[int]:
    rotations = rotation_ids or list(range(len(ENTITY_DEFINITIONS)))
    if any(index < 0 or index >= len(ENTITY_DEFINITIONS) for index in rotations):
        raise ValueError(
            f"rotation ids must be in [0, {len(ENTITY_DEFINITIONS) - 1}]"
        )
    return rotations


def base_row(encoded: EncodedPrompt) -> dict[str, Any]:
    return {
        "rotation_id": encoded.rotation_id,
        "target_id": encoded.target_id,
        "target_entity": encoded.target_entity,
        "target_definition": encoded.target_definition,
        "demonstration_order": list(encoded.demonstration_order),
        "prompt": encoded.prompt,
        "prompt_token_count": len(encoded.ids),
        "target_position": encoded.target_position,
        "target_token": encoded.tokens[encoded.target_position],
        "readout_position": encoded.readout_position,
        "readout_token": encoded.tokens[encoded.readout_position],
        "demonstration_positions": list(encoded.demonstration_positions),
    }


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    rotations = selected_rotations(args.rotation_ids)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir or f"results/qwen_definition_attention_ablation_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    results_path = out_dir / "generations.jsonl"

    device = torch.device(args.device)
    model, tokenizer, layers = load_model(args.model, device)
    n_layers = len(layers)
    if WIDTH > n_layers:
        raise ValueError(f"width {WIDTH} exceeds the model's {n_layers} layers")

    cases = [encode_prompt(tokenizer, rotation_id) for rotation_id in rotations]
    starts = list(range(n_layers - WIDTH + 1))
    if args.start_layers is not None:
        invalid = [start for start in args.start_layers if start not in starts]
        if invalid:
            raise ValueError(
                f"start layers must be in [0, {n_layers - WIDTH}]; got {invalid}"
            )
        starts = args.start_layers
    if args.baseline_only:
        starts = []

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "model": args.model,
        "device": str(device),
        "n_layers": n_layers,
        "layer_definition": "zero-based self-attention module index",
        "span_interval": "start_layer <= layer < end_layer_exclusive",
        "width": WIDTH,
        "intervention": (
            "native SDPA with only the held-out target entity value zeroed for "
            "the final prompt colon query in the selected four layers; attention "
            "weights unchanged, no renormalization, and generated-token queries "
            "unmodified"
        ),
        "prompt_prefix": PROMPT_PREFIX,
        "entity_definitions": [asdict(item) for item in ENTITY_DEFINITIONS],
        "rotations": rotations,
        "start_layers": starts,
        "max_new_tokens": args.max_new_tokens,
        "baseline_only": args.baseline_only,
        "n_baseline_trials": len(cases),
        "n_trials_per_span": len(cases),
        "expected_rows": len(cases) * (1 + len(starts)),
        "scoring_note": (
            "No automatic semantic score is saved. Inspect raw continuations "
            "without layer labels before assigning correct-definition judgments."
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    with results_path.open("x") as output:
        print(f"baselines: {len(cases)}", flush=True)
        for case_index, encoded in enumerate(cases, start=1):
            completion = greedy_completion(
                model,
                tokenizer,
                encoded,
                device,
                args.max_new_tokens,
            )
            write_jsonl(
                output,
                {
                    "condition": "baseline",
                    **base_row(encoded),
                    "width": None,
                    "start_layer": None,
                    "end_layer_exclusive": None,
                    "disabled_layers": [],
                    "attention_ablation_applications": 0,
                    **completion,
                },
            )
            print(
                f"  baseline {case_index:02d}/{len(cases)} "
                f"target={encoded.target_entity}: {completion['completion']!r}",
                flush=True,
            )

        for span_index, start in enumerate(starts, start=1):
            end = start + WIDTH
            print(
                f"span {span_index:02d}/{len(starts)}: "
                f"layers [{start}, {end}), width={WIDTH}",
                flush=True,
            )
            for encoded in cases:
                with disable_target_attention(
                    range(start, end),
                    encoded.target_position,
                    encoded.readout_position,
                ) as state:
                    completion = greedy_completion(
                        model,
                        tokenizer,
                        encoded,
                        device,
                        args.max_new_tokens,
                    )
                    applications = state.applications
                if applications != WIDTH:
                    raise RuntimeError(
                        f"intervention applied {applications} times for width {WIDTH}"
                    )
                write_jsonl(
                    output,
                    {
                        "condition": "target_attention_ablation",
                        **base_row(encoded),
                        "width": WIDTH,
                        "start_layer": start,
                        "end_layer_exclusive": end,
                        "disabled_layers": list(range(start, end)),
                        "attention_ablation_applications": applications,
                        **completion,
                    },
                )

    print(f"saved {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
