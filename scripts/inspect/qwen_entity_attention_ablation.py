"""Ablate direct attention to an entity in consecutive Qwen layers.

Each prompt contains one celebrity. For every width/start-layer span, this
script runs ordinary greedy generation while removing the post-softmax entity
contribution in the selected layers. It does this with the native SDPA backend
by zeroing only the entity value before the weighted sum, which leaves every
attention weight unchanged and performs no renormalization. The intervention
applies to queries after the entity during both prompt prefill and cached
generation. Raw continuations are saved for manual semantic scoring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


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


@dataclass
class AblationState:
    disabled_layers: frozenset[int] = frozenset()
    entity_position: int | None = None
    applications: int = 0


ABLATION = AblationState()


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

    if (
        ABLATION.entity_position is not None
        and module.layer_idx in ABLATION.disabled_layers
    ):
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        if not 0 <= ABLATION.entity_position < key_length:
            raise RuntimeError(
                f"entity position {ABLATION.entity_position} outside key length {key_length}"
            )
        query_start = key_length - query_length
        first_query = max(ABLATION.entity_position + 1 - query_start, 0)
        if first_query < query_length:
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
            ABLATION.applications += 1
            if first_query == 0:
                return zero_output, None

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
            attention_output[:, first_query:, :, :] = zero_output[
                :, first_query:, :, :
            ]
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
def disable_entity_attention(layers: range, entity_position: int):
    if ABLATION.entity_position is not None:
        raise RuntimeError("nested entity-attention ablations are not supported")
    ABLATION.disabled_layers = frozenset(layers)
    ABLATION.entity_position = entity_position
    ABLATION.applications = 0
    try:
        yield ABLATION
    finally:
        ABLATION.disabled_layers = frozenset()
        ABLATION.entity_position = None
        ABLATION.applications = 0


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
) -> EncodedPrompt:
    prompt = template.text.format(entity=entity)
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    start = prompt.index(entity)
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
    parser.add_argument("--baseline-only", action="store_true")
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
        "prompt": encoded.prompt,
        "entity_position": encoded.entity_position,
        "entity_token": encoded.tokens[encoded.entity_position],
    }


def main() -> int:
    args = parse_args()
    if args.baseline_only and args.skip_baselines:
        raise ValueError("--baseline-only and --skip-baselines are mutually exclusive")
    if any(width < 1 for width in args.widths):
        raise ValueError("all widths must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    entity_pool = ENTITY_SETS[args.entity_set]
    if len({entity for entity, _ in entity_pool}) != len(entity_pool):
        raise RuntimeError(f"{args.entity_set} contains duplicate entities")
    selected_templates = select(TEMPLATES, args.template_ids, "template")
    selected_entities = select(entity_pool, args.entity_ids, "entity")
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

    cases = [
        encode_prompt(tokenizer, template, entity_id, entity, entity_group)
        for _, template in selected_templates
        for entity_id, (entity, entity_group) in selected_entities
    ]
    spans = []
    for width in args.widths:
        starts = list(range(n_layers - width + 1))
        if args.start_layers is not None:
            starts = [start for start in starts if start in args.start_layers]
        spans.extend((width, start, start + width) for start in starts)
    if not spans and not args.baseline_only:
        raise ValueError("no valid layer spans selected")
    if args.baseline_only:
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
            "native SDPA with only the entity value zeroed before the weighted "
            "sum for queries after the entity during prefill and cached generation; "
            "attention weights unchanged and no renormalization"
        ),
        "templates": [asdict(template) for _, template in selected_templates],
        "entity_set": args.entity_set,
        "entities": [
            {
                "entity_id": entity_id,
                "entity": entity,
                "entity_group": entity_group,
            }
            for entity_id, (entity, entity_group) in selected_entities
        ],
        "widths": args.widths,
        "start_layers": args.start_layers,
        "max_new_tokens": args.max_new_tokens,
        "baseline_only": args.baseline_only,
        "skip_baselines": args.skip_baselines,
        "n_baseline_trials": 0 if args.skip_baselines else len(cases),
        "n_trials_per_span": len(cases),
        "scoring_note": (
            "entity_in_completion is a whole-word diagnostic; inspect unique raw "
            "completions before reporting semantic accuracy"
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    with results_path.open("x") as output:
        if not args.skip_baselines:
            print(f"baselines: {len(cases)}", flush=True)
            for case_index, encoded in enumerate(cases, start=1):
                completion = completion_fields(
                    greedy_completion(
                        model, tokenizer, encoded, device, args.max_new_tokens
                    ),
                    encoded.entity,
                )
                write_jsonl(output, {
                    "condition": "baseline",
                    **base_row(encoded),
                    "width": None,
                    "start_layer": None,
                    "end_layer_exclusive": None,
                    "disabled_layers": [],
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
                    range(start, end), encoded.entity_position
                ) as state:
                    completion = completion_fields(
                        greedy_completion(
                            model, tokenizer, encoded, device, args.max_new_tokens
                        ),
                        encoded.entity,
                    )
                    applications = state.applications
                if applications < width:
                    raise RuntimeError(
                        f"intervention applied {applications} times for width {width}"
                    )
                write_jsonl(output, {
                    "condition": "entity_attention_ablation",
                    **base_row(encoded),
                    "width": width,
                    "start_layer": start,
                    "end_layer_exclusive": end,
                    "disabled_layers": list(range(start, end)),
                    "attention_ablation_applications": applications,
                    **completion,
                })

    print(f"saved {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
