"""Fixed-width two-stage entity-to-readout relay for Qwen.

For each template, directed entity pair, and (source layer s, relay layer t), this script:
1. copies the donor entity-span state after block s into the recipient prompt;
2. captures the resulting final-prompt-token state after block t;
3. copies that state into a fresh recipient pass after block t; and
4. greedily records the next token and raw continuation.

Layer indices name transformer-block outputs. The default windows satisfy
t - s in {1, 2, 4}. Every donor and recipient entity must occupy exactly one
token. Whole-word name-presence fields are diagnostics, not final semantic
labels; follow qwen_entity_relay_fixed_window.md to analyze a completed run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


MODEL = "Qwen/Qwen3-8B-Base"


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
PAIRS = [
    ("Einstein", "Newton"),
    ("Darwin", "Tesla"),
    ("Mozart", "Bach"),
    ("Shakespeare", "Dickens"),
    ("Tolkien", "Orwell"),
    ("Messi", "Ronaldo"),
    ("Elvis", "Drake"),
    ("Madonna", "Cher"),
    ("Gandhi", "Mandela"),
    ("Lincoln", "Kennedy"),
]


@dataclass(frozen=True)
class EncodedPrompt:
    entity: str
    prompt: str
    ids: list[int]
    tokens: list[str]
    entity_positions: list[int]


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def load_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = list(model.gpt_neox.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
    else:
        raise RuntimeError(f"cannot locate transformer blocks for {model_name}")
    return model, tokenizer, layers


def encode_prompt(tokenizer, template: Template, entity: str) -> EncodedPrompt:
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
    if not positions:
        raise RuntimeError(f"could not locate {entity!r} in its tokenized prompt")
    ids = list(encoded["input_ids"])
    return EncodedPrompt(
        entity=entity,
        prompt=prompt,
        ids=ids,
        tokens=list(tokenizer.convert_ids_to_tokens(ids)),
        entity_positions=positions,
    )


def validate_pair(donor: EncodedPrompt, recipient: EncodedPrompt) -> None:
    if len(donor.entity_positions) != 1 or len(recipient.entity_positions) != 1:
        raise RuntimeError(
            f"{donor.entity}->{recipient.entity}: every entity must be exactly one token "
            f"({donor.entity_positions} vs {recipient.entity_positions})"
        )
    if donor.entity_positions != recipient.entity_positions:
        raise RuntimeError(
            f"{donor.entity}->{recipient.entity}: entity spans do not align "
            f"({donor.entity_positions} vs {recipient.entity_positions})"
        )
    if len(donor.ids) != len(recipient.ids):
        raise RuntimeError(
            f"{donor.entity}->{recipient.entity}: prompt token lengths differ "
            f"({len(donor.ids)} vs {len(recipient.ids)})"
        )
    entity_positions = set(donor.entity_positions)
    other_differences = [
        index
        for index, (donor_id, recipient_id) in enumerate(zip(donor.ids, recipient.ids))
        if donor_id != recipient_id and index not in entity_positions
    ]
    if other_differences:
        raise RuntimeError(
            f"{donor.entity}->{recipient.entity}: tokenization also differs outside "
            f"the entity span at {other_differences}"
        )


def replace_hidden(out: Any, hidden: torch.Tensor) -> Any:
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


@torch.inference_mode()
def capture_donor_entity_states(model, layers, encoded: EncodedPrompt, device: torch.device):
    states: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, out):
            hidden = out[0] if isinstance(out, tuple) else out
            states[layer_index] = hidden[0, encoded.entity_positions, :].detach().clone()

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(state is None for state in states):
        raise RuntimeError("failed to capture every donor layer")
    return states


@torch.inference_mode()
def build_relay_state(
    model,
    layers,
    recipient: EncodedPrompt,
    source_layer: int,
    relay_layer: int,
    source_state: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    relay_state: torch.Tensor | None = None

    def patch_entity(_module, _inputs, out):
        hidden0 = out[0] if isinstance(out, tuple) else out
        hidden = hidden0.clone()
        hidden[0, recipient.entity_positions, :] = source_state.to(
            device=hidden.device, dtype=hidden.dtype
        )
        return replace_hidden(out, hidden)

    def capture_readout(_module, _inputs, out):
        nonlocal relay_state
        hidden = out[0] if isinstance(out, tuple) else out
        relay_state = hidden[0, -1, :].detach().clone()

    patch_handle = layers[source_layer].register_forward_hook(patch_entity)
    relay_handle = layers[relay_layer].register_forward_hook(capture_readout)
    try:
        input_ids = torch.tensor([recipient.ids], dtype=torch.long, device=device)
        model(input_ids=input_ids, use_cache=False)
    finally:
        patch_handle.remove()
        relay_handle.remove()
    if relay_state is None:
        raise RuntimeError(f"failed to capture relay state at layer {relay_layer}")
    return relay_state


@torch.inference_mode()
def greedy_completion(
    model,
    tokenizer,
    layers,
    encoded: EncodedPrompt,
    device: torch.device,
    max_new_tokens: int,
    relay_layer: int | None = None,
    relay_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    handle = None
    prompt_length = len(encoded.ids)
    if relay_layer is not None:
        if relay_state is None:
            raise ValueError("relay_state is required with relay_layer")

        def patch_readout(_module, _inputs, out):
            hidden0 = out[0] if isinstance(out, tuple) else out
            # Generation calls the hook again with one-token cached decoding.
            # Patch only the initial recipient-prompt prefill.
            if hidden0.shape[1] != prompt_length:
                return out
            hidden = hidden0.clone()
            hidden[0, -1, :] = relay_state.to(device=hidden.device, dtype=hidden.dtype)
            return replace_hidden(out, hidden)

        handle = layers[relay_layer].register_forward_hook(patch_readout)

    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    try:
        generated = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    generated_ids = generated.sequences[0, prompt_length:].detach().cpu().tolist()
    if not generated_ids or not generated.scores:
        raise RuntimeError("generation produced no next token")
    next_token_id = int(generated_ids[0])
    return {
        "next_token_id": next_token_id,
        "next_token_text": tokenizer.decode([next_token_id], skip_special_tokens=False),
        "next_token_score": float(generated.scores[0][0, next_token_id].float().cpu()),
        "generated_token_ids": generated_ids,
        "completion": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def normalize_text(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return unicodedata.normalize("NFKC", text).casefold().strip()


def contains_name(text: str, name: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(normalize_text(name))}(?!\w)",
        normalize_text(text),
    ) is not None


def score_completion(
    completion: dict[str, Any], expected: str, alternative: str
) -> dict[str, Any]:
    expected_present = contains_name(completion["completion"], expected)
    alternative_present = contains_name(completion["completion"], alternative)
    if expected_present and alternative_present:
        category = "both"
    elif expected_present:
        category = "expected_only"
    elif alternative_present:
        category = "alternative_only"
    else:
        category = "neither"
    return {
        **completion,
        "expected": expected,
        "alternative": alternative,
        "normalized_completion": normalize_text(completion["completion"]),
        "expected_in_completion": expected_present,
        "alternative_in_completion": alternative_present,
        "output_category": category,
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
        "--source-layers",
        type=parse_ints,
        help="optional comma-separated subset; default is every valid source layer",
    )
    parser.add_argument("--pair-ids", type=parse_ints, help="optional zero-based pair indices")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(width < 1 for width in args.widths):
        raise ValueError("all widths must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    selected_ids = args.pair_ids if args.pair_ids is not None else list(range(len(PAIRS)))
    if any(index < 0 or index >= len(PAIRS) for index in selected_ids):
        raise ValueError(f"pair indices must be in [0, {len(PAIRS) - 1}]")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"results/qwen_entity_relay_fixed_window_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    results_path = out_dir / "generations.jsonl"

    device = torch.device(args.device)
    model, tokenizer, layers = load_model(args.model, device)
    n_layers = len(layers)
    if any(width >= n_layers for width in args.widths):
        raise ValueError(f"widths must be smaller than the model's {n_layers} layers")

    encoded_cases = []
    for template in TEMPLATES:
        for pair_id in selected_ids:
            donor_name, recipient_name = PAIRS[pair_id]
            donor = encode_prompt(tokenizer, template, donor_name)
            recipient = encode_prompt(tokenizer, template, recipient_name)
            validate_pair(donor, recipient)
            encoded_cases.append((template, pair_id, donor, recipient))

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "model": args.model,
        "device": str(device),
        "n_layers": n_layers,
        "layer_definition": "transformer-block output; layer 0 is block 0 output",
        "templates": [asdict(template) for template in TEMPLATES],
        "widths": args.widths,
        "source_layers": args.source_layers,
        "max_new_tokens": args.max_new_tokens,
        "pairs": [
            {"pair_id": pair_id, "donor": PAIRS[pair_id][0], "recipient": PAIRS[pair_id][1]}
            for pair_id in selected_ids
        ],
        "n_relay_trials_per_span": len(TEMPLATES) * len(selected_ids),
        "classification": (
            "case-insensitive whole-word donor/recipient-name presence in the full "
            "generated completion: expected_only, alternative_only, both, or neither"
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    with results_path.open("x") as output:
        for case_number, (template, pair_id, donor, recipient) in enumerate(encoded_cases, start=1):
            print(
                f"case {case_number}/{len(encoded_cases)}: {template.name}, "
                f"{donor.entity}->{recipient.entity}",
                flush=True,
            )
            for condition, encoded in (("donor_baseline", donor), ("recipient_baseline", recipient)):
                expected = donor.entity if condition == "donor_baseline" else recipient.entity
                row = {
                    "condition": condition,
                    "template_id": template.template_id,
                    "template_name": template.name,
                    "pair_id": pair_id,
                    "donor": donor.entity,
                    "recipient": recipient.entity,
                    "source_layer_s": None,
                    "relay_layer_t": None,
                    "width": None,
                    **score_completion(
                        greedy_completion(
                            model, tokenizer, layers, encoded, device, args.max_new_tokens
                        ),
                        expected,
                        recipient.entity if condition == "donor_baseline" else donor.entity,
                    ),
                }
                write_jsonl(output, row)

            donor_states = capture_donor_entity_states(model, layers, donor, device)
            for width in args.widths:
                valid_sources = list(range(n_layers - width))
                if args.source_layers is not None:
                    valid_sources = [s for s in args.source_layers if 0 <= s < n_layers - width]
                for source_layer in valid_sources:
                    relay_layer = source_layer + width
                    relay_state = build_relay_state(
                        model,
                        layers,
                        recipient,
                        source_layer,
                        relay_layer,
                        donor_states[source_layer],
                        device,
                    )
                    completion = score_completion(
                        greedy_completion(
                            model,
                            tokenizer,
                            layers,
                            recipient,
                            device,
                            args.max_new_tokens,
                            relay_layer=relay_layer,
                            relay_state=relay_state,
                        ),
                        donor.entity,
                        recipient.entity,
                    )
                    row = {
                        "condition": "entity_to_final_relay",
                        "template_id": template.template_id,
                        "template_name": template.name,
                        "pair_id": pair_id,
                        "donor": donor.entity,
                        "recipient": recipient.entity,
                        "source_layer_s": source_layer,
                        "relay_layer_t": relay_layer,
                        "width": width,
                        **completion,
                    }
                    write_jsonl(output, row)
                    print(
                        f"  s={source_layer:02d} t={relay_layer:02d} w={width}: "
                        f"{completion['completion']!r}",
                        flush=True,
                    )

    print(f"saved {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
