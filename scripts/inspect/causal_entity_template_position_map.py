"""Template/position causal map for entity-information handoff in Qwen3-8B-Base.

This is same-prompt causal activation patching, not Patchscope target-prompt
injection.  Layer 0 is the cumulative residual output of transformer block 0.
At a chosen layer and same token index, recipient block-output residuals are
replaced by unmodified donor residuals.  Recovery is the established,
unclipped donor-minus-recipient next-token logit-margin normalization.

The program is deliberately resumable and has four modes:

* smoke: global preflight plus the required gated smoke experiment;
* worker --phase phase1: grouped-position scans for one cluster shard;
* worker --phase phase2: complete token-position maps after Phase 1 passes;
* analyze: merge, validate, cluster-bootstrap, plot, and report.

Reciprocal directions are assigned by their canonical unordered cluster, so a
cluster can never be split across workers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch


MODEL = "Qwen/Qwen3-8B-Base"
N_LAYERS = 36
BASELINE_SEPARATION_THRESHOLD = 0.5
BF16_CONTROL_FLOOR = 0.0638297872340425
REFERENCE_RELAY = Path("results/patchscopes/qwen3_8b_base_p02_p10_relay_20260714_030311")
REFERENCE_METADATA = REFERENCE_RELAY / "pair_metadata.csv"
REFERENCE_HANDOFF = Path("results/patchscopes/qwen3_8b_base_causal_entity_position_handoff_20260713_181400")
REFERENCE_GROUPED = REFERENCE_HANDOFF / "grouped_position_results.csv"
SCRIPT_PATH = Path(__file__).resolve()
HOOK_DEFINITION = "cumulative residual state at output of transformer block; layer 0 is output of block 0"
BOOTSTRAP_SEED = 20260811
DEFAULT_BOOTSTRAPS = 2000
SMOKE_LAYERS = [0, 24, 35]
SMOKE_TEMPLATES = ["T0", "T2"]

TEMPLATES = {
    "T0": "Everyone knows {name} was a celebrated {role}. The {role} was",
    "T1": "It is widely known that {name} was a celebrated {role}. The {role} was",
    "T2": "A celebrated {role}, {name}, became widely known. The {role} was",
    "T3": "{name}, who became widely known, was a celebrated {role}. The {role} was",
    "T4": "The celebrated {role} {name} became widely known. The {role} was",
}

# Framing text before the main entity-bearing predication.  Tokens in these
# spans receive source_prefix; all remaining source-clause material that is not
# subject/source-role/boundary receives source_predicate.
SOURCE_PREFIX_TEXT = {
    "T0": "Everyone knows ",
    "T1": "It is widely known that ",
    "T2": "",
    "T3": "",
    "T4": "",
}

# Exact validated natural-sentence filler families from the distance-generality
# experiment.  Leading-space standalone blocks are intentional and checked in
# their prompt context.
FILLER_SENTENCES: dict[str, list[str]] = {
    "meadow": [
        "Light rain crossed the empty meadow.",
        "A narrow stream moved quietly between smooth stones.",
        "Pale clouds drifted above the distant hills.",
        "Small leaves trembled whenever the cool wind returned.",
        "Along the path, shallow puddles reflected the changing sky.",
        "The air remained calm, fresh, and slightly damp.",
        "By midday, warm light reached the shaded grass.",
        "Far away, reeds bent beside a quiet pond.",
        "Evening arrived slowly under a band of silver clouds.",
        "After sunset, the ground held the day's fading warmth.",
        "A faint breeze continued across the open field.",
        "Overhead, the last cloud thinned into the darkening air.",
        "The stream kept its steady pace through the night.",
    ],
    "room": [
        "A plain bowl rested on the shelf.",
        "Morning light entered through a half-open curtain.",
        "The floorboards creaked softly near the doorway.",
        "A small clock marked each minute with a muted click.",
        "Later, clean cups were arranged in an even row.",
        "Fresh air moved gently through the quiet room.",
        "Nothing else changed before the afternoon.",
        "At dusk, a lamp cast a circle across the table.",
        "The remaining objects stayed exactly where they had been.",
        "A closed drawer held paper, string, and spare buttons.",
        "The curtain shifted once and then became still.",
        "Muted shadows gathered along the opposite wall.",
        "The room settled into its usual evening quiet.",
    ],
    "cards": [
        "Blank cards lay beside a spool of thread.",
        "One corner of the table caught the morning light.",
        "A shallow tray held clips of different sizes.",
        "The paper had a smooth surface and square edges.",
        "Nearby, a ruler rested parallel to the table.",
        "Loose fibers gathered beneath the quiet fan.",
        "During the afternoon, the light shifted toward the wall.",
        "No marks appeared on any of the cards.",
        "Before evening, each item returned to its place.",
        "A short length of thread remained beside the tray.",
        "The ruler cast a thin shadow across the paper.",
        "Nothing on the table moved for several hours.",
        "The room then settled into its usual stillness.",
    ],
}
FILLER_FAMILIES = list(FILLER_SENTENCES)
FORBIDDEN_FILLER_TERMS = {
    "einstein", "newton", "darwin", "mozart", "bach", "shakespeare", "dickens",
    "france", "japan", "paris", "tokyo", "google", "apple", "scientist", "composer",
    "writer", "country", "city", "company",
}

SEMANTIC_LABELS = [
    "source_prefix", "subject_entity", "source_predicate", "source_role",
    "source_boundary", "neutral_filler", "query_determiner", "query_role",
    "readout_final", "other",
]
GROUP_NAMES = [
    "subject_entity", "source_role", "complete_source_clause", "source_boundary",
    "neutral_filler", "query_determiner", "query_role", "readout_final",
    "post_subject_excluding_readout", "unrelated_prefix_negative_control",
    "all_prompt_positions_oracle", "identity_control",
]

RAW_FIELDS = [
    "phase", "template_id", "full_prompt", "donor_prompt", "pair_id",
    "reciprocal_cluster_id", "donor_entity", "recipient_entity", "role", "layer",
    "layer_hook_definition", "position_group", "patched_positions", "absolute_position",
    "relative_to_subject", "relative_to_readout", "subject_to_readout_distance",
    "semantic_role", "token_id", "decoded_token", "character_offsets",
    "position_condition", "placement", "filler_family", "target_filler_length",
    "actual_filler_length", "subject_positions", "source_role_positions",
    "source_boundary_positions", "query_determiner_positions", "query_role_positions",
    "readout_position", "source_activation", "donor_baseline_margin",
    "recipient_baseline_margin", "baseline_separation", "normalization_denominator",
    "patched_margin", "donor_logit", "recipient_logit", "normalized_recovery",
    "intervention_batch_size", "notes",
]

PREFLIGHT_FIELDS = [
    "status", "reason", "template_id", "position_condition", "placement", "filler_family",
    "target_filler_length", "actual_filler_length", "pair_id", "reciprocal_cluster_id",
    "donor_entity", "recipient_entity", "role", "donor_prompt", "recipient_prompt",
    "token_length", "differing_positions", "subject_positions", "source_role_positions",
    "query_role_positions", "readout_position", "subject_to_readout_distance",
    "donor_token_id", "recipient_token_id", "donor_baseline_margin",
    "recipient_baseline_margin", "baseline_separation",
]


@dataclass(frozen=True)
class Pair:
    pair_id: str
    cluster: str
    donor: str
    recipient: str
    role: str
    donor_token_id: int
    recipient_token_id: int


@dataclass(frozen=True)
class Condition:
    position_condition: str
    placement: str
    family: str
    target: int
    filler_text: str
    filler_token_ids: tuple[int, ...]
    filler_tokens: tuple[str, ...]

    @property
    def key(self) -> str:
        fam = self.family or "none"
        return f"{self.position_condition.replace(' ', '_').replace('+', 'plus')}__{fam}"


@dataclass
class Encoded:
    template_id: str
    prompt: str
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]
    semantic: list[str]
    subject: list[int]
    source_role: list[int]
    source_clause: list[int]
    source_boundary: list[int]
    filler: list[int]
    query_determiner: list[int]
    query_role: list[int]
    readout: int
    post_subject_excluding_readout: list[int]
    unrelated_prefix: list[int]
    actual_filler_length: int


@dataclass
class Capture:
    block: dict[int, torch.Tensor]
    logits: torch.Tensor


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def append_rows(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def finite_float(value: Any) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite value: {value}")
    return x


def load_pairs() -> list[Pair]:
    rows = read_csv(REFERENCE_METADATA)
    if len(rows) != 16:
        raise RuntimeError(f"canonical metadata has {len(rows)} directions, expected 16")
    pairs = [Pair(
        pair_id=r["pair_id"], cluster=r["unordered_pair_id"], donor=r["donor_entity"],
        recipient=r["recipient_entity"], role=r["role"],
        donor_token_id=int(r["donor_token_id"]), recipient_token_id=int(r["recipient_token_id"]),
    ) for r in rows]
    counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        counts[pair.cluster] += 1
    if len(counts) != 8 or set(counts.values()) != {2}:
        raise RuntimeError(f"invalid reciprocal clusters: {dict(counts)}")
    return pairs


def cluster_assignment(pairs: list[Pair], num_shards: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    clusters = sorted({p.cluster for p in pairs})
    assignment = {cluster: i % num_shards for i, cluster in enumerate(clusters)}
    rows = [{
        "shard_id": assignment[p.cluster], "reciprocal_cluster_id": p.cluster,
        "pair_id": p.pair_id, "donor_entity": p.donor, "recipient_entity": p.recipient,
        "role": p.role,
    } for p in pairs]
    return assignment, rows


def load_model(device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    layers = list(model.model.layers)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} layers, got {len(layers)}")
    return model, tokenizer, layers


def filler_blocks(tokenizer) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for family, sentences in FILLER_SENTENCES.items():
        joined = " ".join(sentences).lower()
        bad = sorted(term for term in FORBIDDEN_FILLER_TERMS if term in joined)
        if bad:
            raise RuntimeError(f"filler family {family} contains tested terms: {bad}")
        candidates = [" " + " ".join(sentences[:i]) for i in range(1, len(sentences) + 1)]
        output[family] = {}
        for target in (32, 128):
            ranked = []
            for text in candidates:
                ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
                ranked.append((abs(len(ids) - target), len(ids), text, ids))
            _, count, text, ids = min(ranked, key=lambda x: (x[0], x[1]))
            output[family][target] = {
                "text": text, "token_ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids),
                "token_count": count,
            }
    return output


def all_conditions(blocks: dict[str, dict[int, dict[str, Any]]]) -> list[Condition]:
    conditions = [Condition("none", "none", "", 0, "", (), ())]
    for placement in ("prefix", "gap"):
        for target in (32, 128):
            for family in FILLER_FAMILIES:
                block = blocks[family][target]
                conditions.append(Condition(
                    f"{placement} +{target}", placement, family, target, block["text"],
                    tuple(block["token_ids"]), tuple(block["tokens"]),
                ))
    return conditions


def phase2_condition(condition: Condition) -> bool:
    return condition.position_condition in {"none", "prefix +128", "gap +128"}


def overlap_positions(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    return [i for i, (a, b) in enumerate(offsets) if (a, b) != (0, 0) and b > start and a < end]


def occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    out = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return out
        out.append((index, index + len(needle)))
        start = index + len(needle)


def build_prompt(template_id: str, entity: str, role: str, condition: Condition) -> tuple[str, dict[str, tuple[int, int] | None]]:
    base = TEMPLATES[template_id].format(name=entity, role=role)
    query = f"The {role} was"
    if not base.endswith(query):
        raise RuntimeError(f"{template_id}: final query construction changed")
    source = base[:-(len(query) + 1)]
    if condition.placement == "none":
        prompt = base
        source_start = 0
        filler_span = None
    elif condition.placement == "prefix":
        prompt = f"{condition.filler_text} {base}"
        source_start = len(condition.filler_text) + 1
        filler_span = (0, len(condition.filler_text))
    elif condition.placement == "gap":
        prompt = f"{source}{condition.filler_text} {query}"
        source_start = 0
        filler_span = (len(source), len(source) + len(condition.filler_text))
    else:
        raise ValueError(condition.placement)
    source_span = (source_start, source_start + len(source))
    boundary_index = source_span[1] - 1
    if prompt[boundary_index] != ".":
        raise RuntimeError(f"{template_id}: source boundary is not period")
    query_start = len(prompt) - len(query)
    if prompt[query_start:] != query:
        raise RuntimeError(f"{template_id}: final query is not exact: {prompt[query_start:]!r}")
    subject_start = prompt.index(entity, source_span[0], source_span[1])
    role_spans = occurrences(prompt, role)
    source_roles = [x for x in role_spans if x[0] < source_span[1]]
    query_roles = [x for x in role_spans if x[0] >= query_start]
    if len(source_roles) != 1 or len(query_roles) != 1:
        raise RuntimeError(f"{template_id}: invalid role occurrences {role_spans}")
    spans: dict[str, tuple[int, int] | None] = {
        "source_clause": source_span,
        "source_boundary": (boundary_index, boundary_index + 1),
        "filler": filler_span,
        "subject": (subject_start, subject_start + len(entity)),
        "source_role": source_roles[0],
        "source_prefix": (source_start, source_start + len(SOURCE_PREFIX_TEXT[template_id])) if SOURCE_PREFIX_TEXT[template_id] else None,
        "query_determiner": (query_start, query_start + len("The")),
        "query_role": query_roles[0],
        "readout": (len(prompt) - len("was"), len(prompt)),
    }
    return prompt, spans


def encode(tokenizer, template_id: str, entity: str, role: str, condition: Condition) -> Encoded:
    prompt, spans = build_prompt(template_id, entity, role, condition)
    data = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(data["input_ids"])
    offsets = [tuple(x) for x in data["offset_mapping"]]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    pos = {name: overlap_positions(offsets, span) if span is not None else [] for name, span in spans.items()}
    required = ["subject", "source_role", "source_clause", "source_boundary", "query_determiner", "query_role", "readout"]
    if any(not pos[name] for name in required):
        raise RuntimeError(f"empty required span(s): {[name for name in required if not pos[name]]}")
    if pos["readout"] != [len(ids) - 1]:
        raise RuntimeError(f"readout is not final single token: {pos['readout']} len={len(ids)}")
    semantic = []
    sets = {name: set(values) for name, values in pos.items()}
    for index in range(len(ids)):
        # Specific roles have priority over broad source-prefix/predicate spans.
        if index in sets["subject"]:
            label = "subject_entity"
        elif index in sets["source_role"]:
            label = "source_role"
        elif index in sets["source_boundary"]:
            label = "source_boundary"
        elif index in sets["filler"]:
            label = "neutral_filler"
        elif index in sets["query_determiner"]:
            label = "query_determiner"
        elif index in sets["query_role"]:
            label = "query_role"
        elif index in sets["readout"]:
            label = "readout_final"
        elif index in sets["source_prefix"]:
            label = "source_prefix"
        elif index in sets["source_clause"]:
            label = "source_predicate"
        else:
            label = "other"
        semantic.append(label)
    subject = pos["subject"]
    readout = pos["readout"][0]
    prefix = [i for i in range(subject[0])]
    unrelated = [prefix[-1]] if prefix else []
    filler_ids = [ids[p] for p in pos["filler"]]
    if filler_ids != list(condition.filler_token_ids):
        raise RuntimeError(
            f"contextual filler IDs differ from matched standalone block: context={filler_ids} standalone={list(condition.filler_token_ids)}"
        )
    return Encoded(
        template_id=template_id, prompt=prompt, ids=ids, tokens=tokens, offsets=offsets,
        semantic=semantic, subject=subject, source_role=pos["source_role"],
        source_clause=pos["source_clause"], source_boundary=pos["source_boundary"],
        filler=pos["filler"], query_determiner=pos["query_determiner"],
        query_role=pos["query_role"], readout=readout,
        post_subject_excluding_readout=list(range(subject[-1] + 1, readout)),
        unrelated_prefix=unrelated, actual_filler_length=len(ids) - len(tokenizer(TEMPLATES[template_id].format(name=entity, role=role), add_special_tokens=True)["input_ids"]),
    )


def leading_space_output_id(tokenizer, entity: str) -> tuple[int | None, list[int], list[str]]:
    ids = list(tokenizer(" " + entity, add_special_tokens=False)["input_ids"])
    return (ids[0] if len(ids) == 1 else None), ids, tokenizer.convert_ids_to_tokens(ids)


def validate_alignment(tokenizer, pair: Pair, template_id: str, condition: Condition) -> tuple[Encoded | None, Encoded | None, str | None]:
    try:
        donor = encode(tokenizer, template_id, pair.donor, pair.role, condition)
        recipient = encode(tokenizer, template_id, pair.recipient, pair.role, condition)
        donor_out, donor_out_ids, donor_out_toks = leading_space_output_id(tokenizer, pair.donor)
        recipient_out, recipient_out_ids, recipient_out_toks = leading_space_output_id(tokenizer, pair.recipient)
        if donor_out is None or donor_out != pair.donor_token_id:
            return donor, recipient, f"donor leading-space output is not canonical single token: ids={donor_out_ids} tokens={donor_out_toks}"
        if recipient_out is None or recipient_out != pair.recipient_token_id:
            return donor, recipient, f"recipient leading-space output is not canonical single token: ids={recipient_out_ids} tokens={recipient_out_toks}"
        if len(donor.ids) != len(recipient.ids):
            return donor, recipient, f"token length mismatch: {len(donor.ids)} vs {len(recipient.ids)}"
        named = [
            ("subject", donor.subject, recipient.subject), ("source_role", donor.source_role, recipient.source_role),
            ("source_boundary", donor.source_boundary, recipient.source_boundary),
            ("query_determiner", donor.query_determiner, recipient.query_determiner),
            ("query_role", donor.query_role, recipient.query_role),
        ]
        for name, a, b in named:
            if not a or not b or a != b:
                return donor, recipient, f"{name} span mismatch: {a} vs {b}"
        if donor.readout != recipient.readout:
            return donor, recipient, f"readout mismatch: {donor.readout} vs {recipient.readout}"
        diffs = [i for i, (a, b) in enumerate(zip(donor.ids, recipient.ids)) if a != b]
        if not diffs or any(i not in set(donor.subject) for i in diffs):
            return donor, recipient, f"token differences not confined to subject: diffs={diffs}, subject={donor.subject}"
        if donor.semantic != recipient.semantic:
            return donor, recipient, "semantic token labels differ between donor and recipient"
        return donor, recipient, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def replace_output(out: Any, hidden: torch.Tensor) -> Any:
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


@torch.inference_mode()
def last_logits(model, ids: torch.Tensor) -> torch.Tensor:
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    return model.lm_head(hidden[:, -1]).detach().float().cpu()


@torch.inference_mode()
def capture(model, layers, ids: list[int], device: torch.device) -> Capture:
    block: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        def hook(_module, _inputs, out, index=layer_index):
            h = out[0] if isinstance(out, tuple) else out
            block[index] = h[0].detach().clone()
        handles.append(layer.register_forward_hook(hook))
    try:
        logits = last_logits(model, torch.tensor([ids], dtype=torch.long, device=device))[0]
    finally:
        for handle in handles:
            handle.remove()
    if len(block) != N_LAYERS:
        raise RuntimeError(f"incomplete block capture: {len(block)}")
    return Capture(block=block, logits=logits)


def margin(logits: torch.Tensor, pair: Pair) -> float:
    return float(logits[pair.donor_token_id] - logits[pair.recipient_token_id])


def grouped_specs(enc: Encoded) -> list[tuple[str, list[int], str, str]]:
    return [
        ("subject_entity", enc.subject, "donor", ""),
        ("source_role", enc.source_role, "donor", ""),
        ("complete_source_clause", enc.source_clause, "donor", ""),
        ("source_boundary", enc.source_boundary, "donor", ""),
        ("neutral_filler", enc.filler, "donor", "empty no-op for none condition" if not enc.filler else ""),
        ("query_determiner", enc.query_determiner, "donor", ""),
        ("query_role", enc.query_role, "donor", ""),
        ("readout_final", [enc.readout], "donor", ""),
        ("post_subject_excluding_readout", enc.post_subject_excluding_readout, "donor", ""),
        ("unrelated_prefix_negative_control", enc.unrelated_prefix, "donor", "declared empty no-op when no pre-subject token exists" if not enc.unrelated_prefix else "causally upstream of subject"),
        ("all_prompt_positions_oracle", list(range(len(enc.ids))), "donor", "singleton exact control"),
        ("identity_control", [enc.readout], "recipient", "singleton recipient-state control"),
    ]


def smoke_token_positions(enc: Encoded) -> list[int]:
    wanted = ["source_prefix", "source_predicate", "subject_entity", "source_role", "query_role", "readout_final"]
    chosen: list[int] = []
    for label in wanted:
        hits = [i for i, value in enumerate(enc.semantic) if value == label]
        if hits and hits[0] not in chosen:
            chosen.append(hits[0])
        if len(chosen) == 5:
            break
    if len(chosen) != 5:
        raise RuntimeError(f"smoke expected five representative positions, got {chosen} labels={enc.semantic}")
    return chosen


def common_row(pair: Pair, template_id: str, condition: Condition, donor_enc: Encoded,
               recipient_enc: Encoded, dm: float, rm: float) -> dict[str, Any]:
    return {
        "template_id": template_id, "full_prompt": recipient_enc.prompt,
        "donor_prompt": donor_enc.prompt, "pair_id": pair.pair_id,
        "reciprocal_cluster_id": pair.cluster, "donor_entity": pair.donor,
        "recipient_entity": pair.recipient, "role": pair.role,
        "layer_hook_definition": HOOK_DEFINITION,
        "subject_to_readout_distance": recipient_enc.readout - recipient_enc.subject[-1],
        "position_condition": condition.position_condition, "placement": condition.placement,
        "filler_family": condition.family or "none", "target_filler_length": condition.target,
        "actual_filler_length": recipient_enc.actual_filler_length,
        "subject_positions": j(recipient_enc.subject), "source_role_positions": j(recipient_enc.source_role),
        "source_boundary_positions": j(recipient_enc.source_boundary),
        "query_determiner_positions": j(recipient_enc.query_determiner),
        "query_role_positions": j(recipient_enc.query_role), "readout_position": recipient_enc.readout,
        "donor_baseline_margin": dm, "recipient_baseline_margin": rm,
        "baseline_separation": dm - rm, "normalization_denominator": dm - rm,
    }


def payload_for_positions(enc: Encoded, positions: list[int]) -> dict[str, str]:
    return {
        "patched_positions": j(positions),
        "semantic_role": j([enc.semantic[p] for p in positions]),
        "token_id": j([enc.ids[p] for p in positions]),
        "decoded_token": j([enc.tokens[p] for p in positions]),
        "character_offsets": j([enc.offsets[p] for p in positions]),
    }


@torch.inference_mode()
def run_specs(model, layers, pair: Pair, enc: Encoded, donor: Capture, recipient: Capture,
              layer_index: int, specs: list[tuple[str, list[int], str, str]], device: torch.device,
              dm: float, rm: float, common: dict[str, Any], phase: str,
              force_singletons: set[str] | None = None) -> list[dict[str, Any]]:
    force_singletons = force_singletons or set()
    # Empty specs are genuine structural no-ops and use the exact recipient baseline.
    nonempty = [spec for spec in specs if spec[1]]
    empty = [spec for spec in specs if not spec[1]]
    singletons = [spec for spec in nonempty if spec[0] in force_singletons]
    regular = [spec for spec in nonempty if spec[0] not in force_singletons]
    batches: list[list[tuple[str, list[int], str, str]]] = []
    if regular:
        batches.append(regular)
    batches.extend([[spec] for spec in singletons])
    results: dict[tuple[str, tuple[int, ...], str], tuple[torch.Tensor, int]] = {}
    for batch in batches:
        x = torch.tensor([enc.ids] * len(batch), dtype=torch.long, device=device)
        def hook(_module, _inputs, out):
            h0 = out[0] if isinstance(out, tuple) else out
            h = h0.clone()
            for bi, (_name, positions, source, _notes) in enumerate(batch):
                values = donor.block[layer_index] if source == "donor" else recipient.block[layer_index]
                h[bi, positions] = values[positions].to(h)
            return replace_output(out, h)
        handle = layers[layer_index].register_forward_hook(hook)
        try:
            logits = last_logits(model, x)
        finally:
            handle.remove()
        for bi, (name, positions, source, _notes) in enumerate(batch):
            results[(name, tuple(positions), source)] = (logits[bi], len(batch))
    for name, positions, source, _notes in empty:
        results[(name, tuple(positions), source)] = (recipient.logits, 1)
    rows = []
    denominator = dm - rm
    for name, positions, source, notes in specs:
        logits, batch_size = results[(name, tuple(positions), source)]
        pm = margin(logits, pair)
        row = dict(common)
        row.update({
            "phase": phase, "layer": layer_index, "position_group": name,
            "absolute_position": "", "relative_to_subject": "", "relative_to_readout": "",
            "source_activation": source, "patched_margin": pm,
            "donor_logit": float(logits[pair.donor_token_id]),
            "recipient_logit": float(logits[pair.recipient_token_id]),
            "normalized_recovery": (pm - rm) / denominator,
            "intervention_batch_size": batch_size, "notes": notes,
        })
        row.update(payload_for_positions(enc, positions))
        rows.append(row)
    return rows


@torch.inference_mode()
def run_token_layer(model, layers, pair: Pair, enc: Encoded, donor: Capture, layer_index: int,
                    positions: list[int], batch_size: int, device: torch.device, dm: float, rm: float,
                    common: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    rows = []
    denominator = dm - rm
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        x = torch.tensor([enc.ids] * len(chunk), dtype=torch.long, device=device)
        def hook(_module, _inputs, out):
            h0 = out[0] if isinstance(out, tuple) else out
            h = h0.clone()
            values = donor.block[layer_index].to(h)
            for bi, position in enumerate(chunk):
                h[bi, position] = values[position]
            return replace_output(out, h)
        handle = layers[layer_index].register_forward_hook(hook)
        try:
            logits = last_logits(model, x)
        finally:
            handle.remove()
        for bi, position in enumerate(chunk):
            result = logits[bi]
            pm = margin(result, pair)
            row = dict(common)
            row.update({
                "phase": phase, "layer": layer_index, "position_group": "token_position",
                "patched_positions": j([position]), "absolute_position": position,
                "relative_to_subject": position - enc.subject[-1],
                "relative_to_readout": position - enc.readout,
                "semantic_role": enc.semantic[position], "token_id": enc.ids[position],
                "decoded_token": enc.tokens[position], "character_offsets": j(enc.offsets[position]),
                "source_activation": "donor", "patched_margin": pm,
                "donor_logit": float(result[pair.donor_token_id]),
                "recipient_logit": float(result[pair.recipient_token_id]),
                "normalized_recovery": (pm - rm) / denominator,
                "intervention_batch_size": len(chunk), "notes": "complete same-position token map" if phase == "phase2" else "smoke representative position",
            })
            rows.append(row)
    return rows


def preflight(model, tokenizer, pairs: list[Pair], conditions: list[Condition], out_dir: Path,
              device: torch.device) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    prompt_cache: dict[tuple[str, str, str], tuple[Encoded, torch.Tensor]] = {}
    for template_id in TEMPLATES:
        for condition in conditions:
            for pair in pairs:
                donor, recipient, reason = validate_alignment(tokenizer, pair, template_id, condition)
                base = {
                    "template_id": template_id, "position_condition": condition.position_condition,
                    "placement": condition.placement, "filler_family": condition.family or "none",
                    "target_filler_length": condition.target, "pair_id": pair.pair_id,
                    "reciprocal_cluster_id": pair.cluster, "donor_entity": pair.donor,
                    "recipient_entity": pair.recipient, "role": pair.role,
                    "donor_token_id": pair.donor_token_id, "recipient_token_id": pair.recipient_token_id,
                }
                if reason is not None or donor is None or recipient is None:
                    rejected.append({**base, "status": "rejected", "reason": reason or "unknown alignment failure"})
                    continue
                try:
                    cache_key_d = (template_id, condition.key, pair.donor)
                    cache_key_r = (template_id, condition.key, pair.recipient)
                    if cache_key_d not in prompt_cache:
                        logits = last_logits(model, torch.tensor([donor.ids], dtype=torch.long, device=device))[0]
                        prompt_cache[cache_key_d] = (donor, logits)
                    if cache_key_r not in prompt_cache:
                        logits = last_logits(model, torch.tensor([recipient.ids], dtype=torch.long, device=device))[0]
                        prompt_cache[cache_key_r] = (recipient, logits)
                    dl = prompt_cache[cache_key_d][1]
                    rl = prompt_cache[cache_key_r][1]
                    dm = margin(dl, pair)
                    rm = margin(rl, pair)
                    separation = dm - rm
                    reasons = []
                    if not math.isfinite(dm) or not math.isfinite(rm):
                        reasons.append("non-finite baseline margin")
                    if separation < BASELINE_SEPARATION_THRESHOLD:
                        reasons.append(f"baseline separation {separation:.6g} below {BASELINE_SEPARATION_THRESHOLD}")
                    if dm <= 0:
                        reasons.append(f"donor baseline does not favor donor: margin={dm:.6g}")
                    if rm >= 0:
                        reasons.append(f"recipient baseline does not favor recipient: donor-minus-recipient margin={rm:.6g}")
                    diffs = [i for i, (a, b) in enumerate(zip(donor.ids, recipient.ids)) if a != b]
                    payload = {
                        **base, "reason": "; ".join(reasons), "donor_prompt": donor.prompt,
                        "recipient_prompt": recipient.prompt, "token_length": len(donor.ids),
                        "differing_positions": j(diffs), "subject_positions": j(donor.subject),
                        "source_role_positions": j(donor.source_role), "query_role_positions": j(donor.query_role),
                        "readout_position": donor.readout,
                        "subject_to_readout_distance": donor.readout - donor.subject[-1],
                        "actual_filler_length": donor.actual_filler_length,
                        "donor_baseline_margin": dm, "recipient_baseline_margin": rm,
                        "baseline_separation": separation,
                    }
                    if reasons:
                        rejected.append({**payload, "status": "rejected"})
                    else:
                        accepted.append({**payload, "status": "accepted"})
                except Exception as exc:
                    rejected.append({**base, "status": "rejected", "reason": f"baseline evaluation failed: {type(exc).__name__}: {exc}"})
            log(f"preflight {template_id} {condition.position_condition}/{condition.family or 'none'} complete")
    write_csv(out_dir / "accepted_combinations.csv", accepted, PREFLIGHT_FIELDS)
    write_csv(out_dir / "rejected_combinations.csv", rejected, PREFLIGHT_FIELDS)
    expected = len(TEMPLATES) * len(conditions) * len(pairs)
    report = {
        "status": "PASS" if len(accepted) + len(rejected) == expected else "FAIL",
        "expected_combinations": expected, "accepted_combinations": len(accepted),
        "rejected_combinations": len(rejected), "baseline_threshold": BASELINE_SEPARATION_THRESHOLD,
        "templates_fixed_before_results": TEMPLATES, "conditions_per_template": len(conditions),
        "completed_at": now(),
    }
    write_json(out_dir / "preflight_validation.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"preflight accounting failure: {report}")
    return report


def accepted_keys(root: Path) -> set[tuple[str, str, str, str]]:
    rows = read_csv(root / "accepted_combinations.csv")
    if not rows:
        raise RuntimeError("missing accepted_combinations.csv; run smoke/preflight first")
    return {(r["template_id"], r["position_condition"], r["filler_family"], r["pair_id"]) for r in rows}


def combo_key(template_id: str, condition: Condition, pair: Pair) -> tuple[str, str, str, str]:
    return template_id, condition.position_condition, condition.family or "none", pair.pair_id


def checkpoint_dir(root: Path, shard_id: int, phase: str, template_id: str,
                   condition: Condition, pair: Pair) -> Path:
    return root / f"shard_{shard_id}" / phase / template_id / condition.key / pair.pair_id


def validate_runtime_baselines(pair: Pair, dm: float, rm: float) -> None:
    if not all(math.isfinite(x) for x in (dm, rm, dm - rm)):
        raise RuntimeError(f"{pair.pair_id}: non-finite runtime baselines")
    if dm <= 0 or rm >= 0 or dm - rm < BASELINE_SEPARATION_THRESHOLD:
        raise RuntimeError(f"{pair.pair_id}: runtime baseline gate changed dm={dm} rm={rm}")


def run_combination(model, tokenizer, layers, pair: Pair, template_id: str, condition: Condition,
                    phase: str, layer_list: list[int], device: torch.device, checkpoint: Path,
                    token_batch_size: int, smoke: bool = False) -> dict[str, Any]:
    complete = checkpoint / "COMPLETE.json"
    if complete.exists() and not smoke:
        return json.loads(complete.read_text())
    checkpoint.mkdir(parents=True, exist_ok=True)
    donor_enc, recipient_enc, reason = validate_alignment(tokenizer, pair, template_id, condition)
    if reason is not None or donor_enc is None or recipient_enc is None:
        raise RuntimeError(f"accepted alignment failed at runtime: {reason}")
    started = time.time()
    donor = capture(model, layers, donor_enc.ids, device)
    recipient = capture(model, layers, recipient_enc.ids, device)
    dm, rm = margin(donor.logits, pair), margin(recipient.logits, pair)
    validate_runtime_baselines(pair, dm, rm)
    common = common_row(pair, template_id, condition, donor_enc, recipient_enc, dm, rm)
    metadata = {
        **common, "donor_token_ids": donor_enc.ids, "recipient_token_ids": recipient_enc.ids,
        "donor_tokens": donor_enc.tokens, "recipient_tokens": recipient_enc.tokens,
        "offsets": recipient_enc.offsets, "semantic_labels": recipient_enc.semantic,
        "filler_positions": recipient_enc.filler,
        "unrelated_prefix_positions": recipient_enc.unrelated_prefix,
    }
    write_json(checkpoint / "prompt_metadata.json", metadata)
    result_path = checkpoint / ("grouped_results.csv" if phase == "phase1" else "token_position_results.csv")
    if result_path.exists():
        result_path.unlink()
    count = 0
    for layer_index in layer_list:
        if phase == "phase1":
            specs = grouped_specs(recipient_enc)
            rows = run_specs(
                model, layers, pair, recipient_enc, donor, recipient, layer_index, specs, device,
                dm, rm, common, phase,
                force_singletons={
                    "subject_entity", "readout_final", "unrelated_prefix_negative_control",
                    "all_prompt_positions_oracle", "identity_control",
                },
            )
        elif phase == "phase2":
            positions = smoke_token_positions(recipient_enc) if smoke else list(range(len(recipient_enc.ids)))
            rows = run_token_layer(
                model, layers, pair, recipient_enc, donor, layer_index, positions,
                token_batch_size, device, dm, rm, common, "smoke_phase2" if smoke else phase,
            )
        else:
            raise ValueError(phase)
        append_rows(result_path, rows, RAW_FIELDS)
        count += len(rows)
        log(f"{phase} {template_id} {condition.key} {pair.pair_id} layer {layer_index} rows={count}")
    details = {
        "phase": phase, "template_id": template_id, "condition_key": condition.key,
        "position_condition": condition.position_condition, "filler_family": condition.family or "none",
        "pair_id": pair.pair_id, "reciprocal_cluster_id": pair.cluster,
        "layers": layer_list, "token_length": len(recipient_enc.ids), "num_rows": count,
        "donor_margin": dm, "recipient_margin": rm, "baseline_separation": dm - rm,
        "runtime_seconds": time.time() - started, "completed_at": now(),
    }
    write_json(complete, details)
    (checkpoint / "SUCCESS").write_text(now() + "\n")
    return details


def reference_curves(pair_ids: set[str], layers: set[int]) -> dict[tuple[str, int, str], float]:
    mapping = {"subject_span": "subject_entity", "final_token": "readout_final"}
    output = {}
    for row in iter_csv(REFERENCE_GROUPED):
        if row["pair_id"] in pair_ids and int(row["layer"]) in layers and row["position_group"] in mapping:
            output[(row["pair_id"], int(row["layer"]), mapping[row["position_group"]])] = float(row["normalized_recovery"])
    return output


def smoke_validate_phase1(rows: list[dict[str, str]], pair_ids: set[str]) -> dict[str, Any]:
    expected = len(pair_ids) * len(SMOKE_TEMPLATES) * 2 * len(SMOKE_LAYERS) * len(GROUP_NAMES)
    finite = all(math.isfinite(float(r["normalized_recovery"])) for r in rows)
    identity = [abs(float(r["normalized_recovery"])) for r in rows if r["position_group"] == "identity_control"]
    oracle = [abs(float(r["normalized_recovery"]) - 1) for r in rows if r["position_group"] == "all_prompt_positions_oracle"]
    unrelated = [abs(float(r["normalized_recovery"])) for r in rows if r["position_group"] == "unrelated_prefix_negative_control"]
    reference = reference_curves(pair_ids, set(SMOKE_LAYERS))
    deltas = []
    new_by: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if row["template_id"] == "T0" and row["position_condition"] == "none" and row["position_group"] in {"subject_entity", "readout_final"}:
            key = (row["pair_id"], int(row["layer"]), row["position_group"])
            new_by[key] = float(row["normalized_recovery"])
            deltas.append(abs(new_by[key] - reference[key]))
    cross_errors = []
    for pair_id in pair_ids:
        new_cross = next((layer for layer in SMOKE_LAYERS if new_by[(pair_id, layer, "readout_final")] >= new_by[(pair_id, layer, "subject_entity")]), None)
        ref_cross = next((layer for layer in SMOKE_LAYERS if reference[(pair_id, layer, "readout_final")] >= reference[(pair_id, layer, "subject_entity")]), None)
        cross_errors.append(abs(int(new_cross) - int(ref_cross)) if new_cross is not None and ref_cross is not None else 999)
    checks = {
        "exact_expected_grouped_row_count": len(rows) == expected,
        "all_values_finite": finite,
        "identity_recovery_near_zero": bool(identity) and max(identity) <= 0.02,
        "oracle_recovery_near_one": bool(oracle) and max(oracle) <= 0.02,
        "unrelated_prefix_within_established_bf16_floor": bool(unrelated) and max(unrelated) <= BF16_CONTROL_FLOOR,
        "zero_filler_T0_reference_within_0p02": len(deltas) == len(pair_ids) * len(SMOKE_LAYERS) * 2 and max(deltas) <= 0.02,
        "zero_filler_T0_crossover_within_one_layer": bool(cross_errors) and max(cross_errors) <= 1,
        "all_group_names_present": set(r["position_group"] for r in rows) == set(GROUP_NAMES),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "expected_grouped_rows": expected, "observed_grouped_rows": len(rows),
        "identity_max_abs_recovery": max(identity) if identity else None,
        "oracle_max_abs_deviation": max(oracle) if oracle else None,
        "unrelated_max_abs_recovery": max(unrelated) if unrelated else None,
        "zero_filler_T0_max_abs_reference_delta": max(deltas) if deltas else None,
        "zero_filler_T0_max_crossover_error_layers": max(cross_errors) if cross_errors else None,
    }


def run_smoke(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, tokenizer, layers = load_model(device)
    pairs = load_pairs()
    blocks = filler_blocks(tokenizer)
    conditions = all_conditions(blocks)
    write_json(root / "filler_blocks.json", blocks)
    write_json(root / "templates.json", TEMPLATES)
    write_json(root / "experiment_config.json", {
        "model": MODEL, "precision": "BF16", "attention_implementation": "eager",
        "layers": N_LAYERS, "hook_definition": HOOK_DEFINITION,
        "baseline_separation_threshold": BASELINE_SEPARATION_THRESHOLD,
        "semantic_labels": SEMANTIC_LABELS, "group_names": GROUP_NAMES,
        "phase1_position_conditions": ["none", "prefix +32", "prefix +128", "gap +32", "gap +128"],
        "phase2_position_conditions": ["none", "prefix +128", "gap +128"],
        "filler_families": FILLER_FAMILIES, "created_at": now(),
    })
    preflight_report = preflight(model, tokenizer, pairs, conditions, root, device)
    accepted = accepted_keys(root)
    first_cluster = pairs[0].cluster
    run_pairs = [p for p in pairs if p.cluster == first_cluster]
    smoke_conditions = [c for c in conditions if c.position_condition == "none" or (c.position_condition == "gap +32" and c.family == "meadow")]
    smoke_root = root / "smoke"
    grouped_rows: list[dict[str, str]] = []
    for template_id in SMOKE_TEMPLATES:
        for condition in smoke_conditions:
            for pair in run_pairs:
                if combo_key(template_id, condition, pair) not in accepted:
                    raise RuntimeError(f"smoke combination rejected by preflight: {combo_key(template_id, condition, pair)}")
                cp = smoke_root / "phase1" / template_id / condition.key / pair.pair_id
                run_combination(model, tokenizer, layers, pair, template_id, condition, "phase1", SMOKE_LAYERS, device, cp, args.token_batch_size, smoke=True)
                grouped_rows.extend(read_csv(cp / "grouped_results.csv"))
    phase1_report = smoke_validate_phase1(grouped_rows, {p.pair_id for p in run_pairs})
    write_json(smoke_root / "phase1_validation.json", phase1_report)
    if phase1_report["status"] != "PASS":
        write_json(smoke_root / "smoke_validation.json", {"status": "FAIL", "preflight": preflight_report, "phase1": phase1_report})
        raise RuntimeError("smoke Phase 1 validation failed; Phase 2 was not launched")
    token_rows: list[dict[str, str]] = []
    for template_id in SMOKE_TEMPLATES:
        for condition in smoke_conditions:
            for pair in run_pairs:
                cp = smoke_root / "phase2" / template_id / condition.key / pair.pair_id
                run_combination(model, tokenizer, layers, pair, template_id, condition, "phase2", SMOKE_LAYERS, device, cp, args.token_batch_size, smoke=True)
                token_rows.extend(read_csv(cp / "token_position_results.csv"))
    expected_token = len(run_pairs) * len(SMOKE_TEMPLATES) * len(smoke_conditions) * len(SMOKE_LAYERS) * 5
    token_finite = all(math.isfinite(float(r["normalized_recovery"])) for r in token_rows)
    # Alignment assertions are independently reconstructed, not inferred from result rows.
    alignment_ok = True
    for template_id in SMOKE_TEMPLATES:
        for condition in smoke_conditions:
            for pair in run_pairs:
                d, r, reason = validate_alignment(tokenizer, pair, template_id, condition)
                if reason or d is None or r is None:
                    alignment_ok = False
                elif any(i not in set(d.subject) for i, (a, b) in enumerate(zip(d.ids, r.ids)) if a != b):
                    alignment_ok = False
    checks = dict(phase1_report["checks"])
    checks.update({
        "donor_recipient_differ_only_in_declared_subject_span": alignment_ok,
        "exact_expected_token_row_count": len(token_rows) == expected_token,
        "token_values_all_finite": token_finite,
    })
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "preflight": preflight_report, "grouped_rows": len(grouped_rows),
        "expected_grouped_rows": phase1_report["expected_grouped_rows"],
        "token_rows": len(token_rows), "expected_token_rows": expected_token,
        "identity_max_abs_recovery": phase1_report["identity_max_abs_recovery"],
        "oracle_max_abs_deviation": phase1_report["oracle_max_abs_deviation"],
        "unrelated_max_abs_recovery": phase1_report["unrelated_max_abs_recovery"],
        "zero_filler_T0_max_abs_reference_delta": phase1_report["zero_filler_T0_max_abs_reference_delta"],
        "zero_filler_T0_max_crossover_error_layers": phase1_report["zero_filler_T0_max_crossover_error_layers"],
        "completed_at": now(),
    }
    write_json(smoke_root / "smoke_validation.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"smoke validation failed: {report}")
    (smoke_root / "SUCCESS").write_text(now() + "\n")
    log("SMOKE PASS " + json.dumps(report, sort_keys=True))
    return 0


def run_worker(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not (root / "smoke" / "SUCCESS").exists():
        raise RuntimeError("full worker refused: smoke SUCCESS is absent")
    if args.phase == "phase2":
        phase1_report = root / "phase1_validation.json"
        if not phase1_report.exists() or json.loads(phase1_report.read_text()).get("status") != "PASS":
            raise RuntimeError("Phase 2 refused: global Phase 1 validation has not passed")
    device = torch.device(args.device)
    model, tokenizer, layers = load_model(device)
    pairs = load_pairs()
    blocks = filler_blocks(tokenizer)
    conditions = all_conditions(blocks)
    accepted = accepted_keys(root)
    assignment, mapping_rows = cluster_assignment(pairs, args.num_shards)
    write_csv(root / "cluster_shard_mapping.csv", mapping_rows, ["shard_id", "reciprocal_cluster_id", "pair_id", "donor_entity", "recipient_entity", "role"])
    run_pairs = [p for p in pairs if assignment[p.cluster] == args.shard_id]
    if {assignment[p.cluster] for p in run_pairs} != {args.shard_id}:
        raise RuntimeError("reciprocal cluster sharding failure")
    run_conditions = [c for c in conditions if args.phase == "phase1" or phase2_condition(c)]
    manifest = []
    shard_root = root / f"shard_{args.shard_id}"
    started = time.time()
    for template_id in TEMPLATES:
        for condition in run_conditions:
            for pair in run_pairs:
                if combo_key(template_id, condition, pair) not in accepted:
                    continue
                cp = checkpoint_dir(root, args.shard_id, args.phase, template_id, condition, pair)
                details = run_combination(
                    model, tokenizer, layers, pair, template_id, condition, args.phase,
                    list(range(N_LAYERS)), device, cp, args.token_batch_size,
                )
                manifest.append(details)
                write_json(shard_root / f"{args.phase}_manifest.partial.json", manifest)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    expected = sum(
        1 for template_id in TEMPLATES for condition in run_conditions for pair in run_pairs
        if combo_key(template_id, condition, pair) in accepted
    )
    report = {
        "status": "PASS" if len(manifest) == expected else "FAIL", "phase": args.phase,
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "physical_gpu": os.getenv("CUDA_VISIBLE_DEVICES"), "process_device": args.device,
        "reciprocal_clusters": sorted({p.cluster for p in run_pairs}),
        "pair_ids": [p.pair_id for p in run_pairs], "expected_combinations": expected,
        "completed_combinations": len(manifest), "rows": sum(int(x["num_rows"]) for x in manifest),
        "runtime_seconds": time.time() - started, "completed_at": now(),
    }
    write_json(shard_root / f"{args.phase}_manifest.json", manifest)
    write_json(shard_root / f"{args.phase}_worker_validation.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"worker accounting failed: {report}")
    (shard_root / f"{args.phase}_SUCCESS").write_text(now() + "\n")
    log(f"WORKER PASS {args.phase} shard={args.shard_id} rows={report['rows']}")
    return 0


def merge_csv_files(paths: list[Path], out: Path, fields: Sequence[str]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    count = 0
    with tmp.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            for row in iter_csv(path):
                writer.writerow({name: row.get(name, "") for name in fields})
                count += 1
    tmp.replace(out)
    return count


def stable_crossover(subject: dict[int, float], readout: dict[int, float]) -> int | None:
    for layer in range(N_LAYERS - 1):
        if readout.get(layer, -math.inf) > subject.get(layer, math.inf) and readout.get(layer + 1, -math.inf) > subject.get(layer + 1, math.inf):
            return layer
    return None


def trapz(values: list[float]) -> float:
    return sum((values[i] + values[i + 1]) / 2 for i in range(len(values) - 1))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    x = q * (len(values) - 1)
    lo, hi = int(math.floor(x)), int(math.ceil(x))
    return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)


def cluster_boot_mean(records: list[dict[str, Any]], value: str, n_boot: int,
                      seed_offset: int = 0) -> dict[str, Any]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in records:
        x = float(row[value])
        if math.isfinite(x):
            by_cluster[str(row["reciprocal_cluster_id"])].append(x)
    cluster_values = {k: statistics.fmean(v) for k, v in by_cluster.items() if v}
    vals = list(cluster_values.values())
    if not vals:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "num_clusters": 0, "num_rows": 0}
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    boots = [statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(n_boot)]
    return {
        "mean": statistics.fmean(float(r[value]) for r in records if math.isfinite(float(r[value]))),
        "ci95_low": quantile(boots, 0.025), "ci95_high": quantile(boots, 0.975),
        "num_clusters": len(vals), "num_rows": len(records),
    }


def phase1_analysis(rows: list[dict[str, str]], n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Direction-level curves and crossovers.
    key_fields = ["template_id", "position_condition", "filler_family", "pair_id", "reciprocal_cluster_id", "donor_entity", "recipient_entity", "role", "actual_filler_length", "subject_to_readout_distance", "subject_positions", "readout_position"]
    curves: dict[tuple[str, ...], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        curves[key][row["position_group"]][int(row["layer"])] = float(row["normalized_recovery"])
    metrics = []
    for key, groups in curves.items():
        info = dict(zip(key_fields, key))
        subject = groups["subject_entity"]
        readout = groups["readout_final"]
        cross = stable_crossover(subject, readout)
        metric = {**info, "stable_crossover_layer": "" if cross is None else cross}
        for group in ("subject_entity", "query_role", "readout_final", "post_subject_excluding_readout"):
            values = [groups[group][layer] for layer in range(N_LAYERS)]
            metric[f"{group}_peak_layer"] = max(range(N_LAYERS), key=lambda x: values[x])
            metric[f"{group}_peak_recovery"] = max(values)
            metric[f"{group}_layer_auc"] = trapz(values)
        metrics.append(metric)
    # Aggregate every grouped curve with cluster bootstrap.
    buckets: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["template_id"], row["position_condition"], row["filler_family"], int(row["layer"]), row["position_group"])
        buckets[key].append({"reciprocal_cluster_id": row["reciprocal_cluster_id"], "value": float(row["normalized_recovery"])})
    aggregates = []
    for index, (key, cell) in enumerate(sorted(buckets.items())):
        summary = cluster_boot_mean(cell, "value", n_boot, index)
        aggregates.append(dict(zip(["template_id", "position_condition", "filler_family", "layer", "position_group"], key)) | summary)
    # Matched prefix-gap curve contrasts at similar final indices.  Boundary
    # tokenization can change the realized filler-only token count even when the
    # same tokenizer-matched text block is used, so retain that delta as metadata
    # instead of requiring equality.
    index: dict[tuple[str, str, int, str, int, str], dict[str, str]] = {}
    for row in rows:
        if row["position_condition"] in {"prefix +32", "gap +32", "prefix +128", "gap +128"}:
            target = int(row["target_filler_length"])
            key = (row["template_id"], row["filler_family"], target, row["pair_id"], int(row["layer"]), row["position_group"])
            index.setdefault(key, {})[row["placement"]] = row
    contrasts = []
    for key, matched in index.items():
        if set(matched) != {"prefix", "gap"}:
            continue
        p, g = matched["prefix"], matched["gap"]
        readout_delta = int(g["readout_position"]) - int(p["readout_position"])
        if abs(readout_delta) > 1:
            raise RuntimeError(f"matched prefix/gap final-index mismatch: {key}")
        contrasts.append({
            "template_id": key[0], "filler_family": key[1], "target_filler_length": key[2],
            "pair_id": key[3], "reciprocal_cluster_id": p["reciprocal_cluster_id"],
            "layer": key[4], "position_group": key[5],
            "prefix_readout_position": p["readout_position"], "gap_readout_position": g["readout_position"],
            "gap_minus_prefix_readout_position": readout_delta,
            "prefix_actual_filler_length": p["actual_filler_length"],
            "gap_actual_filler_length": g["actual_filler_length"],
            "gap_minus_prefix_actual_filler_length": int(g["actual_filler_length"]) - int(p["actual_filler_length"]),
            "gap_subject_position": json.loads(g["subject_positions"])[-1],
            "prefix_subject_position": json.loads(p["subject_positions"])[-1],
            "gap_minus_prefix_recovery": float(g["normalized_recovery"]) - float(p["normalized_recovery"]),
        })
    return metrics, aggregates, contrasts


def semantic_role_analysis(token_path: Path, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, int, int], list[float]]]:
    # Stream the multi-million-row token map into direction-level role/layer sums
    # and representative absolute-position map sums.
    sums: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
    absolute: dict[tuple[str, str, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    # This reciprocal direction is aligned and baseline-valid for every fixed
    # template/condition, including sentence-initial T3.
    representative_pair = "writer_Shakespeare_into_Dickens"
    for row in iter_csv(token_path):
        role = row["semantic_role"]
        key = (
            row["template_id"], row["position_condition"], row["filler_family"], row["pair_id"],
            row["reciprocal_cluster_id"], row["donor_entity"], row["recipient_entity"], row["role"],
            row["actual_filler_length"], row["subject_to_readout_distance"], row["subject_positions"],
            row["readout_position"], role, row["layer"],
        )
        sums[key][0] += float(row["normalized_recovery"])
        sums[key][1] += 1
        if row["pair_id"] == representative_pair and row["filler_family"] in {"none", "meadow"}:
            akey = (row["template_id"], row["position_condition"], int(row["absolute_position"]), int(row["layer"]))
            absolute[akey][0] += float(row["normalized_recovery"])
            absolute[akey][1] += 1
    profiles = []
    for key, (total, count) in sums.items():
        names = [
            "template_id", "position_condition", "filler_family", "pair_id", "reciprocal_cluster_id",
            "donor_entity", "recipient_entity", "role", "actual_filler_length",
            "subject_to_readout_distance", "subject_positions", "readout_position", "semantic_role", "layer",
        ]
        profiles.append(dict(zip(names, key)) | {"mean_token_recovery": total / count, "num_tokens": int(count)})
    by_curve: dict[tuple[str, ...], dict[int, float]] = defaultdict(dict)
    metric_key_names = [
        "template_id", "position_condition", "filler_family", "pair_id", "reciprocal_cluster_id",
        "donor_entity", "recipient_entity", "role", "actual_filler_length",
        "subject_to_readout_distance", "subject_positions", "readout_position", "semantic_role",
    ]
    for row in profiles:
        key = tuple(str(row[name]) for name in metric_key_names)
        by_curve[key][int(row["layer"])] = float(row["mean_token_recovery"])
    metrics = []
    for key, curve in by_curve.items():
        if set(curve) != set(range(N_LAYERS)):
            raise RuntimeError(f"incomplete semantic curve: {key}")
        values = [curve[layer] for layer in range(N_LAYERS)]
        metrics.append(dict(zip(metric_key_names, key)) | {
            "peak_layer": max(range(N_LAYERS), key=lambda x: values[x]),
            "peak_recovery": max(values), "layer_auc": trapz(values),
        })
    return profiles, metrics, absolute


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a)); db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


def correlation_matrix(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[float]]]:
    # Role-aligned vectors under none condition, averaged across all directions.
    means: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in profiles:
        if row["position_condition"] == "none":
            means[(row["template_id"], row["semantic_role"], int(row["layer"]))].append(float(row["mean_token_recovery"]))
    # Use the intersection of role/layer cells, because some templates have no source_prefix tokens.
    common_cells = [
        (role, layer) for role in SEMANTIC_LABELS for layer in range(N_LAYERS)
        if all((template_id, role, layer) in means for template_id in TEMPLATES)
    ]
    vectors = {t: [statistics.fmean(means[(t, role, layer)]) for role, layer in common_cells] for t in TEMPLATES}
    matrix = []
    rows = []
    for a in TEMPLATES:
        line = []
        for b in TEMPLATES:
            value = pearson(vectors[a], vectors[b])
            line.append(value)
            rows.append({"template_a": a, "template_b": b, "correlation": value, "aligned_cells": len(common_cells)})
        matrix.append(line)
    return rows, matrix


def template_contrasts(role_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in role_metrics:
        key = (row["position_condition"], row["filler_family"], row["pair_id"], row["semantic_role"])
        by[key][row["template_id"]] = row
    out = []
    templates = list(TEMPLATES)
    for key, cells in by.items():
        for i, a in enumerate(templates):
            for b in templates[i + 1:]:
                if a not in cells or b not in cells:
                    continue
                ra, rb = cells[a], cells[b]
                out.append({
                    "position_condition": key[0], "filler_family": key[1], "pair_id": key[2],
                    "reciprocal_cluster_id": ra["reciprocal_cluster_id"], "semantic_role": key[3],
                    "template_a": a, "template_b": b,
                    "b_minus_a_peak_layer": float(rb["peak_layer"]) - float(ra["peak_layer"]),
                    "b_minus_a_peak_recovery": float(rb["peak_recovery"]) - float(ra["peak_recovery"]),
                    "b_minus_a_layer_auc": float(rb["layer_auc"]) - float(ra["layer_auc"]),
                })
    return out


def ols_coefficients(metrics: list[dict[str, Any]], outcome: str) -> dict[str, float]:
    import numpy as np

    valid = [r for r in metrics if str(r.get(outcome, "")) != ""]
    names = ["intercept", "absolute_subject_position", "subject_to_readout_distance"] + [f"template_{t}" for t in list(TEMPLATES)[1:]]
    x, y = [], []
    for row in valid:
        subject = json.loads(row["subject_positions"])[-1]
        template = row["template_id"]
        x.append([1.0, float(subject), float(row["subject_to_readout_distance"])] + [1.0 if template == t else 0.0 for t in list(TEMPLATES)[1:]])
        y.append(float(row[outcome]))
    beta, *_ = np.linalg.lstsq(np.asarray(x), np.asarray(y), rcond=None)
    return dict(zip(names, [float(v) for v in beta]))


def regression_bootstrap(metrics: list[dict[str, Any]], outcome: str, n_boot: int) -> list[dict[str, Any]]:
    point = ols_coefficients(metrics, outcome)
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_cluster[row["reciprocal_cluster_id"]].append(row)
    clusters = sorted(by_cluster)
    rng = random.Random(BOOTSTRAP_SEED + 9000)
    boots: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sample = []
        for cluster in rng.choices(clusters, k=len(clusters)):
            sample.extend(by_cluster[cluster])
        values = ols_coefficients(sample, outcome)
        for name, value in values.items():
            boots[name].append(value)
    return [{
        "outcome": outcome, "coefficient": name, "estimate": estimate,
        "ci95_low_cluster_bootstrap": quantile(boots[name], 0.025),
        "ci95_high_cluster_bootstrap": quantile(boots[name], 0.975),
        "num_clusters": len(clusters), "bootstrap_resamples": n_boot,
    } for name, estimate in point.items()]


def make_plots(root: Path, grouped_agg: list[dict[str, Any]], profiles: list[dict[str, Any]],
               crossovers: list[dict[str, Any]], correlation: list[list[float]],
               absolute: dict[tuple[str, str, int, int], list[float]],
               prefix_gap: list[dict[str, Any]], template_diffs: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    colors = {"subject_entity": "#167d8d", "query_role": "#8e44ad", "readout_final": "#c0392b"}

    # Semantic-role heatmaps: Phase-1 char-offset-defined semantic groups cover
    # all five conditions (Phase 2 intentionally covers only none/+128).
    sem_sums: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    semantic_group_names = [
        "subject_entity", "source_role", "source_boundary", "neutral_filler",
        "query_determiner", "query_role", "readout_final",
    ]
    for row in grouped_agg:
        if row["position_group"] in semantic_group_names:
            sem_sums[(row["template_id"], row["position_condition"], row["position_group"], int(row["layer"]))].append(float(row["mean"]))
    for template_id in TEMPLATES:
        for condition in ["none", "prefix +32", "prefix +128", "gap +32", "gap +128"]:
            active_roles = [r for r in semantic_group_names if any((template_id, condition, r, l) in sem_sums for l in range(N_LAYERS))]
            matrix = np.asarray([[statistics.fmean(sem_sums[(template_id, condition, role, layer)]) if (template_id, condition, role, layer) in sem_sums else np.nan for layer in range(N_LAYERS)] for role in active_roles])
            fig, ax = plt.subplots(figsize=(10.5, max(3.2, 0.38 * len(active_roles))))
            im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(0, N_LAYERS, 2)); ax.set_xticklabels(range(0, N_LAYERS, 2), fontsize=7)
            ax.set_yticks(range(len(active_roles))); ax.set_yticklabels(active_roles, fontsize=8)
            ax.set_xlabel("Transformer block output layer"); ax.set_title(f"{template_id} — {condition}: role-aligned causal recovery")
            fig.colorbar(im, ax=ax, label="mean normalized recovery (shared [-1, 1])")
            fig.tight_layout()
            stem = plots / f"semantic_heatmap_{template_id}_{condition.replace(' ', '_').replace('+', 'plus')}"
            fig.savefig(stem.with_suffix(".png"), dpi=180); fig.savefig(stem.with_suffix(".pdf")); plt.close(fig)

    # Subject/query/readout curves, one compact panel per template.
    agg_index = {(r["template_id"], r["position_condition"], int(r["layer"]), r["position_group"]): [] for r in grouped_agg}
    for row in grouped_agg:
        agg_index[(row["template_id"], row["position_condition"], int(row["layer"]), row["position_group"])].append(float(row["mean"]))
    conditions = ["none", "prefix +32", "prefix +128", "gap +32", "gap +128"]
    for template_id in TEMPLATES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, group in zip(axes, colors):
            for condition in conditions:
                ys = [statistics.fmean(agg_index[(template_id, condition, layer, group)]) for layer in range(N_LAYERS)]
                ax.plot(range(N_LAYERS), ys, label=condition, lw=1.5)
            ax.set_title(group); ax.set_xlabel("Layer"); ax.grid(alpha=.2); ax.axhline(0, color=".5", lw=.5)
        axes[0].set_ylabel("Normalized recovery")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.02), fontsize=8)
        fig.suptitle(f"{template_id}: subject, query-role, and readout causal curves")
        fig.tight_layout(rect=[0, .08, 1, .95])
        stem = plots / f"subject_query_readout_curves_{template_id}"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight"); fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

    # Crossover versus realized distance.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    markers = dict(zip(TEMPLATES, ["o", "s", "^", "D", "P"]))
    for template_id in TEMPLATES:
        cell = [r for r in crossovers if r["template_id"] == template_id and str(r["stable_crossover_layer"]) != ""]
        ax.scatter([float(r["subject_to_readout_distance"]) for r in cell], [float(r["stable_crossover_layer"]) for r in cell], alpha=.35, s=22, marker=markers[template_id], label=template_id)
    ax.set_xlabel("Realized subject-to-readout token distance"); ax.set_ylabel("Stable crossover layer")
    ax.grid(alpha=.2); ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, -.14))
    fig.tight_layout(); stem = plots / "crossover_vs_realized_distance"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight"); fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

    # Representative absolute maps (Shakespeare->Dickens, meadow/none), annotated by semantic token labels.
    metadata_index = {}
    for path in root.glob("shard_*/phase2/*/*/writer_Shakespeare_into_Dickens/prompt_metadata.json"):
        m = json.loads(path.read_text())
        if m["filler_family"] in {"none", "meadow"}:
            metadata_index[(m["template_id"], m["position_condition"])] = m
    for template_id in TEMPLATES:
        fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=True)
        for ax, condition in zip(axes, ["none", "prefix +128", "gap +128"]):
            meta = metadata_index[(template_id, condition)]
            n = len(meta["recipient_token_ids"])
            matrix = np.full((n, N_LAYERS), np.nan)
            for position in range(n):
                for layer in range(N_LAYERS):
                    total, count = absolute[(template_id, condition, position, layer)]
                    if count:
                        matrix[position, layer] = total / count
            ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            step = max(1, n // 18)
            ticks = list(range(0, n, step))
            labels = [f"{p}: {meta['recipient_tokens'][p]} [{meta['semantic_labels'][p]}]" for p in ticks]
            ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=5.5)
            ax.set_title(condition); ax.set_xlabel("Layer")
        fig.suptitle(f"{template_id}: representative absolute-position maps with semantic annotations")
        fig.tight_layout(rect=[0, 0, 1, .96]); stem = plots / f"representative_absolute_map_{template_id}"
        fig.savefig(stem.with_suffix(".png"), dpi=180); fig.savefig(stem.with_suffix(".pdf")); plt.close(fig)

    # Cross-template role-profile correlation.
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(np.asarray(correlation), vmin=-1, vmax=1, cmap="coolwarm")
    labels = list(TEMPLATES)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels); ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    for i in range(5):
        for k in range(5): ax.text(k, i, f"{correlation[i][k]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Cross-template role-aligned profile correlation (none)")
    fig.colorbar(im, ax=ax, label="Pearson r"); fig.tight_layout(); stem = plots / "cross_template_role_profile_correlation"
    fig.savefig(stem.with_suffix(".png"), dpi=180); fig.savefig(stem.with_suffix(".pdf")); plt.close(fig)

    # Compact matched contrasts: distance (gap-prefix), template, and absolute subject shift (prefix128-prefix32).
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pg = [r["gap_minus_prefix_recovery"] for r in prefix_gap if r["position_group"] in {"subject_entity", "readout_final"}]
    td = [r["b_minus_a_layer_auc"] for r in template_diffs if r["semantic_role"] in {"subject_entity", "query_role", "readout_final"}]
    # Prefix absolute-shift contrast from grouped aggregate means.
    absolute_shift = []
    lookup = defaultdict(list)
    for r in grouped_agg:
        if r["position_condition"] in {"prefix +32", "prefix +128"} and r["position_group"] in {"subject_entity", "readout_final"}:
            lookup[(r["template_id"], r["layer"], r["position_group"], r["position_condition"])].append(float(r["mean"]))
    for t in TEMPLATES:
        for layer in range(N_LAYERS):
            for group in ("subject_entity", "readout_final"):
                a = lookup.get((t, layer, group, "prefix +32")); b = lookup.get((t, layer, group, "prefix +128"))
                if a and b: absolute_shift.append(statistics.fmean(b) - statistics.fmean(a))
    for ax, values, title, ylabel in zip(axes, [absolute_shift, pg, td], ["Absolute shift\nprefix128 − prefix32", "Distance/filler placement\ngap − prefix", "Template role-profile\npairwise AUC differences"], ["Recovery difference", "Recovery difference", "Layer-AUC difference"]):
        ax.boxplot(values, showfliers=False); ax.axhline(0, color=".4", lw=.7); ax.set_xticks([]); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); stem = plots / "compact_matched_contrasts"
    fig.savefig(stem.with_suffix(".png"), dpi=180); fig.savefig(stem.with_suffix(".pdf")); plt.close(fig)


def write_reports(root: Path, validation: dict[str, Any], crossovers: list[dict[str, Any]],
                  role_metrics: list[dict[str, Any]], correlations: list[dict[str, Any]],
                  regressions: list[dict[str, Any]], prefix_gap: list[dict[str, Any]], runtimes: dict[str, Any]) -> None:
    valid_cross = [float(r["stable_crossover_layer"]) for r in crossovers if str(r["stable_crossover_layer"]) != ""]
    by_template = {}
    for template_id in TEMPLATES:
        vals = [float(r["stable_crossover_layer"]) for r in crossovers if r["template_id"] == template_id and str(r["stable_crossover_layer"]) != ""]
        by_template[template_id] = statistics.fmean(vals) if vals else None
    corr_offdiag = [float(r["correlation"]) for r in correlations if r["template_a"] != r["template_b"]]
    pg_subject = [float(r["gap_minus_prefix_recovery"]) for r in prefix_gap if r["position_group"] == "subject_entity"]
    pg_readout = [float(r["gap_minus_prefix_recovery"]) for r in prefix_gap if r["position_group"] == "readout_final"]
    reg_lines = "\n".join(
        f"- {r['outcome']} / {r['coefficient']}: {r['estimate']:.4g} (95% cluster bootstrap {r['ci95_low_cluster_bootstrap']:.4g}, {r['ci95_high_cluster_bootstrap']:.4g})"
        for r in regressions
    )
    summary = f"""# Scientific summary

This experiment is same-prompt, same-layer residual activation patching in `{MODEL}` with BF16 eager inference. It does not inject states into a target prompt, and it does not interpret raw attention weights as causal evidence.

## Validation and scale

All gates: **{validation['status']}**. Phase 1 contains {validation['grouped_rows']:,} rows; Phase 2 contains {validation['token_rows']:,} rows. Preflight accepted {validation['accepted_combinations']:,} and rejected {validation['rejected_combinations']:,} template/condition/direction combinations. Identity maximum absolute recovery was {validation['identity_max_abs_recovery']:.6g}; oracle maximum absolute deviation from one was {validation['oracle_max_abs_deviation']:.6g}; unrelated-prefix maximum absolute recovery was {validation['unrelated_max_abs_recovery']:.6g}.

## Quantitative findings

Across direction-level conditions with a defined stable crossover, the mean crossover layer was {statistics.fmean(valid_cross):.3f}. Template means were {json.dumps(by_template, sort_keys=True)}. The mean off-diagonal cross-template correlation of role-aligned no-filler causal profiles was {statistics.fmean(corr_offdiag):.4f}.

At matched final indices (equal or one token apart) and matched filler family/target length, mean gap-minus-prefix recovery was {statistics.fmean(pg_subject):.5f} for the subject and {statistics.fmean(pg_readout):.5f} for the readout. Realized filler-token lengths and their deltas are retained in every contrast row because sentence-boundary tokenization can differ by one token. These are the primary matched distance/filler-placement contrasts; full layerwise values and cluster-bootstrap intervals are in the aggregate CSVs.

The joint linear decomposition (absolute subject position, realized subject-to-readout distance, and template indicators; descriptive rather than a fully randomized structural model) gave:

{reg_lines}

Semantic-role peak layers and layer-AUCs for every direction/condition are in `semantic_role_metrics.csv`; their eight-cluster bootstrap aggregates are in `semantic_role_metrics_cluster_bootstrap.csv`. Matched template contrasts, prefix-gap contrasts, crossover-versus-distance rows, and role-profile correlations are retained separately.

## Interpretation

Evidence for absolute position is assessed by prefix shifts and the absolute-position coefficient; evidence for distance is assessed most cleanly by matched gap-versus-prefix conditions with the same final index; syntactic/semantic role is assessed by role-aligned profiles across templates; template-specific computation is assessed by matched template contrasts and residual template coefficients. Exact numerical conclusions should be read from the reported confidence intervals rather than from a single heatmap.

Prefix length also changes preceding context, and moving filler from prefix to gap changes filler placement as well as subject-to-readout distance. Template syntax, token count, and semantic role are not all independently randomized. Normalized recovery is intentionally unclipped and can lie outside [0, 1]. Token-map interventions were batched, so effects at the established BF16 numerical scale are not substantively interpretable. No model comparison, head scan, MLP intervention, training, or attention-weight claim was performed.
"""
    (root / "scientific_summary.md").write_text(summary)
    readme = f"""# Qwen3-8B-Base template/position causal map

Controlled same-prompt causal activation patching tests absolute token position, subject-to-readout distance, semantic/syntactic role, and sentence-template-specific computation across five fixed templates, five position conditions, three validated filler families, all 36 blocks, 16 ordered directions, and eight reciprocal clusters.

Validation: **{validation['status']}**. The root `SUCCESS` marker is written only after preflight, smoke, Phase 1, Phase 2, merge, row-count, finite-value, alignment, control, cluster, analysis, and plot gates pass.

Key files:

- `raw_grouped_results.csv` and `raw_token_position_results.csv`: complete causal rows.
- `accepted_combinations.csv` and `rejected_combinations.csv`: predeclared preflight decisions.
- `pair_and_template_metadata.json`, `filler_blocks.json`, and `cluster_shard_mapping.csv`: design metadata.
- `validation_report.json`, `smoke/smoke_validation.json`, and `phase1_validation.json`: gates.
- Aggregate, bootstrap, matched-contrast, correlation, and regression CSV/JSON files: quantitative analysis.
- `plots/`: every required plot in PNG and PDF.
- `source_snapshot/`: executed script snapshot and diff.
- `commands.txt`, `gpu_mapping.json`, and worker logs: execution provenance.

Total recorded worker runtime: {runtimes.get('total_worker_runtime_seconds', 0):.1f} seconds. See `scientific_summary.md` for conclusions and limitations.
"""
    (root / "README.md").write_text(readme)


def analyze(args: argparse.Namespace) -> int:
    root = Path(args.root)
    pairs = load_pairs()
    accepted = read_csv(root / "accepted_combinations.csv")
    rejected = read_csv(root / "rejected_combinations.csv")
    phase1_paths = sorted(root.glob("shard_*/phase1/*/*/*/grouped_results.csv"))
    if not phase1_paths:
        raise RuntimeError("no Phase 1 checkpoint results found")
    grouped_path = root / "raw_grouped_results.csv"
    grouped_count = merge_csv_files(phase1_paths, grouped_path, RAW_FIELDS)
    grouped = read_csv(grouped_path)
    accepted_keys_set = {(r["template_id"], r["position_condition"], r["filler_family"], r["pair_id"]) for r in accepted}
    expected_grouped = len(accepted_keys_set) * N_LAYERS * len(GROUP_NAMES)
    finite = all(math.isfinite(float(r["normalized_recovery"])) for r in grouped)
    identity = [abs(float(r["normalized_recovery"])) for r in grouped if r["position_group"] == "identity_control"]
    oracle = [abs(float(r["normalized_recovery"]) - 1) for r in grouped if r["position_group"] == "all_prompt_positions_oracle"]
    unrelated = [abs(float(r["normalized_recovery"])) for r in grouped if r["position_group"] == "unrelated_prefix_negative_control"]
    phase1_checks = {
        "smoke_passed": (root / "smoke" / "SUCCESS").exists(),
        "all_workers_passed": all((root / f"shard_{i}" / "phase1_SUCCESS").exists() for i in range(args.num_shards)),
        "exact_expected_grouped_rows": grouped_count == expected_grouped,
        "all_36_layers": {int(r["layer"]) for r in grouped} == set(range(N_LAYERS)),
        "all_group_names": set(r["position_group"] for r in grouped) == set(GROUP_NAMES),
        "all_grouped_values_finite": finite,
        "identity_near_zero": bool(identity) and max(identity) <= 0.02,
        "oracle_near_one": bool(oracle) and max(oracle) <= 0.02,
        "unrelated_within_established_bf16_floor": bool(unrelated) and max(unrelated) <= BF16_CONTROL_FLOOR,
        "eight_reciprocal_clusters": len({r["reciprocal_cluster_id"] for r in grouped}) == 8,
        "sixteen_ordered_directions": len({r["pair_id"] for r in grouped}) == 16,
    }
    phase1_report = {
        "status": "PASS" if all(phase1_checks.values()) else "FAIL", "checks": phase1_checks,
        "expected_rows": expected_grouped, "observed_rows": grouped_count,
        "identity_max_abs_recovery": max(identity) if identity else None,
        "oracle_max_abs_deviation": max(oracle) if oracle else None,
        "unrelated_max_abs_recovery": max(unrelated) if unrelated else None,
        "completed_at": now(),
    }
    write_json(root / "phase1_validation.json", phase1_report)
    if args.phase == "phase1":
        if phase1_report["status"] != "PASS":
            raise RuntimeError(f"Phase 1 validation failed: {phase1_report}")
        log("PHASE 1 GLOBAL PASS")
        return 0
    if phase1_report["status"] != "PASS":
        raise RuntimeError("full analysis refused because Phase 1 does not pass")
    phase2_paths = sorted(root.glob("shard_*/phase2/*/*/*/token_position_results.csv"))
    token_path = root / "raw_token_position_results.csv"
    token_count = merge_csv_files(phase2_paths, token_path, RAW_FIELDS)
    # Expected token rows come from per-combination completion metadata and thus
    # exactly respect preflight exclusions and realized tokenizer lengths.
    expected_token = 0
    for path in sorted(root.glob("shard_*/phase2/*/*/*/COMPLETE.json")):
        info = json.loads(path.read_text())
        expected_token += int(info["token_length"]) * len(info["layers"])
    token_finite = True
    token_layers, token_pairs, token_templates = set(), set(), set()
    required_columns = set(RAW_FIELDS)
    for row in iter_csv(token_path):
        token_finite &= math.isfinite(float(row["normalized_recovery"]))
        token_layers.add(int(row["layer"])); token_pairs.add(row["pair_id"]); token_templates.add(row["template_id"])
        if row["semantic_role"] not in SEMANTIC_LABELS:
            token_finite = False
    full_checks = {
        **phase1_checks,
        "all_phase2_workers_passed": all((root / f"shard_{i}" / "phase2_SUCCESS").exists() for i in range(args.num_shards)),
        "exact_expected_token_rows": token_count == expected_token and token_count > 0,
        "token_values_finite": token_finite,
        "token_all_36_layers": token_layers == set(range(N_LAYERS)),
        "token_all_five_templates": token_templates == set(TEMPLATES),
        "token_all_sixteen_directions": len(token_pairs) == 16,
        "phase2_only_predeclared_conditions": all(r["position_condition"] in {"none", "prefix +128", "gap +128"} for r in iter_csv(token_path)),
        "raw_schema_complete": required_columns == set(next(iter_csv(token_path)).keys()),
    }
    if not all(full_checks.values()):
        report = {"status": "FAIL", "checks": full_checks, "grouped_rows": grouped_count, "token_rows": token_count}
        write_json(root / "validation_report.json", report)
        raise RuntimeError(f"full raw validation failed: {report}")

    crossovers, grouped_agg, prefix_gap = phase1_analysis(grouped, args.bootstrap_resamples)
    profiles, role_metrics, absolute = semantic_role_analysis(token_path, args.bootstrap_resamples)
    correlation_rows, correlation = correlation_matrix(profiles)
    template_diffs = template_contrasts(role_metrics)
    regression_rows = regression_bootstrap(crossovers, "stable_crossover_layer", args.bootstrap_resamples)
    crossover_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in crossovers:
        if str(row["stable_crossover_layer"]) != "":
            crossover_buckets[(row["template_id"], row["position_condition"], row["filler_family"])].append(row)
    crossover_boot = []
    for index, (key, cell) in enumerate(sorted(crossover_buckets.items())):
        summary = cluster_boot_mean(cell, "stable_crossover_layer", args.bootstrap_resamples, 5000 + index)
        crossover_boot.append(dict(zip(["template_id", "position_condition", "filler_family"], key)) | summary)
    # Every semantic-role metric gets an eight-cluster interval within design cell.
    role_buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in role_metrics:
        role_buckets[(row["template_id"], row["position_condition"], row["filler_family"], row["semantic_role"])].append(row)
    role_boot = []
    for index, (key, cell) in enumerate(sorted(role_buckets.items())):
        for value in ("peak_layer", "peak_recovery", "layer_auc"):
            summary = cluster_boot_mean(cell, value, args.bootstrap_resamples, 10000 + index)
            role_boot.append(dict(zip(["template_id", "position_condition", "filler_family", "semantic_role"], key)) | {"metric": value} | summary)
    contrast_buckets: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prefix_gap:
        contrast_buckets[(row["template_id"], row["filler_family"], int(row["target_filler_length"]), int(row["layer"]), row["position_group"])].append(row)
    prefix_gap_boot = []
    for index, (key, cell) in enumerate(sorted(contrast_buckets.items())):
        summary = cluster_boot_mean(cell, "gap_minus_prefix_recovery", args.bootstrap_resamples, 20000 + index)
        prefix_gap_boot.append(dict(zip(["template_id", "filler_family", "target_filler_length", "layer", "position_group"], key)) | summary)
    template_diff_buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in template_diffs:
        template_diff_buckets[(row["position_condition"], row["filler_family"], row["semantic_role"], row["template_a"], row["template_b"])].append(row)
    template_diff_boot = []
    for index, (key, cell) in enumerate(sorted(template_diff_buckets.items())):
        for value in ("b_minus_a_peak_layer", "b_minus_a_peak_recovery", "b_minus_a_layer_auc"):
            summary = cluster_boot_mean(cell, value, args.bootstrap_resamples, 30000 + index)
            template_diff_boot.append(dict(zip(["position_condition", "filler_family", "semantic_role", "template_a", "template_b"], key)) | {"metric": value} | summary)

    write_csv(root / "crossover_and_group_metrics.csv", crossovers, list(crossovers[0]))
    write_csv(root / "crossover_cluster_bootstrap.csv", crossover_boot, list(crossover_boot[0]))
    write_csv(root / "aggregate_grouped_cluster_bootstrap.csv", grouped_agg, list(grouped_agg[0]))
    write_csv(root / "matched_prefix_gap_contrasts.csv", prefix_gap, list(prefix_gap[0]))
    write_csv(root / "matched_prefix_gap_cluster_bootstrap.csv", prefix_gap_boot, list(prefix_gap_boot[0]))
    write_csv(root / "semantic_role_profiles.csv", profiles, list(profiles[0]))
    write_csv(root / "semantic_role_metrics.csv", role_metrics, list(role_metrics[0]))
    write_csv(root / "semantic_role_metrics_cluster_bootstrap.csv", role_boot, list(role_boot[0]))
    write_csv(root / "matched_template_contrasts.csv", template_diffs, list(template_diffs[0]))
    write_csv(root / "matched_template_contrasts_cluster_bootstrap.csv", template_diff_boot, list(template_diff_boot[0]))
    write_csv(root / "cross_template_role_profile_correlations.csv", correlation_rows, list(correlation_rows[0]))
    write_csv(root / "absolute_position_distance_template_regression.csv", regression_rows, list(regression_rows[0]))
    make_plots(root, grouped_agg, profiles, crossovers, correlation, absolute, prefix_gap, template_diffs)

    runtimes = {"total_worker_runtime_seconds": 0.0, "workers": []}
    for path in sorted(root.glob("shard_*/*_worker_validation.json")):
        info = json.loads(path.read_text()); runtimes["workers"].append(info)
        runtimes["total_worker_runtime_seconds"] += float(info["runtime_seconds"])
    write_json(root / "runtime_summary.json", runtimes)
    write_json(root / "pair_and_template_metadata.json", {
        "pairs": [p.__dict__ for p in pairs], "templates": TEMPLATES,
        "semantic_labels": SEMANTIC_LABELS, "group_names": GROUP_NAMES,
        "reference_metadata": str(REFERENCE_METADATA), "reference_handoff": str(REFERENCE_HANDOFF),
    })
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(SCRIPT_PATH, snapshot / SCRIPT_PATH.name)
    proc = subprocess.run(["git", "diff", "--no-index", "/dev/null", str(SCRIPT_PATH)], cwd=SCRIPT_PATH.parents[2], text=True, capture_output=True)
    (snapshot / "causal_entity_template_position_map.patch").write_text(proc.stdout)
    expected_plot_stems = [
        *(f"semantic_heatmap_{t}_{c}" for t in TEMPLATES for c in ["none", "prefix_plus32", "prefix_plus128", "gap_plus32", "gap_plus128"]),
        *(f"subject_query_readout_curves_{t}" for t in TEMPLATES),
        *(f"representative_absolute_map_{t}" for t in TEMPLATES),
        "crossover_vs_realized_distance", "cross_template_role_profile_correlation", "compact_matched_contrasts",
    ]
    plots_ok = all((root / "plots" / f"{stem}.{suffix}").exists() for stem in expected_plot_stems for suffix in ("png", "pdf"))
    full_checks["all_required_png_pdf_plots"] = plots_ok
    full_checks["analysis_outputs_nonempty"] = all((root / name).exists() and (root / name).stat().st_size > 0 for name in [
        "crossover_and_group_metrics.csv", "semantic_role_metrics.csv", "matched_template_contrasts.csv",
        "crossover_cluster_bootstrap.csv", "matched_template_contrasts_cluster_bootstrap.csv",
        "matched_prefix_gap_cluster_bootstrap.csv", "cross_template_role_profile_correlations.csv",
        "absolute_position_distance_template_regression.csv",
    ])
    validation = {
        "status": "PASS" if all(full_checks.values()) else "FAIL", "checks": full_checks,
        "accepted_combinations": len(accepted), "rejected_combinations": len(rejected),
        "grouped_rows": grouped_count, "expected_grouped_rows": expected_grouped,
        "token_rows": token_count, "expected_token_rows": expected_token,
        "identity_max_abs_recovery": max(identity), "oracle_max_abs_deviation": max(oracle),
        "unrelated_max_abs_recovery": max(unrelated), "num_clusters": len({r["reciprocal_cluster_id"] for r in grouped}),
        "num_directions": len({r["pair_id"] for r in grouped}), "completed_at": now(),
    }
    write_json(root / "validation_report.json", validation)
    rejection_reasons: dict[str, int] = defaultdict(int)
    for row in rejected:
        reason = row["reason"]
        category = "alignment_or_tokenization" if any(term in reason for term in ("token length mismatch", "span mismatch", "token differences", "leading-space", "RuntimeError")) else "baseline_gate"
        rejection_reasons[category] += 1
    finite_crossovers = [float(r["stable_crossover_layer"]) for r in crossovers if str(r["stable_crossover_layer"]) != ""]
    write_json(root / "aggregate_summary.json", {
        "validation_status": validation["status"], "row_counts": {"phase1_grouped": grouped_count, "phase2_token": token_count},
        "preflight": {"accepted": len(accepted), "rejected": len(rejected), "rejection_categories": dict(rejection_reasons)},
        "controls": {
            "identity_max_abs_recovery": max(identity), "oracle_max_abs_deviation": max(oracle),
            "unrelated_max_abs_recovery": max(unrelated),
        },
        "stable_crossover_mean": statistics.fmean(finite_crossovers),
        "stable_crossover_cluster_bootstrap": crossover_boot,
        "cross_template_correlations": correlation_rows,
        "absolute_position_distance_template_regression": regression_rows,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": args.bootstrap_resamples,
    })
    write_reports(root, validation, crossovers, role_metrics, correlation_rows, regression_rows, prefix_gap, runtimes)
    if validation["status"] != "PASS":
        raise RuntimeError(f"final validation failed: {validation}")
    (root / "SUCCESS").write_text(now() + "\n")
    log("FULL ANALYSIS PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "worker", "analyze"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=["phase1", "phase2", "full"], default="full")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAPS)
    args = parser.parse_args()
    if args.mode == "worker" and args.phase not in {"phase1", "phase2"}:
        parser.error("worker requires --phase phase1 or phase2")
    if args.mode == "analyze" and args.phase not in {"phase1", "full"}:
        parser.error("analyze requires --phase phase1 or full")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("invalid shard id")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return run_smoke(args)
    if args.mode == "worker":
        return run_worker(args)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
