"""Controlled absolute-position and subject-to-readout-distance causal tests.

This is a new, self-contained extension of the completed Qwen3-8B-Base
position-handoff and component-mediation experiments.  The authoritative pair
list and token IDs are loaded from the completed relay; no dataset is rebuilt.

Layer indexing and hooks are unchanged: layer 0 is the cumulative residual at
the output of transformer block 0.  Direct patches insert same-model donor
block-output states at the same token indices.  Attention sufficiency and
necessity use the established hybrid trajectory: donor P_subject at block-0
output is inserted into the recipient run, and the post-o_proj/pre-residual
attention output at P_readout is exchanged between recipient and hybrid runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


MODEL = "Qwen/Qwen3-8B-Base"
N_LAYERS = 36
REFERENCE_RELAY = Path("/ssd/mh3897/patchscope_results/qwen3_8b_base_p02_p10_relay_20260714_030311")
REFERENCE_METADATA = REFERENCE_RELAY / "pair_metadata.csv"
REFERENCE_HANDOFF = Path("/ssd/mh3897/patchscope_results/qwen3_8b_base_causal_entity_position_handoff_20260713_181400")
HOOK_DEFINITION = "cumulative residual state at output of transformer block; layer 0 is output of block 0"
ATTENTION_HOOK_DEFINITION = "self-attention output after o_proj and before residual addition at P_readout"
TARGET_LENGTHS = [0, 8, 32, 64, 128]
SMOKE_LAYERS = [0, 10, 23, 24, 26, 35]
MATRIX_TARGETS = {0, 32, 128}
EPS = 1e-12
BASELINE_LENGTH = 11

# Each family is made only of complete natural sentences.  At runtime the
# tokenizer chooses the cumulative sentence prefix nearest each target length.
# The leading space is intentional: it makes filler token IDs identical at the
# beginning of a prefix prompt and after the first sentence of a gap prompt.
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

FORBIDDEN_FILLER_TERMS = {
    "einstein", "newton", "darwin", "mozart", "bach", "shakespeare", "dickens",
    "france", "japan", "paris", "tokyo", "google", "apple", "scientist", "composer",
    "writer", "country", "city", "company",
}

DIRECT_FIELDS = [
    "experiment", "placement", "filler_family", "target_added_tokens", "actual_added_tokens",
    "condition_id", "pair_id", "unordered_pair_id", "donor_entity", "recipient_entity", "role",
    "layer", "hook_definition", "position_group", "patched_positions", "patched_token_ids",
    "patched_tokens", "P_subject", "P_readout", "subject_to_readout_distance", "filler_positions",
    "donor_margin", "recipient_margin", "normalization_denominator", "intervention_margin",
    "normalized_recovery", "donor_logit", "recipient_logit", "source_activation", "notes",
]

ATTENTION_FIELDS = [
    "experiment", "placement", "filler_family", "target_added_tokens", "actual_added_tokens",
    "condition_id", "pair_id", "unordered_pair_id", "donor_entity", "recipient_entity", "role",
    "layer", "hook_definition", "position_group", "P_subject", "P_readout",
    "subject_to_readout_distance", "filler_positions", "intervention", "donor_margin",
    "recipient_margin", "normalization_denominator", "intervention_margin", "normalized_recovery",
    "donor_logit", "recipient_logit", "hybrid_end_to_end_recovery", "recipient_vector_norm",
    "hybrid_vector_norm", "delta_vector_norm", "cosine_recipient_hybrid", "notes",
]


@dataclass(frozen=True)
class Pair:
    pair_id: str
    cluster: str
    donor: str
    recipient: str
    role: str
    donor_ids: list[int]
    recipient_ids: list[int]
    donor_token_id: int
    recipient_token_id: int


@dataclass
class Encoded:
    prompt: str
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]
    subject: list[int]
    readout: int
    filler: list[int]
    unrelated_pre_subject: list[int]


@dataclass
class Capture:
    block: dict[int, torch.Tensor]
    attention: dict[int, torch.Tensor]
    logits: torch.Tensor


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
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
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_pairs() -> list[Pair]:
    rows = read_csv(REFERENCE_METADATA)
    if len(rows) != 16:
        raise RuntimeError(f"canonical metadata has {len(rows)} rows, expected 16")
    pairs = [Pair(
        pair_id=r["pair_id"], cluster=r["unordered_pair_id"], donor=r["donor_entity"],
        recipient=r["recipient_entity"], role=r["role"], donor_ids=json.loads(r["donor_token_ids"]),
        recipient_ids=json.loads(r["recipient_token_ids"]), donor_token_id=int(r["donor_token_id"]),
        recipient_token_id=int(r["recipient_token_id"]),
    ) for r in rows]
    clusters: dict[str, int] = {}
    for pair in pairs:
        clusters[pair.cluster] = clusters.get(pair.cluster, 0) + 1
        diffs = [i for i, (a, b) in enumerate(zip(pair.donor_ids, pair.recipient_ids)) if a != b]
        if len(pair.donor_ids) != BASELINE_LENGTH or diffs != [2]:
            raise RuntimeError(f"{pair.pair_id}: canonical baseline alignment changed: len={len(pair.donor_ids)} diffs={diffs}")
    if len(clusters) != 8 or any(v != 2 for v in clusters.values()):
        raise RuntimeError(f"canonical reciprocal clusters invalid: {clusters}")
    return pairs


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


def last_logits(model, ids: torch.Tensor) -> torch.Tensor:
    # Equivalent to AutoModelForCausalLM.forward at the final token, without
    # materializing vocabulary logits at every earlier token.
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    return model.lm_head(hidden[:, -1]).detach().float().cpu()


def filler_blocks(tokenizer) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for family, sentences in FILLER_SENTENCES.items():
        joined = " ".join(sentences).lower()
        bad = sorted(term for term in FORBIDDEN_FILLER_TERMS if term in joined)
        if bad:
            raise RuntimeError(f"filler family {family} contains forbidden tested terms: {bad}")
        candidates = [" " + " ".join(sentences[:i]) for i in range(1, len(sentences) + 1)]
        family_blocks: dict[int, dict[str, Any]] = {0: {"text": "", "token_ids": [], "tokens": [], "token_count": 0}}
        for target in TARGET_LENGTHS[1:]:
            ranked = []
            for text in candidates:
                ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
                ranked.append((abs(len(ids) - target), len(ids), text, ids))
            _, count, text, ids = min(ranked, key=lambda x: (x[0], x[1]))
            family_blocks[target] = {
                "text": text, "token_ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids),
                "token_count": count,
            }
        result[family] = family_blocks
    return result


def span_positions(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if (a, b) != (0, 0) and b > start and a < end]


def build_prompt(entity: str, role: str, placement: str, filler: str) -> tuple[str, tuple[int, int] | None]:
    first = f"Everyone knows {entity} was a celebrated {role}."
    query = f"The {role} was"
    if not filler:
        return f"{first} {query}", None
    if placement == "prefix":
        return f"{filler} {first} {query}", (0, len(filler))
    if placement == "gap":
        return f"{first}{filler} {query}", (len(first), len(first) + len(filler))
    raise ValueError(placement)


def encode(tokenizer, entity: str, role: str, placement: str, filler: str) -> Encoded:
    prompt, filler_chars = build_prompt(entity, role, placement, filler)
    data = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(data["input_ids"])
    offsets = [tuple(x) for x in data["offset_mapping"]]
    subject_start = prompt.index(entity)
    subject = span_positions(offsets, subject_start, subject_start + len(entity))
    filler_pos = span_positions(offsets, *filler_chars) if filler_chars else []
    if not subject:
        raise RuntimeError(f"empty subject span: {prompt}")
    prefix = [i for i, (_a, b) in enumerate(offsets) if b <= subject_start]
    unrelated = [prefix[-1]] if prefix else []
    return Encoded(
        prompt=prompt, ids=ids, tokens=tokenizer.convert_ids_to_tokens(ids), offsets=offsets,
        subject=subject, readout=len(ids) - 1, filler=filler_pos, unrelated_pre_subject=unrelated,
    )


def validate_condition(tokenizer, pair: Pair, placement: str, target: int, block: dict[str, Any]) -> tuple[Encoded, Encoded, dict[str, Any]]:
    donor = encode(tokenizer, pair.donor, pair.role, placement, block["text"])
    recipient = encode(tokenizer, pair.recipient, pair.role, placement, block["text"])
    if len(donor.ids) != len(recipient.ids):
        raise RuntimeError(f"{pair.pair_id}: donor/recipient length mismatch")
    if donor.subject != recipient.subject or donor.readout != recipient.readout or donor.filler != recipient.filler:
        raise RuntimeError(f"{pair.pair_id}: aligned spans differ")
    diffs = [i for i, (a, b) in enumerate(zip(donor.ids, recipient.ids)) if a != b]
    if diffs != donor.subject or len(donor.subject) != 1:
        raise RuntimeError(f"{pair.pair_id}: non-subject mismatch or multi-token subject: diffs={diffs} subject={donor.subject}")
    if target == 0 and (donor.ids != pair.donor_ids or recipient.ids != pair.recipient_ids):
        raise RuntimeError(f"{pair.pair_id}: zero-filler prompt differs from canonical relay")
    donor_filler_ids = [donor.ids[p] for p in donor.filler]
    recipient_filler_ids = [recipient.ids[p] for p in recipient.filler]
    if donor_filler_ids != recipient_filler_ids or donor_filler_ids != block["token_ids"]:
        raise RuntimeError(
            f"{pair.pair_id} {placement} {target}: contextual filler IDs differ from standalone block: "
            f"context={donor_filler_ids} standalone={block['token_ids']}"
        )
    actual_added = len(donor.ids) - BASELINE_LENGTH
    meta = {
        "pair_id": pair.pair_id, "unordered_pair_id": pair.cluster, "placement": placement,
        "filler_family": "", "target_added_tokens": target, "actual_added_tokens": actual_added,
        "filler_text": block["text"], "filler_token_count": block["token_count"],
        "filler_token_ids": block["token_ids"], "filler_tokens": block["tokens"],
        "donor_prompt": donor.prompt, "recipient_prompt": recipient.prompt,
        "donor_token_ids": donor.ids, "recipient_token_ids": recipient.ids,
        "donor_tokens": donor.tokens, "recipient_tokens": recipient.tokens,
        "P_subject": donor.subject, "P_readout": donor.readout,
        "subject_to_readout_distance": donor.readout - donor.subject[-1],
        "filler_positions": donor.filler, "unrelated_pre_subject_positions": donor.unrelated_pre_subject,
        "clean_corrupt_alignment_valid": True,
    }
    return donor, recipient, meta


def replace_output(out: Any, hidden: torch.Tensor) -> Any:
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


@torch.inference_mode()
def capture(model, layers, ids: list[int], device: torch.device, source: tuple[list[int], torch.Tensor] | None = None) -> Capture:
    block: dict[int, torch.Tensor] = {}
    attention: dict[int, torch.Tensor] = {}
    handles = []
    if source is not None:
        positions, values = source
        def source_hook(_module, _inputs, out):
            h0 = out[0] if isinstance(out, tuple) else out
            h = h0.clone(); h[0, positions] = values.to(h)
            return replace_output(out, h)
        handles.append(layers[0].register_forward_hook(source_hook))
    for li, layer in enumerate(layers):
        def block_hook(_module, _inputs, out, index=li):
            h = out[0] if isinstance(out, tuple) else out
            block[index] = h[0].detach().clone()
        def attention_hook(_module, _inputs, out, index=li):
            attention[index] = out[0][0].detach().clone()
        handles.append(layer.register_forward_hook(block_hook))
        handles.append(layer.self_attn.register_forward_hook(attention_hook))
    try:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = last_logits(model, x)[0]
    finally:
        for handle in handles: handle.remove()
    if len(block) != N_LAYERS or len(attention) != N_LAYERS:
        raise RuntimeError(f"incomplete capture: block={len(block)} attention={len(attention)}")
    return Capture(block, attention, logits)


def margin(logits: torch.Tensor, pair: Pair) -> float:
    return float(logits[pair.donor_token_id] - logits[pair.recipient_token_id])


def vector_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.float()).cpu())


def vector_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1))[0].cpu())


def base_row(pair: Pair, placement: str, family: str, target: int, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "placement": placement, "filler_family": family, "target_added_tokens": target,
        "actual_added_tokens": meta["actual_added_tokens"],
        "condition_id": f"{placement}__{family}__target_{target:03d}__actual_{meta['actual_added_tokens']:03d}",
        "pair_id": pair.pair_id, "unordered_pair_id": pair.cluster, "donor_entity": pair.donor,
        "recipient_entity": pair.recipient, "role": pair.role, "P_subject": json.dumps(meta["P_subject"]),
        "P_readout": meta["P_readout"], "subject_to_readout_distance": meta["subject_to_readout_distance"],
        "filler_positions": json.dumps(meta["filler_positions"]),
    }


def direct_specs(enc: Encoded, full_matrix: bool) -> list[tuple[str, list[int], str, str]]:
    post = list(range(enc.subject[-1] + 1, enc.readout + 1))
    specs: list[tuple[str, list[int], str, str]] = [
        ("subject_span", enc.subject, "donor", ""),
        ("readout_token", [enc.readout], "donor", ""),
        ("post_subject_including_readout", post, "donor", ""),
        ("post_subject_excluding_readout", [p for p in post if p != enc.readout], "donor", ""),
        ("all_prompt_positions_oracle", list(range(len(enc.ids))), "donor", ""),
        ("unrelated_pre_subject_control", enc.unrelated_pre_subject, "donor", "pre-subject state is causally upstream of entity"),
        ("identity_control", [enc.readout], "recipient", "recipient activation patched back into recipient"),
    ]
    if enc.filler:
        specs.append(("filler_block_patch", enc.filler, "donor", "semantic filler; causally downstream only in gap placement"))
    if full_matrix:
        specs.extend(("token_position", [p], "donor", "full layer-by-token-position matrix") for p in range(len(enc.ids)))
    return [(name, pos, source, notes) for name, pos, source, notes in specs if pos]


@torch.inference_mode()
def direct_layer(model, layers, pair: Pair, enc: Encoded, donor: Capture, recipient: Capture,
                 layer_index: int, specs: list[tuple[str, list[int], str, str]], device: torch.device,
                 dm: float, rm: float, common: dict[str, Any]) -> list[dict[str, Any]]:
    # Singleton controls retain the baseline BF16 GEMM batch shape. A mixed
    # intervention batch can otherwise move exact controls by about 0.03 in
    # normalized recovery, as the first smoke test demonstrated.
    controls = {"identity_control", "all_prompt_positions_oracle"}
    batches = [[spec] for spec in specs if spec[0] in controls]
    regular = [spec for spec in specs if spec[0] not in controls]
    if regular:
        batches.insert(0, regular)
    logits_by_key: dict[tuple[str, tuple[int, ...], str], torch.Tensor] = {}
    for batch_specs in batches:
        x = torch.tensor([enc.ids] * len(batch_specs), dtype=torch.long, device=device)
        def hook(_module, _inputs, out):
            h0 = out[0] if isinstance(out, tuple) else out
            h = h0.clone()
            for bi, (_name, positions, source, _notes) in enumerate(batch_specs):
                values = donor.block[layer_index] if source == "donor" else recipient.block[layer_index]
                h[bi, positions] = values[positions].to(h)
            return replace_output(out, h)
        handle = layers[layer_index].register_forward_hook(hook)
        try:
            batch_logits = last_logits(model, x)
        finally:
            handle.remove()
        for bi, (name, positions, source, _notes) in enumerate(batch_specs):
            logits_by_key[(name, tuple(positions), source)] = batch_logits[bi]
    rows = []
    denom = dm - rm
    for name, positions, source, notes in specs:
        result_logits = logits_by_key[(name, tuple(positions), source)]
        im = margin(result_logits, pair)
        row = dict(common)
        row.update({
            "experiment": "residual_position_patching", "layer": layer_index,
            "hook_definition": HOOK_DEFINITION, "position_group": name,
            "patched_positions": json.dumps(positions),
            "patched_token_ids": json.dumps([enc.ids[p] for p in positions]),
            "patched_tokens": json.dumps([enc.tokens[p] for p in positions]),
            "donor_margin": dm, "recipient_margin": rm, "normalization_denominator": denom,
            "intervention_margin": im, "normalized_recovery": (im - rm) / denom,
            "donor_logit": float(result_logits[pair.donor_token_id]),
            "recipient_logit": float(result_logits[pair.recipient_token_id]),
            "source_activation": source, "notes": notes,
        })
        rows.append(row)
    return rows


@torch.inference_mode()
def attention_layer(model, layers, pair: Pair, enc: Encoded, donor: Capture, recipient: Capture,
                    hybrid: Capture, layer_index: int, device: torch.device, dm: float, rm: float,
                    common: dict[str, Any], hybrid_recovery: float) -> list[dict[str, Any]]:
    # Batch shape and ordering exactly follow the established component runner:
    # row 0 is recipient+sufficiency; row 1 is hybrid+necessity.
    x = torch.tensor([enc.ids, enc.ids], dtype=torch.long, device=device)
    subject = enc.subject
    readout = enc.readout
    target_rec = recipient.attention[layer_index][readout]
    target_hyb = hybrid.attention[layer_index][readout]
    def source_hook(_module, _inputs, out):
        h0 = out[0] if isinstance(out, tuple) else out
        h = h0.clone(); h[1, subject] = donor.block[0][subject].to(h)
        return replace_output(out, h)
    def attn_hook(_module, _inputs, out):
        h = out[0].clone(); h[0, readout] = target_hyb.to(h); h[1, readout] = target_rec.to(h)
        return (h,) + out[1:]
    handles = [layers[0].register_forward_hook(source_hook), layers[layer_index].self_attn.register_forward_hook(attn_hook)]
    try:
        logits = last_logits(model, x)
    finally:
        for handle in handles: handle.remove()
    denom = dm - rm
    shared = {
        "experiment": "attention_output_mediation", "layer": layer_index,
        "hook_definition": ATTENTION_HOOK_DEFINITION, "position_group": "readout_token",
        "donor_margin": dm, "recipient_margin": rm, "normalization_denominator": denom,
        "hybrid_end_to_end_recovery": hybrid_recovery,
        "recipient_vector_norm": vector_norm(target_rec), "hybrid_vector_norm": vector_norm(target_hyb),
        "delta_vector_norm": vector_norm(target_hyb - target_rec),
        "cosine_recipient_hybrid": vector_cosine(target_rec, target_hyb),
        "notes": "established block-0 P_subject hybrid and post-o_proj/pre-residual attention exchange",
    }
    rows = []
    for bi, intervention in enumerate(("attention_sufficiency", "attention_necessity")):
        im = margin(logits[bi], pair)
        row = dict(common); row.update(shared); row.update({
            "intervention": intervention, "intervention_margin": im,
            "normalized_recovery": (im - rm) / denom,
            "donor_logit": float(logits[bi, pair.donor_token_id]),
            "recipient_logit": float(logits[bi, pair.recipient_token_id]),
        })
        rows.append(row)
    return rows


def condition_dir(out_dir: Path, placement: str, family: str, target: int, actual: int, pair: Pair) -> Path:
    condition = f"{placement}__{family}__target_{target:03d}__actual_{actual:03d}"
    return out_dir / "checkpoints" / condition / pair.pair_id


def run_one(model, tokenizer, layers, pair: Pair, placement: str, family: str, target: int,
            block: dict[str, Any], layer_list: list[int], full_matrix: bool, device: torch.device,
            out_dir: Path) -> dict[str, Any]:
    donor_enc, recipient_enc, meta = validate_condition(tokenizer, pair, placement, target, block)
    meta["filler_family"] = family
    checkpoint = condition_dir(out_dir, placement, family, target, meta["actual_added_tokens"], pair)
    complete = checkpoint / "COMPLETE.json"
    if complete.exists():
        log(f"reuse complete {checkpoint.relative_to(out_dir)}")
        return json.loads(complete.read_text())
    checkpoint.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint / "tokenized_prompt_metadata.json", meta)
    started = time.time()
    donor = capture(model, layers, donor_enc.ids, device)
    recipient = capture(model, layers, recipient_enc.ids, device)
    hybrid = capture(model, layers, recipient_enc.ids, device, (recipient_enc.subject, donor.block[0][recipient_enc.subject]))
    dm, rm = margin(donor.logits, pair), margin(recipient.logits, pair)
    denom = dm - rm
    if not math.isfinite(denom) or abs(denom) < EPS:
        raise RuntimeError(f"{pair.pair_id}: invalid denominator {denom}")
    hybrid_recovery = (margin(hybrid.logits, pair) - rm) / denom
    common = base_row(pair, placement, family, target, meta)
    specs = direct_specs(recipient_enc, full_matrix)
    direct_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    for layer_index in layer_list:
        direct_rows.extend(direct_layer(
            model, layers, pair, recipient_enc, donor, recipient, layer_index, specs, device, dm, rm, common,
        ))
        attention_rows.extend(attention_layer(
            model, layers, pair, recipient_enc, donor, recipient, hybrid, layer_index, device, dm, rm,
            common, hybrid_recovery,
        ))
        write_csv(checkpoint / "direct_results.partial.csv", direct_rows, DIRECT_FIELDS)
        write_csv(checkpoint / "attention_results.partial.csv", attention_rows, ATTENTION_FIELDS)
        log(f"{pair.pair_id} {placement}/{family}/{target} layer {layer_index} complete")
    write_csv(checkpoint / "direct_results.csv", direct_rows, DIRECT_FIELDS)
    write_csv(checkpoint / "attention_results.csv", attention_rows, ATTENTION_FIELDS)
    details = {
        "pair_id": pair.pair_id, "unordered_pair_id": pair.cluster, "placement": placement,
        "filler_family": family, "target_added_tokens": target, "actual_added_tokens": meta["actual_added_tokens"],
        "P_subject": meta["P_subject"], "P_readout": meta["P_readout"],
        "subject_to_readout_distance": meta["subject_to_readout_distance"],
        "layers": layer_list, "full_matrix": full_matrix, "num_direct_rows": len(direct_rows),
        "num_attention_rows": len(attention_rows), "donor_margin": dm, "recipient_margin": rm,
        "normalization_denominator": denom, "hybrid_end_to_end_recovery": hybrid_recovery,
        "runtime_seconds": time.time() - started, "completed_at": now(),
    }
    write_json(complete, details)
    (checkpoint / "SUCCESS").write_text(now() + "\n")
    return details


def previous_subject_map(pair_ids: set[str]) -> dict[tuple[str, int], float]:
    result = {}
    for row in read_csv(REFERENCE_HANDOFF / "grouped_position_results.csv"):
        if row["pair_id"] in pair_ids and row["position_group"] == "subject_span":
            result[(row["pair_id"], int(row["layer"]))] = float(row["normalized_recovery"])
    return result


def validate_smoke(out_dir: Path, pair_ids: set[str], layers: list[int]) -> dict[str, Any]:
    direct: list[dict[str, str]] = []
    attention: list[dict[str, str]] = []
    metadata = []
    for path in out_dir.glob("checkpoints/*/*/direct_results.csv"):
        direct.extend(read_csv(path))
        metadata.append(json.loads((path.parent / "tokenized_prompt_metadata.json").read_text()))
    for path in out_dir.glob("checkpoints/*/*/attention_results.csv"):
        attention.extend(read_csv(path))
    finite = all(math.isfinite(float(r["normalized_recovery"])) for r in direct + attention)
    identity = [abs(float(r["normalized_recovery"])) for r in direct if r["position_group"] == "identity_control"]
    oracle = [float(r["normalized_recovery"]) for r in direct if r["position_group"] == "all_prompt_positions_oracle"]
    previous = previous_subject_map(pair_ids)
    deltas = []
    for row in direct:
        if (row["position_group"] == "subject_span" and int(row["target_added_tokens"]) == 0
                and row["filler_family"] == "meadow"):
            key = (row["pair_id"], int(row["layer"]))
            deltas.append(abs(float(row["normalized_recovery"]) - previous[key]))
    # Compare exact filler-token payloads across matched prefix/gap metadata.
    matched: dict[tuple[str, str, int], dict[str, list[int]]] = {}
    for meta in metadata:
        key = (meta["pair_id"], meta["filler_family"], int(meta["target_added_tokens"]))
        matched.setdefault(key, {})[meta["placement"]] = meta["filler_token_ids"]
    filler_match = all(set(v) == {"prefix", "gap"} and v["prefix"] == v["gap"] for v in matched.values())
    checks = {
        "clean_corrupt_token_alignment_valid": all(m["clean_corrupt_alignment_valid"] for m in metadata),
        "subject_spans_located": all(bool(m["P_subject"]) for m in metadata),
        "readout_positions_located": all(int(m["P_readout"]) == len(m["recipient_token_ids"]) - 1 for m in metadata),
        "matched_prefix_gap_identical_filler_tokens": filler_match,
        "identity_recovery_max_abs_le_0p02": bool(identity) and max(identity) <= 0.02,
        "oracle_recovery_within_0p02_of_one": bool(oracle) and max(abs(x - 1.0) for x in oracle) <= 0.02,
        "all_values_finite": finite,
        "zero_filler_reference_max_abs_delta_le_0p06": bool(deltas) and max(deltas) <= 0.06,
        "required_layers_present": {int(r["layer"]) for r in direct} == set(layers),
        "one_reciprocal_cluster": len(pair_ids) == 2,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "num_directions": len(pair_ids), "num_prompt_metadata": len(metadata),
        "num_direct_rows": len(direct), "num_attention_rows": len(attention),
        "identity_max_abs_recovery": max(identity) if identity else None,
        "oracle_max_abs_deviation_from_one": max(abs(x - 1.0) for x in oracle) if oracle else None,
        "zero_filler_reference_max_abs_delta": max(deltas) if deltas else None,
        "zero_filler_reference_mean_abs_delta": sum(deltas) / len(deltas) if deltas else None,
    }
    write_json(out_dir / "validation_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("smoke validation failed: " + json.dumps(report, sort_keys=True))
    return report


def parse_ints(value: str | None, default: Iterable[int]) -> list[int]:
    return list(default) if not value else [int(x) for x in value.split(",") if x.strip()]


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    write_json(out_dir / "environment.json", {
        "model": MODEL, "python": sys.executable, "torch": torch.__version__, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "device_argument": args.device, "precision": "bfloat16 on CUDA", "attention_implementation": "eager",
        "hook_definition": HOOK_DEFINITION, "attention_hook_definition": ATTENTION_HOOK_DEFINITION,
        "reference_metadata": str(REFERENCE_METADATA), "started_at": now(),
    })
    device = torch.device(args.device)
    model, tokenizer, layers = load_model(device)
    pairs = load_pairs()
    # Revalidate the zero-filler tokenizer output against the authoritative metadata.
    for pair in pairs:
        d = tokenizer(f"Everyone knows {pair.donor} was a celebrated {pair.role}. The {pair.role} was", add_special_tokens=True)["input_ids"]
        r = tokenizer(f"Everyone knows {pair.recipient} was a celebrated {pair.role}. The {pair.role} was", add_special_tokens=True)["input_ids"]
        if list(d) != pair.donor_ids or list(r) != pair.recipient_ids:
            raise RuntimeError(f"{pair.pair_id}: current tokenizer differs from canonical metadata")
    blocks = filler_blocks(tokenizer)
    write_json(out_dir / "filler_blocks.json", blocks)
    placements = ["prefix", "gap"] if args.placement == "both" else [args.placement]
    targets = parse_ints(args.targets, [0, 8] if args.smoke else TARGET_LENGTHS)
    layer_list = parse_ints(args.layers, SMOKE_LAYERS if args.smoke else range(N_LAYERS))
    families = args.families.split(",") if args.families else list(FILLER_SENTENCES)
    if args.smoke:
        first_cluster = pairs[0].cluster
        run_pairs = [pair for pair in pairs if pair.cluster == first_cluster]
    else:
        run_pairs = pairs
    manifest = []
    for placement in placements:
        for family in families:
            for target in targets:
                block = blocks[family][target]
                for pair in run_pairs:
                    details = run_one(
                        model, tokenizer, layers, pair, placement, family, target, block, layer_list,
                        (not args.smoke and target in MATRIX_TARGETS), device, out_dir,
                    )
                    manifest.append(details)
                    write_json(out_dir / "run_manifest.partial.json", manifest)
                    if device.type == "cuda": torch.cuda.empty_cache()
    write_json(out_dir / "run_manifest.json", manifest)
    if args.smoke:
        report = validate_smoke(out_dir, {p.pair_id for p in run_pairs}, layer_list)
        log("SMOKE SUCCESS " + json.dumps(report, sort_keys=True))
    write_json(out_dir / "worker_validation.json", {
        "status": "PASS", "num_canonical_pairs_validated": len(pairs), "num_directions_run": len(run_pairs),
        "num_conditions": len(placements) * len(families) * len(targets), "layers": layer_list,
        "placements": placements, "families": families, "targets": targets,
        "all_completion_markers_present": len(manifest) == len(run_pairs) * len(placements) * len(families) * len(targets),
        "completed_at": now(),
    })
    (out_dir / "SUCCESS").write_text(now() + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--placement", choices=["prefix", "gap", "both"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--families", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
