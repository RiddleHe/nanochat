"""Layer-by-token-position causal entity handoff patching.

This extends PR #15's same-prompt subject-only causal patching experiment.
It is not Patchscope target-prompt injection.

Layer indexing: layer 0 means the output residual state of transformer block 0.
The intervention replaces selected token-position cumulative residual vectors at
that block output with same-model, same-layer, same-position donor vectors. Donor
activations are inserted exactly as recorded; no norm matching is used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


RESULT_ROOT = Path("/ssd/mh3897/patchscope_results")
DEFAULT_MODEL = "Qwen/Qwen3-8B-Base"
TEMPLATE = "Everyone knows {name} was a celebrated {role}. The {role} was"
HOOK_DEFINITION = "cumulative residual state at output of transformer block; layer 0 is output of block 0"
BASELINE_SEPARATION_THRESHOLD = 0.5
REFERENCE_SUBJECT_CURVE = {0: 0.994, 10: 0.920, 22: 0.849, 23: 0.725, 24: 0.411, 35: 0.000}

ENTITIES = [
    ("Einstein", "scientist"),
    ("Newton", "scientist"),
    ("Darwin", "scientist"),
    ("Mozart", "composer"),
    ("Bach", "composer"),
    ("Shakespeare", "writer"),
    ("Dickens", "writer"),
    ("France", "country"),
    ("Japan", "country"),
    ("Paris", "city"),
    ("Tokyo", "city"),
    ("Google", "company"),
    ("Apple", "company"),
]

RESULT_FIELDS = [
    "pair_id", "shard_id", "donor_entity", "recipient_entity", "role", "direction", "prompt", "layer",
    "layer_hook_definition", "position_group", "token_positions", "token_ids", "decoded_tokens", "subject_span",
    "first_role_span", "second_role_span", "donor_token_id", "recipient_token_id", "donor_token_string",
    "recipient_token_string", "original_donor_logit_donor", "original_donor_logit_recipient",
    "original_donor_prob_donor", "original_donor_prob_recipient", "original_recipient_logit_donor",
    "original_recipient_logit_recipient", "original_recipient_prob_donor", "original_recipient_prob_recipient",
    "patched_logit_donor", "patched_logit_recipient", "patched_prob_donor", "patched_prob_recipient",
    "original_donor_margin", "original_recipient_margin", "patched_margin", "baseline_separation",
    "normalized_recovery",
]


@dataclass(frozen=True)
class OrderedPair:
    pair_id: str
    donor: str
    recipient: str
    role: str


@dataclass
class EncodedPrompt:
    pair: OrderedPair
    entity: str
    other_entity: str
    prompt: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    tokens: list[str]
    subject_pos: list[int]
    first_role_pos: list[int]
    second_role_pos: list[int]
    final_pos: list[int]
    post_subject_including_final_pos: list[int]
    post_subject_excluding_final_pos: list[int]
    except_subject_pos: list[int]
    all_pos: list[int]
    unrelated_prefix_pos: list[int]
    entity_token_id: int
    other_entity_token_id: int
    entity_token_string: str
    other_entity_token_string: str


def log(msg: str) -> None:
    print(msg, flush=True)


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_ordered_pairs() -> list[OrderedPair]:
    by_role: dict[str, list[str]] = {}
    for name, role in ENTITIES:
        by_role.setdefault(role, []).append(name)
    pairs: list[OrderedPair] = []
    for role, names in by_role.items():
        for donor in names:
            for recipient in names:
                if donor == recipient:
                    continue
                pairs.append(OrderedPair(f"{role}_{donor}_into_{recipient}", donor, recipient, role))
    return pairs


def expected_pair_count_by_role(pairs: list[OrderedPair]) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in pairs:
        out[pair.role] = out.get(pair.role, 0) + 1
    return out


def build_prompt(name: str, role: str) -> str:
    return TEMPLATE.format(name=name, role=role)


def infer_spans(prompt: str, name: str, role: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    subject_start = prompt.index(name)
    role_spans = []
    start = 0
    while True:
        idx = prompt.find(role, start)
        if idx < 0:
            break
        role_spans.append((idx, idx + len(role)))
        start = idx + len(role)
    if len(role_spans) != 2:
        raise ValueError(f"expected exactly two role mentions for {role!r} in {prompt!r}; found {len(role_spans)}")
    return (subject_start, subject_start + len(name)), role_spans[0], role_spans[1]


def load_hf(name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = list(model.gpt_neox.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
    else:
        raise RuntimeError(f"cannot locate transformer blocks for {name}")
    return model, tok, layers


def encode_with_offsets(tok, prompt: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    tokens = tok.convert_ids_to_tokens(ids)
    return ids, offsets, tokens


def char_span_to_positions(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    return [i for i, (a, b) in enumerate(offsets) if (a, b) != (0, 0) and b > start and a < end]


def leading_space_single_token(tok, name: str) -> tuple[int | None, str, list[int], list[str]]:
    ids = tok(" " + name, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    return (ids[0] if len(ids) == 1 else None), (toks[0] if len(toks) == 1 else ""), ids, toks


def encode_prompt(tok, pair: OrderedPair, entity: str, other: str) -> EncodedPrompt:
    prompt = build_prompt(entity, pair.role)
    ids, offsets, tokens = encode_with_offsets(tok, prompt)
    subject_span, first_role_span, second_role_span = infer_spans(prompt, entity, pair.role)
    subject_pos = char_span_to_positions(offsets, subject_span)
    first_role_pos = char_span_to_positions(offsets, first_role_span)
    second_role_pos = char_span_to_positions(offsets, second_role_span)
    if not subject_pos or not first_role_pos or not second_role_pos:
        raise ValueError(f"empty span mapping for {pair.pair_id} prompt {prompt!r}")
    final_pos = [len(ids) - 1]
    post_subject_including_final = list(range(subject_pos[-1] + 1, len(ids)))
    post_subject_excluding_final = [p for p in post_subject_including_final if p != final_pos[0]]
    except_subject = [i for i in range(len(ids)) if i not in set(subject_pos)]
    prefix = [i for i, (_, b) in enumerate(offsets) if b <= subject_span[0]]
    unrelated_prefix = [prefix[-1]] if prefix else []
    ent_id, ent_tok, ent_ids, ent_toks = leading_space_single_token(tok, entity)
    oth_id, oth_tok, oth_ids, oth_toks = leading_space_single_token(tok, other)
    if ent_id is None:
        raise ValueError(f"entity {entity!r} is not one leading-space token: ids={ent_ids} toks={ent_toks}")
    if oth_id is None:
        raise ValueError(f"entity {other!r} is not one leading-space token: ids={oth_ids} toks={oth_toks}")
    return EncodedPrompt(
        pair=pair,
        entity=entity,
        other_entity=other,
        prompt=prompt,
        ids=ids,
        offsets=offsets,
        tokens=tokens,
        subject_pos=subject_pos,
        first_role_pos=first_role_pos,
        second_role_pos=second_role_pos,
        final_pos=final_pos,
        post_subject_including_final_pos=post_subject_including_final,
        post_subject_excluding_final_pos=post_subject_excluding_final,
        except_subject_pos=except_subject,
        all_pos=list(range(len(ids))),
        unrelated_prefix_pos=unrelated_prefix,
        entity_token_id=ent_id,
        other_entity_token_id=oth_id,
        entity_token_string=ent_tok,
        other_entity_token_string=oth_tok,
    )


def validate_pair(tok, pair: OrderedPair) -> tuple[EncodedPrompt | None, EncodedPrompt | None, str | None]:
    try:
        donor = encode_prompt(tok, pair, pair.donor, pair.recipient)
        recipient = encode_prompt(tok, pair, pair.recipient, pair.donor)
    except Exception as exc:
        return None, None, str(exc)
    if len(donor.ids) != len(recipient.ids):
        return donor, recipient, f"token length mismatch: {len(donor.ids)} vs {len(recipient.ids)}"
    checks = [
        ("subject", donor.subject_pos, recipient.subject_pos),
        ("first_role", donor.first_role_pos, recipient.first_role_pos),
        ("second_role", donor.second_role_pos, recipient.second_role_pos),
        ("final", donor.final_pos, recipient.final_pos),
    ]
    for name, a, b in checks:
        if a != b:
            return donor, recipient, f"{name} positions mismatch: {a} vs {b}"
    subject = set(donor.subject_pos)
    for i, (a, b) in enumerate(zip(donor.ids, recipient.ids)):
        if i not in subject and a != b:
            return donor, recipient, f"non-subject token mismatch at {i}: {donor.tokens[i]!r} vs {recipient.tokens[i]!r}"
    return donor, recipient, None


def validate_all_pairs(tok) -> tuple[list[tuple[OrderedPair, EncodedPrompt, EncodedPrompt]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = make_ordered_pairs()
    role_counts = expected_pair_count_by_role(pairs)
    log("Ordered pair list:")
    for i, pair in enumerate(pairs):
        log(f"  {i:02d}: {pair.pair_id} ({pair.role})")
    if len(pairs) != 16 or role_counts != {"scientist": 6, "composer": 2, "writer": 2, "country": 2, "city": 2, "company": 2}:
        raise RuntimeError(f"ordered pair construction failed: n={len(pairs)} counts={role_counts}")

    valid = []
    meta_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for pair in pairs:
        donor, recipient, reason = validate_pair(tok, pair)
        if reason is not None or donor is None or recipient is None:
            skipped.append({"pair_id": pair.pair_id, "donor": pair.donor, "recipient": pair.recipient, "role": pair.role, "reason": reason or "unknown"})
            continue
        meta_rows.append({
            "pair_id": pair.pair_id,
            "donor": pair.donor,
            "recipient": pair.recipient,
            "role": pair.role,
            "donor_prompt": donor.prompt,
            "recipient_prompt": recipient.prompt,
            "token_len": len(donor.ids),
            "donor_token_id": donor.entity_token_id,
            "recipient_token_id": donor.other_entity_token_id,
            "donor_token_string": donor.entity_token_string,
            "recipient_token_string": donor.other_entity_token_string,
            "subject_span_positions": json.dumps(donor.subject_pos),
            "first_role_positions": json.dumps(donor.first_role_pos),
            "second_role_positions": json.dumps(donor.second_role_pos),
            "final_position": json.dumps(donor.final_pos),
            "tokens": json.dumps(donor.tokens),
            "token_ids": json.dumps(donor.ids),
        })
        valid.append((pair, donor, recipient))
    if len(valid) != 16:
        raise RuntimeError(f"expected 16 valid ordered pairs, got {len(valid)}; skipped={skipped}")
    return valid, meta_rows, skipped


def shard_bounds(n: int, num_shards: int) -> list[tuple[int, int]]:
    base = n // num_shards
    rem = n % num_shards
    out = []
    start = 0
    for sid in range(num_shards):
        size = base + (1 if sid < rem else 0)
        out.append((start, start + size))
        start += size
    return out


def select_shard(items: list[Any], num_shards: int, shard_id: int) -> list[Any]:
    lo, hi = shard_bounds(len(items), num_shards)[shard_id]
    return items[lo:hi]


def pair_assignments(valid: list[tuple[OrderedPair, EncodedPrompt, EncodedPrompt]], num_shards: int) -> list[dict[str, Any]]:
    bounds = shard_bounds(len(valid), num_shards)
    rows = []
    for sid, (lo, hi) in enumerate(bounds):
        for idx in range(lo, hi):
            pair = valid[idx][0]
            rows.append({"shard_id": sid, "pair_index": idx, "pair_id": pair.pair_id, "donor": pair.donor, "recipient": pair.recipient, "role": pair.role})
    return rows


@torch.inference_mode()
def forward_logits(model, ids: list[int], device: torch.device, patcher=None) -> torch.Tensor:
    handle = patcher() if patcher is not None else None
    try:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        return model(x).logits[0, -1, :].detach().float().cpu()
    finally:
        if handle is not None:
            handle.remove()


@torch.inference_mode()
def capture_layer_outputs(model, layers, ids: list[int], device: torch.device) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    acts: dict[int, torch.Tensor] = {}
    handles = []
    def mk(layer: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[layer] = h[0, :len(ids), :].detach().clone()
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(mk(i)))
    try:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x).logits[0, -1, :].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return acts, logits


def make_patcher(layers, layer: int, positions: list[int], donor_acts: dict[int, torch.Tensor], prompt_len: int, device: torch.device):
    pos = list(dict.fromkeys(positions))
    def install():
        def hook(_mod, _inp, out):
            h0 = out[0] if isinstance(out, tuple) else out
            if h0.shape[1] < prompt_len:
                return out
            h = h0.clone()
            donor = donor_acts[layer].to(device=device, dtype=h.dtype)
            for p in pos:
                if 0 <= p < prompt_len:
                    h[:, p, :] = donor[p, :]
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return layers[layer].register_forward_hook(hook)
    return install


def logit_prob(logits: torch.Tensor, token_id: int) -> tuple[float, float]:
    probs = F.softmax(logits, dim=-1)
    return float(logits[token_id].item()), float(probs[token_id].item())


def margin(logits: torch.Tensor, donor_token_id: int, recipient_token_id: int) -> float:
    return float((logits[donor_token_id] - logits[recipient_token_id]).item())


def recovery(patched_margin: float, clean_margin: float, corrupt_margin: float) -> float:
    denom = clean_margin - corrupt_margin
    return float("nan") if abs(denom) < 1e-12 else (patched_margin - corrupt_margin) / denom


def group_positions(enc: EncodedPrompt) -> list[tuple[str, list[int]]]:
    return [
        ("subject_span", enc.subject_pos),
        ("first_role", enc.first_role_pos),
        ("second_role", enc.second_role_pos),
        ("final_token", enc.final_pos),
        ("post_subject_including_final", enc.post_subject_including_final_pos),
        ("post_subject_excluding_final", enc.post_subject_excluding_final_pos),
        ("except_subject", enc.except_subject_pos),
        ("all_prompt_positions_oracle", enc.all_pos),
        ("unrelated_prefix_negative_control", enc.unrelated_prefix_pos),
    ]


def smoke_groups(enc: EncodedPrompt) -> list[tuple[str, list[int]]]:
    wanted = {"subject_span", "final_token", "post_subject_excluding_final", "post_subject_including_final", "all_prompt_positions_oracle", "unrelated_prefix_negative_control"}
    return [(k, v) for k, v in group_positions(enc) if k in wanted and v]


def token_position_groups(enc: EncodedPrompt) -> list[tuple[str, list[int]]]:
    return [(f"token_{i:02d}", [i]) for i in range(len(enc.ids))]


def positions_payload(enc: EncodedPrompt, positions: list[int]) -> tuple[str, str, str]:
    return json.dumps(positions), json.dumps([enc.ids[p] for p in positions]), json.dumps([enc.tokens[p] for p in positions])


def run_pair(model, layers, pair: OrderedPair, donor: EncodedPrompt, recipient: EncodedPrompt, layer_list: list[int], device: torch.device, shard_id: int | None, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donor_acts, donor_logits = capture_layer_outputs(model, layers, donor.ids, device)
    _, recipient_logits = capture_layer_outputs(model, layers, recipient.ids, device)
    donor_token_id = donor.entity_token_id
    recipient_token_id = donor.other_entity_token_id
    clean_margin = margin(donor_logits, donor_token_id, recipient_token_id)
    corrupt_margin = margin(recipient_logits, donor_token_id, recipient_token_id)
    sep = abs(clean_margin - corrupt_margin)
    if sep < BASELINE_SEPARATION_THRESHOLD:
        raise RuntimeError(f"baseline separation too small for {pair.pair_id}: {sep:.4f}")
    # no-patch reproducibility check
    donor_logits_2 = forward_logits(model, donor.ids, device)
    recipient_logits_2 = forward_logits(model, recipient.ids, device)
    no_patch_max_abs_diff = max(float((donor_logits - donor_logits_2).abs().max().item()), float((recipient_logits - recipient_logits_2).abs().max().item()))

    dld, dpd = logit_prob(donor_logits, donor_token_id)
    dlr, dpr = logit_prob(donor_logits, recipient_token_id)
    rld, rpd = logit_prob(recipient_logits, donor_token_id)
    rlr, rpr = logit_prob(recipient_logits, recipient_token_id)
    rows: list[dict[str, Any]] = []
    groups = smoke_groups(recipient) if mode == "smoke" else token_position_groups(recipient) + group_positions(recipient)
    for layer in layer_list:
        for group, positions in groups:
            if not positions:
                continue
            patcher = make_patcher(layers, layer, positions, donor_acts, len(recipient.ids), device)
            patched_logits = forward_logits(model, recipient.ids, device, patcher)
            pld, ppd = logit_prob(patched_logits, donor_token_id)
            plr, ppr = logit_prob(patched_logits, recipient_token_id)
            patched_margin = margin(patched_logits, donor_token_id, recipient_token_id)
            pos_json, ids_json, toks_json = positions_payload(recipient, positions)
            rows.append({
                "pair_id": pair.pair_id,
                "shard_id": "" if shard_id is None else shard_id,
                "donor_entity": pair.donor,
                "recipient_entity": pair.recipient,
                "role": pair.role,
                "direction": f"{pair.donor}_into_{pair.recipient}",
                "prompt": recipient.prompt,
                "layer": layer,
                "layer_hook_definition": HOOK_DEFINITION,
                "position_group": group,
                "token_positions": pos_json,
                "token_ids": ids_json,
                "decoded_tokens": toks_json,
                "subject_span": json.dumps(recipient.subject_pos),
                "first_role_span": json.dumps(recipient.first_role_pos),
                "second_role_span": json.dumps(recipient.second_role_pos),
                "donor_token_id": donor_token_id,
                "recipient_token_id": recipient_token_id,
                "donor_token_string": donor.entity_token_string,
                "recipient_token_string": donor.other_entity_token_string,
                "original_donor_logit_donor": dld,
                "original_donor_logit_recipient": dlr,
                "original_donor_prob_donor": dpd,
                "original_donor_prob_recipient": dpr,
                "original_recipient_logit_donor": rld,
                "original_recipient_logit_recipient": rlr,
                "original_recipient_prob_donor": rpd,
                "original_recipient_prob_recipient": rpr,
                "patched_logit_donor": pld,
                "patched_logit_recipient": plr,
                "patched_prob_donor": ppd,
                "patched_prob_recipient": ppr,
                "original_donor_margin": clean_margin,
                "original_recipient_margin": corrupt_margin,
                "patched_margin": patched_margin,
                "baseline_separation": sep,
                "normalized_recovery": recovery(patched_margin, clean_margin, corrupt_margin),
            })
    meta = {
        "pair_id": pair.pair_id,
        "shard_id": "" if shard_id is None else shard_id,
        "donor": pair.donor,
        "recipient": pair.recipient,
        "role": pair.role,
        "clean_margin": clean_margin,
        "corrupt_margin": corrupt_margin,
        "baseline_separation": sep,
        "no_patch_max_abs_logit_diff": no_patch_max_abs_diff,
    }
    return rows, meta


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        val = numeric(r["normalized_recovery"])
        if math.isnan(val):
            continue
        buckets.setdefault((int(r["layer"]), str(r["position_group"])), []).append(val)
    out = []
    for (layer, group), vals in sorted(buckets.items()):
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        out.append({"layer": layer, "position_group": group, "mean": mean, "std": std, "count": len(vals)})
    return out


def bootstrap_ci(rows: list[dict[str, Any]], out_dir: Path, n_boot: int = 1000) -> list[dict[str, Any]]:
    rng = random.Random(0)
    units = sorted({str(r["pair_id"]) for r in rows})
    records = []
    for key in sorted({(int(r["layer"]), str(r["position_group"])) for r in rows}):
        layer, group = key
        vals_by_pair: dict[str, list[float]] = {}
        for r in rows:
            if int(r["layer"]) == layer and str(r["position_group"]) == group:
                val = numeric(r["normalized_recovery"])
                if not math.isnan(val):
                    vals_by_pair.setdefault(str(r["pair_id"]), []).append(val)
        vals = [sum(vals_by_pair[u]) / len(vals_by_pair[u]) for u in units if u in vals_by_pair and vals_by_pair[u]]
        if not vals:
            continue
        boots = []
        for _ in range(n_boot):
            sample = [rng.choice(vals) for _ in vals]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        records.append({
            "layer": layer,
            "position_group": group,
            "mean": sum(vals) / len(vals),
            "ci_low": boots[int(0.025 * (len(boots) - 1))],
            "ci_high": boots[int(0.975 * (len(boots) - 1))],
            "n_pairs": len(vals),
        })
    write_csv(out_dir / "bootstrap_ci_by_layer_and_position.csv", records)
    return records


def per_pair_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = sorted({str(r["pair_id"]) for r in rows})
    out = []
    crossover = []
    for pid in pairs:
        sub = [r for r in rows if str(r["pair_id"]) == pid]
        by = {(int(r["layer"]), str(r["position_group"])): numeric(r["normalized_recovery"]) for r in sub if not str(r["position_group"]).startswith("token_")}
        layers = sorted({k[0] for k in by})
        cross = ""
        for layer in layers:
            s = by.get((layer, "subject_span"), float("nan"))
            f = by.get((layer, "final_token"), float("nan"))
            if not math.isnan(s) and not math.isnan(f) and f >= s:
                cross = layer
                break
        first = sub[0] if sub else {}
        crossover.append({"pair_id": pid, "donor": first.get("donor_entity", ""), "recipient": first.get("recipient_entity", ""), "role": first.get("role", ""), "crossover_layer_final_ge_subject": cross})
        for layer in layers:
            row = {"pair_id": pid, "layer": layer}
            for group in ["subject_span", "final_token", "post_subject_including_final", "post_subject_excluding_final", "all_prompt_positions_oracle", "unrelated_prefix_negative_control"]:
                row[group] = by.get((layer, group), "")
            out.append(row)
    return out, crossover


def make_plots(rows: list[dict[str, Any]], summary: list[dict[str, Any]], ci_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    if summary:
        layers = sorted({int(r["layer"]) for r in summary})
        groups = ["subject_span", "first_role", "second_role", "final_token", "post_subject_including_final", "post_subject_excluding_final", "except_subject", "all_prompt_positions_oracle", "unrelated_prefix_negative_control"]
        vals = {(int(r["layer"]), str(r["position_group"])): numeric(r["mean"]) for r in summary}
        matrix = [[vals.get((layer, group), float("nan")) for layer in layers] for group in groups]
        fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(layers)), 5.2))
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
        ax.set_xlabel("Layer"); ax.set_title("Mean normalized recovery by patched position group")
        fig.colorbar(im, ax=ax, label="mean normalized recovery")
        fig.tight_layout(); fig.savefig(plots / "mean_layer_by_position_heatmap.png", dpi=180); plt.close(fig)

    curve_groups = ["subject_span", "final_token", "post_subject_including_final", "post_subject_excluding_final", "all_prompt_positions_oracle"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in curve_groups:
        sub = sorted([r for r in summary if r["position_group"] == group], key=lambda r: int(r["layer"]))
        if sub:
            ax.plot([int(r["layer"]) for r in sub], [numeric(r["mean"]) for r in sub], marker="o", ms=2.5, label=group)
    ax.axhline(0, color="0.4", lw=0.6); ax.axhline(1, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean normalized recovery"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(plots / "position_group_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for group, color in [("subject_span", "#2c8a8a"), ("final_token", "#c0392b")]:
        sub = sorted([r for r in summary if r["position_group"] == group], key=lambda r: int(r["layer"]))
        if sub:
            ax.plot([int(r["layer"]) for r in sub], [numeric(r["mean"]) for r in sub], marker="o", ms=3, label=group, color=color)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean normalized recovery"); ax.set_title("Subject vs final-token crossover")
    ax.axhline(0, color="0.4", lw=0.6); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "subject_vs_final_crossover.png", dpi=180); plt.close(fig)

    ci_groups = curve_groups
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in ci_groups:
        sub = sorted([r for r in ci_rows if r["position_group"] == group], key=lambda r: int(r["layer"]))
        if sub:
            xs = [int(r["layer"]) for r in sub]
            ys = [numeric(r["mean"]) for r in sub]
            lo = [numeric(r["ci_low"]) for r in sub]
            hi = [numeric(r["ci_high"]) for r in sub]
            ax.plot(xs, ys, marker="o", ms=2.5, label=group)
            ax.fill_between(xs, lo, hi, alpha=0.12)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean normalized recovery with bootstrap 95% CI"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(plots / "bootstrap_ci_curves.png", dpi=180); plt.close(fig)

    token_rows = [r for r in rows if str(r["position_group"]).startswith("token_")]
    for pid in sorted({str(r["pair_id"]) for r in token_rows}):
        sub = [r for r in token_rows if str(r["pair_id"]) == pid]
        layers = sorted({int(r["layer"]) for r in sub})
        toks = sorted({int(str(r["position_group"]).replace("token_", "")) for r in sub})
        vals: dict[tuple[int, int], list[float]] = {}
        for r in sub:
            vals.setdefault((int(str(r["position_group"]).replace("token_", "")), int(r["layer"])), []).append(numeric(r["normalized_recovery"]))
        matrix = []
        for t in toks:
            row = []
            for layer in layers:
                xs = [x for x in vals.get((t, layer), []) if not math.isnan(x)]
                row.append(sum(xs) / len(xs) if xs else float("nan"))
            matrix.append(row)
        fig, ax = plt.subplots(figsize=(max(9, 0.28 * len(layers)), max(3.8, 0.28 * len(toks))))
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
        ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks)
        ax.set_xlabel("Layer"); ax.set_ylabel("Token position"); ax.set_title(pid)
        fig.colorbar(im, ax=ax, label="normalized recovery")
        fig.tight_layout(); fig.savefig(plots / f"{pid}__token_position_heatmap.png", dpi=180); plt.close(fig)


def write_readme(out_dir: Path, model: str, n_pairs: int, n_skipped: int, mode: str) -> None:
    text = f"""# Qwen3-8B-Base causal entity position handoff

Mode: `{mode}`

Model: `{model}`

Prompt template: `{TEMPLATE}`

Tokenizer: model tokenizer with `add_special_tokens=True` for prompts, matching PR #15.

Metric: PR #15 logit-difference metric `m(x) = logit(donor entity) - logit(recipient entity)` using leading-space single-token entity IDs.

Layer hook: {HOOK_DEFINITION}

Valid ordered pairs: {n_pairs}

Skipped pairs: {n_skipped}

No donor activation normalization is applied.
"""
    (out_dir / "README.md").write_text(text)


def summarize_and_plot(out_dir: Path, rows: list[dict[str, Any]], model: str, mode: str, n_pairs: int, n_skipped: int) -> None:
    grouped = [r for r in rows if not str(r["position_group"]).startswith("token_")]
    token = [r for r in rows if str(r["position_group"]).startswith("token_")]
    write_csv(out_dir / "grouped_position_results.csv", grouped, RESULT_FIELDS)
    write_csv(out_dir / "token_position_results.csv", token, RESULT_FIELDS)
    summary = aggregate_summary(grouped)
    write_csv(out_dir / "summary_by_layer_and_position.csv", summary)
    ci_rows = bootstrap_ci(grouped, out_dir)
    pair_summary, crossover = per_pair_summary(grouped)
    write_csv(out_dir / "per_pair_summary.csv", pair_summary)
    write_csv(out_dir / "per_pair_crossover_layers.csv", crossover)
    make_plots(rows, summary, ci_rows, out_dir)
    write_readme(out_dir, model, n_pairs, n_skipped, mode)


def compare_subject_curve(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer = {int(r["layer"]): numeric(r["mean"]) for r in summary if r["position_group"] == "subject_span"}
    rows = []
    for layer, ref in REFERENCE_SUBJECT_CURVE.items():
        got = by_layer.get(layer, float("nan"))
        rows.append({"layer": layer, "observed_subject_recovery": got, "pr15_reference": ref, "delta": got - ref if not math.isnan(got) else ""})
    return rows


def run_experiment(args: argparse.Namespace, out_dir: Path) -> int:
    device = torch.device(args.device)
    model, tok, layers = load_hf(args.model, device)
    log(f"Loaded {args.model}: n_layer={len(layers)} on {device}")
    valid, meta_rows, skipped = validate_all_pairs(tok)
    log(f"Validated exactly {len(valid)} ordered pairs; skipped={len(skipped)}")

    run_items = valid
    if args.smoke:
        run_items = [x for x in valid if x[0].donor == "Einstein" and x[0].recipient == "Newton"]
        layers_to_run = [0, 10, 23, 24, 28, 35]
    else:
        layers_to_run = list(range(len(layers))) if not args.layers else sorted({int(x) for x in args.layers.split(",")})
        if args.num_shards > 1:
            run_items = select_shard(valid, args.num_shards, args.shard_id)
    selected_ids = {item[0].pair_id for item in run_items}
    assignments = [r for r in pair_assignments(valid, args.num_shards) if args.num_shards == 1 or int(r["shard_id"]) == args.shard_id]
    selected_meta = [r for r in meta_rows if r["pair_id"] in selected_ids]
    write_csv(out_dir / "pair_assignments.csv", assignments)
    write_csv(out_dir / "pair_metadata.csv", selected_meta)
    write_csv(out_dir / "skipped_pairs.csv", skipped, ["pair_id", "donor", "recipient", "role", "reason"])
    log(f"Running {len(run_items)} pairs on layers {layers_to_run[0]}..{layers_to_run[-1]} mode={'smoke' if args.smoke else 'full'} shard={args.shard_id}/{args.num_shards}")

    all_rows: list[dict[str, Any]] = []
    pair_controls = []
    for idx, (pair, donor, recipient) in enumerate(run_items):
        log(f"Pair {idx + 1}/{len(run_items)}: {pair.pair_id}")
        rows, meta = run_pair(model, layers, pair, donor, recipient, layers_to_run, device, args.shard_id if args.num_shards > 1 else None, "smoke" if args.smoke else "full")
        all_rows.extend(rows)
        pair_controls.append(meta)
        log(f"  done {pair.pair_id}: sep={meta['baseline_separation']:.4f} no_patch_max_abs_diff={meta['no_patch_max_abs_logit_diff']:.3g}")

    summarize_and_plot(out_dir, all_rows, args.model, "smoke" if args.smoke else "full-shard" if args.num_shards > 1 else "full", len(run_items), len(skipped))
    summary_rows = read_csv(out_dir / "summary_by_layer_and_position.csv")
    subject_compare = compare_subject_curve(summary_rows)
    write_csv(out_dir / "subject_curve_pr15_comparison.csv", subject_compare)
    oracle_vals = [numeric(r["normalized_recovery"]) for r in all_rows if r["position_group"] == "all_prompt_positions_oracle"]
    neg_vals = [numeric(r["normalized_recovery"]) for r in all_rows if r["position_group"] == "unrelated_prefix_negative_control"]
    subj = {int(r["layer"]): numeric(r["mean"]) for r in summary_rows if r["position_group"] == "subject_span"}
    validation = {
        "model": args.model,
        "num_ordered_pairs_validated": len(valid),
        "num_pairs_run": len(run_items),
        "num_skipped_pairs": len(skipped),
        "layers": layers_to_run,
        "all_position_oracle_mean_recovery": sum(oracle_vals) / len(oracle_vals) if oracle_vals else None,
        "all_position_oracle_min_recovery": min(oracle_vals) if oracle_vals else None,
        "negative_control_mean_abs_recovery": sum(abs(x) for x in neg_vals) / len(neg_vals) if neg_vals else None,
        "subject_curve_by_layer": {str(k): v for k, v in sorted(subj.items())},
        "pair_controls": pair_controls,
        "subject_curve_pr15_comparison": subject_compare,
    }
    (out_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2))
    if args.smoke:
        keep = {"subject_span", "final_token", "post_subject_excluding_final", "post_subject_including_final", "all_prompt_positions_oracle", "unrelated_prefix_negative_control"}
        log("\nSmoke-test table:")
        log("layer | position_group | normalized_recovery | patched_margin")
        for r in all_rows:
            if r["position_group"] in keep:
                log(f"{r['layer']} | {r['position_group']} | {numeric(r['normalized_recovery']):.4f} | {numeric(r['patched_margin']):.4f}")
    return 0


def merge_shards(root: Path, num_shards: int, model: str) -> int:
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for sid in range(num_shards):
        shard = root / f"shard_{sid}"
        required = [shard / "grouped_position_results.csv", shard / "token_position_results.csv", shard / "pair_metadata.csv", shard / "skipped_pairs.csv"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"shard {sid} missing files: {missing}")
        rows.extend(read_csv(shard / "grouped_position_results.csv"))
        rows.extend(read_csv(shard / "token_position_results.csv"))
        meta.extend(read_csv(shard / "pair_metadata.csv"))
        skipped.extend(read_csv(shard / "skipped_pairs.csv"))
        assignments.extend(read_csv(shard / "pair_assignments.csv"))
    pair_ids = [r["pair_id"] for r in meta]
    dupes = sorted({p for p in pair_ids if pair_ids.count(p) > 1})
    if len(pair_ids) != 16 or dupes:
        raise RuntimeError(f"merged pair coverage invalid: n={len(pair_ids)} dupes={dupes} pairs={pair_ids}")
    write_csv(root / "pair_assignments.csv", assignments)
    write_csv(root / "pair_metadata.csv", meta)
    write_csv(root / "skipped_pairs.csv", skipped, ["pair_id", "donor", "recipient", "role", "reason"])
    summarize_and_plot(root, rows, model, "merged-3gpu", len(pair_ids), len(skipped))
    summary_rows = read_csv(root / "summary_by_layer_and_position.csv")
    subject_compare = compare_subject_curve(summary_rows)
    write_csv(root / "subject_curve_pr15_comparison.csv", subject_compare)
    grouped = [r for r in rows if not str(r["position_group"]).startswith("token_")]
    oracle_vals = [numeric(r["normalized_recovery"]) for r in grouped if r["position_group"] == "all_prompt_positions_oracle"]
    neg_vals = [numeric(r["normalized_recovery"]) for r in grouped if r["position_group"] == "unrelated_prefix_negative_control"]
    validation = {
        "model": model,
        "merged_shards": num_shards,
        "num_ordered_pairs": len(pair_ids),
        "pair_ids": pair_ids,
        "all_position_oracle_mean_recovery": sum(oracle_vals) / len(oracle_vals) if oracle_vals else None,
        "all_position_oracle_min_recovery": min(oracle_vals) if oracle_vals else None,
        "negative_control_mean_abs_recovery": sum(abs(x) for x in neg_vals) / len(neg_vals) if neg_vals else None,
        "subject_curve_pr15_comparison": subject_compare,
    }
    (root / "validation_summary.json").write_text(json.dumps(validation, indent=2))
    log(f"Merged {num_shards} shards with exactly {len(pair_ids)} ordered pairs")
    return 0


def ensure_out_dir(base: str | None, model: str) -> Path:
    if base:
        out = Path(base)
    else:
        slug = "qwen3_8b_base_causal_entity_position_handoff"
        out = RESULT_ROOT / f"{slug}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--merge-shards", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir, args.model)
    cmd = " ".join([sys.executable] + sys.argv)
    (out_dir / "command.txt").write_text(cmd + "\n")
    if args.merge_shards:
        return merge_shards(out_dir, args.num_shards, args.model)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    return run_experiment(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
