"""Causal relay from an entity span to the final next-token readout position.

The dataset and token alignment are required to match the completed PR #15
entity-position experiment. Layer indices refer to cumulative residual states
at transformer-block outputs (layer 0 is the output of block 0). Activations
are copied without norm matching or any other normalization.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


MODEL = "Qwen/Qwen3-8B-Base"
TEMPLATE = "Everyone knows {entity} was a celebrated {role}. The {role} was"
RESULT_ROOT = Path("/ssd/mh3897/patchscope_results")
REFERENCE_ROOT = RESULT_ROOT / "qwen3_8b_base_causal_entity_position_handoff_20260713_181400"
REFERENCE_METADATA = REFERENCE_ROOT / "pair_metadata.csv"
REFERENCE_SUMMARY = REFERENCE_ROOT / "summary_by_layer_and_position.csv"
HOOK_DEFINITION = "cumulative residual state at output of transformer block; layer 0 is output of block 0"
N_LAYERS = 36
N_ORDERED_PAIRS = 16
SMOKE_SOURCE_LAYERS = [5, 28]
SMOKE_RELAY_LAYERS = [5, 10, 20, 23, 24, 25, 26, 28, 32, 35]
SELECTED_SOURCE_LAYERS = [0, 5, 10, 15, 20, 23, 26, 28, 32, 35]
BASELINE_EPS = 1e-12

COMMON_FIELDS = [
    "pair_id", "unordered_pair_id", "shard_id", "donor_entity", "recipient_entity", "role",
    "source_layer_s", "relay_layer_t", "relay_condition", "source_positions", "relay_positions",
    "recipient_margin", "donor_margin", "intervention_margin", "baseline_separation",
    "normalized_recovery", "donor_logit", "recipient_logit",
]


@dataclass(frozen=True)
class Pair:
    pair_id: str
    donor: str
    recipient: str
    role: str

    @property
    def unordered_pair_id(self) -> str:
        return f"{self.role}_{'_'.join(sorted((self.donor, self.recipient)))}"


@dataclass
class Encoded:
    prompt: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    tokens: list[str]
    subject_positions: list[int]
    final_positions: list[int]
    post_subject_positions: list[int]
    entity_token_id: int
    other_token_id: int


@dataclass
class Patch:
    layer: int
    batch_index: int
    positions: list[int]
    values: torch.Tensor


def log(message: str) -> None:
    print(message, flush=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_int_list(value: str | None, default: Iterable[int]) -> list[int]:
    if value is None:
        return list(default)
    return sorted({int(item) for item in value.split(",") if item.strip()})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_reference_pairs() -> tuple[list[Pair], dict[str, dict[str, str]]]:
    rows = read_csv(REFERENCE_METADATA)
    if len(rows) != N_ORDERED_PAIRS:
        raise RuntimeError(f"reference dataset must contain exactly 16 ordered pairs, found {len(rows)}")
    required = {"pair_id", "donor", "recipient", "role", "subject_span_positions", "final_position", "tokens", "token_ids"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"reference metadata lacks required columns: {sorted(required - set(rows[0] if rows else {}))}")
    pair_ids = [row["pair_id"] for row in rows]
    if len(set(pair_ids)) != N_ORDERED_PAIRS:
        raise RuntimeError("reference dataset contains duplicate pair IDs")
    pairs = [Pair(row["pair_id"], row["donor"], row["recipient"], row["role"]) for row in rows]
    unordered: dict[str, int] = {}
    for pair in pairs:
        unordered[pair.unordered_pair_id] = unordered.get(pair.unordered_pair_id, 0) + 1
    if len(unordered) != 8 or any(count != 2 for count in unordered.values()):
        raise RuntimeError(f"expected 8 reciprocal unordered clusters, found {unordered}")
    return pairs, {row["pair_id"]: row for row in rows}


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
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} transformer blocks, found {len(layers)}")
    return model, tokenizer, layers


def char_positions(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [i for i, (left, right) in enumerate(offsets) if (left, right) != (0, 0) and right > start and left < end]


def single_entity_token(tokenizer, entity: str) -> int:
    ids = list(tokenizer(" " + entity, add_special_tokens=False)["input_ids"])
    if len(ids) != 1:
        raise RuntimeError(f"entity {entity!r} is not one leading-space token: {ids}")
    return ids[0]


def encode(tokenizer, pair: Pair, entity: str, other: str) -> Encoded:
    prompt = TEMPLATE.format(entity=entity, role=pair.role)
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    tokens = list(tokenizer.convert_ids_to_tokens(ids))
    start = prompt.index(entity)
    subject = char_positions(offsets, start, start + len(entity))
    final = [len(ids) - 1]
    post_subject = list(range(subject[-1] + 1, len(ids))) if subject else []
    if not subject or not post_subject:
        raise RuntimeError(f"failed to derive subject or post-subject positions for {pair.pair_id}")
    return Encoded(
        prompt=prompt,
        ids=ids,
        offsets=offsets,
        tokens=tokens,
        subject_positions=subject,
        final_positions=final,
        post_subject_positions=post_subject,
        entity_token_id=single_entity_token(tokenizer, entity),
        other_token_id=single_entity_token(tokenizer, other),
    )


def validate_dataset(tokenizer) -> tuple[list[tuple[Pair, Encoded, Encoded]], list[dict[str, Any]]]:
    pairs, reference = load_reference_pairs()
    validated: list[tuple[Pair, Encoded, Encoded]] = []
    metadata: list[dict[str, Any]] = []
    failures: list[str] = []
    for pair in pairs:
        try:
            donor = encode(tokenizer, pair, pair.donor, pair.recipient)
            recipient = encode(tokenizer, pair, pair.recipient, pair.donor)
            ref = reference[pair.pair_id]
            checks = {
                "subject alignment": donor.subject_positions == recipient.subject_positions,
                "final alignment": donor.final_positions == recipient.final_positions,
                "length alignment": len(donor.ids) == len(recipient.ids),
                "P02 entity position": donor.subject_positions == [2],
                "P10 final position": donor.final_positions == [10],
                "final token is was": tokenizer.decode([donor.ids[donor.final_positions[0]]]).strip() == "was",
                "reference subject positions": donor.subject_positions == json.loads(ref["subject_span_positions"]),
                "reference final positions": donor.final_positions == json.loads(ref["final_position"]),
                "reference donor IDs": donor.ids == json.loads(ref["token_ids"]),
                "reference donor tokens": donor.tokens == json.loads(ref["tokens"]),
                "reference donor token ID": donor.entity_token_id == int(ref["donor_token_id"]),
                "reference recipient token ID": donor.other_token_id == int(ref["recipient_token_id"]),
            }
            subject_set = set(donor.subject_positions)
            checks["only entity differs"] = all(
                index in subject_set or left == right
                for index, (left, right) in enumerate(zip(donor.ids, recipient.ids))
            )
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                raise RuntimeError(", ".join(failed))
            metadata.append({
                "pair_id": pair.pair_id,
                "unordered_pair_id": pair.unordered_pair_id,
                "donor_entity": pair.donor,
                "recipient_entity": pair.recipient,
                "role": pair.role,
                "donor_prompt": donor.prompt,
                "recipient_prompt": recipient.prompt,
                "token_length": len(donor.ids),
                "source_positions": json.dumps(donor.subject_positions),
                "relay_p10_positions": json.dumps(donor.final_positions),
                "post_subject_positions": json.dumps(donor.post_subject_positions),
                "donor_token_id": donor.entity_token_id,
                "recipient_token_id": donor.other_token_id,
                "donor_tokens": json.dumps(donor.tokens),
                "recipient_tokens": json.dumps(recipient.tokens),
                "donor_token_ids": json.dumps(donor.ids),
                "recipient_token_ids": json.dumps(recipient.ids),
                "hook_definition": HOOK_DEFINITION,
                "add_special_tokens": True,
            })
            validated.append((pair, donor, recipient))
        except Exception as exc:
            failures.append(f"{pair.pair_id}: {exc}")
    if failures or len(validated) != N_ORDERED_PAIRS:
        raise RuntimeError("all 16 PR #15 pairs must validate; failures=" + " | ".join(failures))
    return validated, metadata


def shard_slice(items: list[Any], num_shards: int, shard_id: int) -> list[Any]:
    base, remainder = divmod(len(items), num_shards)
    start = shard_id * base + min(shard_id, remainder)
    size = base + (1 if shard_id < remainder else 0)
    return items[start:start + size]


def replace_output(out: Any, hidden: torch.Tensor) -> Any:
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


@torch.inference_mode()
def capture_outputs(
    model,
    layers,
    ids: list[int],
    device: torch.device,
    source_layer: int | None = None,
    source_positions: list[int] | None = None,
    source_values: torch.Tensor | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    handles = []
    outputs: dict[int, torch.Tensor] = {}
    if source_layer is not None:
        positions = list(source_positions or [])
        if source_values is None:
            raise ValueError("source_values required with source_layer")

        def patch_hook(_module, _inputs, out):
            hidden0 = out[0] if isinstance(out, tuple) else out
            hidden = hidden0.clone()
            hidden[0, positions, :] = source_values.to(device=hidden.device, dtype=hidden.dtype)
            return replace_output(out, hidden)

        # Register before capture hooks: layer s is captured after P02 replacement.
        handles.append(layers[source_layer].register_forward_hook(patch_hook))

    def make_capture(layer_index: int):
        def hook(_module, _inputs, out):
            hidden = out[0] if isinstance(out, tuple) else out
            outputs[layer_index] = hidden[0, :len(ids), :].detach().clone()
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_capture(index)))
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids).logits[0, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if len(outputs) != len(layers):
        raise RuntimeError(f"captured {len(outputs)} of {len(layers)} layer outputs")
    return outputs, logits


@torch.inference_mode()
def patched_batch_logits(model, layers, ids: list[int], patches: list[Patch], device: torch.device) -> torch.Tensor:
    if not patches:
        raise ValueError("patched batch cannot be empty")
    logits = []
    # BF16 GEMM reductions change slightly with batch shape on H100. Literal
    # batch-size-one runs keep identity controls comparable to clean baselines.
    for patch in patches:
        def hook(_module, _inputs, out, current=patch):
            hidden0 = out[0] if isinstance(out, tuple) else out
            hidden = hidden0.clone()
            hidden[0, current.positions, :] = current.values.to(device=hidden.device, dtype=hidden.dtype)
            return replace_output(out, hidden)

        handle = layers[patch.layer].register_forward_hook(hook)
        try:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits.append(model(input_ids=input_ids).logits[0, -1, :].detach().float().cpu())
        finally:
            handle.remove()
    return torch.stack(logits)


def margin(logits: torch.Tensor, donor_id: int, recipient_id: int) -> float:
    return float((logits[donor_id] - logits[recipient_id]).item())


def result_row(
    pair: Pair,
    shard_id: int,
    source_layer: int | str,
    relay_layer: int | str,
    condition: str,
    source_positions: list[int],
    relay_positions: list[int],
    donor_margin: float,
    recipient_margin: float,
    logits: torch.Tensor,
    donor_id: int,
    recipient_id: int,
) -> dict[str, Any]:
    intervention_margin = margin(logits, donor_id, recipient_id)
    separation = donor_margin - recipient_margin
    if abs(separation) < BASELINE_EPS:
        raise RuntimeError(f"zero baseline denominator for {pair.pair_id}: {separation}")
    return {
        "pair_id": pair.pair_id,
        "unordered_pair_id": pair.unordered_pair_id,
        "shard_id": shard_id,
        "donor_entity": pair.donor,
        "recipient_entity": pair.recipient,
        "role": pair.role,
        "source_layer_s": source_layer,
        "relay_layer_t": relay_layer,
        "relay_condition": condition,
        "source_positions": json.dumps(source_positions),
        "relay_positions": json.dumps(relay_positions),
        "recipient_margin": recipient_margin,
        "donor_margin": donor_margin,
        "intervention_margin": intervention_margin,
        "baseline_separation": separation,
        "normalized_recovery": (intervention_margin - recipient_margin) / separation,
        "donor_logit": float(logits[donor_id].item()),
        "recipient_logit": float(logits[recipient_id].item()),
    }


def checkpoint_pair(
    out_dir: Path,
    pair: Pair,
    source_rows: list[dict[str, Any]],
    relay_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    details: dict[str, Any],
) -> None:
    checkpoint = out_dir / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    write_csv(checkpoint / f"{pair.pair_id}__source.csv", source_rows, COMMON_FIELDS)
    write_csv(checkpoint / f"{pair.pair_id}__relay.csv", relay_rows, COMMON_FIELDS)
    write_csv(checkpoint / f"{pair.pair_id}__direct.csv", direct_rows, COMMON_FIELDS)
    write_json(checkpoint / f"{pair.pair_id}__COMPLETE.json", details)


def run_pair(
    model,
    layers,
    pair: Pair,
    donor: Encoded,
    recipient: Encoded,
    source_layers: list[int],
    relay_layers: list[int],
    device: torch.device,
    shard_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    donor_acts, donor_logits = capture_outputs(model, layers, donor.ids, device)
    recipient_acts, recipient_logits = capture_outputs(model, layers, recipient.ids, device)
    donor_id = donor.entity_token_id
    recipient_id = donor.other_token_id
    donor_margin = margin(donor_logits, donor_id, recipient_id)
    recipient_margin = margin(recipient_logits, donor_id, recipient_id)
    separation = donor_margin - recipient_margin
    if abs(separation) < BASELINE_EPS:
        raise RuntimeError(f"zero baseline denominator for {pair.pair_id}")

    source_rows: list[dict[str, Any]] = []
    relay_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    for condition, logits in (("clean_donor", donor_logits), ("no_patch_recipient", recipient_logits)):
        direct_rows.append(result_row(
            pair, shard_id, "", "", condition, recipient.subject_positions, [], donor_margin,
            recipient_margin, logits, donor_id, recipient_id,
        ))

    control_patches: list[Patch] = []
    control_specs: list[tuple[str, int, list[int]]] = []
    for relay_layer in relay_layers:
        for condition, values in (
            ("direct_clean_donor_p10", donor_acts[relay_layer][donor.final_positions, :]),
            ("identity_recipient_p10", recipient_acts[relay_layer][recipient.final_positions, :]),
        ):
            batch_index = len(control_patches)
            control_patches.append(Patch(relay_layer, batch_index, recipient.final_positions, values))
            control_specs.append((condition, relay_layer, recipient.final_positions))
    control_logits = patched_batch_logits(model, layers, recipient.ids, control_patches, device)
    for index, (condition, relay_layer, positions) in enumerate(control_specs):
        direct_rows.append(result_row(
            pair, shard_id, "", relay_layer, condition, recipient.subject_positions, positions,
            donor_margin, recipient_margin, control_logits[index], donor_id, recipient_id,
        ))

    for source_layer in source_layers:
        hybrid_acts, hybrid_logits = capture_outputs(
            model,
            layers,
            recipient.ids,
            device,
            source_layer=source_layer,
            source_positions=recipient.subject_positions,
            source_values=donor_acts[source_layer][donor.subject_positions, :],
        )
        source_rows.append(result_row(
            pair, shard_id, source_layer, "", "source_patch_end_to_end", recipient.subject_positions,
            [], donor_margin, recipient_margin, hybrid_logits, donor_id, recipient_id,
        ))

        relay_patches: list[Patch] = []
        relay_specs: list[tuple[str, int, list[int]]] = []
        for relay_layer in relay_layers:
            if relay_layer < source_layer:
                continue
            for condition, positions in (
                ("p10_relay", recipient.final_positions),
                ("post_subject_relay", recipient.post_subject_positions),
            ):
                batch_index = len(relay_patches)
                relay_patches.append(Patch(
                    relay_layer,
                    batch_index,
                    positions,
                    hybrid_acts[relay_layer][positions, :],
                ))
                relay_specs.append((condition, relay_layer, positions))
        if relay_patches:
            relayed_logits = patched_batch_logits(model, layers, recipient.ids, relay_patches, device)
            for index, (condition, relay_layer, positions) in enumerate(relay_specs):
                relay_rows.append(result_row(
                    pair, shard_id, source_layer, relay_layer, condition, recipient.subject_positions,
                    positions, donor_margin, recipient_margin, relayed_logits[index], donor_id, recipient_id,
                ))
        del hybrid_acts

    details = {
        "pair_id": pair.pair_id,
        "completed_at": now(),
        "donor_margin": donor_margin,
        "recipient_margin": recipient_margin,
        "baseline_separation": separation,
        "source_positions": recipient.subject_positions,
        "p10_positions": recipient.final_positions,
        "post_subject_positions": recipient.post_subject_positions,
        "num_source_rows": len(source_rows),
        "num_relay_rows": len(relay_rows),
        "num_direct_rows": len(direct_rows),
    }
    return source_rows, relay_rows, direct_rows, details


def validate_smoke(
    source_rows: list[dict[str, Any]], relay_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = source_rows + relay_rows + direct_rows
    invalid = [row for row in all_rows if not math.isfinite(number(row["normalized_recovery"])) or abs(number(row["baseline_separation"])) < BASELINE_EPS]
    identity = [abs(number(row["normalized_recovery"])) for row in direct_rows if row["relay_condition"] == "identity_recipient_p10"]
    direct_final = [number(row["normalized_recovery"]) for row in direct_rows if row["relay_condition"] == "direct_clean_donor_p10" and int(row["relay_layer_t"]) == 35]
    source = {int(row["source_layer_s"]): number(row["normalized_recovery"]) for row in source_rows}
    checks = {
        "finite_recoveries_and_nonzero_denominators": not invalid,
        "identity_recipient_p10_max_abs_le_1e-5": bool(identity) and max(identity) <= 1e-5,
        "direct_clean_donor_p10_layer35_ge_0.95": bool(direct_final) and min(direct_final) >= 0.95,
        "early_p02_exceeds_late_by_0.2": 5 in source and 28 in source and source[5] - source[28] >= 0.2,
        "all_required_relay_conditions_present": {row["relay_condition"] for row in relay_rows} == {"p10_relay", "post_subject_relay"},
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "identity_recipient_p10_max_abs_recovery": max(identity) if identity else None,
        "direct_clean_donor_p10_layer35_recovery": direct_final[0] if direct_final else None,
        "source_patch_end_to_end_layer5_recovery": source.get(5),
        "source_patch_end_to_end_layer28_recovery": source.get(28),
        "num_rows": len(all_rows),
    }
    if not report["passed"]:
        raise RuntimeError(f"smoke validation failed: {json.dumps(report, sort_keys=True)}")
    return report


def write_shard_readme(out_dir: Path, mode: str, num_pairs: int, source_layers: list[int], relay_layers: list[int]) -> None:
    (out_dir / "README.md").write_text(f"""# P02-to-P10 causal relay ({mode})

Model: `{MODEL}`

Prompt: `{TEMPLATE}`

Dataset authority: `{REFERENCE_METADATA}`; all 16 PR #15 ordered pairs were validated before sharding.

P02 and P10 are labels confirmed from tokenization, not hardcoded intervention indices. P02 is the entity-name span and P10 is the final `was`, the next-token readout position.

Layer hook: {HOOK_DEFINITION}.

Source layers: `{source_layers}`. Relay layers: `{relay_layers}`. Pairs in this output: {num_pairs}.

No activation norm matching or normalization was applied. Normalized recovery values are not clipped.
""")


def run_experiment(args: argparse.Namespace, out_dir: Path) -> int:
    device = torch.device(args.device)
    model, tokenizer, layers = load_model(args.model, device)
    valid, metadata = validate_dataset(tokenizer)
    log(f"[{now()}] loaded {args.model} on {device}; validated all {len(valid)} PR #15 ordered pairs")
    source_layers = SMOKE_SOURCE_LAYERS if args.smoke else parse_int_list(args.source_layers, range(N_LAYERS))
    relay_layers = SMOKE_RELAY_LAYERS if args.smoke else parse_int_list(args.relay_layers, range(N_LAYERS))
    if any(layer < 0 or layer >= N_LAYERS for layer in source_layers + relay_layers):
        raise ValueError("source and relay layers must be between 0 and 35")
    run_items = [valid[0]] if args.smoke else shard_slice(valid, args.num_shards, args.shard_id)
    selected = {item[0].pair_id for item in run_items}
    write_csv(out_dir / "pair_metadata.csv", [row for row in metadata if row["pair_id"] in selected])
    write_csv(out_dir / "skipped_pairs.csv", [], ["pair_id", "reason"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_shard_readme(out_dir, "smoke" if args.smoke else f"shard {args.shard_id}/{args.num_shards}", len(run_items), source_layers, relay_layers)

    all_source: list[dict[str, Any]] = []
    all_relay: list[dict[str, Any]] = []
    all_direct: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for index, (pair, donor, recipient) in enumerate(run_items, start=1):
        log(f"[{now()}] pair {index}/{len(run_items)} {pair.pair_id} starting")
        source_rows, relay_rows, direct_rows, details = run_pair(
            model, layers, pair, donor, recipient, source_layers, relay_layers, device, args.shard_id,
        )
        checkpoint_pair(out_dir, pair, source_rows, relay_rows, direct_rows, details)
        all_source.extend(source_rows)
        all_relay.extend(relay_rows)
        all_direct.extend(direct_rows)
        controls.append(details)
        # These cumulative files make a running shard usable after every pair.
        write_csv(out_dir / "source_patch_end_to_end.csv", all_source, COMMON_FIELDS)
        write_csv(out_dir / "relay_results.csv", all_relay, COMMON_FIELDS)
        write_csv(out_dir / "direct_p10_control.csv", all_direct, COMMON_FIELDS)
        log(f"[{now()}] pair {pair.pair_id} complete; checkpoint written")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    validation: dict[str, Any] = {
        "mode": "smoke" if args.smoke else "shard",
        "model": args.model,
        "num_ordered_pairs_validated_before_sharding": len(valid),
        "num_pairs_run": len(run_items),
        "num_skipped_pairs": 0,
        "source_layers": source_layers,
        "relay_layers": relay_layers,
        "pair_controls": controls,
    }
    if args.smoke:
        validation["smoke"] = validate_smoke(all_source, all_relay, all_direct)
        log("SMOKE SUCCESS " + json.dumps(validation["smoke"], sort_keys=True))
    write_json(out_dir / "validation_summary.json", validation)
    (out_dir / "SUCCESS").write_text(now() + "\n")
    return 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def cluster_bootstrap_summary(rows: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    rng = random.Random(20260714)
    buckets: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["relay_condition"]), int(row["source_layer_s"]), int(row["relay_layer_t"]))
        buckets.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (condition, source_layer, relay_layer), cell in sorted(buckets.items()):
        cluster_values: dict[str, list[float]] = {}
        values = []
        for row in cell:
            value = number(row["normalized_recovery"])
            if not math.isfinite(value):
                continue
            values.append(value)
            cluster_values.setdefault(str(row["unordered_pair_id"]), []).append(value)
        clusters = sorted(cluster_values)
        if len(values) != N_ORDERED_PAIRS or len(clusters) != 8 or any(len(cluster_values[key]) != 2 for key in clusters):
            raise RuntimeError(f"invalid reciprocal cluster coverage at {(condition, source_layer, relay_layer)}")
        boot = []
        for _ in range(n_boot):
            sampled = [rng.choice(clusters) for _ in clusters]
            sample_values = [value for cluster in sampled for value in cluster_values[cluster]]
            boot.append(statistics.fmean(sample_values))
        summary.append({
            "relay_condition": condition,
            "source_layer_s": source_layer,
            "relay_layer_t": relay_layer,
            "mean_normalized_recovery": statistics.fmean(values),
            "std_normalized_recovery": statistics.stdev(values),
            "ci95_low_cluster_bootstrap": percentile(boot, 0.025),
            "ci95_high_cluster_bootstrap": percentile(boot, 0.975),
            "num_ordered_pairs": len(values),
            "num_unordered_clusters": len(clusters),
            "bootstrap_resamples": n_boot,
            "interval_note": "descriptive; reciprocal directions are dependent and resampled together",
        })
    return summary


def grouped_mean(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], float]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        value = number(row["normalized_recovery"])
        if math.isfinite(value):
            buckets.setdefault(key, []).append(value)
    return {key: statistics.fmean(values) for key, values in buckets.items()}


def save_figure(fig, plots: Path, name: str) -> None:
    fig.savefig(plots / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(plots / f"{name}.pdf", bbox_inches="tight")


def make_plots(relay_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    means = grouped_mean(relay_rows, ["relay_condition", "source_layer_s", "relay_layer_t"])

    def matrix(condition: str) -> np.ndarray:
        result = np.full((N_LAYERS, N_LAYERS), np.nan)
        for source_layer in range(N_LAYERS):
            for relay_layer in range(source_layer, N_LAYERS):
                result[source_layer, relay_layer] = means.get((condition, str(source_layer), str(relay_layer)), np.nan)
        return result

    p10 = matrix("p10_relay")
    post = matrix("post_subject_relay")
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#dddddd")

    def heatmap(data, name: str, title: str, label: str, vmin: float, vmax: float) -> None:
        fig, ax = plt.subplots(figsize=(9.5, 8))
        image = ax.imshow(data, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('P10 relay layer t (final "was" / next-token readout position)')
        ax.set_ylabel("P02 source injection layer s (entity-name position)")
        ax.set_title(title + "\n0 = no donor recovery; 1 = full donor recovery")
        ax.set_xticks(range(0, N_LAYERS, 5)); ax.set_yticks(range(0, N_LAYERS, 5))
        fig.colorbar(image, ax=ax, label=label, shrink=0.82)
        fig.tight_layout()
        save_figure(fig, plots, name)
        plt.close(fig)

    heatmap(p10, "p10_relay_matrix", "Mean normalized donor recovery: P10-only causal relay", "mean normalized donor recovery", 0, 1)
    heatmap(post, "post_subject_relay_matrix", "Mean normalized donor recovery: all positions after P02", "mean normalized donor recovery", 0, 1)
    heatmap(p10 - post, "p10_vs_post_subject_difference", "P10-only minus post-subject relay recovery", "difference in mean normalized recovery", -1, 1)

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, sharey=True)
    for axis, condition, title in zip(axes, ["p10_relay", "post_subject_relay"], ["P10-only relay", "All post-subject positions relay"]):
        for color_index, source_layer in enumerate(SELECTED_SOURCE_LAYERS):
            xs = list(range(source_layer, N_LAYERS))
            ys = [means.get((condition, str(source_layer), str(relay_layer)), float("nan")) for relay_layer in xs]
            axis.plot(xs, ys, marker="o", markersize=2.5, color=colors(color_index), label=f"s={source_layer}")
        axis.axhline(0, color="black", linewidth=0.7); axis.axhline(1, color="black", linewidth=0.7, linestyle="--")
        axis.set_ylabel("Mean normalized donor recovery")
        axis.set_title(title + ": P02 entity-name injection to P10 final `was` readout")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel('Relay/readout layer t at P10 (final "was" / next-token readout position)')
    axes[0].legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("0 = no donor recovery; 1 = full donor recovery", y=1.02)
    fig.tight_layout()
    save_figure(fig, plots, "relay_curves_selected_source_layers")
    plt.close(fig)

    source_means = grouped_mean(source_rows, ["source_layer_s"])
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = list(range(N_LAYERS))
    ax.plot(xs, [source_means.get((str(layer),), float("nan")) for layer in xs], marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.7); ax.axhline(1, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("P02 source injection layer s (entity-name position)")
    ax.set_ylabel("Mean normalized donor recovery at final logits")
    ax.set_title('Source patch end-to-end: P02 entity name to P10 final "was" readout\n0 = no donor recovery; 1 = full donor recovery')
    ax.grid(alpha=0.2)
    fig.tight_layout(); save_figure(fig, plots, "source_patch_end_to_end_curve"); plt.close(fig)

    direct_means = grouped_mean(
        [row for row in direct_rows if row["relay_condition"] == "direct_clean_donor_p10"], ["relay_layer_t"]
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, [direct_means.get((str(layer),), float("nan")) for layer in xs], color="black", linewidth=2.2, label="Direct clean-donor P10 upper bound")
    for color_index, source_layer in enumerate(SELECTED_SOURCE_LAYERS):
        relay_x = list(range(source_layer, N_LAYERS))
        relay_y = [means.get(("p10_relay", str(source_layer), str(layer)), float("nan")) for layer in relay_x]
        ax.plot(relay_x, relay_y, color=colors(color_index), alpha=0.8, label=f"Natural P02→P10 relay, s={source_layer}")
    ax.axhline(0, color="black", linewidth=0.7); ax.axhline(1, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel('P10 intervention layer t (final "was" / next-token readout position)')
    ax.set_ylabel("Mean normalized donor recovery")
    ax.set_title("Direct clean-donor P10 control versus naturally relayed hybrid P10\n0 = no donor recovery; 1 = full donor recovery")
    ax.grid(alpha=0.2); ax.legend(fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); save_figure(fig, plots, "direct_p10_vs_relay"); plt.close(fig)

    for pair_id in sorted({str(row["pair_id"]) for row in relay_rows}):
        pair_rows = [row for row in relay_rows if row["pair_id"] == pair_id and row["relay_condition"] == "p10_relay"]
        pair_mean = grouped_mean(pair_rows, ["source_layer_s", "relay_layer_t"])
        data = np.full((N_LAYERS, N_LAYERS), np.nan)
        for source_layer in range(N_LAYERS):
            for relay_layer in range(source_layer, N_LAYERS):
                data[source_layer, relay_layer] = pair_mean.get((str(source_layer), str(relay_layer)), np.nan)
        heatmap(data, f"p10_relay_pair__{pair_id}", f"P10 relay for {pair_id}", "normalized donor recovery", 0, 1)


def reference_subject_means() -> dict[int, float]:
    rows = read_csv(REFERENCE_SUMMARY)
    return {int(row["layer"]): number(row["mean"]) for row in rows if row["position_group"] == "subject_span"}


def merged_validation(
    relay_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]],
    pair_ids: list[str], num_shards: int,
) -> dict[str, Any]:
    identity = [abs(number(row["normalized_recovery"])) for row in direct_rows if row["relay_condition"] == "identity_recipient_p10"]
    direct_final = [number(row["normalized_recovery"]) for row in direct_rows if row["relay_condition"] == "direct_clean_donor_p10" and int(row["relay_layer_t"]) == 35]
    source_mean = grouped_mean(source_rows, ["source_layer_s"])
    reference = reference_subject_means()
    comparison = []
    for layer in range(N_LAYERS):
        observed = source_mean[(str(layer),)]
        comparison.append({"layer": layer, "observed": observed, "reference": reference[layer], "delta": observed - reference[layer]})
    expected_relay = N_ORDERED_PAIRS * 2 * sum(N_LAYERS - source for source in range(N_LAYERS))
    finite = all(math.isfinite(number(row["normalized_recovery"])) for row in relay_rows + source_rows + direct_rows)
    return {
        "status": "SUCCESS",
        "model": MODEL,
        "merged_shards": num_shards,
        "num_ordered_pairs": len(pair_ids),
        "num_unordered_clusters": len({row["unordered_pair_id"] for row in relay_rows}),
        "pair_ids": pair_ids,
        "num_skipped_pairs": 0,
        "relay_rows": len(relay_rows),
        "expected_relay_rows": expected_relay,
        "source_rows": len(source_rows),
        "direct_control_rows": len(direct_rows),
        "all_normalized_recoveries_finite": finite,
        "identity_recipient_p10_max_abs_recovery": max(identity),
        "direct_clean_donor_p10_layer35_mean_recovery": statistics.fmean(direct_final),
        "direct_clean_donor_p10_layer35_min_recovery": min(direct_final),
        "source_patch_end_to_end_reference_comparison": comparison,
        "source_patch_reference_max_abs_delta": max(abs(row["delta"]) for row in comparison),
        "interval_note": "Cluster-bootstrap intervals are descriptive because reciprocal ordered directions are dependent; unordered entity pairs are resampled with both directions together.",
        "checks": {
            "exactly_16_ordered_pairs": len(pair_ids) == N_ORDERED_PAIRS and len(set(pair_ids)) == N_ORDERED_PAIRS,
            "exact_relay_row_count": len(relay_rows) == expected_relay,
            "finite_values": finite,
            "identity_max_abs_le_1e-5": max(identity) <= 1e-5,
            "direct_final_min_ge_0.95": min(direct_final) >= 0.95,
        },
    }


def write_merged_readme(out_dir: Path, validation: dict[str, Any], n_boot: int) -> None:
    (out_dir / "README.md").write_text(f"""# Qwen3-8B-Base P02-to-P10 causal relay

This experiment asks when information introduced at the entity-name position becomes causally usable at the final next-token readout position.

Model: `{MODEL}`. Prompt: `{TEMPLATE}`. Prompts use the model tokenizer with `add_special_tokens=True`. All {validation['num_ordered_pairs']} ordered pairs from PR #15 validated exactly against `{REFERENCE_METADATA}`; skipped pairs: 0.

P02 and P10 are descriptive labels verified from tokenization for every pair. P02 is the entity-name span. P10 is the final `was`, whose residual state supplies the next-token logits. Source and relay position lists are stored in every row.

At source block output `s`, donor P02 replaces recipient P02. The resulting hybrid run supplies P10 or all strictly post-subject states at block output `t`; those states are relayed into a fresh recipient sequence. `t=s` is included and `t<s` is undefined and was never run. The direct clean-donor P10 condition is an upper-bound control, not the primary relay result.

Metric: `m(x) = logit(donor entity) - logit(recipient entity)`. Recovery: `(m(intervention) - m(recipient)) / (m(donor) - m(recipient))`. Values are not clipped. Plot annotations use 0 for no donor recovery and 1 for full donor recovery.

Layer hook: {HOOK_DEFINITION}. No donor activation normalization was applied. The experiment calls no text-generation API and records only next-token logits.

`relay_summary.csv` contains means and descriptive 95% cluster-bootstrap intervals from {n_boot} resamples. The {validation['num_unordered_clusters']} unordered entity pairs are clusters and both reciprocal directions are retained together when a cluster is sampled. These intervals are descriptive because reciprocal directions are dependent.

Files include raw relay, source end-to-end, direct/identity/baseline control rows, merged metadata, validation, per-pair checkpoints within shards, and PNG/PDF plots under `plots/`.
""")


def merge_shards(args: argparse.Namespace, out_dir: Path) -> int:
    relay_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    merged_checkpoints = out_dir / "checkpoints"
    merged_checkpoints.mkdir(exist_ok=True)
    for shard_id in range(args.num_shards):
        shard = out_dir / f"shard_{shard_id}"
        required = ["SUCCESS", "relay_results.csv", "source_patch_end_to_end.csv", "direct_p10_control.csv", "pair_metadata.csv"]
        missing = [name for name in required if not (shard / name).exists()]
        if missing:
            raise RuntimeError(f"shard {shard_id} is incomplete; missing {missing}")
        relay_rows.extend(read_csv(shard / "relay_results.csv"))
        source_rows.extend(read_csv(shard / "source_patch_end_to_end.csv"))
        direct_rows.extend(read_csv(shard / "direct_p10_control.csv"))
        metadata.extend(read_csv(shard / "pair_metadata.csv"))
        for checkpoint_file in (shard / "checkpoints").glob("*"):
            shutil.copy2(checkpoint_file, merged_checkpoints / checkpoint_file.name)
    pair_ids = [row["pair_id"] for row in metadata]
    if len(pair_ids) != N_ORDERED_PAIRS or len(set(pair_ids)) != N_ORDERED_PAIRS:
        raise RuntimeError(f"merged pair coverage invalid: {pair_ids}")
    write_csv(out_dir / "relay_results.csv", relay_rows, COMMON_FIELDS)
    write_csv(out_dir / "source_patch_end_to_end.csv", source_rows, COMMON_FIELDS)
    write_csv(out_dir / "direct_p10_control.csv", direct_rows, COMMON_FIELDS)
    write_csv(out_dir / "pair_metadata.csv", metadata)
    write_csv(out_dir / "skipped_pairs.csv", [], ["pair_id", "reason"])
    summary = cluster_bootstrap_summary(relay_rows, args.bootstrap_resamples)
    write_csv(out_dir / "relay_summary.csv", summary)
    validation = merged_validation(relay_rows, source_rows, direct_rows, pair_ids, args.num_shards)
    if not all(validation["checks"].values()):
        raise RuntimeError(f"merged validation failed: {json.dumps(validation['checks'])}")
    make_plots(relay_rows, source_rows, direct_rows, out_dir)
    write_json(out_dir / "validation_summary.json", validation)
    write_merged_readme(out_dir, validation, args.bootstrap_resamples)
    log(f"[{now()}] merged {args.num_shards} shards; all validation checks passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir")
    parser.add_argument("--source-layers")
    parser.add_argument("--relay-layers")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard configuration")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        suffix = "smoke" if args.smoke else timestamp()
        out_dir = RESULT_ROOT / f"qwen3_8b_base_p02_p10_relay_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.merge_shards:
        return merge_shards(args, out_dir)
    return run_experiment(args, out_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"[{now()}] FAILED: {type(exc).__name__}: {exc}")
        raise
