"""Three component-level causal-relay experiments for Qwen3-8B-Base.

The dataset authority is the completed P02-to-P10 relay.  This file implements
only: (1) attention-output mediation, (2) matched MLP intervention, and
(3) the predefined layer-23--26 head/source-position path scan.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MODEL = "Qwen/Qwen3-8B-Base"
N_LAYERS = 36
SCAN_LAYERS = [23, 24, 25, 26]
P02 = 2
P10 = 10
N_PAIRS = 16
REFERENCE_RELAY = Path("/ssd/mh3897/patchscope_results/qwen3_8b_base_p02_p10_relay_20260714_030311")
REFERENCE_METADATA = REFERENCE_RELAY / "pair_metadata.csv"
REFERENCE_SOURCE = REFERENCE_RELAY / "source_patch_end_to_end.csv"
REFERENCE_DIRECT = REFERENCE_RELAY / "direct_p10_control.csv"
EPS = 1e-12

FIELDS = [
    "experiment", "pair_id", "unordered_pair_id", "shard_id", "donor_entity", "recipient_entity",
    "role", "layer", "position", "position_label", "component", "head", "kv_head",
    "source_position", "source_position_label", "source_group", "intervention",
    "donor_token_id", "recipient_token_id", "donor_token_label", "recipient_token_label",
    "recipient_margin", "donor_margin", "normalization_denominator", "intervention_margin",
    "normalized_recovery", "donor_logit", "recipient_logit", "recipient_vector_norm",
    "hybrid_vector_norm", "delta_vector_norm", "cosine_recipient_hybrid", "attention_weight_recipient",
    "attention_weight_hybrid", "state_error_max_abs", "notes",
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
    donor_tokens: list[str]
    recipient_tokens: list[str]


@dataclass
class Capture:
    block: dict[int, torch.Tensor]
    r: dict[int, torch.Tensor]
    mlp: dict[int, torch.Tensor]
    attention: dict[int, torch.Tensor]
    heads: dict[int, torch.Tensor]
    weights: dict[int, torch.Tensor]
    values: dict[int, torch.Tensor]
    logits: torch.Tensor


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def replace_output(out: Any, tensor: torch.Tensor) -> Any:
    return (tensor,) + out[1:] if isinstance(out, tuple) else tensor


def scalar(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def norm(x: torch.Tensor) -> float:
    return scalar(torch.linalg.vector_norm(x.float()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return scalar(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1))[0])


def source_label(pair: Pair, position: int) -> str:
    token = pair.recipient_tokens[position]
    return f"P{position:02d}:{token}"


def source_group(position: int) -> str:
    if position == P02:
        return "P02_subject_entity"
    if P02 < position < P10:
        return "P03-P09_intermediate"
    if position == P10:
        return "P10_self"
    return "P00-P01_prefix"


def load_pairs() -> list[Pair]:
    rows = read_csv(REFERENCE_METADATA)
    if len(rows) != N_PAIRS:
        raise RuntimeError(f"expected exactly 16 ordered directions, found {len(rows)}")
    pairs = []
    for row in rows:
        pairs.append(Pair(
            pair_id=row["pair_id"], cluster=row["unordered_pair_id"], donor=row["donor_entity"],
            recipient=row["recipient_entity"], role=row["role"],
            donor_ids=json.loads(row["donor_token_ids"]), recipient_ids=json.loads(row["recipient_token_ids"]),
            donor_token_id=int(row["donor_token_id"]), recipient_token_id=int(row["recipient_token_id"]),
            donor_tokens=json.loads(row["donor_tokens"]), recipient_tokens=json.loads(row["recipient_tokens"]),
        ))
    clusters: dict[str, int] = {}
    for pair in pairs:
        clusters[pair.cluster] = clusters.get(pair.cluster, 0) + 1
        if len(pair.donor_ids) != 11 or len(pair.recipient_ids) != 11:
            raise RuntimeError(f"{pair.pair_id}: token length is not 11")
        diffs = [i for i, (a, b) in enumerate(zip(pair.donor_ids, pair.recipient_ids)) if a != b]
        if diffs != [P02] or pair.donor_ids[P02] != pair.donor_token_id or pair.recipient_ids[P02] != pair.recipient_token_id:
            raise RuntimeError(f"{pair.pair_id}: P02 IDs do not match metadata")
        if pair.donor_tokens[P10].replace("Ġ", "").strip() != "was":
            raise RuntimeError(f"{pair.pair_id}: P10 is not final 'was': {pair.donor_tokens[P10]}")
    if len(clusters) != 8 or any(count != 2 for count in clusters.values()):
        raise RuntimeError(f"reciprocal cluster coverage invalid: {clusters}")
    return pairs


def load_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # Eager attention exposes the exact fixed softmax weights required to
    # decompose each head into value-weighted source-position messages.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    layers = list(model.model.layers)
    config = model.config
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected 36 layers, got {len(layers)}")
    if config.num_attention_heads % config.num_key_value_heads:
        raise RuntimeError("query-head count is not divisible by KV-head count")
    return model, tokenizer, layers, config


def validate_tokenizer(tokenizer, pairs: list[Pair]) -> list[dict[str, Any]]:
    validated = []
    for pair in pairs:
        donor_prompt = f"Everyone knows {pair.donor} was a celebrated {pair.role}. The {pair.role} was"
        recipient_prompt = f"Everyone knows {pair.recipient} was a celebrated {pair.role}. The {pair.role} was"
        d = tokenizer(donor_prompt, add_special_tokens=True)["input_ids"]
        r = tokenizer(recipient_prompt, add_special_tokens=True)["input_ids"]
        if list(d) != pair.donor_ids or list(r) != pair.recipient_ids:
            raise RuntimeError(f"{pair.pair_id}: tokenizer IDs differ from completed relay metadata")
        validated.append({
            "pair_id": pair.pair_id, "unordered_pair_id": pair.cluster, "donor_entity": pair.donor,
            "recipient_entity": pair.recipient, "role": pair.role, "P02": P02, "P10": P10,
            "donor_P02_token_id": pair.donor_token_id, "recipient_P02_token_id": pair.recipient_token_id,
            "P02_donor_label": pair.donor_tokens[P02], "P02_recipient_label": pair.recipient_tokens[P02],
            "P10_token_id": pair.donor_ids[P10], "P10_label": pair.donor_tokens[P10],
            "donor_token_ids": json.dumps(pair.donor_ids), "recipient_token_ids": json.dumps(pair.recipient_ids),
            "donor_tokens": json.dumps(pair.donor_tokens), "recipient_tokens": json.dumps(pair.recipient_tokens),
        })
    return validated


@torch.inference_mode()
def capture_run(model, layers, ids: list[int], device: torch.device, source_p02: torch.Tensor | None = None) -> Capture:
    block: dict[int, torch.Tensor] = {}
    r: dict[int, torch.Tensor] = {}
    mlp: dict[int, torch.Tensor] = {}
    attention: dict[int, torch.Tensor] = {}
    heads: dict[int, torch.Tensor] = {}
    weights: dict[int, torch.Tensor] = {}
    values: dict[int, torch.Tensor] = {}
    handles = []

    if source_p02 is not None:
        def source_hook(_module, _inputs, out):
            x0 = out[0] if isinstance(out, tuple) else out
            x = x0.clone()
            x[0, P02, :] = source_p02.to(x)
            return replace_output(out, x)
        handles.append(layers[0].register_forward_hook(source_hook))

    for layer_index, layer in enumerate(layers):
        def block_hook(_module, _inputs, out, li=layer_index):
            x = out[0] if isinstance(out, tuple) else out
            block[li] = x[0].detach().clone()
        handles.append(layer.register_forward_hook(block_hook))

        def r_pre_hook(_module, inputs, li=layer_index):
            r[li] = inputs[0][0].detach().clone()
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(r_pre_hook))

        def mlp_hook(_module, _inputs, out, li=layer_index):
            mlp[li] = out[0].detach().clone()
        handles.append(layer.mlp.register_forward_hook(mlp_hook))

        def attn_hook(_module, _inputs, out, li=layer_index):
            attention[li] = out[0][0].detach().clone()
            if li in SCAN_LAYERS:
                weights[li] = out[1][0].detach().clone()
        handles.append(layer.self_attn.register_forward_hook(attn_hook))

        if layer_index in SCAN_LAYERS:
            def o_pre_hook(_module, inputs, li=layer_index):
                heads[li] = inputs[0][0].detach().clone()
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(o_pre_hook))

            def v_hook(_module, _inputs, out, li=layer_index):
                values[li] = out[0].detach().clone()
            handles.append(layer.self_attn.v_proj.register_forward_hook(v_hook))

    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if not (len(block) == len(r) == len(mlp) == len(attention) == N_LAYERS):
        raise RuntimeError(f"incomplete capture: block={len(block)} r={len(r)} mlp={len(mlp)} attn={len(attention)}")
    return Capture(block, r, mlp, attention, heads, weights, values, logits)


def baseline(pair: Pair, donor: Capture, recipient: Capture) -> tuple[float, float, float]:
    dm = float(donor.logits[pair.donor_token_id] - donor.logits[pair.recipient_token_id])
    rm = float(recipient.logits[pair.donor_token_id] - recipient.logits[pair.recipient_token_id])
    denom = dm - rm
    if not math.isfinite(denom) or abs(denom) < EPS:
        raise RuntimeError(f"{pair.pair_id}: invalid normalization denominator {denom}")
    return dm, rm, denom


def make_row(pair: Pair, shard_id: int, dm: float, rm: float, logits: torch.Tensor, **values: Any) -> dict[str, Any]:
    dl = float(logits[pair.donor_token_id])
    rl = float(logits[pair.recipient_token_id])
    margin = dl - rl
    row = {field: "" for field in FIELDS}
    row.update({
        "pair_id": pair.pair_id, "unordered_pair_id": pair.cluster, "shard_id": shard_id,
        "donor_entity": pair.donor, "recipient_entity": pair.recipient, "role": pair.role,
        "donor_token_id": pair.donor_token_id, "recipient_token_id": pair.recipient_token_id,
        "donor_token_label": pair.donor_tokens[P02], "recipient_token_label": pair.recipient_tokens[P02],
        "recipient_margin": rm, "donor_margin": dm, "normalization_denominator": dm - rm,
        "intervention_margin": margin, "normalized_recovery": (margin - rm) / (dm - rm),
        "donor_logit": dl, "recipient_logit": rl,
    })
    row.update(values)
    return row


@torch.inference_mode()
def attention_interventions(model, layers, pair: Pair, donor0: torch.Tensor, recipient: Capture, hybrid: Capture,
                            layer_index: int, device: torch.device, dm: float, rm: float, shard_id: int) -> list[dict[str, Any]]:
    ids = torch.tensor([pair.recipient_ids, pair.recipient_ids], dtype=torch.long, device=device)
    target_rec = recipient.attention[layer_index][P10]
    target_hyb = hybrid.attention[layer_index][P10]
    handles = []

    def source_hook(_module, _inputs, out):
        x0 = out[0] if isinstance(out, tuple) else out
        x = x0.clone(); x[1, P02] = donor0.to(x)
        return replace_output(out, x)

    def attn_hook(_module, _inputs, out):
        x = out[0].clone()
        x[0, P10] = target_hyb.to(x)
        x[1, P10] = target_rec.to(x)
        return (x,) + out[1:]

    handles.append(layers[0].register_forward_hook(source_hook))
    handles.append(layers[layer_index].self_attn.register_forward_hook(attn_hook))
    try:
        logits = model(input_ids=ids, use_cache=False).logits[:, -1].detach().float().cpu()
    finally:
        for h in handles: h.remove()
    common = {
        "experiment": "1_attention_output_mediation", "layer": layer_index, "position": P10,
        "position_label": f"P10:{pair.recipient_tokens[P10]}", "component": "attention_output_post_o_proj_pre_residual",
        "recipient_vector_norm": norm(target_rec), "hybrid_vector_norm": norm(target_hyb),
        "delta_vector_norm": norm(target_hyb - target_rec), "cosine_recipient_hybrid": cosine(target_rec, target_hyb),
    }
    return [
        make_row(pair, shard_id, dm, rm, logits[0], intervention="attention_sufficiency", **common),
        make_row(pair, shard_id, dm, rm, logits[1], intervention="attention_necessity", **common),
    ]


@torch.inference_mode()
def mlp_interventions(model, layers, pair: Pair, hybrid: Capture, layer_index: int, position: int,
                      device: torch.device, dm: float, rm: float, shard_id: int) -> list[dict[str, Any]]:
    ids = torch.tensor([pair.recipient_ids], dtype=torch.long, device=device)
    target_r = hybrid.r[layer_index][position]
    target_mlp = hybrid.mlp[layer_index][position]
    # Use the definition itself.  At layer 0/P02 the later source hook replaces
    # block output with donor P02, so the captured hybrid block tensor is not
    # the pre-source r + MLP(norm(r)) state tested by this experiment.
    target_full = target_r + target_mlp
    logits_list = []
    states: dict[str, torch.Tensor] = {}
    # Literal batch-size-one runs avoid BF16 GEMM kernel changes.  Active and
    # full-post-MLP are then numerically identical to the singleton hybrid run.
    for kind in ("mlp_active", "mlp_bypass", "full_post_mlp"):
        handles = []
        if kind == "mlp_active":
            def r_pre_hook(_module, inputs):
                # Qwen3DecoderLayer's local residual aliases this tensor.
                inputs[0][0, position].copy_(target_r.to(inputs[0]))
            handles.append(layers[layer_index].post_attention_layernorm.register_forward_pre_hook(r_pre_hook))

        def block_hook(_module, _inputs, out, current=kind):
            x0 = out[0] if isinstance(out, tuple) else out
            x = x0.clone()
            if current == "mlp_bypass": x[0, position] = target_r.to(x)
            elif current == "full_post_mlp": x[0, position] = target_full.to(x)
            states[current] = x[0, position].detach().clone()
            return replace_output(out, x)
        handles.append(layers[layer_index].register_forward_hook(block_hook))
        try:
            logits_list.append(model(input_ids=ids, use_cache=False).logits[0, -1].detach().float().cpu())
        finally:
            for handle in handles: handle.remove()
    logits = torch.stack(logits_list)
    active_state = states["mlp_active"]; bypass_state = states["mlp_bypass"]
    active_error = scalar((active_state.float() - target_full.float()).abs().max())
    bypass_error = scalar((bypass_state.float() - target_r.float()).abs().max())
    common = {
        "experiment": "2_matched_mlp_intervention", "layer": layer_index, "position": position,
        "position_label": source_label(pair, position), "component": "mlp_branch",
        "recipient_vector_norm": "", "hybrid_vector_norm": norm(target_r),
        "delta_vector_norm": norm(target_mlp), "cosine_recipient_hybrid": cosine(target_r, target_full),
    }
    return [
        make_row(pair, shard_id, dm, rm, logits[0], intervention="mlp_active", state_error_max_abs=active_error,
                 notes="hybrid post-attention r patched before MLP norm; MLP active", **common),
        make_row(pair, shard_id, dm, rm, logits[1], intervention="mlp_bypass", state_error_max_abs=bypass_error,
                 notes="block output equals hybrid r; exactly one MLP branch omitted", **common),
        make_row(pair, shard_id, dm, rm, logits[2], intervention="full_post_mlp", state_error_max_abs=active_error,
                 notes="true hybrid r + MLP(norm(r)) patched at block output", **common),
    ]


def edge_message(capture: Capture, layer_index: int, head: int, source: int, num_kv_groups: int,
                 num_kv_heads: int, head_dim: int) -> torch.Tensor:
    kv_head = head // num_kv_groups
    value = capture.values[layer_index][source].view(num_kv_heads, head_dim)[kv_head]
    weight = capture.weights[layer_index][head, P10, source]
    return weight * value


@torch.inference_mode()
def head_interventions(model, layers, pair: Pair, donor0: torch.Tensor, recipient: Capture, hybrid: Capture,
                       layer_index: int, device: torch.device, dm: float, rm: float, shard_id: int,
                       num_heads: int, head_dim: int) -> list[dict[str, Any]]:
    specs = [("head_sufficiency", h) for h in range(num_heads)] + [("head_necessity", h) for h in range(num_heads)]
    ids = torch.tensor([pair.recipient_ids] * len(specs), dtype=torch.long, device=device)
    rec_heads = recipient.heads[layer_index][P10].view(num_heads, head_dim)
    hyb_heads = hybrid.heads[layer_index][P10].view(num_heads, head_dim)

    def source_hook(_module, _inputs, out):
        x0 = out[0] if isinstance(out, tuple) else out
        x = x0.clone()
        for i, (kind, _h) in enumerate(specs):
            if kind == "head_necessity": x[i, P02] = donor0.to(x)
        return replace_output(out, x)

    def o_pre_hook(_module, inputs):
        x = inputs[0].clone(); shaped = x.view(len(specs), x.shape[1], num_heads, head_dim)
        for i, (kind, head) in enumerate(specs):
            shaped[i, P10, head] = (hyb_heads if kind == "head_sufficiency" else rec_heads)[head].to(x)
        return (x,)

    h1 = layers[0].register_forward_hook(source_hook)
    h2 = layers[layer_index].self_attn.o_proj.register_forward_pre_hook(o_pre_hook)
    try:
        logits = model(input_ids=ids, use_cache=False).logits[:, -1].detach().float().cpu()
    finally:
        h1.remove(); h2.remove()
    rows = []
    groups = num_heads // int(model.config.num_key_value_heads)
    for i, (kind, head) in enumerate(specs):
        rec = rec_heads[head]; hyb = hyb_heads[head]
        rows.append(make_row(
            pair, shard_id, dm, rm, logits[i], experiment="3_head_and_token_position_path_scan",
            layer=layer_index, position=P10, position_label=source_label(pair, P10),
            component="query_head_pre_o_proj", head=head, kv_head=head // groups, intervention=kind,
            recipient_vector_norm=norm(rec), hybrid_vector_norm=norm(hyb), delta_vector_norm=norm(hyb - rec),
            cosine_recipient_hybrid=cosine(rec, hyb),
        ))
    return rows


@torch.inference_mode()
def edge_interventions(model, layers, pair: Pair, donor0: torch.Tensor, recipient: Capture, hybrid: Capture,
                       layer_index: int, device: torch.device, dm: float, rm: float, shard_id: int,
                       num_heads: int, num_kv_heads: int, head_dim: int, chunk_size: int) -> list[dict[str, Any]]:
    groups = num_heads // num_kv_heads
    rec_heads = recipient.heads[layer_index][P10].view(num_heads, head_dim)
    hyb_heads = hybrid.heads[layer_index][P10].view(num_heads, head_dim)
    specs = [(kind, head, source) for kind in ("edge_sufficiency", "edge_necessity")
             for head in range(num_heads) for source in range(P10 + 1)]
    rows = []
    for start in range(0, len(specs), chunk_size):
        chunk = specs[start:start + chunk_size]
        ids = torch.tensor([pair.recipient_ids] * len(chunk), dtype=torch.long, device=device)

        def source_hook(_module, _inputs, out):
            x0 = out[0] if isinstance(out, tuple) else out
            x = x0.clone()
            for i, (kind, _head, _source) in enumerate(chunk):
                if kind == "edge_necessity": x[i, P02] = donor0.to(x)
            return replace_output(out, x)

        def o_pre_hook(_module, inputs):
            x = inputs[0].clone(); shaped = x.view(len(chunk), x.shape[1], num_heads, head_dim)
            for i, (kind, head, source) in enumerate(chunk):
                rec_edge = edge_message(recipient, layer_index, head, source, groups, num_kv_heads, head_dim)
                hyb_edge = edge_message(hybrid, layer_index, head, source, groups, num_kv_heads, head_dim)
                delta = hyb_edge - rec_edge
                shaped[i, P10, head].add_((delta if kind == "edge_sufficiency" else -delta).to(x))
            return (x,)

        h1 = layers[0].register_forward_hook(source_hook)
        h2 = layers[layer_index].self_attn.o_proj.register_forward_pre_hook(o_pre_hook)
        try:
            logits = model(input_ids=ids, use_cache=False).logits[:, -1].detach().float().cpu()
        finally:
            h1.remove(); h2.remove()
        for i, (kind, head, source) in enumerate(chunk):
            rec_edge = edge_message(recipient, layer_index, head, source, groups, num_kv_heads, head_dim)
            hyb_edge = edge_message(hybrid, layer_index, head, source, groups, num_kv_heads, head_dim)
            rows.append(make_row(
                pair, shard_id, dm, rm, logits[i], experiment="3_head_and_token_position_path_scan",
                layer=layer_index, position=P10, position_label=source_label(pair, P10),
                component="value_weighted_edge_message_pre_o_proj", head=head, kv_head=head // groups,
                source_position=source, source_position_label=source_label(pair, source),
                source_group=source_group(source), intervention=kind,
                recipient_vector_norm=norm(rec_edge), hybrid_vector_norm=norm(hyb_edge),
                delta_vector_norm=norm(hyb_edge - rec_edge), cosine_recipient_hybrid=cosine(rec_edge, hyb_edge),
                attention_weight_recipient=scalar(recipient.weights[layer_index][head, P10, source]),
                attention_weight_hybrid=scalar(hybrid.weights[layer_index][head, P10, source]),
            ))
    return rows


@torch.inference_mode()
def all_head_control(model, layers, pair: Pair, donor0: torch.Tensor, recipient: Capture, hybrid: Capture,
                     layer_index: int, device: torch.device, dm: float, rm: float, shard_id: int,
                     num_heads: int, head_dim: int) -> list[dict[str, Any]]:
    specs = ["all_head_sufficiency", "all_head_necessity"]
    ids = torch.tensor([pair.recipient_ids] * 2, dtype=torch.long, device=device)

    def source_hook(_module, _inputs, out):
        x0 = out[0] if isinstance(out, tuple) else out
        x = x0.clone(); x[1, P02] = donor0.to(x)
        return replace_output(out, x)

    def o_pre_hook(_module, inputs):
        x = inputs[0].clone(); shaped = x.view(2, x.shape[1], num_heads, head_dim)
        shaped[0, P10] = hybrid.heads[layer_index][P10].view(num_heads, head_dim).to(x)
        shaped[1, P10] = recipient.heads[layer_index][P10].view(num_heads, head_dim).to(x)
        return (x,)

    h1 = layers[0].register_forward_hook(source_hook)
    h2 = layers[layer_index].self_attn.o_proj.register_forward_pre_hook(o_pre_hook)
    try:
        logits = model(input_ids=ids, use_cache=False).logits[:, -1].detach().float().cpu()
    finally:
        h1.remove(); h2.remove()
    return [make_row(
        pair, shard_id, dm, rm, logits[i], experiment="3_head_and_token_position_path_scan",
        layer=layer_index, position=P10, position_label=source_label(pair, P10),
        component="all_query_heads_pre_o_proj_control", intervention=kind,
    ) for i, kind in enumerate(specs)]


@torch.inference_mode()
def reference_controls(model, layers, pair: Pair, donor: Capture, device: torch.device, dm: float, rm: float,
                       shard_id: int, selected_layers: list[int]) -> list[dict[str, Any]]:
    specs = [(kind, layer) for kind in ("previous_P02_source_curve", "previous_direct_P10_curve") for layer in selected_layers]
    ids = torch.tensor([pair.recipient_ids], dtype=torch.long, device=device)
    logits = []
    for kind, layer_index in specs:
        def hook(_module, _inputs, out, li=layer_index, current=kind):
            x0 = out[0] if isinstance(out, tuple) else out
            x = x0.clone()
            pos = P02 if current == "previous_P02_source_curve" else P10
            x[0, pos] = donor.block[li][pos].to(x)
            return replace_output(out, x)
        handle = layers[layer_index].register_forward_hook(hook)
        try:
            logits.append(model(input_ids=ids, use_cache=False).logits[0, -1].detach().float().cpu())
        finally:
            handle.remove()
    return [make_row(
        pair, shard_id, dm, rm, logits[i], experiment="validation_reference_reproduction", layer=layer,
        position=P02 if kind == "previous_P02_source_curve" else P10,
        position_label=source_label(pair, P02 if kind == "previous_P02_source_curve" else P10),
        component="block_output_residual_reference_control", intervention=kind,
    ) for i, (kind, layer) in enumerate(specs)]


@torch.inference_mode()
def identity_control(model, layers, pair: Pair, recipient: Capture, device: torch.device, dm: float, rm: float,
                     shard_id: int) -> list[dict[str, Any]]:
    # Recipient P02 into recipient P02 is the explicit wrong-entity/identity source control.
    ids = torch.tensor([pair.recipient_ids], dtype=torch.long, device=device)
    def hook(_module, _inputs, out):
        x0 = out[0] if isinstance(out, tuple) else out
        x = x0.clone(); x[0, P02] = recipient.block[0][P02].to(x)
        return replace_output(out, x)
    h = layers[0].register_forward_hook(hook)
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].detach().float().cpu()
    finally:
        h.remove()
    return [make_row(
        pair, shard_id, dm, rm, logits, experiment="validation_control", layer=0, position=P02,
        position_label=source_label(pair, P02), component="block_output_residual_identity",
        intervention="wrong_entity_recipient_P02_identity", notes="recipient entity state; must not recover donor",
    )]


def run_pair(model, layers, config, pair: Pair, device: torch.device, out_dir: Path, shard_id: int,
             layers_to_run: list[int], chunk_size: int, smoke: bool) -> dict[str, Any]:
    started = time.time()
    checkpoint = out_dir / "checkpoints" / pair.pair_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    donor = capture_run(model, layers, pair.donor_ids, device)
    recipient = capture_run(model, layers, pair.recipient_ids, device)
    hybrid = capture_run(model, layers, pair.recipient_ids, device, donor.block[0][P02])
    dm, rm, denom = baseline(pair, donor, recipient)
    num_heads = int(config.num_attention_heads); num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim); groups = num_heads // num_kv_heads

    attention_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    reconstruction: dict[str, Any] = {}
    for li in layers_to_run:
        layer_dir = checkpoint / f"layer_{li:02d}"
        layer_dir.mkdir(exist_ok=True)
        ap = layer_dir / "attention.csv"
        mp = layer_dir / "mlp.csv"
        if ap.exists(): attention_rows.extend(read_csv(ap))
        else:
            rows = attention_interventions(model, layers, pair, donor.block[0][P02], recipient, hybrid, li,
                                           device, dm, rm, shard_id)
            write_csv(ap, rows); attention_rows.extend(rows)
        if mp.exists(): mlp_rows.extend(read_csv(mp))
        else:
            rows = []
            for pos in (P02, P10):
                rows.extend(mlp_interventions(model, layers, pair, hybrid, li, pos, device, dm, rm, shard_id))
            write_csv(mp, rows); mlp_rows.extend(rows)

        if li in SCAN_LAYERS:
            hp = layer_dir / "heads.csv"; ep = layer_dir / "edges.csv"
            if hp.exists(): head_rows.extend(read_csv(hp))
            else:
                rows = head_interventions(model, layers, pair, donor.block[0][P02], recipient, hybrid, li,
                                          device, dm, rm, shard_id, num_heads, head_dim)
                rows += all_head_control(model, layers, pair, donor.block[0][P02], recipient, hybrid, li,
                                         device, dm, rm, shard_id, num_heads, head_dim)
                write_csv(hp, rows); head_rows.extend(rows)
            if ep.exists(): edge_rows.extend(read_csv(ep))
            else:
                rows = edge_interventions(model, layers, pair, donor.block[0][P02], recipient, hybrid, li,
                                          device, dm, rm, shard_id, num_heads, num_kv_heads, head_dim, chunk_size)
                write_csv(ep, rows); edge_rows.extend(rows)

            for name, cap in (("recipient", recipient), ("hybrid", hybrid)):
                target_heads = cap.heads[li][P10].view(num_heads, head_dim)
                rebuilt = torch.stack([
                    torch.stack([edge_message(cap, li, h, j, groups, num_kv_heads, head_dim)
                                 for j in range(P10 + 1)]).sum(0) for h in range(num_heads)
                ])
                # Preserve the original [sequence, hidden] GEMM shape, avoiding
                # a BF16 kernel change from projecting only the P10 row.
                rebuilt_all = cap.heads[li].clone().view(11, num_heads, head_dim)
                rebuilt_all[P10] = rebuilt
                projected = layers[li].self_attn.o_proj(rebuilt_all.reshape(11, -1))[P10].reshape(-1)
                target_projected = cap.attention[li][P10]
                reconstruction[f"layer_{li}_{name}_edge_to_head_max_abs"] = scalar((rebuilt.float() - target_heads.float()).abs().max())
                reconstruction[f"layer_{li}_{name}_edge_to_head_cosine"] = cosine(rebuilt, target_heads)
                reconstruction[f"layer_{li}_{name}_all_head_to_attention_max_abs"] = scalar((projected.float() - target_projected.float()).abs().max())
                reconstruction[f"layer_{li}_{name}_all_head_to_attention_cosine"] = cosine(projected, target_projected)
        log(f"{pair.pair_id} layer {li} checkpoint complete")

    ref_path = checkpoint / "reference_controls.csv"
    if ref_path.exists(): reference_rows = read_csv(ref_path)
    else:
        reference_rows = reference_controls(model, layers, pair, donor, device, dm, rm, shard_id, layers_to_run)
        write_csv(ref_path, reference_rows)
    control_path = checkpoint / "identity_controls.csv"
    if control_path.exists(): control_rows = read_csv(control_path)
    else:
        control_rows = identity_control(model, layers, pair, recipient, device, dm, rm, shard_id)
        write_csv(control_path, control_rows)

    write_csv(checkpoint / "attention_results.csv", attention_rows)
    write_csv(checkpoint / "mlp_results.csv", mlp_rows)
    write_csv(checkpoint / "head_results.csv", head_rows)
    write_csv(checkpoint / "edge_results.csv", edge_rows)
    write_csv(checkpoint / "reference_results.csv", reference_rows)
    write_csv(checkpoint / "control_results.csv", control_rows)
    details = {
        "pair_id": pair.pair_id, "unordered_pair_id": pair.cluster, "completed_at": now(),
        "runtime_seconds": time.time() - started, "donor_margin": dm, "recipient_margin": rm,
        "normalization_denominator": denom, "hybrid_end_to_end_recovery":
            (float(hybrid.logits[pair.donor_token_id] - hybrid.logits[pair.recipient_token_id]) - rm) / denom,
        "layers": layers_to_run, "scan_layers": [x for x in SCAN_LAYERS if x in layers_to_run],
        "num_attention_rows": len(attention_rows), "num_mlp_rows": len(mlp_rows),
        "num_head_rows": len(head_rows), "num_edge_rows": len(edge_rows),
        "num_reference_rows": len(reference_rows), "num_control_rows": len(control_rows),
        "reconstruction": reconstruction,
        "hybrid_norms": {f"layer_{li}": {
            "P02_r": norm(hybrid.r[li][P02]), "P10_r": norm(hybrid.r[li][P10]),
            "P02_mlp": norm(hybrid.mlp[li][P02]), "P10_mlp": norm(hybrid.mlp[li][P10]),
            "P10_attention": norm(hybrid.attention[li][P10]),
        } for li in layers_to_run},
    }
    write_json(checkpoint / "COMPLETE.json", details)
    return details


def shard_slice(items: list[Any], num_shards: int, shard_id: int) -> list[Any]:
    base, extra = divmod(len(items), num_shards)
    start = shard_id * base + min(shard_id, extra)
    return items[start:start + base + (1 if shard_id < extra else 0)]


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    device = torch.device(args.device)
    pairs = load_pairs()
    model, tokenizer, layers, config = load_model(args.model, device)
    metadata = validate_tokenizer(tokenizer, pairs)
    if int(config.num_hidden_layers) != N_LAYERS:
        raise RuntimeError("model config layer count mismatch")
    model_meta = {
        "model": args.model, "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads), "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim), "num_key_value_groups": int(config.num_attention_heads // config.num_key_value_heads),
        "hidden_size": int(config.hidden_size), "attention_implementation": str(config._attn_implementation),
        "torch_version": torch.__version__, "device": str(device), "dtype": str(next(model.parameters()).dtype),
        "source_intervention": "donor P02 block-0 output into recipient P02 block-0 output",
    }
    write_json(out_dir / "model_config.json", model_meta)

    if args.smoke:
        first_cluster = pairs[0].cluster
        selected = [p for p in pairs if p.cluster == first_cluster]
        layers_to_run = SCAN_LAYERS
    else:
        selected = shard_slice(pairs, args.num_shards, args.shard_id)
        layers_to_run = list(range(N_LAYERS))
    selected_ids = {p.pair_id for p in selected}
    write_csv(out_dir / "pair_metadata.csv", [r for r in metadata if r["pair_id"] in selected_ids], list(metadata[0]))
    write_json(out_dir / "run_manifest.json", {
        "mode": "smoke" if args.smoke else "full_shard", "started_at": now(), "shard_id": args.shard_id,
        "num_shards": args.num_shards, "pair_ids": [p.pair_id for p in selected], "layers": layers_to_run,
        "scan_layers": SCAN_LAYERS, "experiments": [
            "1_attention_output_mediation", "2_matched_mlp_intervention", "3_head_and_token_position_path_scan"],
    })
    details = []
    for i, pair in enumerate(selected, 1):
        complete = out_dir / "checkpoints" / pair.pair_id / "COMPLETE.json"
        if complete.exists():
            details.append(json.loads(complete.read_text())); log(f"resuming: {pair.pair_id} already complete"); continue
        log(f"pair {i}/{len(selected)} {pair.pair_id} starting")
        details.append(run_pair(model, layers, config, pair, device, out_dir, args.shard_id,
                                layers_to_run, args.edge_chunk_size, args.smoke))
        torch.cuda.empty_cache() if device.type == "cuda" else None
    write_json(out_dir / "worker_summary.json", {"completed_at": now(), "pairs": details})
    (out_dir / "SUCCESS").write_text(now() + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--edge-chunk-size", type=int, default=96)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard arguments")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(20260715); torch.manual_seed(20260715)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(20260715)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
