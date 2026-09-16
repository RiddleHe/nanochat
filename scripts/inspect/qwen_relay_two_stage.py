"""Two-stage decomposition of the fixed-window entity->readout relay (Qwen).

The relay's late-layer peak has a suspected confound: after the readout state
is transplanted at block T (pass 3), the remaining blocks T+1..35 add large
recipient-context writes that may bury the donor signal. Whether the peak is
"transfer happens late" or "early transfers get buried" cannot be read from
the final generation alone. This script measures the two stages separately,
reusing the collaborator's exact patching code (qwen_entity_relay_fixed_window).

STAGE 1  transfer score (no generation, no pass 3): after pass 2 (donor entity
         state patched at block S, run to block T), compare the readout state
         h_T against the clean recipient r_T and clean donor d_T at the SAME
         block:   transfer = <h_T - r_T, u> / |d_T - r_T|,  u = (d_T - r_T)/|.|
         (1 = fully moved to donor, 0 = untouched). Same-layer comparison, so no
         coordinate drift and nothing downstream can bury it. Also a logit-lens
         normalized recovery on h_T for a second reading. Scanned over the FULL
         (S, T) plane, S < T.
STAGE 2  direct survival (generation): transplant the clean donor readout state
         d_T itself at block T into the recipient (pass 3 only, strongest
         possible donor signal) and generate. Scanned over every T. If mid-T
         transplants flip the output, later blocks cannot bury a donor signal
         and the relay's mid-S failure is a stage-1 fact.
RELAY    the original combined readout at width 4, recomputed in-run so all
         three curves come from identical cases.

PRE-REGISTERED READINGS (2026-09-16, before running):
  A  stage-1 transfer ~0 for mid S and rising only late, stage-2 survival high
     at mid T  -> late transfer is real; overwrite confound irrelevant.
  B  stage-1 ~0 mid, stage-2 low mid  -> late transfer real AND overwrite exists.
  C  stage-1 substantial mid, stage-2 high mid  -> contradiction: the relay's
     pass-3 transplant itself is at fault; inspect code.
  D  stage-1 substantial mid, stage-2 low mid  -> late peak is an overwrite
     artifact; transfer happens earlier.
Check: relay flip(S) should track stage1(S) x stage2(S+4).

Usage:
  CUDA_VISIBLE_DEVICES=N python -m scripts.inspect.qwen_relay_two_stage --out-dir results/relay_two_stage
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import torch

from scripts.inspect.qwen_entity_relay_fixed_window import (
    MODEL, PAIRS, TEMPLATES, EncodedPrompt, build_relay_state, encode_prompt,
    greedy_completion, load_model, score_completion, validate_pair, write_jsonl,
    parse_ints,
)


ATTN_NAME = "sdpa_readout_block"
BLOCK = {"active": False, "layers": frozenset(), "entity_position": None}


def readout_block_attention_forward(module, query, key, value, attention_mask, scaling,
                                    dropout=0.0, **kw):
    """SDPA, except in BLOCK['layers'] the READOUT query rows (final prompt token
    during prefill; every generated token during cached decoding) lose the
    entity's value contribution. Mirrors the collaborator's entity_zero hook
    (weights untouched, no renormalization) but restricted to the readout so
    intermediate tokens keep their access. Falls through to plain SDPA when
    inactive, so every other condition is bit-identical to the default model."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if BLOCK["active"] and module.layer_idx in BLOCK["layers"]:
        pos = BLOCK["entity_position"]
        zero_value = value.clone()
        zero_value[:, :, pos, :] = 0
        zout, _ = sdpa_attention_forward(module, query, key, zero_value, attention_mask,
                                         scaling=scaling, dropout=dropout, **kw)
        if query.shape[-2] == 1:          # cached decoding: this query is a generated token
            return zout, None
        out, _ = sdpa_attention_forward(module, query, key, value, attention_mask,
                                        scaling=scaling, dropout=dropout, **kw)
        out = out.clone()
        out[:, -1, :, :] = zout[:, -1, :, :]   # only the final prompt token row
        return out, None
    return sdpa_attention_forward(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kw)


class block_readout:
    """Context: block the readout's direct entity read in `layers`."""
    def __init__(self, layers, entity_position):
        self.layers, self.pos = frozenset(layers), entity_position
    def __enter__(self):
        BLOCK.update(active=True, layers=self.layers, entity_position=self.pos)
    def __exit__(self, *a):
        BLOCK.update(active=False, layers=frozenset(), entity_position=None)


def load_model_with_block(model_name: str, device: torch.device, cpu_threads: int):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    AttentionInterface.register(ATTN_NAME, readout_block_attention_forward)
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=ATTN_NAME).to(device).eval()
    return model, tokenizer, list(model.model.layers)


@torch.inference_mode()
def capture_states(model, layers, encoded: EncodedPrompt, device):
    """Per block: entity-position state and final-prompt-token state."""
    ent = [None] * len(layers)
    fin = [None] * len(layers)
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            ent[i] = h[0, encoded.entity_positions, :].detach().clone()
            fin[i] = h[0, -1, :].detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(mk(i)))
    try:
        model(input_ids=torch.tensor([encoded.ids], device=device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return ent, fin


def answer_token_id(tokenizer, name: str):
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    return ids[0], len(ids) == 1


@torch.inference_mode()
def logit_lens_margin(model, state: torch.Tensor, donor_id: int, recip_id: int) -> float:
    x = model.model.norm(state.unsqueeze(0).unsqueeze(0).to(model.dtype))
    logits = model.lm_head(x)[0, -1].float()
    return float(logits[donor_id] - logits[recip_id])


def transfer_scores(h, r, d):
    h, r, d = h.float(), r.float(), d.float()
    diff = d - r
    n = diff.norm()
    if n == 0:
        return 0.0, 0.0
    proj = float(torch.dot(h - r, diff / n) / n)
    cos = float(torch.nn.functional.cosine_similarity(h - r, diff, dim=0))
    return proj, cos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--template-ids", type=parse_ints, default=parse_ints("0,1,2,3"))
    ap.add_argument("--pair-ids", type=parse_ints, default=None)
    ap.add_argument("--relay-width", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--width-only", action="store_true",
                    help="stage 1 only at T=S+width instead of the full (S,T) plane")
    ap.add_argument("--stage2-layers", type=parse_ints, default=None,
                    help="subset of T for stage 2 (default: every layer)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-threads", type=int, default=48)
    ap.add_argument("--conditions", default="stage1,stage2,relay,stage2_block,relay_block",
                    help="comma subset of: stage1,stage2,relay,stage2_block,relay_block")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    args.stage1_full_plane = not args.width_only
    conds = set(args.conditions.split(","))

    device = torch.device(args.device)
    # custom attention name: identical to SDPA unless a block_readout() context is active
    model, tokenizer, layers = load_model_with_block(args.model, device, args.cpu_threads)
    L = len(layers)
    stage2_T = args.stage2_layers if args.stage2_layers is not None else list(range(L))
    pair_ids = args.pair_ids if args.pair_ids is not None else list(range(len(PAIRS)))
    templates = [t for t in TEMPLATES if t.template_id in args.template_ids]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": [sys.executable] + sys.argv, "model": args.model, "n_layers": L,
            "templates": [t.name for t in templates], "pairs": [PAIRS[i] for i in pair_ids],
            "relay_width": args.relay_width, "protocol": "reuses qwen_entity_relay_fixed_window hooks"}
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    with (out / "rows.jsonl").open("w") as f:
        for tmpl in templates:
            for pid in pair_ids:
                dn, rn = PAIRS[pid]
                donor, recip = encode_prompt(tokenizer, tmpl, dn), encode_prompt(tokenizer, tmpl, rn)
                validate_pair(donor, recip)
                d_tok, d_single = answer_token_id(tokenizer, dn)
                r_tok, r_single = answer_token_id(tokenizer, rn)
                base = {"template_id": tmpl.template_id, "template_name": tmpl.name, "pair_id": pid,
                        "donor": dn, "recipient": rn, "answer_tokens_single": d_single and r_single}
                print(f"case {tmpl.name} {dn}->{rn}", flush=True)

                # baselines
                for cond, enc, exp, alt in (("donor_baseline", donor, dn, rn),
                                            ("recipient_baseline", recip, rn, dn)):
                    row = score_completion(greedy_completion(model, tokenizer, layers, enc, device,
                                                             args.max_new_tokens), exp, alt)
                    write_jsonl(f, {"condition": cond, **base, **row})

                d_ent, d_fin = capture_states(model, layers, donor, device)
                _, r_fin = capture_states(model, layers, recip, device)
                m_d = [logit_lens_margin(model, d_fin[t], d_tok, r_tok) for t in range(L)]
                m_r = [logit_lens_margin(model, r_fin[t], d_tok, r_tok) for t in range(L)]

                ent_pos = recip.entity_positions[0]

                # STAGE 2: direct donor readout-state transplant at each selected T
                # (+ control: same transplant, readout's direct entity read blocked after T)
                for T in stage2_T:
                    if "stage2" in conds:
                        row = score_completion(greedy_completion(
                            model, tokenizer, layers, recip, device, args.max_new_tokens,
                            relay_layer=T, relay_state=d_fin[T]), dn, rn)
                        write_jsonl(f, {"condition": "stage2_direct", "relay_layer_t": T, **base, **row})
                    if "stage2_block" in conds:
                        with block_readout(range(T + 1, L), ent_pos):
                            row = score_completion(greedy_completion(
                                model, tokenizer, layers, recip, device, args.max_new_tokens,
                                relay_layer=T, relay_state=d_fin[T]), dn, rn)
                        write_jsonl(f, {"condition": "stage2_block", "relay_layer_t": T,
                                        "blocked_layers": [T + 1, L - 1], **base, **row})

                # STAGE 1 (+ relay readout at width w, + relay with post-T block): (S,T) plane
                for S in range(L - 1):
                    for T in range(S + 1, L):
                        if not args.stage1_full_plane and T - S != args.relay_width:
                            continue
                        if T - S != args.relay_width and "stage1" not in conds:
                            continue
                        h = build_relay_state(model, layers, recip, S, T, d_ent[S], device)
                        if T - S == args.relay_width and "relay_block" in conds:
                            with block_readout(range(T + 1, L), ent_pos):
                                gen = score_completion(greedy_completion(
                                    model, tokenizer, layers, recip, device, args.max_new_tokens,
                                    relay_layer=T, relay_state=h), dn, rn)
                            write_jsonl(f, {"condition": "relay_block", "source_layer_s": S,
                                            "relay_layer_t": T, "blocked_layers": [T + 1, L - 1],
                                            **base, **gen})
                        if "stage1" not in conds and "relay" not in conds:
                            continue
                        proj, cos = transfer_scores(h, r_fin[T], d_fin[T])
                        m_h = logit_lens_margin(model, h, d_tok, r_tok)
                        denom = m_d[T] - m_r[T]
                        rec = (m_h - m_r[T]) / denom if abs(denom) > 1e-6 else None
                        dr = (d_fin[T].float() - r_fin[T].float()).norm().item()
                        row = {"condition": "stage1_transfer", "source_layer_s": S, "relay_layer_t": T,
                               "transfer_proj": proj, "transfer_cos": cos, "ll_margin_h": m_h,
                               "ll_margin_r": m_r[T], "ll_margin_d": m_d[T], "ll_recovery": rec,
                               # natural divergence of the clean runs at the readout token, same layer:
                               # tiny => nothing entity-specific has reached the final token by T
                               "clean_diff_norm": dr, "clean_diff_rel": dr / (r_fin[T].float().norm().item() + 1e-6),
                               "h_shift_norm": (h.float() - r_fin[T].float()).norm().item(), **base}
                        if T - S == args.relay_width and "relay" in conds:
                            gen = score_completion(greedy_completion(
                                model, tokenizer, layers, recip, device, args.max_new_tokens,
                                relay_layer=T, relay_state=h), dn, rn)
                            write_jsonl(f, {"condition": "relay", "source_layer_s": S,
                                            "relay_layer_t": T, **base, **gen})
                        if "stage1" in conds:
                            write_jsonl(f, row)
    print("saved", out / "rows.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
