"""Two missing controls for the two-stage relay (pre-registered 2026-09-16).

C1 block_only  -- block the readout's direct entity read after layer T with NO
   patch of any kind, then generate. Experiments 4/5 (stage2_block, relay_block)
   compare "patched + blocked" against "patched", but blocking alone damages the
   model (outputs drift to "John"/"**"). Without this baseline we cannot tell how
   much of the blocked-condition flip rate is rescue and how much is the model
   losing its own answer. Reading: if block_only already destroys the recipient
   answer at mid T, the mid-T rescue claim is weak; if block_only keeps the
   recipient answer at late T while blocked+patched flips to donor, the late-T
   overwrite claim is clean.

C2 mismatch_transfer -- stage-1 transfer score, but the patched entity state
   comes from an UNRELATED pair's donor (e.g. patch Mozart into the
   Einstein->Newton case) while still projecting onto THIS case's (d - r) axis.
   The transfer score must be near zero: otherwise a high score merely reflects
   "patching anything at the entity position perturbs the readout along the
   entity axis", not "this donor's identity arrived".

Usage: CUDA_VISIBLE_DEVICES=N python -m scripts.inspect.qwen_relay_two_stage_controls --out-dir results/relay_controls_v2
"""
from __future__ import annotations
import argparse, datetime as dt, json, sys
from pathlib import Path
import torch

from scripts.inspect.qwen_entity_relay_fixed_window import (
    MODEL, PAIRS, TEMPLATES, encode_prompt, greedy_completion, score_completion,
    validate_pair, write_jsonl, parse_ints, build_relay_state)
from scripts.inspect.qwen_relay_two_stage import (
    capture_states, transfer_scores, block_readout, load_model_with_block)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--template-ids", type=parse_ints, default=parse_ints("0,1,2,3"))
    ap.add_argument("--pair-ids", type=parse_ints, default=None)
    ap.add_argument("--relay-width", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--block-layers", type=parse_ints,
                    default=parse_ints("0,4,8,12,16,20,24,26,28,29,30,31,32,34"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-threads", type=int, default=48)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer, layers = load_model_with_block(args.model, device, args.cpu_threads)
    L = len(layers)
    pair_ids = args.pair_ids if args.pair_ids is not None else list(range(len(PAIRS)))
    templates = [t for t in TEMPLATES if t.template_id in args.template_ids]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(
        {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "command": [sys.executable] + sys.argv, "model": args.model, "n_layers": L,
         "controls": ["block_only", "mismatch_transfer"],
         "templates": [t.name for t in templates], "pairs": [PAIRS[i] for i in pair_ids]},
        ensure_ascii=False, indent=2) + "\n")

    with (out / "rows.jsonl").open("w") as f:
        for tmpl in templates:
            for pid in pair_ids:
                dn, rn = PAIRS[pid]
                donor, recip = encode_prompt(tokenizer, tmpl, dn), encode_prompt(tokenizer, tmpl, rn)
                validate_pair(donor, recip)
                base = {"template_id": tmpl.template_id, "template_name": tmpl.name,
                        "pair_id": pid, "donor": dn, "recipient": rn}
                ent_pos = recip.entity_positions[0]
                print(f"case {tmpl.name} {dn}->{rn}", flush=True)

                # C1: block only, no patch. Scored against the RECIPIENT's own answer:
                # expected=recipient (does it still say Newton?), alternative=donor.
                for T in args.block_layers:
                    with block_readout(range(T + 1, L), ent_pos):
                        row = score_completion(greedy_completion(
                            model, tokenizer, layers, recip, device, args.max_new_tokens), rn, dn)
                    write_jsonl(f, {"condition": "block_only", "relay_layer_t": T,
                                    "blocked_layers": [T + 1, L - 1], **base, **row})

                # C2: mismatched donor -> this case's axis
                mm_pid = pair_ids[(pair_ids.index(pid) + 1) % len(pair_ids)]
                mm_name = PAIRS[mm_pid][0]
                mm = encode_prompt(tokenizer, tmpl, mm_name)
                if len(mm.entity_positions) == 1 and mm.entity_positions == recip.entity_positions \
                        and len(mm.ids) == len(recip.ids):
                    mm_ent, _ = capture_states(model, layers, mm, device)
                    _, d_fin = capture_states(model, layers, donor, device)
                    _, r_fin = capture_states(model, layers, recip, device)
                    for S in range(0, L - args.relay_width):
                        T = S + args.relay_width
                        h = build_relay_state(model, layers, recip, S, T, mm_ent[S], device)
                        proj, cos = transfer_scores(h, r_fin[T], d_fin[T])
                        write_jsonl(f, {"condition": "mismatch_transfer", "source_layer_s": S,
                                        "relay_layer_t": T, "mismatch_donor": mm_name,
                                        "transfer_proj": proj, "transfer_cos": cos, **base})
                else:
                    write_jsonl(f, {"condition": "mismatch_skipped", "mismatch_donor": mm_name, **base})
    print("saved", out / "rows.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
