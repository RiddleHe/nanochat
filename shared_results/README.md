# Shared result files (data + figures)

Copies of the key result files (the `results/` dir is gitignored). Raw numbers
are in the jsons; replot however you like. Scripts that produced each file are
in `scripts/inspect/` (see `STEPS_README.md` there for setups/commands).

## Translator (linear map makes late states patchscope-readable)
- `Qwen_Qwen3-8B-Base__translator.json` / `.png` — clean verified run.
  rows: per source layer {raw, translated, ceiling} patchscope accuracy
  (Qwen3-8B, ref target L10, held-out templates). raw 0.24-0.37 -> translated
  0.61-0.68 = ceiling. Script: `entity_translator.py` (PR #14).
- `step_bc_qwen.json`, `step_b_translator.png` — original exploratory run
  (same numbers), plus Step C direction-alignment data (`step_c_direction.png`).
- `step_b_matrix.json` / `.png` — 18x18 source x target heatmaps, raw vs
  translated vs improvement (grid layers 0,2,...,34).
- `step_b_anatomy.json` / `.png` — shift/scale/rotate decomposition of the
  translator + geometry stats (|A-I|, stable rank, rotation angle).

## Per-layer entity probe (Step 1)
- `{Attention,BoV,Qwen3-8B}__step1.json`, `Qwen3-8B__step1.png` — held-out-
  template probe accuracy per layer + within-category + random-split +
  random-label floor. Script: `probe_step1_robust.py` / `entity_probe.py`.

## Probe vs patchscope (Step 2)
- `step2_qwen.json` / `.png` — same hidden states read by probe vs candidate-
  logit patchscope (target sweep, generation, random floor).
- `step2b_qwen.json`, `step2b_matrix.png` — free-generation patchscope + full
  36x36 source x target matrix.
- `authentic_vs_probe.png` — the ORIGINAL few-shot description patchscope
  (friend's script) on Qwen3-8B vs the probe curve.

## Causal boundary (Step D) + attention check
- `step_d_qwen.json` / `step_d_causal.png` — subject-interchange patching,
  logit-diff recovery per layer (16 pairs, 13 sentences). Cliff L23-26.
- `step_d_probs.json` / `.png` (added when the run finishes) — same experiment,
  softmax PROBABILITY view: P(correct entity) and P(wrong entity) per patched
  layer, two lines, plus clean/corrupt-run reference levels.
- `attn_to_subject.json` / `.png` — no patching: final token's attention mass
  per layer (name span mean + max-over-heads, sink, self, rest). Max-head on
  the name is 0.2-0.44 through L7-23, collapses after ~L24; sink takes over.
- `d12__boundary.json`, `d24__boundary.json` — same causal boundary measured
  on nanochat baselines (d12: L8, d24: L14). Script: `step_d_nanochat.py`.
- `boundary_vs_quality.json` / `.png` — swap-start sweep val_bpb vs measured
  boundary (d12 complete; d24 missing points training now).

## Sliding-window direction (E1/E2, Qwen3-8B)
- `deep_window_mask.json` / `.png` — E1: inference-time window masking of
  layers >= start (W sweep, sink 4 vs 0), wikitext-103 val NLL. No cliff at
  the boundary: deep layers (L24+) are the MOST expensive to window
  (+0.082 nats alone vs +0.036 extra for all of L8-23). W=1024 nearly free
  (+0.007). Sink must stay visible. Script: `deep_window_mask.py`.
- `deep_window_bands.json` — E1b: band masking [start,end). Per masked layer:
  mid L8-24 ~0.0030 nats, deep L24-36 ~0.0068 (2.3x). Windows are cheaper in
  the middle, full attention matters most LATE.
- `handoff_distance.json` / `.png` — E2: entity-pos vs last-pos patching with
  N filler tokens between entity and readout (N=0..1400, 16 pairs). Hand-off
  ONSET is distance-invariant (L23-24); COMPLETION is not: local N=0 arrives
  at readout by L26, distant (N>=100) only by L31-34 while the entity position
  keeps ~0.4 causal influence through L30+. Effect saturates by N~100 (two
  regimes: local vs far). Kills the "deep layers need no distant tokens"
  hypothesis; supports the flipped placement (windows early/mid, full late).

## E3 — window placement trained from scratch (val_bpb, equal FLOPs, equal 3:1 S:L)
| layout | d12 @1.5e18 | d24 @3.91e19 |
|---|---|---|
| interleaved SSSL (baseline) | 0.8540 | 0.7218 |
| full-attn LATE (S...SLLL)   | 0.8541 | 0.7229 |
| full-attn EARLY (LLLS...S)  | 0.8612 | 0.7237 |
EARLY is worst at both scales (predicted by the hand-off/distance measurements:
distant reads happen late). LATE ties interleaved at d12, slightly behind at
d24. Single runs, no seed variance. Checkpoints: arch_{d12,d24}_gpt_base_full_attn_{late,early}*.
Pipeline: run_placement_pipeline.sh (log /tmp/placement_pipeline.log).
