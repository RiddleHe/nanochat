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
