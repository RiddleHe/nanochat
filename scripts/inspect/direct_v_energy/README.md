# Direct V component energy

Matched step-7809 GPTBase and GPTBaseNoEmbedNorm. This diagnostic uses no SVD.
It replays the exact 128 saved BOS-bestfit sequences (262,144 occurrences,
including BOS) from the original separate-source attribution, preserving context
boundaries and token order. This is a deterministic validation sample, not the
full held-out shard used by the separate raw-norm experiment.

At receiving layers 12,14,16,18,20,22, capture the actual BF16 residual before
receiving RMSNorm, and Emb plus every preceding separate attention/MLP output.
Emb is the actual first block input: post-initial-RMSNorm in the baseline and
raw in the variant. For every source, use the full actual residual denominator
sqrt(mean(x²)+BF16 epsilon), then apply the forward-cast V matrix in FP32.
Store the sum of squared coordinates for each of 12 heads of dimension 128.
Neither sourcewise normalization nor head averaging occurs in the heatmaps.

Stages:

- `prepare --gpu INDEX`: creates a new comparison directory with per-model
  frozen model/probe code, checkpoints, tokenizer and kernel dependencies,
  identical tokens, configs and hashes. Model snapshots come from the validated
  baseline probe and the variant training run respectively.
  `--layers 1 2 ... 23` selects an explicit ordered list of receiving layers;
  omit it for the original six-layer selection.
- `run --run-dir ROOT`: sequential single-GPU collection through each frozen
  environment; chunks support resume only under matching provenance.
- `summarize --run-dir ROOT`: validates coverage, hashes and checks; aggregates
  per-token energy, norm, denominators and energy shares using FP64 population
  moments. Energy shares divide by the entire Emb+attention+MLP bank per token.
- `plot --run-dir ROOT --output /absolute/comparison.html`: new-model-first
  six-layer heatmaps, then baseline; identical linear color scale across both.
  Add `--model no_embed_norm` to display only the variant, sharing its scale
  across the six layers.
- `plot_embedding_energy --run-dir ROOT --output /absolute/line.html`: dark
  two-line chart of the embedding's token-mean squared V-projection norm,
  averaged equally across 12 heads. Uses existing validated summaries; no replay.
  Add `--metric share` for embedding energy divided by summed Emb+attention+MLP
  component energies per token/head, then averaged over tokens and heads.
  Add `--uniform-reference` with `--metric share` for the equal-energy reference
  `1/(2L+1)` across Emb and the `L` preceding attention and `L` MLP sources.
  Add `--include-first-attention` with `--metric share` to include attention-0
  from the no-norm model, using the same full-bank denominator and averaging.

All stages are modules under `scripts.inspect.direct_v_energy`, using
`/hdd/mh3897/nanochat/.venv/bin/python` from the repository root.

Checks: unchanged logits on every sequence; embedding and actual V-input
identity; source-sum/residual and FP32/forward projection reconstruction with
the original tolerances; FP32 linear additivity; head/full energy identity;
energy versus squared norm; and every baseline source/projected norm versus
the original saved probe. BF16 accumulation/normalization rounding is reported
separately from FP32 linearity error. No tolerance is inferred from a new run.

The heatmaps display Emb+attention; MLP remains in normalization and share
denominators. Token standard deviations and shares are in hover/tap details.
Shares of component energy are not additive shares of actual output energy,
which includes cross terms. Trained-model comparisons include learned adaptation.

Completed comparison:
`/local-ssd/mh3897/analysis_runs/d24_no_embed_norm_v_comparison/step7809_20261102T213632Z`.
Both models passed checks on all 262,144 occurrences. Baseline collection took
64.0 s including comparisons to saved chunks; the variant took 27.0 s on GPU 0
(`GPU-68dc475f-848f-5a98-5166-ef16491754b4`). Embedding mean projected energy
increased in all 72 receiver/head pairs, by 16.8505–72.8137 times. Per-head data
and full-bank shares are in `comparison.json`; this is a trained-model association.

Expanded receiving layers 1–23:
`/local-ssd/mh3897/analysis_runs/d24_no_embed_norm_v_comparison/step7809_20261102T235407Z`.
All sequences passed validation for both models. `compare_replay --run-dir ROOT
--reference-run PRIOR_ROOT` verifies the overlapping layers; their energy,
energy-share and denominator summary arrays are exactly equal to the six-layer
run. The expanded dark line chart uses every measured layer 1–23.

## Squared output norm relative to the average source

`relative_energy --run-dir MODEL_RUN --alignment-run MATCHED_ALL_LAYER_ALIGNMENT`
derives `q_s = energy_s / mean_all_available_sources(energy)` for each token and
head. The average includes Emb and every prior attention and MLP output,
including zero-energy sources. It uses all 128 V-head output coordinates,
without a square root, SVD truncation, or normalization by the source's norm.
The common receiving RMSNorm factor cancels from this ratio.

For L1–L23, `q = (2L+1) * energy_share`, so validated energy-share means and
population SDs are multiplied by `2L+1` (variances by its square). This preserves
per-token division before token averaging. The CPU derivation checks this
identity against direct per-token energies on one saved sequence at every
layer. L0's unit Emb ratio is included only after checking positive alignment
onto nonzero V singular directions for every token/head in the matching
all-layer probe; this proves its full V output is nonzero.

The entry point freezes and executes its code under a fresh
`derived/source_mean_relative_energy/` directory. It records matching checkpoint
and token hashes, validated input hashes, L0 coverage, conversion errors, and
all-source/head statistics. Family plots retain that full-bank denominator,
select the two largest head-averaged source means separately for attention and
MLP at each receiving layer, and connect equally weighted means of 12 heads.
L0 has only Emb; L1 has only one available output in each family.
