# Full-validation raw activation norms

GPTBase and GPTBaseNoEmbedNorm D24 step 7,809, with the unmodified BF16 forward. Each source has 1,536
coordinates. Sources are the input embedding **before initial RMSNorm**, then
attention and MLP outputs from layers 0–23, both before residual addition.
These are write/source norms, not residual-stream norms. No centering is applied.

ScalarResQKVNoEmbedNorm uses `scalar_res_config.json` at final step 7,773.
Its raw Emb and attention/MLP outputs are measured before scalar receiver
weighting and RMSNorm. Attention returns a tensor; the collector preserves its
batch axis, while also supporting the tuple output of earlier models. Model code
is frozen from the completed training run, checked against its source hashes.

```bash
python -m scripts.inspect.raw_activation_norms.prepare --gpu GPU_INDEX --config scripts/inspect/raw_activation_norms/scalar_res_config.json --token-reference /local-ssd/mh3897/analysis_runs/d24_raw_activation_norms/gpt_base_step7809_fullval_20261102T195302Z
python -m scripts.inspect.raw_activation_norms.run --run-dir RUN_DIR --stage collect
python -m scripts.inspect.raw_activation_norms.run --run-dir RUN_DIR --stage summarize
```

The ScalarRes run reuses all 54,053,190 reference occurrences and identical
context boundaries, including the final partial context. The microbatch is two
sequences; population moments are independent of this batch grouping. The
commands above stop after validated numerical summaries. Plotting is separate;
its Emb label must state no initial RMSNorm, and any sqrt(width) line is a
theoretical reference only.

`prepare.py --gpu INDEX` requires an idle GPU and freezes model/metadata, the held-out
Parquet shard, tokenizer, model and diagnostic code, package versions, and FA3
kernel dependencies. It executes tokenization from the frozen code. The run's
`environment_overrides.json` must be applied when executing frozen modules under
`code/`. Remove inherited distributed rank variables and PYTHONPATH.

For the no-initial-RMSNorm model, select `--config
scripts/inspect/raw_activation_norms/no_embed_norm_config.json`. Add
`--token-reference BASELINE_RUN` to copy the validated baseline's complete token
stream, offsets, frequencies, tokenizer, shard and kernels under verified hashes;
the context length and token policy must match. Its raw embedding is also its
first-block input. The ideal sqrt(1536) horizontal line is a reference only.
`run --run-dir RUN --stage collect` and `run --run-dir RUN --stage summarize`
launch the corresponding modules through the frozen code/environment and archive
console logs. All entry points are modules under `scripts.inspect.raw_activation_norms`.

Tokenization reads every document in Parquet row order, prepends one BOS per
document, and concatenates all documents into `inputs/token_ids.u32`. Document
offsets and vocabulary frequencies are saved. Non-overlapping contexts have
2,048 tokens; the final shorter context is retained without padding. Every token
is observed exactly once, including tokens that would only be targets in a
shifted training batch. Context resets occur at fixed stream offsets; contexts
can span document boundaries and need not start with BOS. This differs from
the sampled BOS-bestfit probes, which crop document tails.

`collect.py --run-dir RUN` verifies frozen hashes, runs two full-context checks
requiring bitwise-identical logits with/without hooks, and checks raw embedding
norms against direct table lookup. Read-only hooks compute FP32 full-vector L2
norms. Chunk files contain `[token_occurrence, source]` scalar norms; JSON sidecars
identify exact contiguous token ranges, sequence shapes, and hashes. Resume
requires matching manifest, inputs, frozen code, dependencies, and saved chunks.

`summarize.py --run-dir RUN` verifies all chunks, gapless coverage, document BOS
offsets and vocabulary frequencies. FP64 parallel moment merging computes the
population mean, variance (`ddof=0`) and standard deviation. An independent
frequency-weighted FP64 embedding-table calculation validates its mean/variance.
The precision difference from FP32 GPU norm computation is bounded explicitly.

`plot.py --run-dir RUN --output /absolute/inline.html` consumes validated results.
Add `--y-scale log` for a logarithmic y-axis. Its positive floor is derived from
the visible positive band bounds; lower bounds at or below zero are clipped at
that explicitly labeled floor. Measured arithmetic means and SDs are unchanged.
Between points, the hover mean follows the line in the chosen axis coordinates.
Use `--families attention` to show only the raw embedding and attention outputs.
Add `--embedding-references` for dashed horizontal references at the measured
raw embedding mean and the ideal post-RMSNorm L2 norm, sqrt(1536). The latter
ignores small epsilon and BF16 rounding departures and is labeled approximate.
The single dark chart defaults to linear axes, a shared scale, separate attention/MLP
lines, and shaded mean ±1 population SD. The y-axis rescales to the visible
series when a legend item is toggled. Negative lower band bounds are clipped
at zero and labeled if clipping occurs. Emb is the first attention-series point;
the MLP series begins at layer 0. Hover retains variance, SD and count. Between
layers, hover interpolation is labeled; tap/click pins exact layer statistics.
Bands show token variation, not confidence intervals. Plot data, code, settings
and hashes are archived independently under the run's `visualization/`.

## Completed reference

`/local-ssd/mh3897/analysis_runs/d24_raw_activation_norms/gpt_base_step7809_fullval_20261102T195302Z`
contains 54,053,190 token occurrences across 84,992 documents (including one BOS
per document), 26,393 full contexts and a final 326-token context. Collection
took 350 seconds on GPU UUID `GPU-21c8ffaa-d17a-cdfe-7210-ec8462eb9baf`.
Coverage, forward equivalence, and independent embedding checks passed.
The validated chart is archived in `visualization/mean_norm_with_std_51c406efea52/`.
The final MLP output dominates the shared linear scale; toggling MLP off rescales
the embedding/attention view. Earlier plotting drafts remain archived separately.

## Completed no-initial-RMSNorm run

`/local-ssd/mh3897/analysis_runs/d24_raw_activation_norms/gpt_base_no_embed_norm_step7809_fullval_20261103T021543Z`
uses exactly the reference's full token stream and context boundaries. All
54,053,190 occurrences are covered. Collection took 346.5 seconds on GPU UUID
`GPU-68dc475f-848f-5a98-5166-ef16491754b4`. Forward, coverage and independent
embedding checks passed. Emb mean/std are 279.167183/63.624669; attention-0
mean/std are 707.394077/38.005879. The ratio of their mean norms is 2.533944.
The attention-only log chart includes the measured embedding-mean reference
and the ideal RMSNorm reference, which is not applied to this model's embedding.

`plot_compare --baseline-run BASELINE_RUN --variant-run VARIANT_RUN --output
/absolute/comparison.html` overlays the raw Emb + attention curves with SD bands
and no horizontal references. It defaults to log Y, verifies identical full token
streams and context policies, and reads validated summaries without replay.
Add `--family mlp` to compare Emb + MLP outputs using the same conventions.

`normalized_embedding --run-dir BASELINE_RUN --gpu INDEX` derives post-initial-
RMSNorm embedding norms using the frozen baseline model's actual BF16 operation,
including its epsilon and rounding. It weights every vocabulary entry by the
full validation token frequencies and checks exact agreement against first-block
input norms in two actual forward contexts. Code, environment, per-vocabulary
norms and validation are saved in a fresh `derived/normalized_embedding_*` folder.
Pass its `statistics.json` to `plot_compare --normalized-embedding PATH` to place
that measured value and SD at GPTBase's Emb point, retain the variant's raw Emb,
and show one dashed reference for GPTBase's raw embedding mean.
