# Input alignment and raw-source output energy

`prepare`, `compute_svd`, `collect`, and `summarize` are separate stages. Preparation
freezes model code, probe code, inputs, and configuration in a fresh run. Stage
CLIs re-execute from that snapshot. Rendering uses validated summaries and archives
its own source, template, settings, and input hashes under `visualization/`.

## ScalarResQKV without initial embedding RMSNorm

`scalar_res_config.json` selects final step 7,773, K/V receivers
12/14/16/18/20/22, and the 50% squared-singular-value threshold. Preparation
freezes the completed training run's model code/checkpoint, exact 128 × 2,048
reference tokens, tokenizer, and kernel dependencies. New model-specific SVDs
are required. Sources include raw Emb and all attention/MLP bank outputs;
future attention-family figures select Emb + attention from these measurements.

`scalar_res.py` observes every `_scalar_res` invocation and verifies source
identity and coefficient ordering. Collection checks bitwise-identical logits
against an uninstrumented forward and source norms from independent module hooks.
Alignment uses unweighted sources: nonzero scalar gates cancel from the ratio.
All head/source/token axes and undefined masks are retained.

Collection also saves `summaries/scalar_coefficients.npz` and its JSON schema:
active raw logits, FP32 gates, BF16-applied coefficients (stored losslessly as
FP32), and receiver offsets for Q/K/V, MLP, and final readout. These coefficients
are token/head independent, positive, unnormalized, and may exceed one. Padding
is excluded. `validation/scalar_coefficients.json` records the independent FP64
gate check; per-sequence bank checks verify their use in the real forward.

```bash
python -m scripts.inspect.qkv_input_alignment.prepare --config scripts/inspect/qkv_input_alignment/scalar_res_config.json --output-root /local-ssd/mh3897/analysis_runs/d24_scalar_res_input_alignment --gpu GPU_INDEX
python -m scripts.inspect.qkv_input_alignment.compute_svd --run-dir RUN_DIR --device cpu
python -m scripts.inspect.qkv_input_alignment.collect --run-dir RUN_DIR
python -m scripts.inspect.qkv_input_alignment.summarize --run-dir RUN_DIR
```

These commands collect and summarize only. Rendering is a separate requested
stage; no figure is created automatically. Probe regression checks:
`python -m pytest --import-mode=importlib tests/test_scalar_res_inspection.py scripts/inspect/qkv_input_alignment/test_alignment.py -q`.

## AttnRes QKV

`attn_res_config.json` selects checkpoint 7,738 and the exact saved validation
tokens from the AttnRes source-attribution reference. It collects source alignment
for all Q/K/V heads at the 50% squared-singular-value threshold, plus raw-source
squared output norms for V. The forward replay is shared by both measurements.

- Sources are the normalized embedding and attention/MLP module outputs in the
  actual routing bank. The adapter observes bank identity and preserves routing.
- Alignment is the per-token fraction of a source's squared norm in the selected
  head's input singular subspace. Nonzero scalar routing weights and receiving
  RMS denominators cancel. Zero sources have undefined alignment.
- `raw_energy_V_LL` is the per-token squared norm of the full head matrix applied
  to the raw source. It uses all output directions, without routing weights or
  receiving RMS normalization. Zero sources contribute zero. It is distinct from
  the weighted, normalized component norms in the attribution reference.
- Summaries average after measuring each token, preserve individual heads, and
  accumulate means and population standard deviations in FP64.
- Checks cover unchanged logits, bank identities, reference source norms, SVD
  reconstruction/orthogonality, alignment bounds and masks, and the identity
  between direct output energy and the full squared SVD-coordinate sum (first
  eight tokens in every source batch, every receiving V head).

From the repository root, using the research Python environment:

```bash
python -m scripts.inspect.qkv_input_alignment.prepare --config scripts/inspect/qkv_input_alignment/attn_res_config.json --output-root /local-ssd/mh3897/analysis_runs/d24_attn_res_qkv_input_alignment --gpu GPU_INDEX
python -m scripts.inspect.qkv_input_alignment.compute_svd --run-dir RUN_DIR --device cuda
python -m scripts.inspect.qkv_input_alignment.collect --run-dir RUN_DIR
python -m scripts.inspect.qkv_input_alignment.summarize --run-dir RUN_DIR
python -m scripts.inspect.qkv_input_alignment.plot_heatmap --run-dir RUN_DIR --output ALIGNMENT_HTML --layers 12 14 16 18 20 22 --kind V --family attention --threshold 0.5 --transpose
python -m scripts.inspect.qkv_input_alignment.plot_raw_energy --run-dir RUN_DIR --output ENERGY_HTML --layers 12 14 16 18 20 22 --kind V --family attention
```

Select an available GPU explicitly. The `attention` plotting family includes Emb.
Both plots show heads on Y and source layers on X, with one linear color scale per
figure. `plot_raw_energy` shares the rendering template in the sibling
`qkv_source_contributions` package; the plot archive includes that template.

GPTBase uses `config.json` / `source_50_90_config.json`. Its actual undivided-input
mode uses `full_input_config.json` / `full_input_50_90_config.json`; that mode is
currently rejected for AttnRes because its Q/K/V mixtures differ. Existing SVD
bases can be reused only with matching checkpoint provenance. A new alignment
threshold requires replay unless the required per-direction measurements exist.

## GPTBase without initial embedding RMSNorm

`no_embed_norm_config.json` selects the trained step-7809 variant, V receivers
12, 14, 16, 18, 20, 22, and the 50% squared-singular-value threshold. Preparation
adapts the validated `direct_v_energy` reference, freezes its checkpoint/model,
identical 262,144 token occurrences, tokenizer/kernel dependencies and current
probe code, and records validated reference source-norm chunks. No baseline SVDs
are reused. Only the requested receiving matrices are decomposed and measured.

Emb is the raw first-block input, checked exactly against the embedding lookup;
block/final RMSNorm remain in the unmodified BF16 forward. Alignment divides the
top-input-subspace squared norm by the full source squared norm per token.
Scalar receiver normalization cancels from this ratio. All available sources
are retained in measurements; the requested heatmap displays Emb + attention.

```bash
python -m scripts.inspect.qkv_input_alignment.prepare --config scripts/inspect/qkv_input_alignment/no_embed_norm_config.json --output-root /local-ssd/mh3897/analysis_runs/d24_no_embed_norm_input_alignment --gpu GPU_INDEX
python -m scripts.inspect.qkv_input_alignment.compute_svd --run-dir RUN_DIR --device cuda
python -m scripts.inspect.qkv_input_alignment.collect --run-dir RUN_DIR
python -m scripts.inspect.qkv_input_alignment.summarize --run-dir RUN_DIR
python -m scripts.inspect.qkv_input_alignment.plot_heatmap --run-dir RUN_DIR --output ALIGNMENT_HTML --layers 12 14 16 18 20 22 --kind V --family attention --threshold 0.5 --transpose --color-max-floor 0.07
```

The last option preserves the reference figure's 0–7% range when possible and
expands it if required, with no clipping. Each head retains its own cutoff `k`,
captured squared-singular-value fraction and isotropic reference `k/1536`.
Hover/tap includes mean, token population standard deviation and valid count.

Completed no-embedding-RMSNorm alignment run:
`/local-ssd/mh3897/analysis_runs/d24_no_embed_norm_input_alignment/gpt_base_no_embed_norm_step7809_20261103T010023Z`.
All 128 sequences passed unchanged-logit, raw-embedding identity and reference
source-norm checks. All 72 V-head SVDs passed reconstruction/orthogonality checks;
the 50% cutoff selects 31–51 input directions. Zero-norm source occurrences have
undefined alignment, excluded from means/stds with valid counts retained.

`no_embed_norm_all_layers_config.json` extends the same measurement to V receivers
L0–L23, all 12 heads, and every available Emb/attention/MLP source. It retains the
same checkpoint, 262,144 token occurrences, and 50% squared-singular-value cutoff.
Use this config with the preparation command above and a fresh output root.

Completed all-layer run:
`/local-ssd/mh3897/analysis_runs/d24_no_embed_norm_input_alignment_all_layers/gpt_base_no_embed_norm_step7809_20261106T015706Z`.
All 288 head SVDs and 128 sequence replay checks passed. Per-token squared
alignment is stored in `tokens/`; population means, SDs, and valid counts are in
`summaries/statistics.npz`. Projection-length ratios require taking the square
root of each token's alignment before aggregation. L0 has only Emb; at L1, A0
and the previous attention output are the same source. The combined scatter
and its derivation, input hashes, and browser checks are archived under this
run's `visualization/` directory.

## Alignment relative to the average source

`relative_alignment` defaults to `alignment_s / mean_valid_sources(alignment)`
from the saved per-token **squared projection ratios**, with no square root.
The denominator includes
Emb and every available prior attention and MLP output for that token and head,
including the source being measured. Normalize before averaging over tokens.
One means alignment equal to that head's average source; this is relative
geometry, not a share of projected V-output energy.

```bash
python -m scripts.inspect.qkv_input_alignment.relative_alignment --run-dir RUN_DIR --threshold 0.5 --metric squared
```

This CPU-only stage snapshots its code into a fresh
`derived/source_mean_relative_squared_alignment/` directory, then executes the
frozen copy. `--metric length` reproduces the earlier square-root transformation
and uses `derived/source_mean_relative_alignment/` instead. Use
`--output-dir FRESH` for another derivation. It saves FP64 population
means, SDs, and counts for every available source/head in `statistics.npz`,
denominator statistics and valid-source count ranges, plus selected Emb/A0/
previous-attention display data in `data.json`. Input hashes and per-token
source-average identities are validated. Undefined source alignments are
excluded from the denominator and source count; finite zero projections are
included. A zero denominator is undefined. L0 has mean 1 and SD 0 by
construction, and unavailable sources are omitted from display data.
Family-specific plots select attention or MLP sources from these statistics;
the denominator remains the full available bank, so Emb has identical values
in both views. Select the strongest sources within each family by the equal
mean of their per-head token means, then retain all heads for those sources.

### Static ScalarRes coefficients

`python -m scripts.inspect.qkv_input_alignment.plot_scalar_coefficients --run-dir RUN --output HTML`
renders saved Q/K/V gates in fixed light mode, ordered Emb, attention 0, MLP 0,
and so on. Use `--scale per-query-log` for independent Q/K/V logarithmic color
scales spanning each query's positive coefficient range, with separate colorbars
and a separate `visualization/qkv_scalar_coefficients_all_sources_per_query_log/`
archive. Use `--scale per-query-linear` for separate linear scales from zero to
each query's maximum, archived with a `_per_query_linear` suffix.
The default is `--scale shared-linear`.
It verifies coefficient/manifest hashes and receiver offsets, preserves
unnormalized coefficient values, and distinguishes FP32 gates from
BF16-applied coefficients on hover. No activation replay is required. Data,
renderer, template, and settings are archived under
`visualization/qkv_scalar_coefficients_all_sources/`. The companion
`check_scalar_coefficients.cjs ARCHIVE` checks a `preview.html` produced by the
inline visualization wrapper, including responsive labels and hover/pin details.
