# Target-layer self-attention

Probe GPTBaseScalarResQKVNoEmbedNorm checkpoint 7773 at the original vocabulary
VO target layers: L9–15, L18, L21–23. Inputs are the saved 128 deterministic
BOS-bestfit validation sequences of length 2048, including internal BOS tokens.
Sequence position 0 is recorded but excluded from summary statistics because
it has only itself available. No extra document-boundary mask is introduced.

The unchanged model supplies BF16 Q/K after RoPE, per-head RMSNorm, and 1.2 scaling
to the frozen FA3 backend. Request its normal output plus FP32 softmax LSE, and
return the normal output onward. Compute diagonal logits by an FP32 dot product
scaled by 1/sqrt(128); probability = exp(diagonal_logit - LSE). No additional
normalization or clipping is applied. Minor numerical overshoots are checked
against a fixed tolerance and retained. No Q/K or full attention matrices are saved.

Validation checks bitwise-equal logits against an uninstrumented forward for all
128 sequences. At sequences 0/64/127, check every target/head against explicit
FP64 masked softmax at nine query positions including the sliding-window boundary;
compare LSE, diagonal probability, row sums, masked entries, and bitwise-identical
FA3 outputs when requesting LSE. Fixed probability/LSE absolute tolerances: 2e-5.

```bash
/hdd/mh3897/nanochat/.venv/bin/python -m scripts.inspect.self_attention_mass.prepare --gpu 0
# Use the printed absolute RUN_DIR for all stages:
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/probe/collect.py --run-dir RUN_DIR
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/probe/summarize.py --run-dir RUN_DIR
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/probe/plot.py --run-dir RUN_DIR --output-dir ABSOLUTE_OUTPUT_DIR
```

Run provenance freezes all inputs, model/probe code and FA3 cache. Each sequence
chunk stores float32 [11 target layers, 2048 positions, 12 heads], about 132 MiB
total. Inputs retain token IDs and target selection. FP64 summaries preserve
per-head mean, population SD, and count; the curve equally averages head means.
The dark inline plot uses a full percentage scale, small deterministic horizontal
head offsets, and token SD/count on hover. Its data, code, settings, hashes and
browser checks belong in the run's visualization directory.

## First-token attention sinks

Prepare with `--config scripts/inspect/self_attention_mass/sink_config.json` and
`--output-root /local-ssd/mh3897/analysis_runs/d24_first_token_attention`.
Then use frozen `collect_sink.py`, `summarize_sink.py`, and `plot_sink.py`.
All 24 layers and 12 heads are measured. Save A[t,0] and A[t,t] in FP32;
A[t,0] is zero where the first key is outside the actual sliding window.
Exclude t=0 from averages. Store sequence means, token population SD, counts,
first-key availability and eligible-key count. Conditional first-token means
when visible are retained separately. The main chart uses unconditional means
including masked zeros, compared with mean attention over available keys (1/n).
The second comparison is identical across heads by definition. All input windows
are 2048 tokens, so the mean sequence length is 2048, not mean document length.
Self-attention and mean over non-first eligible keys are also available in
summaries for later comparisons. Explicit validation checks both saved columns.

## Self-attention excluding keys 0–3

Prepare with `--config scripts/inspect/self_attention_mass/exclude_prefix_config.json`
and `--output-root /local-ssd/mh3897/analysis_runs/d24_self_attention_exclude_prefix`.
Use frozen `collect_exclude_prefix.py`, `summarize_exclude_prefix.py`, then
`plot_exclude_prefix.py`. The original model forward and BF16 FA3 output remain
unchanged. Actual Q/K and FA3 LSE yield raw diagonal probabilities and the mass
on absolute keys 0–3. For queries t>=4, divide raw self probability by remaining
mass before averaging. Queries 0–3 are undefined and excluded; t=4 is included.
Use FP64 dot products and expm1 subtraction; if remaining mass <0.1 or the ratio
is invalid, compute remaining-key logsumexp directly in FP64, with all masks.
Save raw/adjusted probabilities, pre-fallback remaining mass, and fallback flags.
Validate against explicit remaining-key FP64 softmax at the recorded positions
on sequences 0/64/127 across all heads/layers (absolute adjusted tolerance 1e-4).
Uninstrumented logits must match bitwise on all 128 sequences. Uniform baseline
is 1/remaining eligible-key count per query, then averaged over identical queries.
Shapes distinguish the original target layers (triangles) from others (circles).
The layer curve equally averages heads; the dashed uniform curve is analytic.
