# Vocabulary V→O cosine

For each valid vocabulary embedding row, compute `(E @ W_V.T) @ W_O.T`
and its cosine with E. This combines all heads and excludes attention mixing,
residual addition, and centering. Positive scalar weighting and a common
receiver RMS divisor cancel from cosine. This is a matrix-path diagnostic,
not a contextual attention output or causal attribution.

The default configuration uses GPTBaseScalarResQKVNoEmbedNorm, step 7773.
Among candidate layers 8–23, select only those whose actual BF16-applied V
coefficient for Emb strictly exceeds all competing attention and MLP coefficients.
Save all selection decisions and verify them against the prior coefficient probe.

Every vocabulary row below config.vocab_size, including special tokens, has equal
weight. Raw embeddings and matrices are cast to BF16 then FP32. Sequential
projections and cosine reductions use FP32 with TF32 disabled. Zero input/output
norms produce undefined cosines. Do not round the intermediate V projection to
BF16: this intentionally measures the linear map of forward-cast matrices.

```bash
/hdd/mh3897/nanochat/.venv/bin/python -m scripts.inspect.vocab_vo_cosine.prepare --gpu 0
# Use the printed absolute RUN_DIR below.
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/analyze.py --run-dir RUN_DIR
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/plot.py --run-dir RUN_DIR --output ABSOLUTE_OUTPUT.html
```

Preparation freezes the checkpoint, metadata, tokenizer, analysis package and
model reference files. Analysis verifies hashes and uses the selected GPU UUID.
It stores all token IDs, cosines, input/output norms, validity masks, and descending
sort indices, with token-ID tie breaks. Validation compares sequential projections
with the composed matrix, the sum of head contributions, and FP64 reference
calculations on a deterministic vocabulary subset, in addition to analytic cosine
checks, full-population bounds/masks, and sorting.

Plotting sorts each layer independently. A percentile is therefore not a common
token across curves. Display-only adaptive decimation bounds absolute cosine
interpolation error by 1e-4; full unsorted and sorted-index measurements remain in
`derived/statistics.npz`. No smoothing is applied. Presentation provenance is
archived in `visualization/sorted_cosine_dark/`.
