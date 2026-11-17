# Vocabulary embedding spectrum

This CPU diagnostic analyzes every valid vocabulary row of
`transformer.wte.weight` with equal weight, including special tokens. It excludes
padding beyond `model_config.vocab_size`. It does not analyze `lm_head`,
contextual activations, or the RMS-normalized `Emb` source used by other probes.

`prepare.py` freezes the relevant checkpoint tensor, all vocabulary IDs, metadata,
analysis code, checkpoint hash, and environment versions in a fresh run directory.
`analyze.py` verifies those hashes and computes direct FP64 SVDs of the uncentered
and column-centered tables. Run the latter from the frozen copy:

```bash
python -m scripts.inspect.vocab_embedding_spectrum.prepare \
  --checkpoint /absolute/path/model_007809.pt \
  --metadata /absolute/path/meta_007809.json \
  --run-dir /absolute/path/fresh_run
python /absolute/path/fresh_run/code/analyze.py \
  --run-dir /absolute/path/fresh_run --threads 16
```

Numerical rank uses `max(shape) * eps(float64) * largest_singular_value`.
Stable rank is `sum(s**2) / max(s**2)`. The explicitly named energy-entropy rank
is `exp(-sum(p*log(p)))`, and participation ratio is `1/sum(p**2)`, with
`p=s**2/sum(s**2)`. These describe the stored quantized table and do not establish
semantic or causal dimensionality.

PCA subtracts the mean vocabulary vector, without standardizing coordinates.
Eigenvalues are `centered_s**2/(vocab_size-1)`. Threshold counts are the strict
minimum number of components reaching 50%, 80%, 90%, 95%, and 99% cumulative
variance; they are not tie-extended subspace selections.

`derived/spectrum.npz` stores both singular spectra, the mean, all principal
directions as rows of `components`, and individual/cumulative explained variance.
`results.json` stores summary metrics. `validation/summary.json` records SVD
energy identities, component orthogonality, and reconstruction on 128 evenly
spaced vocabulary rows. Numerical errors must be below 1e-10. No corpus,
tokenizer decoding, GPU inference, or rendering is needed for this weight probe.
