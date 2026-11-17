# Vocabulary VO output PCA

Uses the completed `vocab_vo_cosine` run as a frozen parent. The raw embedding
reference and the same 11 selected layers use all 32,768 valid vocabulary rows,
including special tokens, with equal weights. Reproduce `(E @ W_V.T) @ W_O.T`
in the parent's FP32 batches from BF16-cast matrices; no attention mixing,
residual addition, receiving scalar, or output normalization.

Compute FP64 means and both explicitly centered and uncentered Gram matrices.
Symmetric eigendecomposition gives squared singular values and direction vectors.
This avoids storing tall left singular-vector matrices. Negative eigenvalues
within a recorded FP64 roundoff tolerance are clipped to zero; extremely small
singular values are less accurate than direct SVD. Threshold counts and effective
ranks use the full nonnegative energy spectrum, without a numerical-rank cutoff.

Each NPZ stores the mean vector, mean-energy fraction, both singular-value spectra,
fractions, cumulative fractions, directions (rows), and centered PC variances.
Variances use N-1. Fractions use the sum of squared singular values. Mean-energy
fraction is N*||mean||²/||Y||². Effective rank is exp(-sum(p log p)); participation
rank is 1/sum(p²). Counts use the smallest k reaching 50/90/99%, without tie extension.
Full output matrices and token PC scores are not stored.

```bash
/hdd/mh3897/nanochat/.venv/bin/python -m scripts.inspect.vocab_vo_pca.prepare --gpu 0
# Substitute printed absolute RUN_DIR:
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/analyze.py --run-dir RUN_DIR
/hdd/mh3897/nanochat/.venv/bin/python RUN_DIR/code/plot.py --run-dir RUN_DIR --output-dir ABSOLUTE_OUTPUT_DIR
```

Analysis executes frozen code on exactly one selected GPU. Inputs include a copy
of the parent's checkpoint, tokenizer, selected-layer decisions, and measurements.
Validation checks reproduced token norms/cosines for layers available in the parent, spectral
energy and centering identities, orthogonality, eigen residuals, reconstruction,
and a direct FP64 SVD comparison on a deterministic row subset. All tolerances
and errors are recorded. Plots show all 1,536 directions on a logarithmic x-axis
and shared linear [0,1] y-axes. The CSV table and presentation provenance live in
`visualization/cumulative_spectra_dark/`. Replotting does not recompute spectra.

For all 24 layers, prepare with `--config scripts/inspect/vocab_vo_pca/all_layers_config.json`.
Run the frozen analyzer as above, then frozen `plot_grouped.py` with the same
`--run-dir` and `--output-dir` arguments. Its centered PCA chart uses three
legend groups: Emb, Target (original selected L9–15, L18, L21–23), and Non-target
(all remaining layers). Target retains the original L8–23 candidate restriction.
All layers are independently checked against FP64 composed VO projections on the
recorded token subset (relative L2 error below 2e-6). The original 11 also retain
full-vocabulary parent norm/cosine checks. Presentation provenance is archived in
`visualization/centered_grouped_dark/`.
