# Looping Step 1: local diagnostics and controlled LoRA adaptation

This directory packages the three experiments completed on 2026-10-20.
It does not change the baseline architecture, training code, checkpoint, or data.
The target branch is `reasearch-looping` (spelling intentional), not `master`.

This is a code-only package. Results, logs, figures, PDF documentation, datasets,
and checkpoints are stored separately and are not tracked in this directory.

## The question

Do different uses of a shared Transformer block need different parameters?
We measured local gradient directions, output-scale sensitivity, and the effect
of giving each loop its own low-rank adapter.

This is a single-seed, final-checkpoint screen, not an early-training predictor
or a matched-total-FLOPs comparison.

## File map

| File | Responsibility |
| --- | --- |
| `common.py` | Fixed settings; verified baseline loading; data and evaluation helpers |
| `prepare.py` | Deterministic packing, exact-text deduplication, and source/data freeze |
| `interventions.py` | Per-use gradient hooks, output scaling, and per-loop/shared LoRA |
| `experiments.py` | The three experiment routines and original-model evaluation |
| `launch.py` | Three GPU workers, locks, shared LoRA queue, and completion plots |
| `tests.py` | Tiny-model CPU tests and real-checkpoint GPU preflight |
| `test_statistics.py` | Direction examples, mean-order counterexample, optional external-result audit |
| `plot_results.py` | Five measured-result figure families; no model required |
| `analyze_results.py` | Validate evaluation records; paired confirmation comparisons |

The baseline implementation is `nanochat/model/gpt_loop.py`, built on
`nanochat/model/gpt_base.py`. It is not reimplemented here.

## Shared setup

- Baseline commit: `ad896a41c7e0426d7c4d349d3e4f64f3c9ebe421`.
- Checkpoint: step 19,074; **24 physical blocks x 4 loops = 96 block executions**.
- Width 1,536; 780,140,569 unique parameters; 12 Q/K/V heads; context 2,048.
- Original pretraining budget: 10,000,269,312 prediction tokens.
- Full causal attention, final-loop cross-entropy only, no early-exit gate.
- Smear embedding gate retained; no value embeddings, x0 reinjection,
  attention-residual mechanism, or intermediate-loop auxiliary loss.
- BF16 activations and matrix multiplies; FP32 master weights and loss/statistics.
- Original run: PyTorch 2.10.0+cu128, deterministic FA3 backward on H100.
- All experiments use the same fixed checkpoint; original parameters do not update.

### Data and what a sequence means

A sequence is an ordered list of token IDs, not a hidden-state vector. Each
packed record stores 2,049 IDs. The first 2,048 are inputs; the last 2,048 are
the one-token-shifted targets. Multiple documents may occur in one record,
separated by BOS tokens; attention is not reset at document boundaries.

Source: the baseline's ClimbMix shards. Diagnostics use shard 300; adaptation
uses an ordered stream from shards 301-319, stopping when enough tokens are
packed. Development and confirmation use separate document selections from
validation shard 6542. Exact document-text hashes are deduplicated across all
experiment splits; this is not a near-duplicate or pretraining-contamination audit.
The last document's unused suffix is discarded at each split boundary.

This packing differs from baseline pretraining. Compare models on the SAME
Step 1 selection; do not compare these BPBs directly to old training-log BPBs.

## Experiment 1: per-use gradient contributions

32 fixed mini-batches x 2 sequences x 2,048 tokens = 131,072 prediction tokens.
For each batch, backpropagate the mean final-output token loss once. Capture
the contribution of each of four uses of each of 144 matrices (24 layers x
Q/K/V/O/MLP expansion/MLP projection). No optimizer step is taken.

For a linear operation `Z = X W^T`, flatten batch and token positions into rows:
`G_r,b = (dL_b/dZ_r)^T X_r`. This is a matrix-shaped contribution to the shared
weight gradient, not the hidden-state gradient. The token-loss average is
already included in `dL_b/dZ_r`; do not divide by token count again. The hook
matches the forward dtype, then computes statistics in FP32.

Flatten each gradient MATRIX into a vector without averaging its entries.
Within batch b, compute `c_b(r,s) = dot(g_r,b, g_s,b) / (norm(g_r,b) norm(g_s,b))`.
The heatmap displays `mean_b c_b`, NOT `cos(mean_b g_r,b, mean_b g_s,b)`.
The second figure averages within-batch norm shares. Small- and large-gradient
batches have equal weight in the cosine mean. Invalid near-zero norms are
masked.

Blue is aligned (+1); white is orthogonal (0); red is opposed (-1).
Rows are physical layers, columns are loop pairs, and panels are matrix types.
These are raw gradient directions, not Adam-preconditioned update directions.
Negative cosine describes local contributions; it does not prove that untying helps.

Mean-gradient cosine was NOT measured. It cannot be recovered from the stored
per-batch cosines and norms; new gradient measurement, not retraining, is needed.

The sum of four contributions is checked against autograd's shared-weight
gradient on every batch.

## Experiment 2: output-scale sensitivity

Use 128 fixed sequences (262,144 prediction tokens), no weight updates.
Test 4 loops x 2 module classes (attention/MLP) x 2 scales (0.9/1.1) = 16 conditions.
Scale the selected module's output before residual addition in ALL 24 blocks of
ONE loop. This is not a layer-local intervention and not a change to the entire
hidden state. Include the unmodified baseline and an alpha=1.0 sham, checked
bitwise on the first two sequences.

Report the change in final cross-entropy on identical sequences, with 2,000
paired sequence-bootstrap draws (seed 42) for descriptive 95% intervals.
Negative change is better. Intervals are not multiplicity-adjusted and do not
measure training-seed variability. Output-scale gains do not establish sharing harm.

## Experiment 3: controlled LoRA adaptation

For each of Q/K, V/O, and MLP, insert adapters into all 24 blocks:
`y = W x + B_r A_r x`, with W frozen. A compresses to rank k and B maps back
to the output width. Only A and B update; gradients still pass through W.
All B matrices start at zero, so the starting model is exactly the baseline.
Independent copies start with matching A values, not independently chosen noise.

| Variant | Q/K or V/O rank | MLP rank | Added trainable parameters |
| --- | ---: | ---: | ---: |
| Shared small | 20 | 8 | 2,949,120 |
| Independent per loop | 20 per loop | 8 per loop | 11,796,480 |
| Shared large | 80 | 32 | 11,796,480 |

Each group targets two matrices per block. Equal parameter counts across groups
come from the different ranks and matrix shapes. Shared-large uses 4x the
adapter-forward matrix FLOPs of independent, NOT 4x whole-model FLOPs. Do not
describe these runs as matched-total-training-FLOPs experiments.

- 9 runs; 200 updates each; 524,288 tokens/update; 104,857,600 tokens/run.
- One GPU/run; micro-batch 2 x 2,048 tokens; accumulate 128 micro-batches/update.
- AdamW: lr 1e-4, betas (0.9,0.95), epsilon 1e-8; 20-update warmup, then cosine
  decay to 1e-5. No weight decay, dropout, or finite gradient clipping.
- Seed 42, identical training record order, final-loop mean token CE only.
- Evaluate/save at updates 0/50/100/200 on 4,194,304 development tokens.
- Final update 200 is evaluated on a separate 4,194,304-token confirmation set.
  Do not select a different checkpoint using confirmation scores.
- Evaluate the original model on both exact selections as the common reference.

BPB = sum of loss in nats on byte-bearing target tokens / (ln(2) x target bytes).
Zero-byte special-token losses are excluded from the BPB numerator. CE includes
all prediction tokens. Paired confirmation intervals resample the same sequence
indices for both models and recompute the ratio of sums, not an unweighted mean
of per-sequence BPBs.

Gains over the original model include extra parameters and extra training,
not just relaxed sharing. Compare independent adapters with both shared controls.

## Inspect and reproduce the analysis (CPU only)

From the repository root, test without downloading results:

```bash
python research/looping_step1/test_statistics.py
python research/looping_step1/tests.py cpu
```

The first command uses only the standard library; its measured-data audit is
skipped unless an external run directory is supplied. The last command needs
PyTorch 2.10 and the repository CPU dependencies; it runs 41 tiny-model checks.

To analyze a completed run, install NumPy and Matplotlib and point to its
`results/` and `experiment_manifest.json` (not included in Git):

```bash
python research/looping_step1/test_statistics.py --input /absolute/path/to/completed-run
python research/looping_step1/plot_results.py --input /absolute/path/to/completed-run
python research/looping_step1/analyze_results.py --input /absolute/path/to/completed-run
```

Plots go to `research/looping_step1/runs/local/figures`; comparisons and analysis
go to `research/looping_step1/runs/analysis`. These outputs are ignored by Git.

## Reproduce measurement/training (explicit GPU action)

The original baseline checkpoint, tokenizer, frozen `code/`, dataset shards,
and baseline `manifest.json` are required. They are intentionally not in Git.
Use a NEW output directory; never point it at the completed frozen run.

```bash
export LOOPING_BASELINE_DIR=/absolute/path/to/b24_r4_10b_20261015
export LOOPING_STEP1_DIR=/absolute/path/to/a-new-step1-run
python research/looping_step1/prepare.py
python research/looping_step1/tests.py cpu
CUDA_VISIBLE_DEVICES=0 python research/looping_step1/tests.py gpu
python research/looping_step1/prepare.py --freeze
python research/looping_step1/launch.py
```

Only run the last two GPU-using commands when GPU access is approved and those
devices are free. The launcher is intentionally limited to GPUs 0/1/2. First
jobs are gradients on 0, sensitivity on 1, and Q/K shared-small on 2. Finished
workers claim the remaining LoRA jobs, then baseline evaluation. Lock checks
prevent duplicate cooperative launches; no failed job is automatically retried.
This is a same-host cooperative lock, not a cluster reservation service.

The packaged launcher uses the active Python interpreter. Source and run-output
directories are now separate; paths come from the two environment variables.
Numerical settings and intervention math are unchanged. Original-run GPU tests
are historical evidence, not a fresh H100 test of this packaging revision.

## Limits and next decision

One pretraining/adaptation seed, one final checkpoint, one shared optimizer
schedule, all-layer module interventions, and no matched-total-compute control.
Gradient disagreement is batch-scale-dependent. Confirmation intervals do not
account for seed variability, document clustering, or all exploratory choices.
The next hypothesis is whether a LOCAL high-conflict position benefits from
untying more than a low-conflict control, after controlling for output scaling.
No follow-up GPU experiment is started automatically by this package.
