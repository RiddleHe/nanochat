# Weighted loop selection: A/B screen

Read `scripts/train_weighted_loop_ab.py` first: it contains scoring, top-k weights,
residual application, balancing, the training objective, and the entry point.
The shared trainer handles data, optimizer, evaluation and exact resume.

| Variant | Choice per token/loop | Selected weights | Training objective |
| --- | --- | --- | --- |
| A | 12 of 24 | 12 × softmax(selected scores) | Language-model CE |
| B | Same as A | Same as A | CE + 0.01 × usage balance |

24 physical layers are shared over four loops; order is unchanged. A and B train
from the same seed42 initialization and data order. They differ only in the
balance coefficient. The scorer is a shared24×1536 matrix plus one bias vector
per loop. Weights apply to the whole block residual, not the carried hidden state.
Top-k membership remains discrete; selected weights use ordinary differentiation.
The coefficient sum12 does not match hidden-state or parameter-update norms.

Balance pools all four loops in each four-sequence microbatch. Assignment counts
are divided by tokens×loops×k; all-layer softmax probabilities are averaged over
tokens×loops. The coefficient-weighted term is logged separately from CE and is
never added to evaluation BPB. Selection counts and coefficient mass are logged
at the first step, evaluation intervals and the final step. Uniform usage is a hypothesis, not a success metric.

## Run after preflight and GPU allocation

The supplied recipes use two workers, context2048, microbatch4 per worker,
globalbatch1048576, BF16/FA3, and approximately3.91e19 estimated model FLOPs.
Set `NANOCHAT_BASE_DIR` to the immutable tokenizer/data root used by the references.
The loader uses the final sorted parquet shard for primary validation; confirmation
shards must be outside that directory. Freeze file/tokenizer/source identities first.

```bash
NANOCHAT_DTYPE=bfloat16 torchrun --standalone --nproc_per_node=2   -m scripts.train_weighted_loop_ab --config configs/weighted12/a.json --output runs-local/weighted12-a
NANOCHAT_DTYPE=bfloat16 torchrun --standalone --nproc_per_node=2   -m scripts.train_weighted_loop_ab --config configs/weighted12/b.json --output runs-local/weighted12-b
```

Run sequentially. `--stop-after 3` is a correctness pilot with the full schedule;
`--resume-checkpoint <run>/checkpoints/step_000003` restores a completed checkpoint.
Use new output directories for production. Do not change world size mid-run.
A one-worker adaptation needs verified virtual-rank data/evaluation parity; it is
not supplied by simply changing world_size. No training is launched by this PR.

## What the comparison can establish

B−A tests this balance term in one paired seed. A−the previous hard16 model changes
both k and the gating rule. Add a hard12 control only if a promising result needs
attribution. Compare final BPB (lower is better), curves vs model compute/tokens,
selection frequency, coefficient mass, latency and memory. A higher usage entropy
without better BPB is not evidence of improvement. Use new excluded confirmation
text and independent replication before making a general improvement claim.

## Efficiency boundary and next decision

All96 candidate blocks still compute. Keeping48 residual contributions does NOT
halve FLOPs or permit twice as many steps at the same budget. The estimator counts
all candidates. Scalar gating/balance, optimizer, evaluation and compilation costs
are not included in model FLOPs; report actual time separately.

Continue Idea2 as a bounded architecture screen, not an open-ended coefficient
search. Before full runs, check forward/gradient/cache/resume correctness and measure
cost. To pursue efficiency, a separate implementation must actually skip work.
A possible first target is computing MLPs only for selected token positions while
preserving dense attention K/V; correctness and GPU dispatch costs need measurement.
Skipping entire attention blocks per token changes available K/V and is not an
exact drop-in replacement. Do not mix that semantic change into the A/B comparison.

Success may be equal quality with lower inference cost, or fewer training FLOPs to
reach a fixed BPB, or better quality at fixed training FLOPs. Equal training FLOPs
and equal BPB with more steps alone is a tie, not yet a demonstrated efficiency
advantage. Report extra training tokens and actual throughput as well.

CPU tests cover mathematical/model behavior; Hopper BF16/FA3, multiworker updates,
and exact full-trainer A/B resume remain launch gates until run on the target stack.
