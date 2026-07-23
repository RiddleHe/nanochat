# Finding-1 daily report — 2026-07-23 (postmortem: chunk deep-KV closed)

## Decisive seq-4096 result
| run | val_bpb |
|---|---|
| baseline @4096 | 0.86086 |
| chunk deep-KV v2-slim @4096 | 0.86295 (+0.00209) |

The bet was that the branch's benefit grows with sequence length while its
cost share stays fixed. It did not: the gap WIDENED vs 2048 (+0.00124 ->
+0.00209). Per the pre-registered rule, the chunk deep-KV architecture line
is CLOSED. No further variants.

## Why it failed (best supported reading)
- The branch's attention cost scales with sequence length (each shortcut layer
  compares against ALL previous chunks), so cost grew in step with, or faster
  than, any benefit.
- The per-token benefit measured at 2048 (+0.001, controls-verified) is real
  but small, and did not scale with length as hypothesized. Early layers may
  need capacity/training beyond d12 to exploit distant processed states, but
  we do not have evidence for that and are not spending more compute on it.

## What survives
- Scientific findings (all controls-verified, single-seed): early layers CAN
  use processed distant content (+0.001/token at equal tokens); the benefit is
  specifically from PROCESSED content (same-layer control is negative); exact
  single-pass chunk-recurrent training works (bitwise-equivalent at gate=0).
- Negative result, cleanly diagnosed: at d12, seq 2048-4096, augmenting early
  layers with previous chunks' final-layer KV does not pay for its own compute.
  Cost-benefit slope is negative per shortcut layer at both lengths.
- Reusable infra: chunk-recurrent trainer, elastic idle-GPU pipelines, the
  full checkpoint set for anyone re-analyzing.

## Next (already running, per team decision)
Route A "minimum global layers": d12 SSSL has 3 global layers (L3/L7/L11);
our measurements put d12's distant reads at L11 (some L7). Training now at
equal FLOPs: 2 globals (L7,L11) and 1 global (L11) vs baseline 0.85400.
If quality holds, that is a positive claim: same quality with 1/3 to 2/3
fewer global-attention layers, placement chosen by causal measurement.
