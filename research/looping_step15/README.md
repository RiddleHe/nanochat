# Looping Step 1.5: local validation of relaxed sharing

Code-only package for Step 1.5, run 2026-10-26 on cci-p124 (3x H100 NVL).
Results, logs and checkpoints stay outside Git (see "Where results live").

## The question

Does giving each loop round its own adapter help at ONE location, more than a shared
adapter with the same parameter count? Do high gradient-conflict locations benefit more?

Scope: frozen checkpoint 19,074 of the 24-block x 4-round model; MLP expansion
(`mlp.c_fc`, 1536 -> 6144) only; one location adapted per run.

## Protocol (as run)

1. `gradient`: per-round weight gradients of all 24 expansions on D1 and D2
   (32 batches x 2 x 2,048 tokens each, Step 2's selections and probe).
   A = mean within-batch cosine; B = cosine of FP32 batch-mean gradients.
   C = -mean(B) over six round pairs. Selection frozen in `selection.json` before training:
   two highest C, two lowest C, two random of the remaining 20 (`default_rng(2026)`).
   Selected: high L14, L19; low L07, L10; random L04, L21 (D1/D2 Spearman 0.975).
2. 18 adapter runs: 6 locations x {shared-small rank 8 (61,440 params), independent
   4 x rank 8 (245,760), shared-large rank 32 (245,760)}. Base frozen; only A/B train.
   200 updates x 524,288 tokens, AdamW lr 1e-4, 20 warmup, cosine to 1e-5, seed 42,
   Step 1 training order. Dev eval at 0/50/100/200; confirmation at 200 only.
3. `scaling`: one layer's MLP output x {0.9, 1.1} in one round (48 conditions) on a
   256-sequence dev subset; best per location evaluated once on confirmation.
4. `baseline_eval`: unchanged model on dev and confirmation.

Data: Step 1 train/dev reused unchanged. A NEW confirmation set (2,048 sequences,
shard 6542 rows 13069-19934) excludes every Step 1 split and D1/D2 by exact-text hash,
because the Step 1 confirmation had already been inspected.

## Files

| File | Responsibility |
| --- | --- |
| `shared15.py` | Paths, frozen `PROTOCOL`, verification, data access, selection rule |
| `local15.py` | One-location LoRA (reuses Step 1 `LoopLoRA`) and one-layer one-round scaling |
| `run15.py` | Jobs: `gradient`, `L??_<variant>`, `scaling`, `baseline_eval` |
| `prepare15.py` | Reserve new confirmation; `--freeze` writes `manifest.json` after tests pass |
| `tests15.py` | `cpu` tiny-model checks (34); `gpu` real-checkpoint preflight |
| `launch15.py` | Queue on explicit idle GPUs, locks, no automatic retry |
| `threshold15.py` | Derive the minimum useful BPB gain from Step 1 results only |
| `analyze15.py` | Per-location outcomes, paired bootstrap CIs, classes under the frozen rule |

Reuses `step1_20261020/common.py`, `interventions.py` and `step2_early_20261025/probe.py`
by path; neither directory is modified.

## Run

```bash
cd /hdd/mh3897/cc/looping-experiments/step15_local_20261026   # code is copied here
PY=/hdd/mh3897/cc/nanochat/.venv/bin/python
CUDA_VISIBLE_DEVICES= $PY tests15.py cpu
CUDA_VISIBLE_DEVICES= $PY prepare15.py
CUDA_VISIBLE_DEVICES=0 $PY tests15.py gpu
CUDA_VISIBLE_DEVICES= $PY prepare15.py --freeze
CUDA_VISIBLE_DEVICES=0 $PY run15.py gradient
$PY launch15.py --gpus 0,1,2
```

Measured: ~28 s per update, ~1.7 h per run; 18 runs on 3 GPUs took ~10.5 h.

## Where results live

`/hdd/mh3897/cc/looping-experiments/step15_local_20261026/` on cci-p124:

- `selection.json`, `results/gradient/results.json`: A/B/C for all 24 locations, D1 and D2
- `results/L??_<variant>/`: config, training log, dev evals, `confirmation_200.json`, `results.json`
- `results/scaling/`, `results/baseline_eval/`
- `analysis/frozen_rule.json`: decision rule (delta = 0.5e-3 BPB), frozen before outcomes were opened
- `analysis/summary.json`: output of `analyze15.py`

## Headline (confirmation, update 200)

- Independent minus shared-large: shared-large better at all six locations by 20-30e-3 BPB;
  no relation to C (Spearman 0.03).
- Independent vs shared-small (same per-use rank): independent better by 5.4-7.8e-3 BPB
  everywhere; correlates with C (0.77) but equally with depth (0.71).
- Caveat: shared-large has 4x per-use rank and, without alpha/rank scaling, a larger
  effective step; dev BPB was still falling at update 200.
