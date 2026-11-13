# Per-loop Muon (12 blocks x 4 loops)

Does it help to orthogonalize each loop's gradient separately before combining the 4 loops?
This also asks whether the direction of the update or its size matters.

## Variants

The baseline is native Muon: momentum(G) → Polar Express (PE) → NorMuon, where G = g1+g2+g3+g4.

| mode | update for each shared matrix |
| --- | --- |
| `native` | the baseline (used only to check the code) |
| `v1` | per-loop momentum(g_k) → PE per loop → sum → NorMuon → rescale to ‖A‖ |
| `v2` | per-loop momentum(g_k) → PE per loop → sum → PE → NorMuon (no rescale) |

- A is the native update at the same step, from a shadow native path that keeps its own
  momentum and NorMuon state.
- v1 rescales B to ‖A‖ for each matrix, so v1 changes only the direction of the update.
- lr × sqrt(max(1, rows/cols)) and cautious weight decay are unchanged. AdamW (embeddings,
  lm_head, smear) is unchanged.
- Logged every step: ‖B‖/‖A‖ before rescaling and cos(A, B). Per-matrix values are written
  every 50 steps to `matrix_stats.jsonl`.

## Matched to the baseline

The baseline is `/hdd/mh3897/cc/looping-runs/b12_r4_10b_20261019`. It was not retrained.

| | value |
| --- | --- |
| Init | baseline `model_000000.pt` |
| Steps | 19,074 |
| Batch | 524,288 tokens |
| Data | baseline data in the same order |
| Schedules | LR, momentum and weight decay (0.084483) as in the baseline |
| Compute | 3.095e9 FLOPs/token, 3.095e19 FLOPs total (same as the baseline) |

- Extra optimizer work (5 PE passes instead of 1) is about 1e13 FLOPs per step, under 0.01% of
  the 1.6e18 FLOPs per step of forward and backward.
- The baseline ran on 2 GPUs. On 1 GPU the script reads the data as rank 0 and rank 1 did,
  runs both ranks' 8 microbatches, and divides by 16. This is the same as DDP's average.
- Validation reads both rank views, so val BPB can be compared directly with the baseline's
  `train.log`.

Per-loop gradients come from an unrolled twin model with 48 block copies. Its forward is the
same as GPTLoop's, and the copies are re-synced to the 12 shared blocks after every step.
Checkpoints are standard GPTLoop state dicts, so `scripts.base_eval` works on them unchanged.

## Files

| File | Responsibility |
| --- | --- |
| `perloop.py` | twin model and sync; Muon stages split from `muon_step_fused`; `PerLoopMuonAdamW` |
| `train_perloop.py` | 1-GPU training matched to the baseline; eval; checkpoints; exact resume |
| `tests_perloop.py` | `cpu`: math checks on a tiny model; `gpu`: reproduce baseline steps 0–9, v1/v2 smoke, resume |

## Run

```bash
cd /hdd/mh3897/cc/looping-experiments/perloop_muon_20261112
export PYTHONPATH=/hdd/mh3897/cc/looping-runs/b12_r4_10b_20261019/code \
       NANOCHAT_BASE_DIR=/hdd/mh3897/cc/looping-runs/b12_r4_10b_20261019 NANOCHAT_DTYPE=bfloat16
PY=/hdd/mh3897/cc/nanochat/.venv/bin/python
CUDA_VISIBLE_DEVICES= $PY tests_perloop.py cpu
CUDA_VISIBLE_DEVICES=4 $PY tests_perloop.py gpu preflight/
# one GPU per run, in tmux:
CUDA_VISIBLE_DEVICES=4 $PY -u train_perloop.py --mode v1 --out runs/v1 2>&1 | tee runs/v1.log
CUDA_VISIBLE_DEVICES=7 $PY -u train_perloop.py --mode v2 --out runs/v2 2>&1 | tee runs/v2.log
# resume after a crash: add --resume <last saved step>
```

- Each run is expected to take about 24 h on 1 H100. The baseline took 11.7 h on 2 GPUs.
- Models are saved at steps 10, 50, 100, 250, every 500 steps, and the final step.
- Optimizer and data state are kept only for the latest checkpoint.
