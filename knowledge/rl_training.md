# RL training

A small RL training path for code-generation tasks. The reference task is
`rstar_seed`, built from the `microsoft/rStar-Coder` `seed_testcase` split.

## Files

- `nanorl/data.py` — JSONL dataset loader, distributed prompt loader, reward worker pool, code verifier
- `nanorl/sandbox.py` — sandboxed Python subprocess execution for reward computation
- `nanorl/rollout.py` — rollout helpers, logprob computation, batch packing, weight sync
- `nanorl/loss.py` — GRPO, DAPO, REINFORCE
- `nanorl/scripts/train.py` — main trainer
- `nanorl/scripts/rollout_worker.py` — standalone HTTP rollout worker (for `remote_vllm`)
- `nanorl/runs/train.sh` — launcher for remote-vLLM runs

## Data

The canonical JSONL is referenced by absolute path from
`_RL_DATASET_PATHS` in `nanorl/data.py`. On this machine:

```
rstar_seed / train → /local-ssd/mh3897/data/rl/rstar_seed_train_filtered.jsonl
```

The file is pre-filtered: the top-20% of rows by byte size were dropped to
cap per-row payload (the original had rows with 1 GB+ stress-test stdin
inputs). Result: 23,349 rows, 4.2 GB on disk, ~5 GB resident when
eager-loaded.

Each row:

```json
{
  "id": "rstar/seed_123",
  "prompt": "<chat-formatted prompt>",
  "kind": "code_call_based",
  "payload": {
    "inputs": [...],
    "outputs": [...],
    "fn_name": "add",
    "time_limit_s": 4.0,
    "memory_limit_mb": 256
  },
  "meta": { "source": "rstar_seed", "n_tests": 10 }
}
```

To add a new task, add an entry to `_RL_DATASET_PATHS` and write a prep
script that produces a JSONL in this schema.

## Running training

Rollout generation runs in a separate vLLM worker process on its own GPU;
the trainer runs under `torchrun` across the remaining GPUs. Use the
launcher:

```bash
MODEL=Qwen/Qwen3-0.6B \
ROLLOUT_GPU=7 \
TRAIN_GPUS=4,5,6 \
NUM_STEPS=200 \
PROMPTS_PER_STEP=12 \
NUM_SAMPLES=4 \
TRAIN_BATCH_SIZE=4 \
MAX_NEW_TOKENS=256 \
REWARD_WORKERS=8 \
K_TESTS=10 \
EVAL_EVERY=20 \
./nanorl/runs/train.sh
```

The launcher starts a rollout worker on `ROLLOUT_GPU`, waits for its
`/health`, runs `torchrun` across `TRAIN_GPUS`, and tears the worker down
on exit.

The trainer and the worker communicate over localhost HTTP (default
`127.0.0.1:8047`). The worker owns one vLLM engine; the trainer only
consumes it.

## Strict-sync semantics

1. Step `t` rollouts use weights `W_t`
2. Trainer updates to `W_{t+1}`
3. Trainer writes a checkpoint to one of two alternating slots
4. Rollout worker reloads `W_{t+1}` in-place via `collective_rpc("reload_weights", ...)`
5. Step `t+1` begins only after reload completes

## Logs

`nanorl/scripts/train.py` prints per-step phase timings (fetch, rollout, reward,
pack, broadcast, old-logprobs, update, sync) — useful for identifying
whether step time is dominated by the reward sandbox, vLLM generation, or
weight sync.

## Scope

Intentionally small. Easy to read, easy to run locally, enough for code-RL
experiments. Not a general-purpose RLHF platform.
