# RL Training

This repo includes a small RL training path for code-generation tasks.

The current reference task is `rstar_seed`, built from the official
`microsoft/rStar-Coder` `seed_testcase` split.

## What is included

- `nanochat/rl_data.py`
  - canonical JSONL dataset loader
  - distributed prompt loader
  - reward worker pool
- `nanochat/rl_sandbox.py`
  - isolated code execution for reward computation
- `nanochat/rl_rollout.py`
  - rollout helpers
  - logprob computation
  - batch packing
  - strict-sync rollout weight refresh
- `scripts/rl_train.py`
  - main trainer
  - supports `hf`, colocated `vllm`, and `remote_vllm`
- `scripts/rl_rollout_worker.py`
  - standalone rollout worker for `remote_vllm`
- `runs/rl_train_remote_vllm.sh`
  - one-command launcher for `3` trainer GPUs + `1` rollout GPU

## Data format

`rl_data.py` reads canonical JSONL files from:

```text
<NANOCHAT_BASE_DIR>/data/rl/<task>_<split>.jsonl
```

Each row looks like:

```json
{
  "id": "rstar/seed_123",
  "prompt": "<chat-formatted prompt>",
  "kind": "code_call_based",
  "payload": {
    "inputs": [[1, 2], [3, 4]],
    "outputs": [3, 7],
    "fn_name": "add",
    "time_limit_s": 4.0,
    "memory_limit_mb": 256
  },
  "meta": {
    "source": "rstar_seed",
    "n_tests": 10
  }
}
```

Large JSONL files are loaded lazily through a sidecar offsets file:

```text
<dataset>.offsets.u64
```

This keeps the full dataset usable without loading the whole file into RAM.

## Preparing rStar-Coder

There are two steps.

### 1. Filter the official source dataset

```bash
cd /path/to/nanochat

python -m scripts.filter_rstar_local \
  --output /path/to/cache/rstar_filtered_rows_full.jsonl \
  --bucket-dir /path/to/cache/rstar_buckets
```

This script:
- streams the official dataset
- drops rows with too few tests
- caps large test suites
- classifies rows as `code_call_based` or `code_stdin_stdout`
- does a deterministic coarse shuffle

### 2. Convert to canonical RL JSONL

```bash
cd /path/to/nanochat

NANOCHAT_BASE_DIR=/path/to/cache \
python -m scripts.prepare_rstar \
  --tokenizer Qwen/Qwen3-0.6B \
  --input-jsonl /path/to/cache/rstar_filtered_rows_full.jsonl \
  --output /path/to/cache/data/rl/rstar_seed_train.jsonl
```

After this, `rl_data.py` can load the dataset directly.

## Running training

There are two supported ways to run training.

### Option A: simple debug path with HF rollout

This is the easiest path for debugging on a small model or a small number of
steps.

```bash
cd /path/to/nanochat

NANOCHAT_BASE_DIR=/path/to/cache \
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 -m scripts.rl_train \
  --model Qwen/Qwen3-0.6B \
  --algorithm grpo \
  --task rstar_seed \
  --rollout-backend hf \
  --num-steps 2 \
  --prompts-per-step 2 \
  --num-samples 1 \
  --train-batch-size 1 \
  --max-new-tokens 64 \
  --reward-workers 0 \
  --k-tests 1
```

### Option B: recommended path with remote vLLM rollout

This is the main path for multi-GPU training.

```bash
cd /path/to/nanochat

NANOCHAT_BASE_DIR=/path/to/cache \
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
./runs/rl_train_remote_vllm.sh
```

The launcher:
- starts one rollout worker
- waits for worker health
- starts trainer ranks with `torchrun`
- shuts down the worker when training exits

## Required launcher inputs

The launcher works through environment variables.

The important ones are:

- `NANOCHAT_BASE_DIR`
  - base directory containing `data/rl/<task>_<split>.jsonl`
- `MODEL`
  - Hugging Face model name or local checkpoint path
- `ROLLOUT_GPU`
  - one GPU reserved for the rollout worker
- `TRAIN_GPUS`
  - comma-separated trainer GPUs
- `TASK`
  - dataset name, default `rstar_seed`
- `ALGORITHM`
  - RL loss, default `grpo`

Everything else has a default.

## Rollout backends

`scripts.rl_train` supports:

- `hf`
  - simplest path
  - no external worker
  - slower generation
- `vllm`
  - colocated vLLM engine on rank 0
  - useful for experiments, but less robust for multi-GPU training
- `remote_vllm`
  - separate rollout worker process
  - recommended path for multi-GPU runs

## Strict-sync semantics

The `remote_vllm` path is strict synchronous:

1. rollout for step `t` uses weights `W_t`
2. trainer updates weights to `W_{t+1}`
3. rollout worker reloads `W_{t+1}`
4. step `t+1` starts only after reload finishes

This keeps rollout on-policy step by step.

The current implementation updates the worker in place with `reload_weights`,
so sync is much faster than cold-reloading a fresh vLLM engine every step.

## Logs

`scripts.rl_train.py` prints per-step phase timings:

- prompt fetch
- rollout
- reward
- batch packing
- broadcast/shard
- old logprobs
- update
- rollout sync

This is useful for telling apart:
- normal slowness
- reward bottlenecks
- rollout bottlenecks
- sync bottlenecks

## Current scope

This RL path is intentionally small.

It is meant to be:
- easy to read
- easy to run locally
- enough for code-RL experiments

It is not trying to be a large general-purpose RLHF platform.
