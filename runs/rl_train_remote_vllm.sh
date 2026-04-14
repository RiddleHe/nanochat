#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
BASE_DIR="${NANOCHAT_BASE_DIR:-$ROOT_DIR/.nanochat}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TASK="${TASK:-rstar_seed}"
ALGORITHM="${ALGORITHM:-grpo}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6}"

NUM_STEPS="${NUM_STEPS:-200}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-12}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REWARD_WORKERS="${REWARD_WORKERS:-8}"
K_TESTS="${K_TESTS:-10}"
EVAL_EVERY="${EVAL_EVERY:-20}"
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_SYNC_DIR="${ROLLOUT_SYNC_DIR:-$BASE_DIR/rl_rollout_sync}"
SAVE_DIR="${SAVE_DIR:-$BASE_DIR/rl_checkpoints}"
WORKER_LOG="${WORKER_LOG:-$BASE_DIR/rl_rollout_worker.log}"
TRAIN_LOG="${TRAIN_LOG:-$BASE_DIR/rl_train_remote_vllm.log}"

export PYTHONPATH="$ROOT_DIR"
export NANOCHAT_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT

if [[ -z "${TRAIN_NPROC:-}" ]]; then
  IFS=, read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
  TRAIN_NPROC="${#TRAIN_GPU_LIST[@]}"
fi

IFS=, read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
for gpu in "${TRAIN_GPU_LIST[@]}"; do
  if [[ "$gpu" == "$ROLLOUT_GPU" ]]; then
    echo "ROLLOUT_GPU ($ROLLOUT_GPU) must not overlap TRAIN_GPUS ($TRAIN_GPUS)" >&2
    exit 1
  fi
done

WORKER_PID=""

cleanup() {
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "$(dirname "$WORKER_LOG")" "$(dirname "$TRAIN_LOG")" "$ROLLOUT_SYNC_DIR" "$SAVE_DIR"

echo "[launcher] starting rollout worker on GPU $ROLLOUT_GPU -> $WORKER_LOG"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/rl_rollout_worker.py" \
    --model "$MODEL" \
    --host "$ROLLOUT_HOST" \
    --port "$ROLLOUT_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    >"$WORKER_LOG" 2>&1 &
WORKER_PID="$!"

echo "[launcher] waiting for rollout worker health"
"$PYTHON_BIN" - <<'PY'
import json
import os
import sys
import time
import urllib.request

url = f"http://{os.environ['ROLLOUT_HOST']}:{os.environ['ROLLOUT_PORT']}/health"
deadline = time.time() + 300
last_err = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("ok"):
            print(payload["model_path"])
            sys.exit(0)
    except Exception as e:
        last_err = e
    time.sleep(1.0)
raise SystemExit(f"rollout worker did not become healthy: {last_err}")
PY

echo "[launcher] starting trainer on GPUs $TRAIN_GPUS -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_NPROC" -m scripts.rl_train \
    --model "$MODEL" \
    --algorithm "$ALGORITHM" \
    --task "$TASK" \
    --rollout-backend remote_vllm \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --rollout-sync-dir "$ROLLOUT_SYNC_DIR" \
    --save-dir "$SAVE_DIR" \
    --num-steps "$NUM_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" \
    --num-samples "$NUM_SAMPLES" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --reward-workers "$REWARD_WORKERS" \
    --k-tests "$K_TESTS" \
    --eval-every "$EVAL_EVERY" \
    --lr "$LR" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    2>&1 | tee "$TRAIN_LOG"
