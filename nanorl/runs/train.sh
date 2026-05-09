#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${NANOCHAT_BASE_DIR:-$ROOT_DIR/.nanochat}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-3}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.9}"
WEIGHT_TRANSFER_BACKEND="${WEIGHT_TRANSFER_BACKEND:-nccl}"
NUM_STEPS="${NUM_STEPS:-2}"
SAVE_EVERY="${SAVE_EVERY:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LR="${LR:-1e-6}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-16}"
EVAL_EVERY="${EVAL_EVERY:-2}"
EVAL_K="${EVAL_K:-4}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-8192}"
USE_WANDB="${USE_WANDB:-true}"

RUN_DIR="$BASE_DIR/rl/$TAG"
SAVE_DIR="$RUN_DIR/checkpoints"
WORKER_LOG="$RUN_DIR/rollout_worker.log"
TRAIN_LOG="$RUN_DIR/train.log"

export PYTHONPATH="$ROOT_DIR"
export NANOCHAT_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT

IFS=, read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
IFS=, read -r -a ROLLOUT_GPU_LIST <<< "$ROLLOUT_GPUS"
TRAIN_NPROC="${#TRAIN_GPU_LIST[@]}"
ROLLOUT_TP="${#ROLLOUT_GPU_LIST[@]}"
for gpu in "${TRAIN_GPU_LIST[@]}"; do
  for rgpu in "${ROLLOUT_GPU_LIST[@]}"; do
    if [[ "$gpu" == "$rgpu" ]]; then
      echo "ROLLOUT_GPUS ($ROLLOUT_GPUS) must not overlap TRAIN_GPUS ($TRAIN_GPUS)" >&2
      exit 1
    fi
  done
done

WORKER_PID=""

# Recursively kill a process and all its descendants (children first).
kill_subtree() {
  local pid=$1
  local children
  children=$(pgrep -P "$pid" 2>/dev/null) || true
  for child in $children; do
    kill_subtree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  if [[ -n "$WORKER_PID" ]]; then
    echo "[launcher] killing rollout worker subtree (root pid=$WORKER_PID)"
    kill_subtree "$WORKER_PID"
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "$RUN_DIR" "$SAVE_DIR"

echo "[launcher] run tag: $TAG"
echo "[launcher] run dir: $RUN_DIR"
echo "[launcher] starting rollout worker on GPUs $ROLLOUT_GPUS (tp=$ROLLOUT_TP) -> $WORKER_LOG"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS" \
  uv run python "$ROOT_DIR/nanorl/scripts/rollout_worker.py" \
    --model "$MODEL" \
    --host "$ROLLOUT_HOST" \
    --port "$ROLLOUT_PORT" \
    --gpu-memory-utilization "$ROLLOUT_GPU_MEM_UTIL" \
    --tensor-parallel-size "$ROLLOUT_TP" \
    --weight-transfer-backend "$WEIGHT_TRANSFER_BACKEND" \
    >"$WORKER_LOG" 2>&1 &
WORKER_PID="$!"

echo "[launcher] waiting for rollout worker health"
HEALTH_URL="http://$ROLLOUT_HOST:$ROLLOUT_PORT/health"
for _ in $(seq 300); do
  if curl -sf "$HEALTH_URL" | grep -q '"ok": *true'; then
    echo "[launcher] rollout worker healthy"
    break
  fi
  sleep 1
done
curl -sf "$HEALTH_URL" | grep -q '"ok": *true' || { echo "rollout worker did not become healthy" >&2; exit 1; }

echo "[launcher] starting trainer on GPUs $TRAIN_GPUS -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  uv run torchrun --standalone --nproc_per_node="$TRAIN_NPROC" -m nanorl.scripts.train \
    --model "$MODEL" \
    --algorithm dapo \
    --run-name "$TAG" \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --rollout-worker-world-size "$ROLLOUT_TP" \
    --save-dir "$SAVE_DIR" \
    --num-steps "$NUM_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" \
    --num-samples 8 \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --max-new-tokens 8192 \
    --max-seq-len 12288 \
    --reward-workers 8 \
    --eval-every "$EVAL_EVERY" \
    --eval-k "$EVAL_K" \
    --eval-max-tokens "$EVAL_MAX_TOKENS" \
    --save-every "$SAVE_EVERY" \
    --lr "$LR" \
    --kl-coeff 0.0 \
    --temperature 1.0 \
    --top-k -1 \
    $( [[ "$USE_WANDB" == "true" ]] && echo "--wandb" ) \
    2>&1 | tee "$TRAIN_LOG"
