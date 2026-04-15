#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="$ROOT_DIR/.nanochat"

MODEL="Qwen/Qwen3-0.6B"
ROLLOUT_HOST="127.0.0.1"
ROLLOUT_PORT="8047"
ROLLOUT_GPU="3"
TRAIN_GPUS="0,1,2"

RUN_DIR="$BASE_DIR/rl/$TAG"
ROLLOUT_SYNC_DIR="$RUN_DIR/rollout_sync"
SAVE_DIR="$RUN_DIR/checkpoints"
WORKER_LOG="$RUN_DIR/rollout_worker.log"
TRAIN_LOG="$RUN_DIR/train.log"

export PYTHONPATH="$ROOT_DIR"
export NANOCHAT_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT

IFS=, read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
TRAIN_NPROC="${#TRAIN_GPU_LIST[@]}"
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

mkdir -p "$RUN_DIR" "$ROLLOUT_SYNC_DIR" "$SAVE_DIR"

echo "[launcher] run tag: $TAG"
echo "[launcher] run dir: $RUN_DIR"
echo "[launcher] starting rollout worker on GPU $ROLLOUT_GPU -> $WORKER_LOG"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU" \
  python "$ROOT_DIR/scripts/rl_rollout_worker.py" \
    --model "$MODEL" \
    --host "$ROLLOUT_HOST" \
    --port "$ROLLOUT_PORT" \
    --gpu-memory-utilization 0.45 \
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
  torchrun --standalone --nproc_per_node="$TRAIN_NPROC" -m scripts.rl_train \
    --model "$MODEL" \
    --algorithm dapo \
    --task rstar_seed \
    --run "$TAG" \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --rollout-sync-dir "$ROLLOUT_SYNC_DIR" \
    --save-dir "$SAVE_DIR" \
    --num-steps 200 \
    --prompts-per-step 16 \
    --num-samples 8 \
    --train-batch-size 4 \
    --max-new-tokens 8192 \
    --max-seq-len 16384 \
    --reward-workers 8 \
    --k-tests 10 \
    --eval-every 20 \
    --lr 1e-6 \
    --kl-coeff 0.0 \
    --temperature 1.0 \
    --top-k -1 \
    2>&1 | tee "$TRAIN_LOG"
