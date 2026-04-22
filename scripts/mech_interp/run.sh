#!/usr/bin/env bash
# Launch vLLM rollout worker + entropy/grad-norm analysis.
# Mirrors the split GPU layout in nanorl/runs/train.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${NANOCHAT_BASE_DIR:-$ROOT_DIR/.nanochat}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
ANALYSIS_GPUS="${ANALYSIS_GPUS:-0}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.85}"
BATCH_SIZE="${BATCH_SIZE:-2}"

OUT_DIR="$BASE_DIR/mech_interp"
OUTPUT="${OUTPUT:-$OUT_DIR/entropy_grad_norm.json}"
WORKER_LOG="$OUT_DIR/rollout_worker.log"
ANALYSIS_LOG="$OUT_DIR/analysis.log"

mkdir -p "$OUT_DIR"
export PYTHONPATH="$ROOT_DIR"
export NANOCHAT_BASE_DIR="$BASE_DIR"

# Prefer the project's uv venv if present (has vllm, transformers, torch).
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi
echo "[mech_interp] using python: $PYTHON"

IFS=, read -r -a ROLLOUT_GPU_LIST <<< "$ROLLOUT_GPUS"
ROLLOUT_TP="${#ROLLOUT_GPU_LIST[@]}"

WORKER_PID=""
cleanup() {
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[mech_interp] starting rollout worker on GPUs $ROLLOUT_GPUS -> $WORKER_LOG"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS" \
  "$PYTHON" "$ROOT_DIR/nanorl/scripts/rollout_worker.py" \
    --model "$MODEL" \
    --host "$ROLLOUT_HOST" \
    --port "$ROLLOUT_PORT" \
    --gpu-memory-utilization "$ROLLOUT_GPU_MEM_UTIL" \
    --tensor-parallel-size "$ROLLOUT_TP" \
    >"$WORKER_LOG" 2>&1 &
WORKER_PID="$!"

HEALTH_URL="http://$ROLLOUT_HOST:$ROLLOUT_PORT/health"
echo "[mech_interp] waiting for rollout worker health at $HEALTH_URL"
for _ in $(seq 600); do
  if curl -sf "$HEALTH_URL" | grep -q '"ok": *true'; then
    echo "[mech_interp] rollout worker healthy"
    break
  fi
  sleep 1
done
curl -sf "$HEALTH_URL" | grep -q '"ok": *true' \
  || { echo "rollout worker did not become healthy" >&2; exit 1; }

echo "[mech_interp] starting analysis on GPU $ANALYSIS_GPUS -> $ANALYSIS_LOG"
CUDA_VISIBLE_DEVICES="$ANALYSIS_GPUS" \
  "$PYTHON" -m scripts.mech_interp.entropy_grad_norm \
    --model "$MODEL" \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT" \
    2>&1 | tee "$ANALYSIS_LOG"
