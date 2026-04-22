#!/bin/bash

# SFT the GPT baseline, AttnRes, and AttnRes with load balancing loss base checkpoints.
#
# Usage:
#   DEPTH=12 NPROC_PER_NODE=4 bash runs/compare_attn_res_balanced_sft.sh
#   DEPTH=24 NPROC_PER_NODE=2 DEVICE_BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=2,5 \
#       NANOCHAT_BASE_DIR=/local-ssd/mh3897 bash runs/compare_attn_res_balanced_sft.sh

set -eo pipefail

DEPTH="${DEPTH:-12}"
if [ "$DEPTH" -ge 24 ]; then
    DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
else
    DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/attn_res_balanced_comparison_d${DEPTH}"
mkdir -p "$RESULTS_DIR"

MODEL_TYPES=("gpt_nolambda" "attn_res" "attn_res_balanced")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_d${DEPTH}_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_sft.log"
    CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/${TAG}"
    SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/${TAG}"

    if ! ls "$CKPT_DIR"/model_*.pt &>/dev/null; then
        log "Skipping $model_type (no base checkpoint in $CKPT_DIR)"
        continue
    fi

    if ls "$SFT_DIR"/model_*.pt &>/dev/null; then
        log "Skipping $model_type (SFT checkpoint already exists in $SFT_DIR)"
        continue
    fi

    log "=============================================="
    log "SFT: $model_type (depth=$DEPTH, device_batch_size=$DEVICE_BATCH_SIZE)"
    log "=============================================="

    if torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
        --model-tag="${TAG}" \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        --run="${WANDB_RUN}" \
        2>&1 | tee "$LOG_FILE"; then
        log "Finished SFT: $model_type"
    else
        log "FAILED SFT: $model_type (see $LOG_FILE)"
    fi
done

log "=============================================="
log "AttnRes Balanced SFT Comparison Complete (d=$DEPTH)"
log "=============================================="
