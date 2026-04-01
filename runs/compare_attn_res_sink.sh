#!/bin/bash

# Compare GPT baseline, AttnRes, and AttnRes with sink logits.
#
# Usage:
#   DEPTH=12 NPROC_PER_NODE=4 bash runs/compare_attn_res_sink.sh
#   DEPTH=24 NPROC_PER_NODE=4 bash runs/compare_attn_res_sink.sh

set -eo pipefail

DEPTH="${DEPTH:-12}"
if [ "$DEPTH" -ge 24 ]; then
    FLOPS="${FLOPS:-1.5e19}"
    DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
else
    FLOPS="${FLOPS:-1e18}"
    DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
fi
EVAL_EVERY=50
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.dataset -n 8
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
fi
python -m nanochat.dataset -n 170 &
DATASET_PID=$!

RESULTS_DIR="$NANOCHAT_BASE_DIR/attn_res_sink_comparison_d${DEPTH}"
mkdir -p "$RESULTS_DIR"

MODEL_TYPES=("gpt" "attn_res" "attn_res_sink")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait $DATASET_PID

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_d${DEPTH}_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/${TAG}"

    if ls "$CKPT_DIR"/model_*.pt &>/dev/null; then
        log "Skipping $model_type (checkpoint found in $CKPT_DIR)"
        continue
    fi

    log "=============================================="
    log "Training: $model_type (depth=$DEPTH, target_flops=$FLOPS)"
    log "=============================================="

    if torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$DEPTH \
        --model-type=$model_type \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        --target-flops=$FLOPS \
        --target-param-data-ratio=-1 \
        --eval-every=$EVAL_EVERY \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --model-tag="${TAG}" \
        --run="${WANDB_RUN}" \
        2>&1 | tee "$LOG_FILE"; then
        log "Finished training: $model_type"
    else
        log "FAILED training: $model_type (see $LOG_FILE)"
    fi
done

log "=============================================="
log "AttnRes Sink Comparison Complete (d=$DEPTH)"
log "=============================================="
