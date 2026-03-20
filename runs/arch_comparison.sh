#!/bin/bash

# Compare three architecture variants at a fixed compute budget (1e18 FLOPs):
#   1. gpt          - baseline GPT
#   2. attn_res     - Full Attention Residuals
#   3. gated_attn_res - Gated Full Attention Residuals
#
# All models are trained at depth 12 with frequent eval checkpoints for
# plotting FLOP-controlled validation loss curves.
#
# Usage:
#   bash runs/arch_comparison.sh
#   # or with wandb logging:
#   WANDB_RUN=arch_cmp bash runs/arch_comparison.sh

set -e

FLOPS=1e18
DEPTH=12
EVAL_EVERY=50          # eval val bpb every N steps (dense curve)
SAVE_EVERY=200         # save intermediate checkpoints
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/arch_comparison"
mkdir -p "$RESULTS_DIR"

MODEL_TYPES=("gpt" "attn_res" "gated_attn_res")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# Train all three models
# =============================================================================

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

    log "=============================================="
    log "Training: $model_type (depth=$DEPTH, target_flops=$FLOPS)"
    log "=============================================="

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$DEPTH \
        --model-type=$model_type \
        --target-flops=$FLOPS \
        --target-param-data-ratio=-1 \
        --eval-every=$EVAL_EVERY \
        --save-every=$SAVE_EVERY \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --model-tag="${TAG}" \
        --run="${WANDB_RUN}_${TAG}" \
        2>&1 | tee "$LOG_FILE"

    log "Finished training: $model_type"
done

# =============================================================================
# Evaluate all three models (CORE + BPB + samples)
# =============================================================================

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_eval.log"

    log "=============================================="
    log "Evaluating: $model_type"
    log "=============================================="

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
        --model-tag="${TAG}" \
        2>&1 | tee "$LOG_FILE"

    log "Finished evaluating: $model_type"
done

# =============================================================================
# Extract FLOP-controlled val loss curves from training logs
# =============================================================================

log "Extracting validation loss curves..."

CURVE_FILE="$RESULTS_DIR/val_loss_curves.csv"
echo "model_type,step,flops,val_bpb" > "$CURVE_FILE"

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

    # Extract flops_per_token and batch_size from the log
    FLOPS_PER_TOKEN=$(grep "Estimated FLOPs per token:" "$LOG_FILE" | head -1 | grep -oP '[\d.]+e\+?\d+')
    BATCH_SIZE=$(grep "Total batch size" "$LOG_FILE" | head -1 | grep -oP '[\d,]+' | head -1 | tr -d ',')

    if [ -z "$FLOPS_PER_TOKEN" ] || [ -z "$BATCH_SIZE" ]; then
        log "WARNING: Could not extract FLOPs/batch info for $model_type"
        continue
    fi

    # Extract each "Step NNNNN | Validation bpb: X.XXXXXX" line
    grep "Validation bpb:" "$LOG_FILE" | while read -r line; do
        step=$(echo "$line" | grep -oP 'Step \K\d+')
        bpb=$(echo "$line" | grep -oP 'bpb: \K[\d.]+')
        # Compute FLOPs at this step: step * flops_per_token * batch_size
        flops_at_step=$(python3 -c "print(f'{int($step) * $FLOPS_PER_TOKEN * $BATCH_SIZE:.6e}')")
        echo "$model_type,$step,$flops_at_step,$bpb" >> "$CURVE_FILE"
    done

    log "Extracted $(grep -c "^$model_type," "$CURVE_FILE") eval points for $model_type"
done

log "=============================================="
log "Architecture Comparison Complete"
log "=============================================="
log "Training logs: $RESULTS_DIR/*_train.log"
log "Eval logs:     $RESULTS_DIR/*_eval.log"
log "Loss curves:   $CURVE_FILE"
echo ""
echo "Loss curve data:"
column -t -s',' "$CURVE_FILE" | head -30
