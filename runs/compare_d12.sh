#!/bin/bash

# Compare four architecture variants at a fixed compute budget (1e18 FLOPs):
#   1. gpt_nolambda  - vanilla GPT (no lambdas, no backout)
#   2. gpt           - GPT with resid_lambdas + x0_lambdas
#   3. attn_res      - Full Attention Residuals
#   4. gated_attn_res - Gated Full Attention Residuals
#
# Training logs contain everything needed:
#   - FLOP-controlled val loss curves (--eval-every)
#   - Final CORE benchmark metrics (--core-metric-every=999999 = final step)
#
# Usage:
#   bash runs/attn_res_arch_compare.sh
#   # or with wandb logging:
#   WANDB_RUN=arch_cmp bash runs/attn_res_arch_compare.sh

set -eo pipefail

FLOPS=1e18
DEPTH=12
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EVAL_EVERY=50          # eval val bpb every N steps (dense curve)
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Download dataset (need ≥8 shards for tokenizer training, 170 for pretraining)
python -m nanochat.dataset -n 8
# Train tokenizer if not already present
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
fi
# Download remaining shards in background while training starts
python -m nanochat.dataset -n 170 &
DATASET_PID=$!

RESULTS_DIR="$NANOCHAT_BASE_DIR/arch_comparison_d${DEPTH}"
mkdir -p "$RESULTS_DIR"

MODEL_TYPES=("gpt_nolambda" "gpt" "attn_res")
# MODEL_TYPES=("gpt_nolambda" "gpt" "attn_res" "gated_attn_res")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Wait for dataset download to complete before training
wait $DATASET_PID

# =============================================================================
# Train all three models
# =============================================================================

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_d${DEPTH}_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/${TAG}"

    # Skip if checkpoint already exists
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

# =============================================================================
# Extract results from training logs
# =============================================================================

log "Extracting results..."

# 1. FLOP-controlled val loss curves
CURVE_FILE="$RESULTS_DIR/val_loss_curves.csv"
echo "model_type,step,flops,val_bpb" > "$CURVE_FILE"

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_d${DEPTH}_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

    if [ ! -f "$LOG_FILE" ]; then
        log "WARNING: No training log for $model_type"
        continue
    fi

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
        flops_at_step=$(python3 -c "print(f'{int(\"$step\") * $FLOPS_PER_TOKEN * $BATCH_SIZE:.6e}')")
        echo "$model_type,$step,$flops_at_step,$bpb" >> "$CURVE_FILE"
    done

    log "Extracted $(grep -c "^$model_type," "$CURVE_FILE") eval points for $model_type"
done

# 2. Final benchmark metrics
METRICS_FILE="$RESULTS_DIR/final_metrics.csv"
echo "model_type,final_val_bpb,core_metric" > "$METRICS_FILE"

for model_type in "${MODEL_TYPES[@]}"; do
    TAG="arch_d${DEPTH}_${model_type}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

    if [ ! -f "$LOG_FILE" ]; then
        continue
    fi

    FINAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP 'bpb: \K[\d.]+')
    CORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

    if [ -n "$FINAL_BPB" ] && [ -n "$CORE" ]; then
        echo "$model_type,$FINAL_BPB,$CORE" >> "$METRICS_FILE"
    else
        log "WARNING: Could not extract final metrics for $model_type"
    fi
done

log "=============================================="
log "Architecture Comparison Complete"
log "=============================================="
log "Training logs: $RESULTS_DIR/*_train.log"
log "Loss curves:   $CURVE_FILE"
log "Final metrics: $METRICS_FILE"
echo ""
echo "Final metrics:"
column -t -s',' "$METRICS_FILE"
echo ""
echo "Loss curve data (first 30 rows):"
column -t -s',' "$CURVE_FILE" | head -30
