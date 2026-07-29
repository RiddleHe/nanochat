#!/bin/bash
# Chunk deep-KV experiment launcher.
#
# Replaces the four ad-hoc scripts used during the original run
# (run_chunkkv_pipeline / _v2 / _v2slim / run_equaltoken_baseline).
#
# Usage:
#   scripts/run_chunk_deep_kv.sh <variant> [depth] [target_flops] [seq_len]
#
#   variant: baseline | deep_kv | same_kv | v2 | v2_slim | equaltoken
#   depth:        default 12
#   target_flops: default 1.5e18
#   seq_len:      default 2048
#
# Examples (these reproduce the numbers in research_chunk_deep_kv.md):
#   scripts/run_chunk_deep_kv.sh baseline                    # 0.85401
#   scripts/run_chunk_deep_kv.sh v2_slim                     # 0.85524
#   scripts/run_chunk_deep_kv.sh baseline 12 1.5e18 4096     # 0.86086
#   scripts/run_chunk_deep_kv.sh v2_slim  12 1.5e18 4096     # 0.86295
#
# Every variant is compared at EQUAL FLOPs: base_train derives the step count
# from --target-flops using estimate_flops(), which charges the branch its real
# cost (v1 1.414x, v2 1.081x, v2-slim 1.040x baseline). The `equaltoken` variant
# is the extra control that holds STEPS fixed instead, to separate "the branch
# helps per token" from "the branch is worth its FLOPs".
#
# Waits for genuinely idle GPUs (3 samples over 20s, util 0 and <100MiB used)
# and is idempotent: re-running after a completed run is a no-op.
set -u

VARIANT="${1:?usage: $0 <baseline|deep_kv|same_kv|v2|v2_slim|equaltoken> [depth] [flops] [seq]}"
DEPTH="${2:-12}"
FLOPS="${3:-1.5e18}"
SEQ="${4:-2048}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/local-ssd/$USER}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
CKPT="$NANOCHAT_BASE_DIR/base_checkpoints"
PY="$REPO/.venv/bin"

case "$VARIANT" in
  baseline)   MODEL_TYPE=gpt_base ;;
  deep_kv)    MODEL_TYPE=gpt_base_chunk_deep_kv ;;
  same_kv)    MODEL_TYPE=gpt_base_chunk_same_kv ;;
  v2)         MODEL_TYPE=gpt_base_chunk_deep_kv_v2 ;;
  v2_slim)    MODEL_TYPE=gpt_base_chunk_deep_kv_v2_slim ;;
  equaltoken) MODEL_TYPE=gpt_base ;;   # steps pinned below, not FLOPs
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

TAG="arch_d${DEPTH}_${VARIANT}_seq${SEQ}_${FLOPS}"
LOG="/tmp/chunkkv_${TAG}.log"
log() { echo "[$(date +%F\ %T)] $*" >> "$LOG"; }

# Return a comma-separated list of GPUs idle across three samples 10s apart.
idle_gpus() {
  local a b c
  a=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<100 {print $1}')
  sleep 10
  b=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<100 {print $1}')
  sleep 10
  c=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<100 {print $1}')
  comm -12 <(echo "$a" | sort) <(echo "$b" | sort) | comm -12 - <(echo "$c" | sort) | sort -n | head -4 | paste -sd,
}

{
log "=== $TAG start (model_type=$MODEL_TYPE) ==="
if ls "$CKPT/$TAG"/meta_*.json >/dev/null 2>&1; then
  log "checkpoint exists, skipping training"
else
  while true; do G=$(idle_gpus); [ -n "$G" ] && break; log "no idle GPU, wait 120s"; sleep 120; done
  N=$(echo "$G" | awk -F, '{print NF}')
  log "training on GPUs $G (nproc=$N)"

  # equaltoken pins the step count to the v1 deep_kv run so the two see the same
  # number of TOKENS; every other variant is equal-FLOPs via --target-flops.
  BUDGET=(--target-flops="$FLOPS")
  [ "$VARIANT" = "equaltoken" ] && BUDGET=(--num-iterations=2663)

  CUDA_VISIBLE_DEVICES=$G "$PY/torchrun" --standalone --nproc_per_node=$N \
    -m scripts.base_train -- \
    --depth="$DEPTH" --model-type="$MODEL_TYPE" --model-tag="$TAG" \
    --max-seq-len="$SEQ" "${BUDGET[@]}" --device-batch-size=32 --run=dummy
  log "exit=$?"
fi

m=$(ls "$CKPT/$TAG"/meta_*.json 2>/dev/null | tail -1)
[ -n "$m" ] && "$PY/python" -c "
import json; d=json.load(open('$m'))
print('RESULT $TAG: step', d['step'], 'val_bpb', d.get('val_bpb'))" >> "$LOG"
log "=== $TAG done ==="
} >> "$LOG" 2>&1
