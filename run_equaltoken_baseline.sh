#!/bin/bash
# Equal-token baseline for the chunk deep-KV decomposition: plain gpt_base at
# target-flops 1.06e18 (= 1.5e18 / 1.414) => same ~2663 steps as the chunk
# variants but with a COMPLETED lr schedule. Elastic idle-GPU queue, idempotent.
set -u
cd /hdd/mh3897/cc/nanochat-patchscope
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=/tmp/equaltoken_baseline.log
CKPT=/local-ssd/mh3897/base_checkpoints
log() { echo "[$(date +%F\ %T)] $*" >> "$LOG"; }
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
log "=== equal-token baseline start ==="
if ls "$CKPT/arch_d12_gpt_base_equaltoken_1.06e18"/meta_*.json >/dev/null 2>&1; then
  log "exists, skip"
else
  while true; do G=$(idle_gpus); [ -n "$G" ] && break; log "no idle GPU, wait 120s"; sleep 120; done
  N=$(echo "$G" | awk -F, '{print NF}')
  log "training on GPUs $G nproc=$N"
  CUDA_VISIBLE_DEVICES=$G /hdd/mh3897/cc/nanochat/.venv/bin/torchrun --standalone --nproc_per_node=$N \
    -m scripts.base_train -- --depth=12 --model-type=gpt_base \
    --model-tag=arch_d12_gpt_base_equaltoken_1.06e18 \
    --target-flops=1.06e18 --device-batch-size=32 --run=dummy
  log "exit=$?"
fi
log "=== equal-token baseline done ==="
} >> "$LOG" 2>&1
