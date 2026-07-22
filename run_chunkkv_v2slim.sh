#!/bin/bash
set -u
cd /hdd/mh3897/cc/nanochat-patchscope
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=/tmp/chunkkv_v2slim.log
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
log "=== chunkkv v2slim start ==="
if ls "$CKPT/arch_d12_gpt_base_chunk_deep_kv_v2_slim_1.5e18"/meta_*.json >/dev/null 2>&1; then log "exists, skip"; else
  while true; do G=$(idle_gpus); [ -n "$G" ] && break; log "no idle GPU, wait 120s"; sleep 120; done
  N=$(echo "$G" | awk -F, '{print NF}')
  log "training on GPUs $G nproc=$N"
  CUDA_VISIBLE_DEVICES=$G /hdd/mh3897/cc/nanochat/.venv/bin/torchrun --standalone --nproc_per_node=$N \
    -m scripts.base_train -- --depth=12 --model-type=gpt_base_chunk_deep_kv_v2_slim_slim \
    --model-tag=arch_d12_gpt_base_chunk_deep_kv_v2_slim_1.5e18 \
    --target-flops=1.5e18 --device-batch-size=32 --run=dummy
  log "exit=$?"
fi
m=$(ls "$CKPT/arch_d12_gpt_base_chunk_deep_kv_v2_slim_1.5e18"/meta_*.json 2>/dev/null | tail -1)
[ -n "$m" ] && /hdd/mh3897/cc/nanochat/.venv/bin/python -c "import json;d=json.load(open('$m'));print('v2slim: step',d['step'],'val_bpb',d.get('val_bpb'))" >> "$LOG"
log "=== chunkkv v2slim done ==="
} >> "$LOG" 2>&1
