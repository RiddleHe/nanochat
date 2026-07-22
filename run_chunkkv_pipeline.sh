#!/bin/bash
# Finding-1 phase 3: train chunk deep-KV d12 variants at equal FLOPs vs the
# SSSL baseline (arch_d12_gpt_base_1.5e18, val_bpb 0.8540). Elastic idle-GPU
# queue (0% util AND <100MiB, 3x20s samples), 1..4 GPUs, idempotent.
set -u
WT=/hdd/mh3897/cc/nanochat-patchscope
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=/tmp/chunkkv_pipeline.log
CKPT=/local-ssd/mh3897/base_checkpoints
cd "$WT"
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
wait_gpus() { local g; while true; do g=$(idle_gpus); [ -n "$g" ] && { echo "$g"; return; }; log "no idle GPU, wait 120s"; sleep 120; done; }
train_variant() {
  if ls "$CKPT/$2"/meta_*.json >/dev/null 2>&1; then log "$2 exists, skip"; return; fi
  local G N; G=$(wait_gpus); N=$(echo "$G" | awk -F, '{print NF}')
  log "training $2 ($1) on GPUs $G nproc=$N"
  CUDA_VISIBLE_DEVICES=$G /hdd/mh3897/cc/nanochat/.venv/bin/torchrun --standalone --nproc_per_node=$N \
    -m scripts.base_train -- --depth=12 --model-type="$1" --model-tag="$2" \
    --target-flops=1.5e18 --device-batch-size=32 --run=dummy
  log "training $2 exit=$?"
}
{
log "=== chunkkv pipeline start ==="
train_variant gpt_base_chunk_deep_kv arch_d12_gpt_base_chunk_deep_kv_1.5e18
train_variant gpt_base_chunk_same_kv arch_d12_gpt_base_chunk_same_kv_1.5e18
for tag in arch_d12_gpt_base_1.5e18 arch_d12_gpt_base_chunk_deep_kv_1.5e18 arch_d12_gpt_base_chunk_same_kv_1.5e18; do
  m=$(ls "$CKPT/$tag"/meta_*.json 2>/dev/null | tail -1)
  [ -n "$m" ] && echo "$tag: $(/hdd/mh3897/cc/nanochat/.venv/bin/python -c "import json;d=json.load(open('$m'));print('step',d['step'],'val_bpb',d.get('val_bpb'))")" >> "$LOG"
done
log "=== chunkkv pipeline done ==="
} >> "$LOG" 2>&1
