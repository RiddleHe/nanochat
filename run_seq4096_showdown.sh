#!/bin/bash
# Finding-1 decisive experiment: does the chunk deep-KV benefit grow with
# sequence length? baseline vs v2-slim at seq 4096, equal FLOPs (1.5e18).
set -u
cd /hdd/mh3897/cc/nanochat-patchscope
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=/tmp/seq4096_showdown.log
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
train_one() {  # $1=model_type $2=tag
  if ls "$CKPT/$2"/meta_*.json >/dev/null 2>&1; then log "$2 exists, skip"; return; fi
  local G N
  while true; do G=$(idle_gpus); [ -n "$G" ] && break; log "no idle GPU, wait 120s"; sleep 120; done
  N=$(echo "$G" | awk -F, '{print NF}')
  log "training $2 on GPUs $G nproc=$N"
  CUDA_VISIBLE_DEVICES=$G /hdd/mh3897/cc/nanochat/.venv/bin/torchrun --standalone --nproc_per_node=$N \
    -m scripts.base_train -- --depth=12 --model-type="$1" --model-tag="$2" \
    --max-seq-len=4096 --target-flops=1.5e18 --device-batch-size=16 --run=dummy
  log "training $2 exit=$?"
}
{
log "=== seq4096 showdown start ==="
train_one gpt_base                        arch_d12_gpt_base_seq4096_1.5e18
train_one gpt_base_chunk_deep_kv_v2_slim  arch_d12_gpt_base_chunk_v2slim_seq4096_1.5e18
for tag in arch_d12_gpt_base_seq4096_1.5e18 arch_d12_gpt_base_chunk_v2slim_seq4096_1.5e18; do
  m=$(ls "$CKPT/$tag"/meta_*.json 2>/dev/null | tail -1)
  [ -n "$m" ] && echo "RESULT $tag: $(/hdd/mh3897/cc/nanochat/.venv/bin/python -c "import json;d=json.load(open('$m'));print('step',d['step'],'val_bpb',d.get('val_bpb'))")" >> "$LOG"
done
log "=== seq4096 showdown done ==="
} >> "$LOG" 2>&1
