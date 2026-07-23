#!/bin/bash
# Minimum-global-layers experiment (route A): SSSL baseline has 3 full-attention
# layers at L3/L7/L11 (d12). Measurement says d12's distant reads concentrate at
# L11 (with some at L7). Test: can we cut to 2 (L7,L11) or 1 (L11) global layers
# at equal FLOPs without losing quality? Fewer global layers = real long-context
# savings if quality holds. Requires the full-length window_pattern support.
set -u
cd /hdd/mh3897/cc/nanochat-patchscope
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=/tmp/min_global.log
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
train_one() {  # $1=pattern $2=tag
  if ls "$CKPT/$2"/meta_*.json >/dev/null 2>&1; then log "$2 exists, skip"; return; fi
  local G N
  while true; do G=$(idle_gpus); [ -n "$G" ] && break; log "no idle GPU, wait 120s"; sleep 120; done
  N=$(echo "$G" | awk -F, '{print NF}')
  log "training $2 (pattern=$1) on GPUs $G nproc=$N"
  CUDA_VISIBLE_DEVICES=$G /hdd/mh3897/cc/nanochat/.venv/bin/torchrun --standalone --nproc_per_node=$N \
    -m scripts.base_train -- --depth=12 --model-type=gpt_base --window-pattern="$1" \
    --model-tag="$2" --target-flops=1.5e18 --device-batch-size=32 --run=dummy
  log "training $2 exit=$?"
}
{
log "=== min-global start ==="
train_one "SSSSSSSLSSSL" arch_d12_gpt_base_global2_L7L11_1.5e18
train_one "SSSSSSSSSSSL" arch_d12_gpt_base_global1_L11_1.5e18
for tag in arch_d12_gpt_base_1.5e18 arch_d12_gpt_base_global2_L7L11_1.5e18 arch_d12_gpt_base_global1_L11_1.5e18; do
  m=$(ls "$CKPT/$tag"/meta_*.json 2>/dev/null | tail -1)
  [ -n "$m" ] && echo "RESULT $tag: $(/hdd/mh3897/cc/nanochat/.venv/bin/python -c "import json;d=json.load(open('$m'));print('step',d['step'],'val_bpb',d.get('val_bpb'))")" >> "$LOG"
done
log "=== min-global done ==="
} >> "$LOG" 2>&1
