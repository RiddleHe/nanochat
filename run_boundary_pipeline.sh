#!/bin/bash
# Boundary pipeline: Phase 1 (measure stop-reading boundary on d12/d24) +
# Phase 2 (complete the d24 swap-start sweep: train the 2 missing variants) +
# summary plot. Elastic GPU queue: before each job, grab every FULLY idle GPU
# (0% util AND <100MiB, sampled 3x over 20s), use 1..4 of them. Never touches
# a GPU with any activity. Jobs are idempotent (skip if output exists).
set -u
WT=/hdd/mh3897/cc/nanochat-patchscope
PY=/hdd/mh3897/cc/nanochat/.venv/bin/python
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=/tmp/boundary_pipeline.log
CKPT=/local-ssd/mh3897/base_checkpoints
cd "$WT"

log() { echo "[$(date +%F\ %T)] $*" >> "$LOG"; }

idle_gpus() {  # print comma-separated fully-idle GPU ids (3 samples, 10s apart)
  local a b c
  a=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<100 {print $1}')
  sleep 10
  b=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<100 {print $1}')
  sleep 10
  c=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0 && $3<100 {print $1}')
  comm -12 <(echo "$a" | sort) <(echo "$b" | sort) | comm -12 - <(echo "$c" | sort) | sort -n | head -4 | paste -sd,
}

wait_gpus() {  # block until at least 1 fully idle gpu; echo the list
  local g
  while true; do
    g=$(idle_gpus)
    [ -n "$g" ] && { echo "$g"; return; }
    log "no fully-idle GPU, waiting 120s..."
    sleep 120
  done
}

{
log "=== pipeline start ==="

# ---- Phase 1a: boundary on d12 baseline ----
if [ ! -f results/boundary/d12__boundary.json ]; then
  G=$(wait_gpus); G1=${G%%,*}
  log "phase1 d12 boundary on GPU $G1"
  CUDA_VISIBLE_DEVICES=$G1 $PY -m scripts.inspect.step_d_nanochat \
    --model-tag arch_d12_gpt_base_1.5e18 --label d12 --out results/boundary
  log "phase1 d12 exit=$?"
else log "phase1 d12 boundary exists, skip"; fi

# ---- Phase 1b: boundary on d24 baseline ----
if [ ! -f results/boundary/d24__boundary.json ]; then
  G=$(wait_gpus); G1=${G%%,*}
  log "phase1 d24 boundary on GPU $G1"
  CUDA_VISIBLE_DEVICES=$G1 $PY -m scripts.inspect.step_d_nanochat \
    --model-tag arch_d24_gpt_base --label d24 --out results/boundary
  log "phase1 d24 exit=$?"
else log "phase1 d24 boundary exists, skip"; fi

# ---- interim plot with whatever exists (d12 sweep is fully pre-trained) ----
$PY -m scripts.inspect.boundary_vs_quality >> "$LOG" 2>&1
log "interim boundary_vs_quality plotted"

# ---- Phase 2: train the two missing d24 variants (elastic 1-4 GPUs) ----
train_variant() {  # $1=model_type  $2=model_tag
  if ls "$CKPT/$2"/meta_*.json >/dev/null 2>&1; then log "$2 exists, skip"; return; fi
  G=$(wait_gpus); N=$(echo "$G" | awk -F, '{print NF}')
  log "training $2 on GPUs $G (nproc=$N), target_flops=3.91e19"
  CUDA_VISIBLE_DEVICES=$G /hdd/mh3897/cc/nanochat/.venv/bin/torchrun \
    --standalone --nproc_per_node=$N -m scripts.base_train -- \
    --depth=24 --model-type="$1" --model-tag="$2" \
    --target-flops=3.91e19 --run=dummy
  log "training $2 exit=$?"
}
train_variant gpt_base_v_from_value_emb_learn_last_two_thirds arch_d24_gpt_base_v_from_value_emb_learn_last_two_thirds
train_variant gpt_base_v_from_value_emb_learn_every_layer     arch_d24_gpt_base_v_from_value_emb_learn_every_layer

# ---- final plot ----
$PY -m scripts.inspect.boundary_vs_quality >> "$LOG" 2>&1
log "=== pipeline done ==="
} >> "$LOG" 2>&1
