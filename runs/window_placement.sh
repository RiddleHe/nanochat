#!/bin/bash
# Window-placement ablation: train variants at equal FLOPs and equal 3:1
# short:long layer ratio (matching the SSSL baselines); only WHERE the
# full-attention (L) layers sit differs.
#   full_attn_late : S...S LLL  (full attention only in the deep layers)
#   full_attn_early: LLL S...S  (full attention only in the early layers)
# Interleaved baselines already exist (window_pattern=SSSL):
#   arch_d12_gpt_base_1.5e18 (val_bpb 0.8540), arch_d24_gpt_base (0.7218).
# Requires the gpt_base change that respects full-length window patterns
# (len(pattern) == n_layer disables the final-layer-always-L override).
#
# Elastic GPU queue: before each job grab every FULLY idle GPU (0% util AND
# <100MiB, 3 samples over 20s), use 1..4. Jobs are idempotent (skip if a
# checkpoint exists). Run from the repo root with NANOCHAT_BASE_DIR set:
#   NANOCHAT_BASE_DIR=/local-ssd/<user> bash runs/window_placement.sh
set -u
LOG=${LOG:-/tmp/window_placement.log}
CKPT="$NANOCHAT_BASE_DIR/base_checkpoints"

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

wait_gpus() {
  local g
  while true; do
    g=$(idle_gpus)
    [ -n "$g" ] && { echo "$g"; return; }
    log "no fully-idle GPU, waiting 120s..."
    sleep 120
  done
}

train_variant() {  # $1=depth $2=pattern $3=tag $4=target_flops $5=device_batch
  if ls "$CKPT/$3"/meta_*.json >/dev/null 2>&1; then log "$3 exists, skip"; return; fi
  local G N
  G=$(wait_gpus); N=$(echo "$G" | awk -F, '{print NF}')
  log "training $3 (d$1, pattern=$2) on GPUs $G (nproc=$N), flops=$4"
  CUDA_VISIBLE_DEVICES=$G uv run torchrun --standalone --nproc_per_node="$N" \
    -m scripts.base_train -- \
    --depth="$1" --model-type=gpt_base --window-pattern="$2" \
    --model-tag="$3" --target-flops="$4" --device-batch-size="$5" --run=dummy
  log "training $3 exit=$?"
}

{
log "=== window placement pipeline start ==="
train_variant 12 "SSSSSSSSSLLL" arch_d12_gpt_base_full_attn_late_1.5e18  1.5e18  32
train_variant 12 "LLLSSSSSSSSS" arch_d12_gpt_base_full_attn_early_1.5e18 1.5e18  32
train_variant 24 "SSSSSSSSSSSSSSSSSSLLLLLL" arch_d24_gpt_base_full_attn_late  3.91e19 16
train_variant 24 "LLLLLLSSSSSSSSSSSSSSSSSS" arch_d24_gpt_base_full_attn_early 3.91e19 16

uv run python - <<'EOF' >> "$LOG" 2>&1
import json, glob, os
ckpt = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "base_checkpoints")
rows = [
    ("d12 interleaved SSSL (baseline)", "arch_d12_gpt_base_1.5e18"),
    ("d12 full-attn LATE",  "arch_d12_gpt_base_full_attn_late_1.5e18"),
    ("d12 full-attn EARLY", "arch_d12_gpt_base_full_attn_early_1.5e18"),
    ("d24 interleaved SSSL (baseline)", "arch_d24_gpt_base"),
    ("d24 full-attn LATE",  "arch_d24_gpt_base_full_attn_late"),
    ("d24 full-attn EARLY", "arch_d24_gpt_base_full_attn_early"),
]
print("=== window placement results (val_bpb, lower better) ===")
for label, tag in rows:
    metas = sorted(glob.glob(f"{ckpt}/{tag}/meta_*.json"))
    if not metas:
        print(f"{label:38s} MISSING")
        continue
    d = json.load(open(metas[-1]))
    print(f"{label:38s} step {d['step']:6d}  val_bpb {d.get('val_bpb')}")
EOF
log "=== window placement pipeline done ==="
} >> "$LOG" 2>&1
