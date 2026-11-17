#!/usr/bin/env bash
# Prepare by default; only --launch starts training. Two fixed Drexel GPUs.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES=0,1
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base_scalar_res_qkv_no_embed_norm \
    --tag-prefix bov_d24_f3p91e19_gpt_base_scalar_res_qkv_no_embed_norm_sqrt_recovery_nosmear_s42 \
    --device-batch-size 2 "$@" --gpu-count 2
