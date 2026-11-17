#!/usr/bin/env bash
# Prepare only by default. No training or queue starts without --launch.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base_pooled_qk_embed_no_embed_norm --norm-value-input --sum-head-attention \
    --device-batch-size 2 \
    --tag-prefix bov_d24_f3p91e19_pooled_qk_norm_input_sum_nosmear_s42 "$@"
