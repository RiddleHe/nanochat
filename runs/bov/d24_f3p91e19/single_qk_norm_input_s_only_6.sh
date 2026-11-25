#!/usr/bin/env bash
# SSSL placement control: normalized single-head values on the six interleaved-11 S targets.
# Prepare only by default. Training requires explicit --launch.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base_single_qk_embed_no_embed_norm --norm-value-input \
    --target-layers 1 5 9 13 17 21 \
    --device-batch-size 2 \
    --tag-prefix bov_d24_f3p91e19_single_qk_norm_input_s_only_6_nosmear_s42 "$@"
