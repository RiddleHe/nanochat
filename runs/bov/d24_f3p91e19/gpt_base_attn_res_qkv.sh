#!/usr/bin/env bash
# Prepare by default. Hardware batch size can be overridden on the command line.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base_attn_res_qkv \
    --tag-prefix bov_d24_f3p91e19_gpt_base_attn_res_qkv_nosmear_s42 "$@"
