#!/usr/bin/env bash
# Prepare by default. Training requires --launch and explicit GPU selection.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base_split_embed_head_out_no_embed_norm --expect-all-flags-false \
    --device-batch-size 2 \
    --tag-prefix bov_d24_f3p91e19_gpt_base_split_embed_head_out_no_embed_norm_nosmear_s42 "$@"
