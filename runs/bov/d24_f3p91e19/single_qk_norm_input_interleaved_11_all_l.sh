#!/usr/bin/env bash
# All-L placement control: normalized single-head values on odd layers 1..21.
# Prepare only by default. Training requires explicit --launch.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common_all_l.json" \
    --model-type gpt_base_single_qk_embed_no_embed_norm --norm-value-input \
    --target-layers 1 3 5 7 9 11 13 15 17 19 21 \
    --device-batch-size 2 \
    --tag-prefix bov_d24_f3p91e19_single_qk_norm_input_interleaved_11_all_l_nosmear_s42 "$@"
