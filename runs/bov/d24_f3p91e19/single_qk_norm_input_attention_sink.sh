#!/usr/bin/env bash
# Learned zero-value softmax sink on the normalized single-head value branch.
# Prepare only by default. Training requires explicit --launch.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base_single_qk_embed_no_embed_norm \
    --norm-value-input --attention-sink \
    --device-batch-size 2 \
    --tag-prefix bov_d24_f3p91e19_single_qk_norm_input_attention_sink_nosmear_s42 "$@"
