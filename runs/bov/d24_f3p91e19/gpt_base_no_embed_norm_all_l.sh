#!/usr/bin/env bash
# All-L baseline: full causal attention at every layer, raw initial embeddings.
# Prepare by default. Training requires --launch and explicit CUDA_VISIBLE_DEVICES.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common_all_l.json" \
    --model-type gpt_base_no_embed_norm --expect-all-flags-false \
    --device-batch-size 2 \
    --tag-prefix bov_d24_f3p91e19_gpt_base_no_embed_norm_all_l_nosmear_s42 "$@"
