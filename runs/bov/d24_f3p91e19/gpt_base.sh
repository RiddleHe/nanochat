#!/usr/bin/env bash
# Prepare by default. Training requires --launch and explicit CUDA_VISIBLE_DEVICES.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/runs/lib/pretrain.py" \
    --recipe "$recipe_dir/common.json" \
    --model-type gpt_base --expect-all-flags-false \
    --tag-prefix bov_d24_f3p91e19_gpt_base_nosmear_s42 "$@"
