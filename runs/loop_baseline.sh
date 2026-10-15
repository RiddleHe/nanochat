#!/usr/bin/env bash
# 24 actual blocks x 4 passes = 96 executed layers. Width remains 1536.
# Run from the repository root after reserving GPUs. No early-exit gate.
# Set MODEL_TYPE=gpt_loop, gpt_loop_untied (matched init), or gpt_base (ordinary init).
# Example: CUDA_VISIBLE_DEVICES=2 NANOCHAT_BASE_DIR=/path/to/data NUM_ITERATIONS=20 \
#          TOTAL_BATCH_SIZE=2048 DEVICE_BATCH_SIZE=1 bash runs/loop_baseline.sh
# Smoke runs only check plumbing; use a separately agreed budget for research runs.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select idle GPUs explicitly}"
: "${NANOCHAT_BASE_DIR:?Set the directory containing tokenizer and dataset}"
: "${NUM_ITERATIONS:?Set the FULL optimizer-step budget, including when resuming}"
: "${TOTAL_BATCH_SIZE:?Set the global batch size in tokens, identical across comparisons}"
model_type=${MODEL_TYPE:-gpt_loop}
loop_args=()
if [[ "$model_type" == gpt_loop || "$model_type" == gpt_loop_untied ]]; then
    loop_args=(--num-loops 4)
fi
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-1}" \
    -m scripts.base_train --model-type "$model_type" "${loop_args[@]}" \
    --depth 24 --depth-semantics physical --aspect-ratio 64 --head-dim 128 --max-seq-len 2048 --window-pattern L \
    --num-iterations "$NUM_ITERATIONS" --total-batch-size "$TOTAL_BATCH_SIZE" \
    --device-batch-size "${DEVICE_BATCH_SIZE:-1}" --seed "${SEED:-42}" \
    --save-every "${SAVE_EVERY:-1000}" \
    --exact-resume --check-finite --model-tag "${MODEL_TAG:-${model_type}_24blocks_baseline}" "$@"
