#!/bin/bash
# Launch d12 attn_res_qkv training on GPUs 4-7.
cd /hdd/mh3897/cc/nanochat
export NANOCHAT_BASE_DIR=/local-ssd/mh3897
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec .venv/bin/torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
  --depth=12 \
  --model-type=attn_res_qkv \
  --model-tag=arch_d12_attn_res_qkv \
  --target-flops=1.5e18 \
  --device-batch-size=8 \
  --run=dummy
