# Canonical d24 baseline recipes

`bov/d24_f3p91e19/common.json` preserves the BoV baseline comparison recipe:
24 layers, width 1,536, 12 heads of width 128, context 2,048, SSSL attention,
seed 42, BF16, **1,048,576 tokens per optimizer step**, and **3.91e19 estimated
training FLOPs**. Optimizer and evaluation settings are explicit in that file.

Two model entry points share this recipe:

- `gpt_base.sh`: no value-embedding tables, residual/x0 lambdas, or embedding smear;
  retains the initial embedding RMSNorm.
- `gpt_base_no_embed_norm.sh`: additionally omits the initial embedding RMSNorm;
  block and final RMSNorm remain.

Both models resolve to 7,809 steps / 8,188,329,984 tokens with the reference
32,768-token vocabulary. FLOPs count matrix and attention work; elementwise
normalization is excluded, so removing embedding RMSNorm does not change the
estimated budget. Whole-step rounding slightly changes the realized estimate.

Prepare using an existing Python environment, tokenizer, and dataset:

```bash
PYTHON=/path/to/venv/bin/python \
NANOCHAT_BASE_DIR=/path/to/existing/data \
bash runs/bov/d24_f3p91e19/gpt_base_no_embed_norm.sh --gpu-count 2 --dry-run
```

Preparation uses CPU/meta tensors and does not install dependencies, download
inputs, launch training, or start a queue. The global token batch stays fixed
across GPU counts; only gradient accumulation changes. With two GPUs and four
sequences per GPU, accumulation is 64. `--device-batch-size` can reduce the
microbatch if needed, provided the global batch remains divisible.

`--manifest /absolute/path/manifest.json` writes a new manifest recording the
resolved command/config, source and tokenizer hashes, dataset inventory, runtime,
and planned outputs. Its parent directory must exist. A manifest records source
hashes; it does not itself freeze or copy the source. Preserve a matching source
snapshot before executing a reproducible research run.

Training requires explicit `--launch` and `CUDA_VISIBLE_DEVICES` containing
exactly `--gpu-count` devices. Check their availability before launching. The
launcher refuses existing log/checkpoint destinations; manifests are never
overwritten. Checkpoints are saved every 1,000 steps plus the final step.
Resuming requires compatible model/config and all optimizer shards; exact
replay of data/RNG state is unsupported.

The older `ablations.sh` remains available for generic comparisons with automatic
global batch selection. Use these recipes for the fixed BoV comparison budget.
