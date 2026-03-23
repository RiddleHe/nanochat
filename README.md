# nanochat

![nanochat logo](dev/nanochat.png)

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). This fork is an experimental harness for pretraining architecture research, with a model registry system for comparing transformer variants at controlled compute budgets.

## Adding a new model architecture

1. Create a new file `nanochat/gpt_yourmodel.py` with a config dataclass and model class. The model must implement the same interface as `GPT`: `forward(idx, targets, kv_cache, loss_reduction)`, `init_weights()`, `setup_optimizer(...)`, `estimate_flops()`, `num_scaling_params()`, `generate(...)`, `get_device()`.

2. Register it in `nanochat/model_registry.py`:
```python
def _register_variants():
    from nanochat.gpt_yourmodel import YourModelConfig, YourModel
    register("yourmodel", YourModelConfig, YourModel)
```

3. Train it:
```bash
torchrun --nproc_per_node=4 -m scripts.base_train -- --depth=12 --model-type=yourmodel
```

The `model_type` is saved in checkpoint metadata, so evaluation auto-detects the right model class.

## Training

```bash
# Setup (one-time)
uv sync --extra gpu
source .venv/bin/activate
python -m nanochat.dataset -n 170   # download data
python -m scripts.tok_train          # train tokenizer

# Train a model (e.g. depth 12, 4 GPUs, compute-optimal horizon)
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --model-type=gpt \
    --model-tag=my_experiment

# Train with a fixed FLOP budget instead of compute-optimal
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --model-type=attn_res \
    --target-flops=1e18 \
    --target-param-data-ratio=-1 \
    --model-tag=my_attn_res
```

Key flags:
- `--depth`: number of transformer layers (determines model size)
- `--model-type`: architecture variant from the registry
- `--device-batch-size`: reduce if OOM (default 32, try 16, 8, 4)
- `--eval-every`: validate every N steps (default 250)
- `--target-flops` / `--target-param-data-ratio`: training horizon control

## Evaluation

```bash
# Evaluate a trained model (CORE benchmark + BPB + samples)
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- \
    --model-tag=my_experiment
```

## Architecture experiments

### Residual connection variants

We compare four approaches to information flow across transformer depth, all at a fixed 1e18 FLOP budget on depth-12 models:

| Model | Residual mechanism |
|---|---|
| `gpt_nolambda` | Standard `h + f(h)` (true vanilla baseline) |
| `gpt` | `resid_lambda * h + x0_lambda * x0 + f(h)` (learned per-layer scaling) |
| `attn_res` | Softmax attention over all prior sublayer outputs ([Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals)) |
| `gated_attn_res` | AttnRes + sigmoid bottleneck gate on the aggregated output (inspired by [Gated Attention](https://arxiv.org/abs/2505.06708)) |

Run the full comparison:
```bash
NPROC_PER_NODE=4 bash runs/attn_res_arch_compare.sh
```

**Early finding: GPT (with lambdas) vs AttnRes.** With backout enabled in GPT and disabled in AttnRes (first run, not fully controlled), GPT with lambdas converges to 0.848 BPB while AttnRes reaches 1.004 BPB at 1e18 FLOPs. The gap is consistent throughout training, suggesting a structural advantage for the simple learned-scalar approach over softmax-over-depth at this scale. A fairer 4-model comparison (with backout removed from all variants) is in progress.

**Gated AttnRes.** Adding a sigmoid bottleneck gate (`d -> d//4 -> d`) after AttnRes aggregation, inspired by the gated attention paper's finding that post-softmax gating adds beneficial non-linearity and sparsity. Initial results showed a training plateau from the gate's zero-init (sigmoid(0)=0.5 halves the signal); switching to uniform init (matching the paper's normal(0, 0.02) approach) is expected to help.

## Cite

If you build on this fork:

```bibtex
@misc{nanochat-arch,
  author = {Muyu He},
  title = {nanochat-arch: Architecture experiments on nanochat},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RiddleHe/nanochat}
}
```

This fork is based on:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
