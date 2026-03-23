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

The `model_type` is saved in checkpoint metadata, so evaluation auto-detects the right model class.

## Training

```bash
# Setup (one-time)
uv sync --extra gpu
source .venv/bin/activate
python -m nanochat.dataset -n 170   # download data
python -m scripts.tok_train          # train tokenizer

# Train a model
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 --model-type=gpt --model-tag=my_experiment
```

## Evaluation

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- --model-tag=my_experiment
```

## Architecture experiments

We compare transformer variants that differ in how information flows across depth, using FLOP-controlled training.

| Model | `--model-type` | Residual mechanism |
|---|---|---|
| Vanilla GPT | `gpt_nolambda` | Standard `h + f(h)` |
| GPT + lambdas | `gpt` | Learned per-layer residual and embedding scaling |
| AttnRes | `attn_res` | Softmax attention over all prior sublayer outputs |
| Gated AttnRes | `gated_attn_res` | AttnRes + sigmoid bottleneck gate |

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
