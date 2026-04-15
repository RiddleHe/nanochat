# nanochat

![nanochat logo](dev/nanochat_rl.png)

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). This fork makes it easy to do pretraining architecture research and RL algo research.

# nanochat

Pretraining architecture research on the nanochat stack.

## Adding a new architecture

1. Create a new file `nanochat/model/gpt_yourmodel.py` with a config dataclass and model class. The model must implement the same interface as `GPT`: `forward(idx, targets, kv_cache, loss_reduction)`, `init_weights()`, `setup_optimizer(...)`, `estimate_flops()`, `num_scaling_params()`, `generate(...)`, `get_device()`.

2. Register it in `nanochat/model_registry.py`:
```python
def _register_variants():
    from nanochat.model.gpt_yourmodel import YourModelConfig, YourModel
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

## Leaderboard

FLOP-controlled, depth-24 runs. Validation bpb is the minimum reached over training (lower is better). 

| Model | Mechanism | Min val bpb |
|---|---|---|
| karpathy/nanochat default | — | 0.750794 |
| AttnRes | Softmax attention over all prior sublayer outputs | 0.742577 |
| karpathy/nanochat autoresearch 2 | — | 0.71800 |
| AttnRes + load balancing | AttnRes with load-balancing aux loss | **0.698283** |

# nanoRL

Standalone RL package (`nanorl/`) that runs on top of HuggingFace base models (e.g. Qwen3-0.6B) with vLLM for rollouts. See [knowledge/rl_training.md](knowledge/rl_training.md) for full docs.

## Adding a loss

Add a function to `nanorl/loss.py` with signature `loss_fn(logprobs, old_logprobs, rewards, **kwargs) -> scalar` and register it in `ALGORITHMS`. If it needs a non-standard advantage estimator, add a branch to `compute_advantages`.

## Adding data

The pipeline consumes JSONL with rows `{id, prompt, kind, payload, meta}` where `kind` selects a verifier (`code_call_based`, `code_stdin_stdout`, ...). To add a task:

1. Write a prep script in `nanorl/scripts/` that emits the JSONL.
2. Register the path in `_RL_DATASET_PATHS` in `nanorl/data.py`.
3. For a new `kind`, add an elif branch in `nanorl.data.verify` and a sibling helper.

## Training

```bash
./nanorl/runs/train.sh [tag]
```

Launches a vLLM rollout worker on `ROLLOUT_GPU` and a `torchrun` trainer on `TRAIN_GPUS` (set inside the script). Strict-sync: each step's rollouts come from the just-checkpointed weights.

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
