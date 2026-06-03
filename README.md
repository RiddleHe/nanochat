![nanochat logo](dev/nanochat_rl.png)

This repo hosts two distinct training frameworks that are both extremely hackable: `nanochat/` manages pretraining runs for architecture ablations, and `nanorl/` manages RL runs for objective-function ablations. Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat).

# nanochat

## How to add a new architecture

Most variants are best expressed by adding a boolean flag on the config dataclass in `nanochat/model/gpt_base.py`, a corresponding branch in the attention or block forward (often gated to the last third of layers via the `is_target` helper), and any new learnable scalars in `__init__` with their starting values set in `init_weights`. Then register a small config subclass and a name in `nanochat/model_registry.py` so the training script can find it. If a variant is large enough to warrant its own file, create `nanochat/model/gpt_<name>.py` with a matching config and model class and register that instead — the checkpoint metadata records the model type, so evaluation auto-detects the right class.

## How to train and evaluate the model

Pretraining is launched from `scripts/base_train.py` via `torchrun`, which handles distributed setup, learning-rate schedule, gradient accumulation, and periodic validation. For ablation studies, `runs/ablations.sh` wraps a batch of variants at the depth-appropriate compute budget (FLOPs defaults baked in for d12 and d24), skips anything already trained, and aggregates val-loss curves into a single CSV for direct comparison. After training, `scripts/base_eval.py` scores the model on the DCLM CORE suite — 21 ICL tasks listed in `configs/core.yaml` using fixtures from `eval_bundle/` — and produces per-task accuracy plus a centered aggregate.

# nanoRL

## How to add a new objective function

Add a function to `nanorl/loss.py` and register it in the `ALGORITHMS` dictionary. The function receives the current and reference log-probabilities, rewards, and any extra kwargs, and returns a scalar loss. If the objective requires a non-standard advantage estimator (e.g. token-level instead of trajectory-level), extend `compute_advantages` with a matching branch.

## How to train the model

Training is driven by the script under `nanorl/runs/`, which launches a vLLM rollout worker for fast batched sampling and a separate `torchrun` trainer; the two stay strictly in sync so every step's rollouts come from the just-checkpointed weights. See `knowledge/rl_training.md` for the longer guide.

# Cite

```bibtex
@misc{nanochat-arch,
  author = {Muyu He, Yuchen Liu},
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
