"""Visualize per-layer block output magnitude and variance.

Runs validation samples through one or more models and plots (overlaid):
  1. Mean L2 magnitude of each block's output (attn + mlp) vs layer index
  2. Variance of block output magnitudes vs layer index

Usage:
    python -m scripts.inspect.visualize_block_magnitudes \
        --model-tags arch_d12_gpt_nolambda arch_d12_attn_res \
        --labels "Baseline" "AttnRes" \
        --num-samples 100 --device-type cuda
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

parser = argparse.ArgumentParser()
parser.add_argument("--model-tags", type=str, nargs="+", required=True, help="model tags in base_checkpoints")
parser.add_argument("--labels", type=str, nargs="+", default=None, help="legend labels (one per model tag)")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.labels is None:
    args.labels = args.model_tags
assert len(args.labels) == len(args.model_tags), "Number of labels must match number of model tags"

if args.output is None:
    args.output = f"results/block_magnitudes_{'_vs_'.join(args.model_tags)}.png"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)

# Collect results per model
all_results = []  # list of (label, n_layer, mean_mag, var_mag)

for model_tag, label in zip(args.model_tags, args.labels):
    print(f"\n=== {label} ({model_tag}) ===")
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=args.step)
    model.eval()

    config = meta["model_config"]
    n_layer = config["n_layer"]

    mag_sum = torch.zeros(n_layer, device="cpu")
    mag_sq_sum = torch.zeros(n_layer, device="cpu")
    mag_count = torch.zeros(n_layer, device="cpu")

    # Hook storage — capture both attn and mlp outputs
    attn_outputs = {}
    mlp_outputs = {}

    def make_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            # Some attention modules (e.g. gpt_base CausalSelfAttention) return (y, v_init)
            if isinstance(output, tuple):
                output = output[0]
            attn_outputs[layer_idx] = output.detach()
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            mlp_outputs[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, config["sequence_len"], split="val", device=device
    )

    print(f"Running {args.num_samples} validation samples...")

    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            x, y = next(val_loader)
            model(x)

            for layer_idx in range(n_layer):
                block_out = attn_outputs[layer_idx].float() + mlp_outputs[layer_idx].float()  # (B, T, D)
                magnitudes = block_out.norm(dim=-1)  # (B, T)
                mag_sum[layer_idx] += magnitudes.sum().cpu()
                mag_sq_sum[layer_idx] += (magnitudes ** 2).sum().cpu()
                mag_count[layer_idx] += magnitudes.numel()

            attn_outputs.clear()
            mlp_outputs.clear()

            if (sample_idx + 1) % 10 == 0:
                print(f"  {sample_idx + 1}/{args.num_samples}")

    for h in hooks:
        h.remove()

    mean_mag = (mag_sum / mag_count).numpy().astype(np.float64)
    var_mag = (mag_sq_sum / mag_count).numpy().astype(np.float64) - mean_mag ** 2
    all_results.append((label, n_layer, mean_mag, var_mag))

    del model
    torch.cuda.empty_cache()

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

for label, n_layer, mean_mag, var_mag in all_results:
    layers = np.arange(1, n_layer + 1)
    std_mag = np.sqrt(np.maximum(var_mag, 0))
    line, = ax.plot(layers, mean_mag, marker='o', linewidth=2, label=label)
    ax.fill_between(layers, np.maximum(mean_mag - std_mag, 1e-1), mean_mag + std_mag, alpha=0.2, color=line.get_color())

ax.set_yscale("log")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("L2 Magnitude", fontsize=12)
ax.set_title("Block Output Magnitude", fontsize=14)
ax.set_xticks(np.arange(1, max(r[1] for r in all_results) + 1))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved to {args.output}")
