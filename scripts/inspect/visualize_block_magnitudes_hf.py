"""Visualize per-layer block output magnitude for HuggingFace models.

Runs validation samples through one or more HF models and plots:
  Mean L2 magnitude of each block's output (attn + mlp) vs layer index,
  with +/- 1 std shaded region.

Usage:
    python -m scripts.inspect.visualize_block_magnitudes_hf \
        --models Qwen/Qwen3-8B \
        --labels "Qwen3 8B" \
        --num-samples 100 --device-type cuda
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanochat.common import autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, nargs="+", required=True, help="HuggingFace model IDs")
parser.add_argument("--labels", type=str, nargs="+", default=None, help="legend labels (one per model)")
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.labels is None:
    args.labels = args.models
assert len(args.labels) == len(args.models), "Number of labels must match number of models"

if args.output is None:
    tag = "_vs_".join(m.replace("/", "_") for m in args.models)
    args.output = f"results/block_magnitudes_hf_{tag}.png"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)

# Use nanochat tokenizer + val data for consistent comparison
nc_tokenizer = get_tokenizer()

all_results = []  # list of (label, n_layer, mean_mag, var_mag)

for model_id, label in zip(args.models, args.labels):
    print(f"\n=== {label} ({model_id}) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    n_layer = model.config.num_hidden_layers
    seq_len = min(getattr(model.config, "max_position_embeddings", 2048), 2048)

    mag_sum = torch.zeros(n_layer, device="cpu")
    mag_sq_sum = torch.zeros(n_layer, device="cpu")
    mag_count = torch.zeros(n_layer, device="cpu")

    # Hook storage
    attn_outputs = {}
    mlp_outputs = {}

    def make_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            # HF attention returns (hidden_states, attn_weights, ...) or just hidden_states
            out = output[0] if isinstance(output, tuple) else output
            attn_outputs[layer_idx] = out.detach()
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, input, output):
            mlp_outputs[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        nc_tokenizer, 1, seq_len, split="val", device=device
    )

    print(f"Running {args.num_samples} validation samples ({n_layer} layers, seq_len={seq_len})...")

    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            x, y = next(val_loader)
            model(x)

            for layer_idx in range(n_layer):
                block_out = attn_outputs[layer_idx].float() + mlp_outputs[layer_idx].float()
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
    line, = ax.plot(layers, mean_mag, marker='o', markersize=3, linewidth=2, label=label)
    ax.fill_between(layers, np.maximum(mean_mag - std_mag, 1e-1), mean_mag + std_mag, alpha=0.2, color=line.get_color())

ax.set_yscale("log")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("L2 Magnitude", fontsize=12)
ax.set_title("Block Output Magnitude", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved to {args.output}")
