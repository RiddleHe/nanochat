"""Visualize cross-layer angular distance (curse of depth metric).

Computes angular distance between consecutive layer inputs, averaged over
validation tokens. Compares multiple models side by side.

Reference: "The Curse of Depth in Large Language Models" (arxiv 2502.05795)

Usage:
    python -m scripts.visualize_angular_distance \
        --model-tags arch_d12_gpt_nolambda arch_d12_attn_res_lr001 \
        --labels "GPT baseline" "AttnRes (lr=0.001)" \
        --num-samples 50 \
        --output results/angular_distance_d12.png
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tags", type=str, nargs="+", required=True)
parser.add_argument("--labels", type=str, nargs="+", default=None)
parser.add_argument("--num-samples", type=int, default=50)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default="results/angular_distance.png")
args = parser.parse_args()

if args.labels is None:
    args.labels = args.model_tags
assert len(args.labels) == len(args.model_tags)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
tokenizer = get_tokenizer()

colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

fig, ax = plt.subplots(figsize=(10, 6))

for model_idx, (model_tag, label) in enumerate(zip(args.model_tags, args.labels)):
    print(f"\nProcessing: {label} ({model_tag})")
    model, _, meta = load_model("base", device, phase="eval", model_tag=model_tag)
    model.eval()
    config = meta["model_config"]
    n_layer = config["n_layer"]
    seq_len = config["sequence_len"]

    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, seq_len, split="val", device=device
    )

    # Accumulate angular distances: (n_layer,) — distance between layer i and i+1 inputs
    angular_dist_sum = torch.zeros(n_layer)
    angular_dist_count = 0

    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            x_input, _ = next(val_loader)

            # Capture the input to each attention sublayer via forward hooks
            # This works for both GPT (Block.forward calls self.attn) and AttnRes
            # (which calls block.attn directly). Either way, attn.forward gets called.
            layer_inputs = []

            def make_hook(layer_inputs_list):
                def hook_fn(module, input, output):
                    # input[0] is the normed hidden state entering attention
                    layer_inputs_list.append(input[0].detach())
                return hook_fn

            hooks = []
            for block in model.transformer.h:
                hooks.append(block.attn.register_forward_hook(make_hook(layer_inputs)))

            # Forward pass
            model(x_input)

            # Remove hooks
            for h in hooks:
                h.remove()

            # layer_inputs has n_layer entries, each (B, T, D)
            # Compute angular distance between consecutive layers
            for i in range(n_layer - 1):
                x_l = layer_inputs[i].float()      # (B, T, D)
                x_l1 = layer_inputs[i + 1].float()  # (B, T, D)

                # Cosine similarity per token
                cos_sim = torch.nn.functional.cosine_similarity(x_l, x_l1, dim=-1)  # (B, T)
                cos_sim = cos_sim.clamp(-1, 1)  # numerical safety
                ang_dist = torch.acos(cos_sim) / torch.pi  # (B, T), range [0, 1]
                angular_dist_sum[i] += ang_dist.mean().cpu()

            # Also compute distance for the last layer output vs its input
            # Use the model output (post-norm) vs last layer input
            angular_dist_count += 1
            layer_inputs.clear()

            if (sample_idx + 1) % 10 == 0:
                print(f"  {sample_idx + 1}/{args.num_samples}")

    angular_dist_avg = angular_dist_sum / angular_dist_count

    # Plot: x-axis is layer index, y-axis is angular distance to next layer
    layers = list(range(1, n_layer))
    ax.plot(layers, angular_dist_avg[:n_layer-1].numpy(),
            label=label, color=colors[model_idx % len(colors)], linewidth=2, marker='o', markersize=4)

    # Clean up model
    del model
    torch.cuda.empty_cache() if device_type == "cuda" else None

ax.set_xlabel("Layer", fontsize=13)
ax.set_ylabel("Angular Distance (0=identical, 1=orthogonal)", fontsize=13)
ax.set_title("Cross-Layer Angular Distance (Curse of Depth)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"\nSaved to {args.output}")
