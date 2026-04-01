"""Visualize cross-layer angular distance heatmaps (curse of depth metric).

For each model, produces a heatmap where cell (i, j) shows the angular distance
between the hidden state at layer i and layer j, averaged over validation tokens.

Reference: "The Curse of Depth in Large Language Models" (arxiv 2502.05795)

Usage:
    python -m scripts.visualize_angular_distance \
        --model-tags arch_d12_gpt_nolambda arch_d12_attn_res \
        --labels "GPT baseline" "AttnRes" \
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

n_models = len(args.model_tags)
fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
if n_models == 1:
    axes = [axes]

# Shared colorbar range
all_matrices = []

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

    # Accumulate pairwise angular distances: (n_layer+1, n_layer+1)
    # Index 0 = embedding input, index i = input to layer i (1-indexed)
    n_points = n_layer + 1  # embedding + n_layer block inputs
    ang_dist_sum = torch.zeros(n_points, n_points)
    count = 0

    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            x_input, _ = next(val_loader)

            # Capture input to each attention sublayer via forward hooks
            layer_inputs = []

            def make_hook(layer_inputs_list):
                def hook_fn(module, input, output):
                    layer_inputs_list.append(input[0].detach())
                return hook_fn

            hooks = []
            for block in model.transformer.h:
                hooks.append(block.attn.register_forward_hook(make_hook(layer_inputs)))

            model(x_input)

            for h in hooks:
                h.remove()

            # layer_inputs[i] is the normed input to attention at layer i
            # Compute all pairwise angular distances
            reps = [li.float() for li in layer_inputs]  # n_layer tensors of (B, T, D)

            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    cos_sim = torch.nn.functional.cosine_similarity(reps[i], reps[j], dim=-1)
                    cos_sim = cos_sim.clamp(-1, 1)
                    ang_dist = torch.acos(cos_sim) / torch.pi
                    avg = ang_dist.mean().cpu()
                    ang_dist_sum[i, j] += avg
                    ang_dist_sum[j, i] += avg

            count += 1
            layer_inputs.clear()

            if (sample_idx + 1) % 10 == 0:
                print(f"  {sample_idx + 1}/{args.num_samples}")

    ang_dist_avg = (ang_dist_sum / count).numpy()
    all_matrices.append(ang_dist_avg)

    del model
    if device_type == "cuda":
        torch.cuda.empty_cache()

# Find shared color range
vmax = max(m.max() for m in all_matrices)

for model_idx, (label, matrix) in enumerate(zip(args.labels, all_matrices)):
    ax = axes[model_idx]
    n_points = matrix.shape[0]
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_xticks(range(0, n_points, max(1, n_points // 8)))
    ax.set_yticks(range(0, n_points, max(1, n_points // 8)))

fig.colorbar(im, ax=axes, shrink=0.8, label="Angular Distance (0=identical, 1=orthogonal)")
fig.suptitle("Cross-Layer Angular Distance", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved to {args.output}")
