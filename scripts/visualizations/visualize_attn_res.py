"""Visualize AttnRes depth-wise attention weights on validation data.

Produces heatmaps (Pre-Attn and Pre-MLP) showing which source layers
each block attends to, averaged across samples and token positions.

If the checkpoint dir has multiple checkpoints, each is a row of subplots
(sorted by step), with step shown in the row title.

Works with any AttnRes variant (attn_res, attn_res_balanced, etc.) as long
as the model has `attn_res_queries` and the standard `_attn_res` mechanism.

Usage:
    python -m scripts.visualizations.visualize_attn_res --model-tag arch_d12_attn_res --num-samples 100
    python -m scripts.visualizations.visualize_attn_res --model-tag arch_d12_attn_res_balanced --num-samples 100
"""
import argparse
import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.model.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True, help="model tag in base_checkpoints (e.g. arch_d12_attn_res, arch_d12_attn_res_balanced)")
parser.add_argument("--step", type=int, default=None, help="specific step (default: all checkpoints)")
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.output is None:
    args.output = f"results/attn_res_heatmap_{args.model_tag}.png"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)

# Discover checkpoints
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
if args.step is not None:
    steps = [args.step]
else:
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    steps = sorted(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in checkpoint_files)
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

print(f"Found {len(steps)} checkpoint(s): {steps}")


def collect_attn_weights(model, config, val_loader, num_samples):
    """Run validation samples and collect averaged attention weights."""
    n_layer = config["n_layer"]
    n_queries = 2 * n_layer + 1
    max_sources = n_queries

    attn_weights_sum = torch.zeros(n_queries, max_sources)
    attn_weights_count = torch.zeros(n_queries)

    with torch.no_grad():
        for sample_idx in range(num_samples):
            x, y = next(val_loader)
            B, T = x.size()

            tok_emb = model.transformer.wte(x)
            tok_emb = tok_emb.to(COMPUTE_DTYPE)
            tok_emb = norm(tok_emb)

            if hasattr(model, 'smear_lambda'):
                gate = model.smear_lambda.to(tok_emb.dtype) * torch.sigmoid(model.smear_gate(tok_emb[:, 1:, :24]))
                tok_emb = torch.cat([tok_emb[:, :1], tok_emb[:, 1:] + gate * tok_emb[:, :-1]], dim=1)

            cos_sin = model.cos[:, :T], model.sin[:, :T]

            v_list = [tok_emb]
            qi = 0

            for i, block in enumerate(model.transformer.h):
                # Pre-attn AttnRes
                query = model.attn_res_queries[qi]
                V = torch.stack(v_list, dim=0)
                K = norm(V)
                logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
                weights = logits.softmax(dim=0)

                w_avg = weights.mean(dim=(1, 2))
                n_sources = len(v_list)
                attn_weights_sum[qi, :n_sources] += w_avg.cpu()
                attn_weights_count[qi] += 1

                h = torch.einsum('n b t, n b t d -> b t d', weights, V)
                qi += 1

                ve = model.value_embeds[str(i)](x).to(h.dtype) if str(i) in model.value_embeds else None
                attn_out = block.attn(norm(h), ve, cos_sin, model.window_sizes[i], None)
                v_list.append(attn_out)

                # Pre-MLP AttnRes
                query = model.attn_res_queries[qi]
                V = torch.stack(v_list, dim=0)
                K = norm(V)
                logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
                weights = logits.softmax(dim=0)

                w_avg = weights.mean(dim=(1, 2))
                n_sources = len(v_list)
                attn_weights_sum[qi, :n_sources] += w_avg.cpu()
                attn_weights_count[qi] += 1

                h = torch.einsum('n b t, n b t d -> b t d', weights, V)
                qi += 1

                mlp_out = block.mlp(norm(h))
                v_list.append(mlp_out)

            # Final query
            query = model.attn_res_queries[qi]
            V = torch.stack(v_list, dim=0)
            K = norm(V)
            logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
            weights = logits.softmax(dim=0)
            w_avg = weights.mean(dim=(1, 2))
            n_sources = len(v_list)
            attn_weights_sum[qi, :n_sources] += w_avg.cpu()
            attn_weights_count[qi] += 1

            if (sample_idx + 1) % 10 == 0:
                print(f"  {sample_idx + 1}/{num_samples}")

    attn_weights_avg = attn_weights_sum / attn_weights_count.unsqueeze(1).clamp(min=1)
    return attn_weights_avg, n_layer


def split_and_mask(attn_weights_avg, n_layer):
    """Split into pre-attn/pre-mlp and mask invalid entries."""
    max_sources = attn_weights_avg.shape[1]
    pre_attn_indices = list(range(0, 2 * n_layer, 2))
    pre_mlp_indices = list(range(1, 2 * n_layer, 2))

    pre_attn = attn_weights_avg[pre_attn_indices].numpy()
    pre_mlp = attn_weights_avg[pre_mlp_indices].numpy()

    for i in range(n_layer):
        n_sources_attn = 2 * i + 1
        n_sources_mlp = 2 * i + 2
        pre_attn[i, n_sources_attn:] = np.nan
        pre_mlp[i, n_sources_mlp:] = np.nan

    return pre_attn, pre_mlp


# Collect data for all steps
all_results = []
val_loader = None

for step in steps:
    print(f"\n=== Step {step} ===")
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=step)
    model.eval()
    config = meta["model_config"]

    if val_loader is None:
        val_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, 1, config["sequence_len"], split="val", device=device
        )

    attn_weights_avg, n_layer = collect_attn_weights(model, config, val_loader, args.num_samples)
    pre_attn, pre_mlp = split_and_mask(attn_weights_avg, n_layer)
    all_results.append((step, pre_attn, pre_mlp, n_layer))

    del model
    torch.cuda.empty_cache()

# Plot: rows = Pre-Attn / Pre-MLP, columns = checkpoints
n_cols = len(all_results)
fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols + 2, 14))
if n_cols == 1:
    axes = axes[:, np.newaxis]  # ensure 2D indexing

# Global vmax for consistent color scale
vmax = max(
    max(np.nanmax(pre_attn), np.nanmax(pre_mlp))
    for _, pre_attn, pre_mlp, _ in all_results
)

cmap = plt.cm.Blues.copy()
cmap.set_bad(color='white', alpha=0)

for col, (step, pre_attn, pre_mlp, n_layer) in enumerate(all_results):
    max_sources = pre_attn.shape[1]
    pre_attn_masked = np.ma.array(pre_attn, mask=np.isnan(pre_attn))
    pre_mlp_masked = np.ma.array(pre_mlp, mask=np.isnan(pre_mlp))

    ax1, ax2 = axes[0, col], axes[1, col]

    ax1.imshow(pre_attn_masked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax1.set_title(f"Step {step:,} — Pre-Attn", fontsize=14)
    ax1.set_xlabel("Source Index", fontsize=12)
    if col == 0:
        ax1.set_ylabel("Layer", fontsize=12)
    ax1.set_yticks(range(n_layer))
    ax1.set_yticklabels(range(1, n_layer + 1))

    for i in range(n_layer):
        n_sources_attn = 2 * i + 1
        if n_sources_attn < max_sources:
            ax1.add_patch(plt.Rectangle((n_sources_attn - 0.5, i - 0.5), max_sources - n_sources_attn, 1,
                                         fill=False, hatch='///', edgecolor='gray', linewidth=0))

    ax2.imshow(pre_mlp_masked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax2.set_title(f"Step {step:,} — Pre-MLP", fontsize=14)
    ax2.set_xlabel("Source Index", fontsize=12)
    if col == 0:
        ax2.set_ylabel("Layer", fontsize=12)
    ax2.set_yticks(range(n_layer))
    ax2.set_yticklabels(range(1, n_layer + 1))

    for i in range(n_layer):
        n_sources_mlp = 2 * i + 2
        if n_sources_mlp < max_sources:
            ax2.add_patch(plt.Rectangle((n_sources_mlp - 0.5, i - 0.5), max_sources - n_sources_mlp, 1,
                                         fill=False, hatch='///', edgecolor='gray', linewidth=0))

plt.suptitle(f"AttnRes Heatmap: {args.model_tag}", fontsize=16, y=1.0)
plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved to {args.output}")
