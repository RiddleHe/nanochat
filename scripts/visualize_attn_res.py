"""Visualize AttnRes depth-wise attention weights on validation data.

Produces two heatmaps (Pre-Attn and Pre-MLP) showing which source layers
each block attends to, averaged across samples and token positions.

Works with any AttnRes variant (attn_res, attn_res_balanced, etc.) as long
as the model has `attn_res_queries` and the standard `_attn_res` mechanism.

Usage:
    python -m scripts.visualize_attn_res --model-tag arch_d12_attn_res --num-samples 100
    python -m scripts.visualize_attn_res --model-tag arch_d12_attn_res_balanced --num-samples 100
"""
import argparse
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
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.output is None:
    args.output = f"results/attn_res_heatmap_{args.model_tag}.png"

# Load model
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
model.eval()

config = meta["model_config"]
n_layer = config["n_layer"]
n_queries = 2 * n_layer + 1  # pre-attn + pre-mlp + final
max_sources = n_queries  # max number of sources for the final query

# Storage for attention weights: [query_idx, source_idx] accumulated
# Pre-attn queries: indices 0, 2, 4, ... (even)
# Pre-mlp queries: indices 1, 3, 5, ... (odd)
# Final query: index 2*n_layer
attn_weights_sum = torch.zeros(n_queries, max_sources)
attn_weights_count = torch.zeros(n_queries)

# Load validation data
val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, 1, config["sequence_len"], split="val", device=device
)

print(f"Running {args.num_samples} validation samples...")

with torch.no_grad():
    for sample_idx in range(args.num_samples):
        x, y = next(val_loader)
        B, T = x.size()

        # Reproduce the forward pass to capture attention weights
        # Embed tokens
        tok_emb = model.transformer.wte(x)
        tok_emb = tok_emb.to(COMPUTE_DTYPE)
        tok_emb = norm(tok_emb)

        # Smear
        if hasattr(model, 'smear_lambda'):
            gate = model.smear_lambda.to(tok_emb.dtype) * torch.sigmoid(model.smear_gate(tok_emb[:, 1:, :24]))
            tok_emb = torch.cat([tok_emb[:, :1], tok_emb[:, 1:] + gate * tok_emb[:, :-1]], dim=1)

        # Rotary embeddings
        cos_sin = model.cos[:, :T], model.sin[:, :T]

        v_list = [tok_emb]
        qi = 0

        for i, block in enumerate(model.transformer.h):
            # Pre-attn AttnRes
            query = model.attn_res_queries[qi]
            V = torch.stack(v_list, dim=0)
            K = norm(V)
            logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
            weights = logits.softmax(dim=0)  # (N, B, T)

            # Average weights across batch and tokens
            w_avg = weights.mean(dim=(1, 2))  # (N,)
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
            print(f"  {sample_idx + 1}/{args.num_samples}")

# Average across samples
attn_weights_avg = attn_weights_sum / attn_weights_count.unsqueeze(1).clamp(min=1)

# Split into pre-attn (even indices) and pre-mlp (odd indices)
pre_attn_indices = list(range(0, 2 * n_layer, 2))  # 0, 2, 4, ...
pre_mlp_indices = list(range(1, 2 * n_layer, 2))    # 1, 3, 5, ...

pre_attn_weights = attn_weights_avg[pre_attn_indices].numpy()  # (n_layer, max_sources)
pre_mlp_weights = attn_weights_avg[pre_mlp_indices].numpy()    # (n_layer, max_sources)

# Mask out invalid entries (upper triangle — future sources)
for i in range(n_layer):
    n_sources_attn = 2 * i + 1       # embedding + prior sublayer outputs
    n_sources_mlp = 2 * i + 2        # +1 for the attn output of this layer
    pre_attn_weights[i, n_sources_attn:] = np.nan
    pre_mlp_weights[i, n_sources_mlp:] = np.nan

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Use masked arrays for proper handling of NaN
pre_attn_masked = np.ma.array(pre_attn_weights, mask=np.isnan(pre_attn_weights))
pre_mlp_masked = np.ma.array(pre_mlp_weights, mask=np.isnan(pre_mlp_weights))

vmax = max(np.nanmax(pre_attn_weights), np.nanmax(pre_mlp_weights))

cmap = plt.cm.Blues.copy()
cmap.set_bad(color='white', alpha=0)

im1 = ax1.imshow(pre_attn_masked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
ax1.set_title("Full AttnRes, Pre-Attn", fontsize=14)
ax1.set_xlabel("Source Index", fontsize=12)
ax1.set_ylabel("Layer", fontsize=12)
ax1.set_yticks(range(n_layer))
ax1.set_yticklabels(range(1, n_layer + 1))

# Add hatching for invalid region
for i in range(n_layer):
    n_sources_attn = 2 * i + 1
    if n_sources_attn < max_sources:
        ax1.add_patch(plt.Rectangle((n_sources_attn - 0.5, i - 0.5), max_sources - n_sources_attn, 1,
                                     fill=False, hatch='///', edgecolor='gray', linewidth=0))

im2 = ax2.imshow(pre_mlp_masked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
ax2.set_title("Full AttnRes, Pre-MLP", fontsize=14)
ax2.set_xlabel("Source Index", fontsize=12)
ax2.set_ylabel("Layer", fontsize=12)
ax2.set_yticks(range(n_layer))
ax2.set_yticklabels(range(1, n_layer + 1))

for i in range(n_layer):
    n_sources_mlp = 2 * i + 2
    if n_sources_mlp < max_sources:
        ax2.add_patch(plt.Rectangle((n_sources_mlp - 0.5, i - 0.5), max_sources - n_sources_mlp, 1,
                                     fill=False, hatch='///', edgecolor='gray', linewidth=0))

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved to {args.output}")
