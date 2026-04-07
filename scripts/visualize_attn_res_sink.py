"""Visualize AttnRes+Sink depth-wise attention weights on validation data.

Same as visualize_attn_res.py but includes the sink weight as the first column
(source index 0 = sink, source index 1+ = real layer outputs).

Usage:
    python -m scripts.visualize_attn_res_sink --model-tag arch_d12_attn_res_sink --num-samples 100
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.model.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.output is None:
    args.output = f"results/attn_res_sink_heatmap_{args.model_tag}.png"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
model, _, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
model.eval()

config = meta["model_config"]
n_layer = config["n_layer"]
n_queries = 2 * n_layer + 1
# +1 for sink column
max_sources = n_queries + 1  # sink + real sources

tokenizer = get_tokenizer()
val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, 1, config["sequence_len"], split="val", device=device
)

# Storage: [query_idx, source_idx] where source 0 = sink, source 1+ = real
attn_weights_sum = torch.zeros(n_queries, max_sources)
attn_weights_count = torch.zeros(n_queries)

print(f"Running {args.num_samples} validation samples...")

with torch.no_grad():
    for sample_idx in range(args.num_samples):
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
            # Pre-attn AttnRes with sink
            query = model.attn_res_queries[qi]
            sink_logit = model.sink_logits[qi]
            V = torch.stack(v_list, dim=0)
            K = norm(V)
            logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
            sink = sink_logit.to(logits.dtype).expand(1, B, T)
            logits_with_sink = torch.cat([logits, sink], dim=0)  # (N+1, B, T)
            weights = logits_with_sink.softmax(dim=0)  # (N+1, B, T)

            # Store weights: col 0 = sink (last in softmax), col 1+ = real sources
            w_avg = weights.mean(dim=(1, 2))  # (N+1,)
            n_real = len(v_list)
            # Rearrange: sink first, then real sources
            attn_weights_sum[qi, 0] += w_avg[-1].cpu()  # sink
            attn_weights_sum[qi, 1:n_real+1] += w_avg[:-1].cpu()  # real sources
            attn_weights_count[qi] += 1

            # Compute h without sink for forward pass
            weights_no_sink = weights[:-1]
            h = torch.einsum('n b t, n b t d -> b t d', weights_no_sink, V)
            qi += 1

            ve = model.value_embeds[str(i)](x).to(h.dtype) if str(i) in model.value_embeds else None
            attn_out = block.attn(norm(h), ve, cos_sin, model.window_sizes[i], None)
            v_list.append(attn_out)

            # Pre-MLP AttnRes with sink
            query = model.attn_res_queries[qi]
            sink_logit = model.sink_logits[qi]
            V = torch.stack(v_list, dim=0)
            K = norm(V)
            logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
            sink = sink_logit.to(logits.dtype).expand(1, B, T)
            logits_with_sink = torch.cat([logits, sink], dim=0)
            weights = logits_with_sink.softmax(dim=0)

            w_avg = weights.mean(dim=(1, 2))
            n_real = len(v_list)
            attn_weights_sum[qi, 0] += w_avg[-1].cpu()
            attn_weights_sum[qi, 1:n_real+1] += w_avg[:-1].cpu()
            attn_weights_count[qi] += 1

            weights_no_sink = weights[:-1]
            h = torch.einsum('n b t, n b t d -> b t d', weights_no_sink, V)
            qi += 1

            mlp_out = block.mlp(norm(h))
            v_list.append(mlp_out)

        # Final query
        query = model.attn_res_queries[qi]
        sink_logit = model.sink_logits[qi]
        V = torch.stack(v_list, dim=0)
        K = norm(V)
        logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
        sink = sink_logit.expand(1, B, T)
        logits_with_sink = torch.cat([logits, sink], dim=0)
        weights = logits_with_sink.softmax(dim=0)

        w_avg = weights.mean(dim=(1, 2))
        n_real = len(v_list)
        attn_weights_sum[qi, 0] += w_avg[-1].cpu()
        attn_weights_sum[qi, 1:n_real+1] += w_avg[:-1].cpu()
        attn_weights_count[qi] += 1

        if (sample_idx + 1) % 10 == 0:
            print(f"  {sample_idx + 1}/{args.num_samples}")

attn_weights_avg = attn_weights_sum / attn_weights_count.unsqueeze(1).clamp(min=1)

# Split into pre-attn and pre-mlp
pre_attn_indices = list(range(0, 2 * n_layer, 2))
pre_mlp_indices = list(range(1, 2 * n_layer, 2))

pre_attn_weights = attn_weights_avg[pre_attn_indices].numpy()
pre_mlp_weights = attn_weights_avg[pre_mlp_indices].numpy()

# Mask invalid entries (source index > available sources + sink)
for i in range(n_layer):
    n_sources_attn = 2 * i + 1 + 1  # real sources + sink
    n_sources_mlp = 2 * i + 2 + 1
    pre_attn_weights[i, n_sources_attn:] = np.nan
    pre_mlp_weights[i, n_sources_mlp:] = np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

cmap = plt.cm.Blues.copy()
cmap.set_bad(color='white', alpha=0)

vmax = max(np.nanmax(pre_attn_weights), np.nanmax(pre_mlp_weights))

im1 = ax1.imshow(pre_attn_weights, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
ax1.set_title("Full AttnRes+Sink, Pre-Attn", fontsize=14)
ax1.set_xlabel("Source Index (0=sink)", fontsize=12)
ax1.set_ylabel("Layer", fontsize=12)
ax1.set_yticks(range(n_layer))
ax1.set_yticklabels(range(1, n_layer + 1))

# Hatching for invalid region
for i in range(n_layer):
    n_valid = 2 * i + 1 + 1
    if n_valid < max_sources:
        ax1.add_patch(plt.Rectangle((n_valid - 0.5, i - 0.5), max_sources - n_valid, 1,
                                     fill=False, hatch='///', edgecolor='gray', linewidth=0))

im2 = ax2.imshow(pre_mlp_weights, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
ax2.set_title("Full AttnRes+Sink, Pre-MLP", fontsize=14)
ax2.set_xlabel("Source Index (0=sink)", fontsize=12)
ax2.set_ylabel("Layer", fontsize=12)
ax2.set_yticks(range(n_layer))
ax2.set_yticklabels(range(1, n_layer + 1))

for i in range(n_layer):
    n_valid = 2 * i + 2 + 1
    if n_valid < max_sources:
        ax2.add_patch(plt.Rectangle((n_valid - 0.5, i - 0.5), max_sources - n_valid, 1,
                                     fill=False, hatch='///', edgecolor='gray', linewidth=0))

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved to {args.output}")
