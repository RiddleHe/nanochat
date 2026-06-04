"""Visualize per-Q/K/V depth-attention weights for the attn_res_qkv model.

In attn_res_qkv (see nanochat/model/gpt_attn_res_qkv.py), q, k, and v at each
layer each have their own AttnRes pseudo-query that attends over all previous
sublayer outputs. This script replays the forward pass, captures the softmax
weights of each q/k/v (and the pre-MLP) query, averages them over tokens and
samples, and plots one heatmap per stream:

    cell (layer i, source j) = how much q/k/v at layer i reads from sublayer j.

Concentration near the diagonal (recent layers) => the stream wants accumulated
context; concentration on source 0 / early layers => it wants the original token.

Usage:
    python -m scripts.inspect.visualize_attn_res_qkv \
        --model-tag arch_d12_attn_res_qkv --num-samples 100 \
        --output results/attn_res_qkv_heatmap.png
"""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.model.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, default="arch_d12_attn_res_qkv")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default="results/attn_res_qkv_heatmap.png")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)


def _weights(query, v_list):
    """Softmax depth-attention weights of `query` over the stacked sublayer outputs."""
    V = torch.stack(v_list, dim=0)              # (N, B, T, D)
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
    w = logits.softmax(dim=0)                   # (N, B, T)
    h = torch.einsum('n b t, n b t d -> b t d', w, V)
    return w, h


@torch.no_grad()
def collect(model, config, val_loader, num_samples):
    n_layer = config["n_layer"]
    n_queries = 4 * n_layer + 1
    wsum = torch.zeros(n_queries, n_queries)
    wcount = torch.zeros(n_queries)

    for s in range(num_samples):
        x, _ = next(val_loader)
        B, T = x.size()
        tok_emb = norm(model.transformer.wte(x).to(COMPUTE_DTYPE))
        if hasattr(model, 'smear_lambda'):
            g = model.smear_lambda.to(tok_emb.dtype) * torch.sigmoid(model.smear_gate(tok_emb[:, 1:, :24]))
            tok_emb = torch.cat([tok_emb[:, :1], tok_emb[:, 1:] + g * tok_emb[:, :-1]], dim=1)
        cos_sin = model.cos[:, :T], model.sin[:, :T]

        v_list = [tok_emb]
        qi = 0
        for i, block in enumerate(model.transformer.h):
            hs = {}
            for which in ("q", "k", "v"):
                w, h = _weights(model.attn_res_queries[qi], v_list)
                wsum[qi, :len(v_list)] += w.mean(dim=(1, 2)).cpu(); wcount[qi] += 1
                hs[which] = h; qi += 1
            ve = model.value_embeds[str(i)](x).to(hs["q"].dtype) if str(i) in model.value_embeds else None
            attn_out = block.attn(norm(hs["q"]), norm(hs["k"]), norm(hs["v"]), ve, cos_sin, model.window_sizes[i], None)
            v_list.append(attn_out)

            w, h = _weights(model.attn_res_queries[qi], v_list)   # pre-MLP
            wsum[qi, :len(v_list)] += w.mean(dim=(1, 2)).cpu(); wcount[qi] += 1
            qi += 1
            v_list.append(block.mlp(norm(h)))

        w, _ = _weights(model.attn_res_queries[qi], v_list)       # final
        wsum[qi, :len(v_list)] += w.mean(dim=(1, 2)).cpu(); wcount[qi] += 1

        if (s + 1) % 10 == 0:
            print(f"  {s + 1}/{num_samples}", flush=True)

    return (wsum / wcount.unsqueeze(1).clamp(min=1)), n_layer


def stream(avg, n_layer, offset):
    """Extract one stream (q=0, k=1, v=2, mlp=3) as a (n_layer, sources) array,
    masking sources that don't exist at each layer."""
    idx = [4 * i + offset for i in range(n_layer)]
    arr = avg[idx].numpy().copy()
    for i in range(n_layer):
        n_src = (2 * i + 1) if offset < 3 else (2 * i + 2)  # q/k/v see 2i+1; mlp sees 2i+2
        arr[i, n_src:] = np.nan
    return arr


# ---- run ----
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
model.eval()
config = meta["model_config"]
val_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 1, config["sequence_len"], split="val", device=device)
avg, n_layer = collect(model, config, val_loader, args.num_samples)

streams = [("Q", 0), ("K", 1), ("V", 2), ("MLP", 3)]
arrs = [(name, stream(avg, n_layer, off)) for name, off in streams]
# Adaptive cap so the diagonal band + source-0 column are visible in deep rows
# (early layers with a single source saturate near 1.0; we don't let that wash
# out the rest). White=low, dark blue=high, matching the AttnRes figure.
allvalid = np.concatenate([a[~np.isnan(a)] for _, a in arrs])
vmax = float(np.clip(np.percentile(allvalid, 96), 0.4, 0.85))
cmap = plt.cm.Blues.copy(); cmap.set_bad(color='#ededed')  # masked = light gray

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, (name, arr) in zip(axes, arrs):
    m = np.ma.array(arr, mask=np.isnan(arr))
    im = ax.imshow(m, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, origin='upper')
    ax.set_title(f"{name} depth-attention")
    ax.set_xlabel("source sublayer (0 = token emb)")
    ax.set_ylabel("layer index")
    ax.set_yticks(range(n_layer))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
fig.suptitle(f"attn_res_qkv ({args.model_tag}): which sublayers each of Q/K/V reads from "
             f"(white=low, blue=high; gray=causally masked)", y=1.02)
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
fig.tight_layout()
fig.savefig(args.output, dpi=150, bbox_inches="tight")
print(f"saved {args.output}", flush=True)
