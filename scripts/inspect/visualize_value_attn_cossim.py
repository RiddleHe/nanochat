"""Visualize per-layer cosine similarity between value vectors and attention outputs.

For each layer i and each (B, T, head) slot, captures:
  - v_i:       output of c_v (the value projection),  reshaped (B, T, n_head, head_dim)
               (broadcast across query heads when n_kv_head < n_head)
  - y_i:       input to c_proj (attention output, post softmax + value aggregation,
               BEFORE the output projection),         shape (B, T, n_head, head_dim)
Cosine similarity is computed PER HEAD over the head_dim axis, then averaged across
(heads, batch, tokens, samples). Plots three subplots:
  1. cos_sim(v_i, y_i)  per layer
  2. cos_sim(v_0, v_i)  per layer
  3. cos_sim(v_0, y_i)  per layer

Usage:
    python -m scripts.inspect.visualize_value_attn_cossim \\
        --model-tag arch_d12_gpt_base --num-samples 100
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit


parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.output is None:
    args.output = f"results/value_attn_cossim_{args.model_tag}.png"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)

model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
model.eval()

config = meta["model_config"]
n_layer = config["n_layer"]
n_head = config["n_head"]
n_kv_head = config["n_kv_head"]
n_embd = config["n_embd"]
head_dim = n_embd // n_head
group_size = n_head // n_kv_head

# Hook storage: per-layer captured tensors (one batch at a time)
v_outputs = {}  # layer_idx -> (B, T, n_kv_head, head_dim)
y_outputs = {}  # layer_idx -> (B, T, n_head,    head_dim)


def make_v_hook(layer_idx):
    def hook_fn(module, _input, output):
        # c_v output: (B, T, n_kv_head * head_dim)
        B, T, _ = output.shape
        v_outputs[layer_idx] = output.detach().view(B, T, n_kv_head, head_dim)
    return hook_fn


def make_y_hook(layer_idx):
    def hook_fn(module, _input, _output):
        # c_proj input is the post-attention pre-projection y, shape (B, T, n_embd)
        x = _input[0].detach()
        B, T, _ = x.shape
        y_outputs[layer_idx] = x.view(B, T, n_head, head_dim)
    return hook_fn


hooks = []
for i, block in enumerate(model.transformer.h):
    hooks.append(block.attn.c_v.register_forward_hook(make_v_hook(i)))
    hooks.append(block.attn.c_proj.register_forward_hook(make_y_hook(i)))

# Per-layer running sums for averaging
sum_v_y = torch.zeros(n_layer)
sum_v0_v = torch.zeros(n_layer)
sum_v0_y = torch.zeros(n_layer)
n_batches = 0

val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, 1, config["sequence_len"], split="val", device=device
)

print(f"Running {args.num_samples} validation samples...")

with torch.no_grad():
    for sample_idx in range(args.num_samples):
        x, _y = next(val_loader)
        model(x)

        # v_0 broadcast to per-query-head shape: (B, T, n_head, head_dim)
        v0 = v_outputs[0].repeat_interleave(group_size, dim=2).float()

        for i in range(n_layer):
            v_i = v_outputs[i].repeat_interleave(group_size, dim=2).float()  # (B, T, n_head, head_dim)
            y_i = y_outputs[i].float()                                       # (B, T, n_head, head_dim)

            # Cos sim along head_dim, then mean over (B, T, n_head)
            sum_v_y[i]  += F.cosine_similarity(v_i, y_i, dim=-1).mean().cpu()
            sum_v0_v[i] += F.cosine_similarity(v0,  v_i, dim=-1).mean().cpu()
            sum_v0_y[i] += F.cosine_similarity(v0,  y_i, dim=-1).mean().cpu()

        n_batches += 1
        v_outputs.clear()
        y_outputs.clear()

        if (sample_idx + 1) % 10 == 0:
            print(f"  {sample_idx + 1}/{args.num_samples}")

for h in hooks:
    h.remove()

cos_v_y  = (sum_v_y  / n_batches).numpy()
cos_v0_v = (sum_v0_v / n_batches).numpy()
cos_v0_y = (sum_v0_y / n_batches).numpy()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
layers = np.arange(n_layer)

for ax, data, title in [
    (axes[0], cos_v_y,  "cos_sim(v_i, attn_out_i)"),
    (axes[1], cos_v0_v, "cos_sim(v_0, v_i)"),
    (axes[2], cos_v0_y, "cos_sim(v_0, attn_out_i)"),
]:
    ax.plot(layers, data, marker='o', linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("avg cos sim", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)

fig.suptitle(f"Value/Attention Cosine Similarity: {args.model_tag}", fontsize=14)
plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved to {args.output}")
