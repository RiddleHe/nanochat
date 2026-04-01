"""Visualize gated AttnRes gate outputs (sigmoid values) per dim per layer.

Produces two heatmaps:
  1. Gate values (layer × dim), dims sorted by mean gate value
  2. Per-layer histogram of gate values

Usage:
    python -m scripts.visualize_gate_values --model-tag arch_d12_gated_attn_res --num-samples 50
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--num-samples", type=int, default=50)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

if args.output is None:
    args.output = f"results/gate_values_{args.model_tag}.png"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)

model, _, meta = load_model("base", device, phase="eval", model_tag=args.model_tag)
model.eval()
config = meta["model_config"]
n_layer = config["n_layer"]
n_embd = config["n_embd"]
seq_len = config["sequence_len"]
n_queries = 2 * n_layer + 1

tokenizer = get_tokenizer()
val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, 1, seq_len, split="val", device=device
)

# Accumulate gate sigmoid values: (n_queries, n_embd)
gate_sum = torch.zeros(n_queries, n_embd)
gate_count = 0

print(f"Running {args.num_samples} validation samples...")

with torch.no_grad():
    for sample_idx in range(args.num_samples):
        x_input, _ = next(val_loader)
        B, T = x_input.size()

        # Reproduce forward to capture gate values
        x = model.transformer.wte(x_input)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear
        if hasattr(model, 'smear_lambda'):
            gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        cos_sin = model.cos[:, :T], model.sin[:, :T]
        v_list = [x]
        qi = 0

        for i, block in enumerate(model.transformer.h):
            # Pre-attn AttnRes + gate
            query = model.attn_res_queries[qi]
            V = torch.stack(v_list, dim=0)
            K = norm(V)
            logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
            weights = logits.softmax(dim=0)
            h = torch.einsum('n b t, n b t d -> b t d', weights, V)

            # Capture gate sigmoid values
            gate_module = model.attn_res_gates[qi]
            gate_logits = gate_module.up(gate_module.down(h))
            gate_sigmoid = torch.sigmoid(gate_logits)  # (B, T, D)
            gate_sum[qi] += gate_sigmoid.mean(dim=(0, 1)).cpu()

            h = gate_module(h)
            qi += 1

            ve = model.value_embeds[str(i)](x_input).to(h.dtype) if str(i) in model.value_embeds else None
            attn_out = block.attn(norm(h), ve, cos_sin, model.window_sizes[i], None)
            v_list.append(attn_out)

            # Pre-MLP AttnRes + gate
            query = model.attn_res_queries[qi]
            V = torch.stack(v_list, dim=0)
            K = norm(V)
            logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
            weights = logits.softmax(dim=0)
            h = torch.einsum('n b t, n b t d -> b t d', weights, V)

            gate_module = model.attn_res_gates[qi]
            gate_logits = gate_module.up(gate_module.down(h))
            gate_sigmoid = torch.sigmoid(gate_logits)
            gate_sum[qi] += gate_sigmoid.mean(dim=(0, 1)).cpu()

            h = gate_module(h)
            qi += 1

            mlp_out = block.mlp(norm(h))
            v_list.append(mlp_out)

        # Final gate
        query = model.attn_res_queries[qi]
        V = torch.stack(v_list, dim=0)
        K = norm(V)
        logits = torch.einsum('d, n b t d -> n b t', query.to(V.dtype), K)
        weights = logits.softmax(dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)

        gate_module = model.attn_res_gates[qi]
        gate_logits = gate_module.up(gate_module.down(h))
        gate_sigmoid = torch.sigmoid(gate_logits)
        gate_sum[qi] += gate_sigmoid.mean(dim=(0, 1)).cpu()

        gate_count += 1

        if (sample_idx + 1) % 10 == 0:
            print(f"  {sample_idx + 1}/{args.num_samples}")

gate_avg = (gate_sum / gate_count).numpy()  # (n_queries, n_embd)

# Sort each row ascending so x-axis becomes cumulative count
gate_sorted = np.sort(gate_avg, axis=1)

fig, ax = plt.subplots(figsize=(16, 6))

im = ax.imshow(gate_sorted, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
ax.set_xlabel("Dimension (sorted ascending per row)", fontsize=12)
ax.set_ylabel("Gate (layer/sublayer)", fontsize=12)
ax.set_title("Gate Sigmoid Values", fontsize=14)

ytick_labels = []
for i in range(n_queries):
    if i < 2 * n_layer:
        layer = i // 2
        sublayer = "A" if i % 2 == 0 else "M"
        ytick_labels.append(f"L{layer}{sublayer}")
    else:
        ytick_labels.append("Final")
ax.set_yticks(range(n_queries))
ax.set_yticklabels(ytick_labels, fontsize=7)
fig.colorbar(im, ax=ax, label="Sigmoid value (0=fully gated, 1=pass-through)")

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved to {args.output}")
