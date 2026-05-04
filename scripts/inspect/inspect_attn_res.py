"""Inspect AttnRes pseudo-query vectors from a checkpoint.

Usage:
    python -m scripts.inspect.inspect_attn_res --checkpoint /path/to/model_NNNNNN.pt
    python -m scripts.inspect.inspect_attn_res --model-tag arch_attn_res  # auto-find latest
"""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="path to model_*.pt file")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to auto-find checkpoint")
args = parser.parse_args()

if args.checkpoint:
    model_data = torch.load(args.checkpoint, map_location="cpu")
elif args.model_tag:
    from nanochat.common import get_base_dir
    from nanochat.checkpoint_manager import find_last_step
    import os
    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
    step = find_last_step(ckpt_dir)
    path = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
    print(f"Loading: {path}")
    model_data = torch.load(path, map_location="cpu")
else:
    raise ValueError("Provide --checkpoint or --model-tag")

# Strip torch.compile prefix if present
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

queries = model_data["attn_res_queries"].float()
n_total, d = queries.shape
print(f"Query tensor shape: ({n_total}, {d})")

# Infer n_queries from model config: 2 * n_layer + 1, padded to multiple of 8
# Try to find n_layer from the checkpoint
n_layer = None
for key in model_data:
    if "transformer.h." in key:
        idx = int(key.split("transformer.h.")[1].split(".")[0])
        if n_layer is None or idx + 1 > n_layer:
            n_layer = idx + 1
n_queries = 2 * n_layer + 1 if n_layer else n_total
print(f"Detected n_layer={n_layer}, n_queries={n_queries} (padded to {n_total})")

row_norms = queries.norm(dim=1)
queries = queries[:n_queries]
row_norms = row_norms[:n_queries]

print(f"\nPer-query L2 norms:")
for i in range(n_queries):
    if i < 2 * n_layer:
        layer = i // 2
        sublayer = "attn" if i % 2 == 0 else "mlp"
        label = f"Layer {layer:2d} {sublayer:4s}"
    else:
        label = "Final output  "
    print(f"  Query {i:2d} ({label}): norm = {row_norms[i]:.6f}")

print(f"\nMean: {row_norms.mean():.6f}  Std: {row_norms.std():.6f}  "
      f"Min: {row_norms.min():.6f}  Max: {row_norms.max():.6f}")

if row_norms.max() < 0.01:
    print("\n*** Queries are near-zero — effectively frozen (no routing learned) ***")
elif row_norms.max() < 0.1:
    print("\n*** Queries have small norms — minimal routing learned ***")
else:
    print("\n*** Queries have significant norms — active routing ***")
