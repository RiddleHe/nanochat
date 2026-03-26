"""Inspect AttnRes pseudo-query vectors from a checkpoint.

Usage:
    python -m scripts.inspect_attn_res --checkpoint /path/to/model_NNNNNN.pt
    python -m scripts.inspect_attn_res --model-tag arch_attn_res  # auto-find latest
"""
import argparse
import torch
import numpy as np

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

# Figure out how many are real (not padding)
# Padded rows are all zeros if never updated
row_norms = queries.norm(dim=1)
n_active = (row_norms > 1e-8).sum().item()
print(f"Active queries: {n_active} / {n_total} (rest are padding)")

queries = queries[:n_active]
row_norms = row_norms[:n_active]

print(f"\n{'='*60}")
print("Per-query L2 norms:")
print(f"{'='*60}")
n_layers = (n_active - 1) // 2
for i in range(n_active):
    if i < 2 * n_layers:
        layer = i // 2
        sublayer = "attn" if i % 2 == 0 else "mlp"
        label = f"Layer {layer:2d} {sublayer:4s}"
    else:
        label = "Final output  "
    print(f"  Query {i:2d} ({label}): norm = {row_norms[i]:.6f}")

print(f"\n{'='*60}")
print("Summary statistics:")
print(f"{'='*60}")
print(f"  Mean norm:   {row_norms.mean():.6f}")
print(f"  Std norm:    {row_norms.std():.6f}")
print(f"  Min norm:    {row_norms.min():.6f}  (query {row_norms.argmin().item()})")
print(f"  Max norm:    {row_norms.max():.6f}  (query {row_norms.argmax().item()})")
print(f"  Total norm:  {queries.norm():.6f}")

# Check if queries are effectively zero (frozen)
if row_norms.max() < 0.01:
    print(f"\n  *** Queries are near-zero — effectively frozen (no routing learned) ***")
elif row_norms.max() < 0.1:
    print(f"\n  *** Queries have small norms — minimal routing learned ***")
else:
    print(f"\n  *** Queries have significant norms — active routing ***")

# Cosine similarity between adjacent queries (are they learning different things?)
print(f"\n{'='*60}")
print("Cosine similarity between consecutive queries:")
print(f"{'='*60}")
for i in range(min(n_active - 1, 10)):
    cos = torch.nn.functional.cosine_similarity(queries[i:i+1], queries[i+1:i+2]).item()
    print(f"  Query {i:2d} vs {i+1:2d}: {cos:.4f}")
if n_active > 11:
    print(f"  ... ({n_active - 11} more)")

# What do the softmax weights look like for a uniform input?
print(f"\n{'='*60}")
print("Softmax weights (uniform unit-norm input):")
print(f"{'='*60}")
print("If queries learned routing, weights should be non-uniform.")
fake_keys = torch.ones(n_active, d) / d**0.5  # uniform keys
for qi in [0, n_active // 4, n_active // 2, 3 * n_active // 4, n_active - 1]:
    n_sources = qi + 1 if qi < n_active - 1 else n_active
    logits = queries[qi] @ fake_keys[:n_sources].T
    weights = torch.softmax(logits, dim=0)
    uniform = 1.0 / n_sources
    max_w = weights.max().item()
    min_w = weights.min().item()
    entropy = -(weights * weights.log()).sum().item()
    max_entropy = np.log(n_sources)
    print(f"  Query {qi:2d} ({n_sources:2d} sources): max={max_w:.4f} min={min_w:.4f} "
          f"entropy={entropy:.3f}/{max_entropy:.3f} (uniform={uniform:.4f})")
