"""Print sink logit values from an AttnRes+Sink checkpoint.

Usage:
    python -m scripts.inspect.inspect_sink --model-tag arch_d12_attn_res_sink
"""
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
args = parser.parse_args()

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_last_step

base_dir = get_base_dir()
ckpt_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
step = find_last_step(ckpt_dir)
path = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
print(f"Loading: {path}")
model_data = torch.load(path, map_location="cpu")
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

sinks = model_data["sink_logits"].float()
print(f"\nSink logits ({sinks.shape[0]} values, zero-init):\n")

n_layer = None
for key in model_data:
    if "transformer.h." in key:
        idx = int(key.split("transformer.h.")[1].split(".")[0])
        if n_layer is None or idx + 1 > n_layer:
            n_layer = idx + 1
n_queries = 2 * n_layer + 1

for i in range(n_queries):
    if i < 2 * n_layer:
        layer = i // 2
        sublayer = "attn" if i % 2 == 0 else "mlp"
        label = f"Layer {layer:2d} {sublayer:4s}"
    else:
        label = "Final output  "
    print(f"  {label}: {sinks[i]:.6f}")

print(f"\nMean: {sinks[:n_queries].mean():.6f}  Std: {sinks[:n_queries].std():.6f}  "
      f"Min: {sinks[:n_queries].min():.6f}  Max: {sinks[:n_queries].max():.6f}")

if sinks[:n_queries].abs().max() < 0.01:
    print("\n*** Sinks are near-zero — effectively not learned ***")
elif sinks[:n_queries].abs().max() < 0.1:
    print("\n*** Sinks have small values — minimal effect ***")
else:
    print("\n*** Sinks have significant values — actively absorbing attention ***")
