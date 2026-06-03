"""
Layer-removal probe ("Curse of Depth").

For each model and each layer L, we replace block L with the identity map
(skipping its attention + MLP, so its output equals its input) and measure the
resulting Performance Drop:

    Delta P(L) = metric_pruned(L) - metric_original

A small drop means layer L contributes little (the deep layers of Pre-LN
transformers are known to be near-identity and prunable). We compare the
standard Attention model against BoV to see whether replacing the deep-layer
value computation with a context-free lookup changes how much those layers
matter.

Metrics:
  - val BPB     (lower is better; positive Delta = removing the layer HURTS quality)
  - CORE score  (higher is better; we plot Delta on the metric directly, so
                 a negative Delta = removing the layer HURTS)

References:
  - "The Curse of Depth in Large Language Models" (arxiv 2502.05795), Fig. 2.
  - "The Unreasonable Ineffectiveness of the Deeper Layers" (arxiv 2403.17887).

Usage:
  python -m scripts.inspect.layer_removal \
    --model-tags arch_d24_gpt_base_100B arch_d24_gpt_base_v_from_value_emb_learn_100B \
    --labels "Attention" "BoV" \
    --metric both --bpb-tokens 2097152 --max-per-task 200 \
    --output results/layer_removal_d24_100B.png
"""
import os
import json
import types
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from scripts.base_eval import evaluate_core


def _identity_block(self, x, x0, ve, cos_sin, kv_cache):
    """Drop-in replacement for Block.forward: pass the residual through unchanged.
    Returns (x, None); the second element (captured layer-0 value) is unused by
    the gpt_base and v_from_value_emb models evaluated here."""
    return x, None


class drop_layer:
    """Context manager that makes block L behave as the identity, then restores it."""
    def __init__(self, model, layer_idx):
        self.block = model.transformer.h[layer_idx]
        self._orig = None

    def __enter__(self):
        self._orig = self.block.forward
        self.block.forward = types.MethodType(_identity_block, self.block)
        return self

    def __exit__(self, *exc):
        self.block.forward = self._orig


@torch.inference_mode()
def measure_bpb(model, tokenizer, device, bpb_tokens, device_batch_size=16):
    seq_len = model.config.sequence_len
    token_bytes = get_token_bytes(device=device)
    tokens_per_step = device_batch_size * seq_len
    steps = max(1, bpb_tokens // tokens_per_step)
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_batch_size, seq_len, "val", device=device)
    return float(evaluate_bpb(model, loader, steps, token_bytes))


@torch.inference_mode()
def measure_core(model, tokenizer, device, max_per_task):
    return float(evaluate_core(model, tokenizer, device, max_per_task=max_per_task)["core_metric"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tags", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--metric", choices=["bpb", "core", "both"], default="both")
    p.add_argument("--bpb-tokens", type=int, default=2_097_152, help="tokens per BPB eval")
    p.add_argument("--max-per-task", type=int, default=200, help="examples per CORE task")
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--device-type", type=str, default="")
    p.add_argument("--output", type=str, default="results/layer_removal.png")
    p.add_argument("--results-json", type=str, default=None)
    args = p.parse_args()

    labels = args.labels or args.model_tags
    assert len(labels) == len(args.model_tags), "labels must match model-tags"
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    results = {}
    for tag, label in zip(args.model_tags, labels):
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=tag)
        model.eval()
        n_layer = model.config.n_layer
        print(f"\n=== {label} ({tag}): {n_layer} layers ===", flush=True)

        rec = {"tag": tag, "n_layer": n_layer}
        if args.metric in ("bpb", "both"):
            base_bpb = measure_bpb(model, tokenizer, device, args.bpb_tokens, args.device_batch_size)
            d_bpb = []
            for L in range(n_layer):
                with drop_layer(model, L):
                    b = measure_bpb(model, tokenizer, device, args.bpb_tokens, args.device_batch_size)
                d_bpb.append(b - base_bpb)
                print(f"  [bpb] drop L{L:02d}: {b:.5f}  (dP={b-base_bpb:+.5f})", flush=True)
            rec["base_bpb"] = base_bpb
            rec["delta_bpb"] = d_bpb
        if args.metric in ("core", "both"):
            base_core = measure_core(model, tokenizer, device, args.max_per_task)
            d_core = []
            for L in range(n_layer):
                with drop_layer(model, L):
                    c = measure_core(model, tokenizer, device, args.max_per_task)
                d_core.append(c - base_core)
                print(f"  [core] drop L{L:02d}: {c:.4f}  (dP={c-base_core:+.4f})", flush=True)
            rec["base_core"] = base_core
            rec["delta_core"] = d_core
        results[label] = rec
        del model
        torch.cuda.empty_cache()

    # Save raw numbers
    json_path = args.results_json or os.path.splitext(args.output)[0] + ".json"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {json_path}", flush=True)

    # Plot Delta P vs layer index
    panels = [m for m in ("bpb", "core") if (args.metric == m or args.metric == "both")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, m in zip(axes[0], panels):
        key = f"delta_{m}"
        for label, rec in results.items():
            if key in rec:
                ax.plot(range(rec["n_layer"]), rec[key], marker="o", ms=3, label=label)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("removed layer index")
        ax.set_ylabel(("Δ val BPB (↑ = layer matters)" if m == "bpb"
                       else "Δ CORE (↓ = layer matters)"))
        ax.set_title(f"Layer removal: {m}")
        ax.legend()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
