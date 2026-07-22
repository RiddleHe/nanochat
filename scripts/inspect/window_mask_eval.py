"""Inference-time sliding-window masking of a chosen layer range, on a model
trained with full attention (Qwen3-8B-Base). No training.

Hypothesis under test: cross-position reads finish by the measured hand-off
boundary (~L23-26 on Qwen3-8B), so masking attention in layers >= start_layer
to a local window should not hurt language-model loss when start_layer is at or
past the boundary, and should hurt increasingly when moved earlier.

Method: forward-pre-hook on each decoder layer >= start_layer replaces its
attention_mask with a causal sliding-window mask (last W keys visible). The
first `sink` positions stay visible in all layers: Qwen3 deep layers park most
attention mass on position 0 (attention sink), and blinding the sink would
crash the model for reasons unrelated to long-range information (StreamingLLM).

Two sweeps, both evaluated as mean next-token NLL on wikitext-103 validation:
  1. start-layer sweep at fixed window
  2. window-size sweep at fixed start layer (boundary), with sink 4 vs 0

Usage:
  python -m scripts.inspect.window_mask_eval --hf-model Qwen/Qwen3-8B-Base
  # smoke: --n-windows 2 --start-layers 24 --windows 256
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#c0392b", "#2c8a8a", "#888888"]


def build_window_mask(T, window, sink, device, dtype):
    """(1,1,T,T) additive mask: causal, key j visible to query i iff
    i-window < j <= i, or j < sink."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    visible = (j <= i) & ((i - j < window) | (j < sink))
    mask = torch.zeros(T, T, device=device, dtype=dtype)
    mask[~visible] = torch.finfo(dtype).min
    return mask[None, None]


def install_hooks(model, start_layer, mask, end_layer=None):
    """Window-mask layers in [start_layer, end_layer); end_layer None = to the top."""
    handles = []

    def pre_hook(_m, args, kwargs):
        kwargs["attention_mask"] = mask
        return args, kwargs

    end = len(model.model.layers) if end_layer is None else end_layer
    for idx, layer in enumerate(model.model.layers):
        if start_layer <= idx < end:
            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
    return handles


@torch.inference_mode()
def eval_nll(model, batches, device):
    tot, n = 0.0, 0
    for ids in batches:
        x = ids.to(device)[None]
        logits = model(x).logits.float()
        loss = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum")
        tot += float(loss)
        n += x.shape[1] - 1
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=40, help="eval sequences")
    ap.add_argument("--window", type=int, default=256, help="window for the start-layer sweep")
    ap.add_argument("--start-layers", type=int, nargs="*",
                    default=[0, 4, 8, 12, 16, 20, 22, 24, 26, 28, 32])
    ap.add_argument("--fixed-start", type=int, default=24, help="start layer for the window sweep")
    ap.add_argument("--windows", type=int, nargs="*", default=[1024, 512, 256, 128, 64, 16])
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--bands", nargs="*", default=[],
                    help="extra band configs 'start:end' masking only layers "
                         "[start,end), e.g. 4:24 keeps deep layers full")
    ap.add_argument("--boundary", type=int, nargs=2, default=[23, 26])
    ap.add_argument("--out", default="results/window_mask")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    n_layer = model.config.num_hidden_layers
    print(f"{args.hf_model}: {n_layer} layers", flush=True)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    batches = [ids[k * args.seq_len:(k + 1) * args.seq_len]
               for k in range(args.n_windows)]
    assert all(len(b) == args.seq_len for b in batches)
    print(f"eval: {len(batches)} x {args.seq_len} tokens", flush=True)

    base = eval_nll(model, batches, device)
    print(f"baseline NLL {base:.4f} nats/token", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer, "seq_len": args.seq_len,
           "n_windows": args.n_windows, "sink": args.sink, "baseline_nll": base,
           "start_sweep": [], "window_sweep": [], "band_sweep": []}

    def run(start, window, sink, end=None):
        mask = build_window_mask(args.seq_len, window, sink, device, model.dtype)
        hs = install_hooks(model, start, mask, end)
        try:
            return eval_nll(model, batches, device)
        finally:
            for h in hs:
                h.remove()

    for band in args.bands:
        s, e = (int(v) for v in band.split(":"))
        nll = run(s, args.window, args.sink, e)
        res["band_sweep"].append({"start_layer": s, "end_layer": e,
                                  "window": args.window, "sink": args.sink, "nll": nll})
        print(f"band L{s}-{e} W{args.window}: {nll:.4f} (Δ {nll - base:+.4f})", flush=True)

    for s in args.start_layers:
        nll = run(s, args.window, args.sink)
        res["start_sweep"].append({"start_layer": s, "window": args.window,
                                   "sink": args.sink, "nll": nll})
        print(f"start L{s:2d} W{args.window}: {nll:.4f} (Δ {nll - base:+.4f})", flush=True)

    for w in args.windows:
        for sink in (args.sink, 0):
            nll = run(args.fixed_start, w, sink)
            res["window_sweep"].append({"start_layer": args.fixed_start, "window": w,
                                        "sink": sink, "nll": nll})
            print(f"start L{args.fixed_start} W{w} sink{sink}: {nll:.4f} "
                  f"(Δ {nll - base:+.4f})", flush=True)

    with open(os.path.join(args.out, "window_mask_eval.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = [r["start_layer"] for r in res["start_sweep"]]
    ys = [r["nll"] - base for r in res["start_sweep"]]
    ax.plot(xs, ys, "-o", color=PALETTE[0])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvspan(args.boundary[0], args.boundary[1], color="red", alpha=0.12,
               label=f"hand-off boundary L{args.boundary[0]}-{args.boundary[1]}")
    ax.set_xlabel("first masked layer (window applied from here up)")
    ax.set_ylabel("Δ NLL vs full attention (nats/token)")
    ax.set_title(f"deep-layer window W={args.window}, sink={args.sink}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for sink, color in ((args.sink, PALETTE[0]), (0, PALETTE[1])):
        pts = [(r["window"], r["nll"] - base) for r in res["window_sweep"]
               if r["sink"] == sink]
        if pts:
            pts.sort()
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                     color=color, label=f"sink={sink}")
    ax2.set_xscale("log", base=2)
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_xlabel("window size (tokens)")
    ax2.set_ylabel("Δ NLL vs full attention")
    ax2.set_title(f"window sweep, start L{args.fixed_start}")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle(f"{args.hf_model}: inference-time sliding window on deep layers only")
    fig.tight_layout()
    p = os.path.join(args.out, "window_mask_eval.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
