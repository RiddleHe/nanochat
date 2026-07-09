"""Causal 'stop-reading' boundary for nanochat checkpoints.

Same design as step_d_causal.py (subject-token interchange patching), adapted
to nanochat models and multi-token names: clean/corrupted sentences differ
only in the entity name (same-length token spans required); at each layer we
overwrite the corrupted run's name-span hidden states with the clean run's and
measure the logit-diff recovery at the final position. The layer where
recovery collapses = where the model has finished reading the entity position.

Usage:
  NANOCHAT_BASE_DIR=... CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect.step_d_nanochat \
      --model-tag arch_d12_gpt_base_1.5e18 --label d12 --out results/boundary
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#e8a87c", "#2c8a8a", "#5aa75a"]

ENTITIES = [
    ("Einstein", "scientist"), ("Newton", "scientist"), ("Darwin", "scientist"),
    ("Mozart", "composer"), ("Beethoven", "composer"),
    ("Shakespeare", "writer"), ("Dickens", "writer"),
    ("France", "country"), ("Japan", "country"),
    ("Paris", "city"), ("Tokyo", "city"),
    ("Google", "company"), ("Apple", "company"),
]
PREFIX = "Everyone knows"
SUFFIX = " was a celebrated {role}. The {role} was"


def load_nanochat(tag, step, device):
    from nanochat.checkpoint_manager import load_model
    model, tok, meta = load_model("base", device, phase="eval", model_tag=tag, step=step)
    model.eval()
    return model, tok, meta["model_config"]["n_layer"], tok.get_bos_token_id()


def build_ids(tok, bos, name, role):
    """Piecewise encoding so the name span positions are known exactly."""
    pre = [bos] + tok.encode(PREFIX)
    span = tok.encode(" " + name)
    suf = tok.encode(SUFFIX.format(role=role))
    return pre + span + suf, len(pre), len(pre) + len(span)


@torch.inference_mode()
def run_capture(model, ids, s0, s1, device, n_layer):
    """Forward; capture hidden states of the name span at every layer, and the
    final-position logits."""
    feats = [None] * n_layer
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, s0:s1, :].detach().clone()
        return hook

    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    return feats, logits


@torch.inference_mode()
def run_patched(model, ids, L, s0, s1, span_states, device):
    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[0, s0:s1, :] = span_states.to(h.dtype)
        return (h,) + out[1:] if is_t else h

    handle = model.transformer.h[L].register_forward_hook(hook)
    try:
        return model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1, :].float()
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/boundary")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    label = args.label or args.model_tag

    model, tok, n_layer, bos = load_nanochat(args.model_tag, args.step, device)
    print(f"{label} ({args.model_tag}): {n_layer} layers", flush=True)

    # within-category ordered pairs with equal name-span length
    cats = {}
    for name, role in ENTITIES:
        cats.setdefault(role, []).append(name)
    pairs = []
    for role, names in cats.items():
        for a in names:
            for b in names:
                if a != b and len(tok.encode(" " + a)) == len(tok.encode(" " + b)):
                    pairs.append((a, b, role))
    print(f"{len(pairs)} usable pairs", flush=True)

    recov = torch.zeros(n_layer)
    n_used = 0
    for clean, corrupt, role in pairs:
        ids_c, s0, s1 = build_ids(tok, bos, clean, role)
        ids_x, _, _ = build_ids(tok, bos, corrupt, role)
        ct = tok.encode(" " + clean)[0]
        xt = tok.encode(" " + corrupt)[0]
        clean_states, clean_log = run_capture(model, ids_c, s0, s1, device, n_layer)
        _, corr_log = run_capture(model, ids_x, s0, s1, device, n_layer)
        clean_d = float(clean_log[ct] - clean_log[xt])
        corr_d = float(corr_log[ct] - corr_log[xt])
        denom = clean_d - corr_d
        if abs(denom) < 0.5:  # skip pairs the model can't meaningfully separate
            continue
        for L in range(n_layer):
            pl = run_patched(model, ids_x, L, s0, s1, clean_states[L], device)
            recov[L] += (float(pl[ct] - pl[xt]) - corr_d) / denom
        n_used += 1
    assert n_used >= 6, f"only {n_used} usable pairs; model too weak for this probe"
    recov /= n_used
    print(f"used {n_used} pairs", flush=True)
    for L in range(n_layer):
        print(f"  L{L:02d} recovery {recov[L]:.2f}", flush=True)
    # boundary = last layer with recovery >= 0.5
    above = [L for L in range(n_layer) if recov[L] >= 0.5]
    boundary = max(above) if above else -1
    print(f"stop-reading boundary (last layer with recovery>=0.5): L{boundary}", flush=True)

    res = {"model": args.model_tag, "label": label, "n_layer": n_layer,
           "n_pairs": n_used, "recovery": recov.tolist(), "boundary": boundary}
    with open(os.path.join(args.out, f"{label}__boundary.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(n_layer), recov.tolist(), "-o", ms=3, color=PALETTE[3])
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.axvline(boundary, color="r", lw=1, ls="--", label=f"boundary L{boundary}")
    ax.set_xlabel("patched layer (subject-name span, clean into corrupted)")
    ax.set_ylabel("logit-diff recovery (1 = fully flipped)")
    ax.set_title(f"{label}: where does the model stop reading the entity position?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{label}__boundary.png"), dpi=150, bbox_inches="tight")
    print(f"saved {os.path.join(args.out, f'{label}__boundary.png')}", flush=True)


if __name__ == "__main__":
    main()
