"""Linear translator: is a late-layer hidden state one linear map away from
being Patchscope-readable?

Background: the linear probe (entity_probe.py) reads the bound entity at every
layer, but Patchscope stops reading it in late layers. Two explanations: the
late state is fundamentally unusable by the model, or it merely sits in a
different layer-specific coordinate system. This experiment distinguishes them.

Method: train a linear translator A: h_src -> h_ref (MSE; trained WITHOUT any
entity labels — it only sees pairs of hidden states of the same sentence at
two layers, so it cannot inject entity knowledge). Then Patchscope-read three
things at the reference target layer, on held-out templates:

  raw         the late hidden state as-is           (baseline)
  translated  A(h_src)                              (the test)
  true h_ref  the genuine reference-layer state     (ceiling)

If translated recovers to the ceiling, the entity info was fully present in
the late state and only the coordinate system differed.

Readout: candidate-logit Patchscope — patch the vector (norm-matched to the
target position) into the `x` slot of a repeat-format prompt at the reference
layer, and rank the logits of the 15 entity name tokens at the next position.
Few-shot names in the prompt are outside the entity set.

HF models only (the readout needs a capable model).

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect.entity_translator \
      --hf-model Qwen/Qwen3-8B-Base --ref-layer 10 --src-layers 10 20 24 28 32
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.entity_probe import (
    PALETTE, ENTITIES, TEMPLATES, N_TEMPLATES, FOLD, capture_last)

FEWSHOT = "Madrid: Madrid\nLincoln: Lincoln\nFerrari: Ferrari\nx:"


# ---------------------------------------------------------------- model I/O

def load_hf(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16).to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        layers = model.transformer.h
    return model, tok, layers, len(layers)


def build_source(model, tok, layers, n_layer, device):
    X, y, tmpl = [], [], []
    for ei, (name, role, typ) in enumerate(ENTITIES):
        for ti, t in enumerate(TEMPLATES[typ][:N_TEMPLATES]):
            ids = tok(t.format(name=name, role=role), add_special_tokens=True)["input_ids"]
            X.append(capture_last(model, layers, ids, device, n_layer))
            y.append(ei); tmpl.append(ti)
    return torch.stack(X, 0), torch.tensor(y), torch.tensor(tmpl)


# ---------------------------------------------------------------- patchscope

def target_positions(tok):
    ids = tok(FEWSHOT, add_special_tokens=True)["input_ids"]
    patch_pos, read_pos = len(ids) - 2, len(ids) - 1
    assert "x" in tok.decode([ids[patch_pos]]), "patch position must be the 'x' token"
    return ids, patch_pos, read_pos


@torch.inference_mode()
def ref_norm_at(model, layers, target_ids, T, patch_pos, device):
    box = {}

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        box["n"] = float(h[0, patch_pos, :].norm())

    handle = layers[T].register_forward_hook(hook)
    try:
        model(torch.tensor([target_ids], device=device))
    finally:
        handle.remove()
    return box["n"]


@torch.inference_mode()
def patch_readout(model, layers, target_ids, T, patch_pos, read_pos, src, cand, device):
    """Patch src (B, D) at (layer T, patch_pos); return argmax entity idx (B,)."""
    B = src.shape[0]
    x = torch.tensor([target_ids], device=device).expand(B, -1)
    srcd = src.to(device)

    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[:, patch_pos, :] = srcd.to(h.dtype)
        return (h,) + out[1:] if is_t else h

    handle = layers[T].register_forward_hook(hook)
    try:
        logits = model(x).logits[:, read_pos, :].float()
    finally:
        handle.remove()
    return logits[:, cand.to(device)].argmax(1).cpu()


# ---------------------------------------------------------------- translator

def train_translator(Hs, Ht, device, steps=500):
    A = nn.Linear(Hs.shape[1], Hs.shape[1]).to(device)
    opt = torch.optim.Adam(A.parameters(), lr=1e-3, weight_decay=1e-3)
    Hs, Ht = Hs.to(device), Ht.to(device)
    for _ in range(steps):
        loss = F.mse_loss(A(Hs), Ht)
        opt.zero_grad(); loss.backward(); opt.step()
    return A


def template_folds(tmpl):
    for start in range(0, N_TEMPLATES, FOLD):
        te = (tmpl >= start) & (tmpl < start + FOLD)
        yield ~te, te


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--ref-layer", type=int, default=10,
                    help="reference target layer (patch into, translate toward)")
    ap.add_argument("--src-layers", type=int, nargs="+", default=[10, 20, 24, 28, 32])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/translator")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers, ref target L{args.ref_layer}", flush=True)
    X, y, tmpl = build_source(model, tok, layers, n_layer, device)
    target_ids, patch_pos, read_pos = target_positions(tok)
    cand = torch.tensor([tok(" " + n, add_special_tokens=False)["input_ids"][0]
                         for n, _, _ in ENTITIES])
    assert len(set(cand.tolist())) == len(cand), "candidate first-tokens collide"
    Lt = args.ref_layer
    ref = ref_norm_at(model, layers, target_ids, Lt, patch_pos, device)

    def acc(H, yte):
        Hn = H / H.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
        pred = patch_readout(model, layers, target_ids, Lt, patch_pos, read_pos,
                             Hn, cand, device)
        return float((pred == yte).float().mean())

    rows = []
    for Ls in args.src_layers:
        raw, trans, upper = [], [], []
        for trm, tem in template_folds(tmpl):
            A = train_translator(X[trm, Ls, :], X[trm, Lt, :], device)
            with torch.no_grad():
                HA = A(X[tem, Ls, :].to(device)).cpu()
            raw.append(acc(X[tem, Ls, :], y[tem]))
            trans.append(acc(HA, y[tem]))
            upper.append(acc(X[tem, Lt, :], y[tem]))
        rows.append({"src": Ls,
                     "raw": sum(raw) / len(raw),
                     "translated": sum(trans) / len(trans),
                     "ceiling": sum(upper) / len(upper)})
        print(f"  src L{Ls:02d} -> target L{Lt}: raw {rows[-1]['raw']:.2f}  "
              f"translated {rows[-1]['translated']:.2f}  "
              f"ceiling {rows[-1]['ceiling']:.2f}", flush=True)

    slug = args.hf_model.replace("/", "_")
    with open(os.path.join(args.out, f"{slug}__translator.json"), "w") as f:
        json.dump({"model": args.hf_model, "n_layer": n_layer, "ref_layer": Lt,
                   "chance": 1.0 / len(ENTITIES), "rows": rows}, f, indent=1)

    xs = [r["src"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, [r["raw"] for r in rows], "-o", color=PALETTE[1], label="raw late state")
    ax.plot(xs, [r["translated"] for r in rows], "-s", color=PALETTE[2],
            label="translated (linear A)")
    ax.plot(xs, [r["ceiling"] for r in rows], "--", color="0.5",
            label=f"ceiling (true L{Lt} state)")
    ax.axhline(1.0 / len(ENTITIES), color="0.7", lw=0.8, ls=":", label="chance")
    ax.set_xlabel(f"source layer (translated toward L{Lt})")
    ax.set_ylabel("patchscope accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{args.hf_model}: linear translation restores patchscope readability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, f"{slug}__translator.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
