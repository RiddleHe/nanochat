"""Step 2: linear probe vs Patchscope on the SAME hidden states (Qwen3-8B only).

For each source layer L we take the same last-token hidden state
h_L("scientist" | Einstein context) and read the entity two ways:

  (1) linear probe   : h_L -> entity (held-out-template clean, from Step 1).
  (2) Patchscope     : patch h_L into a repeat-format target prompt, read the
                       15 entity candidate logits (and, at the best target
                       layer, greedy generation).

If the probe stays high in late layers while Patchscope drops, then late-layer
hidden states DO contain the entity but Patchscope cannot read them ->
Patchscope failure is a readout limitation, not information loss.

Controls (per the plan):
  - norm matching   : rescale the patched vector to the target-position norm.
  - target sweep    : patch into several target layers, not one.
  - candidate-logit : primary readout (robust; no free generation needed).
  - random-vector patch : floor (patch noise -> should be chance).
  - generation      : at the best target layer, does the model actually say it.

Qwen3-8B only (small models cannot do the natural-language Patchscope readout).

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.step2_probe_vs_patchscope \
      --hf-model Qwen/Qwen3-8B-Base --out results/step2
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import PALETTE
from scripts.inspect.probe_step1_robust import (
    ENTITIES, TEMPLATES, N_TEMPLATES, norm_rows, kfold_template)

# Repeat-format target prompt; few-shot names are OUTSIDE the 15-entity set to
# avoid leakage. Patch the 'x' slot, read the token after the final ':'.
FEWSHOT = "Madrid: Madrid\nLincoln: Lincoln\nFerrari: Ferrari\nx:"
TARGET_LAYERS_FRAC = [0.12, 0.3, 0.5, 0.7]   # early / mid / late target layers


def load_hf(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    layers = model.model.layers
    return model, tok, layers, len(layers)


@torch.inference_mode()
def capture_all(model, layers, ids, device, n_layer):
    feats = [None] * n_layer
    hs = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, -1, :].detach().float().cpu()
        return hook

    for i, l in enumerate(layers):
        hs.append(l.register_forward_hook(mk(i)))
    try:
        model(torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        for h in hs:
            h.remove()
    return torch.stack(feats, 0)


def build_source(model, tok, layers, n_layer, device):
    X, y = [], []
    for ei, (name, role, typ) in enumerate(ENTITIES):
        for t in TEMPLATES[typ][:N_TEMPLATES]:
            ids = tok(t.format(name=name, role=role), add_special_tokens=True)["input_ids"]
            X.append(capture_all(model, layers, ids, device, n_layer))
            y.append(ei)
    return torch.stack(X, 0), torch.tensor(y)


def candidate_ids(tok):
    ids = [tok(" " + name, add_special_tokens=False)["input_ids"][0] for name, _, _ in ENTITIES]
    assert len(set(ids)) == len(ids), "candidate first-tokens collide"
    return torch.tensor(ids)


def target_positions(tok):
    ids = tok(FEWSHOT, add_special_tokens=True)["input_ids"]
    # 'x' is the second-to-last token, ':' the last
    toks = [tok.decode([i]) for i in ids]
    patch_pos = len(ids) - 2
    read_pos = len(ids) - 1
    assert "x" in toks[patch_pos], f"patch_pos token is {toks[patch_pos]!r}, expected 'x'"
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
    """src: (B, D) already norm-matched. Returns predicted entity idx (B,)."""
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


@torch.inference_mode()
def patch_generate(model, layers, target_ids, T, patch_pos, src, device, n_new=4):
    """Greedy generate n_new tokens with the patch re-applied each step.
    Returns list of decoded strings (B,)."""
    B = src.shape[0]
    srcd = src.to(device)
    seq = torch.tensor([target_ids], device=device).expand(B, -1).clone()

    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[:, patch_pos, :] = srcd.to(h.dtype)
        return (h,) + out[1:] if is_t else h

    for _ in range(n_new):
        handle = layers[T].register_forward_hook(hook)
        try:
            nxt = model(seq).logits[:, -1, :].argmax(-1, keepdim=True)
        finally:
            handle.remove()
        seq = torch.cat([seq, nxt], dim=1)
    return seq[:, len(target_ids):].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/step2")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers", flush=True)
    X, y = build_source(model, layers, n_layer, device) if False else build_source(model, tok, layers, n_layer, device)
    cand = candidate_ids(tok)
    target_ids, patch_pos, read_pos = target_positions(tok)
    tgt_layers = sorted(set(int(f * n_layer) for f in TARGET_LAYERS_FRAC))
    print(f"target layers (patch into): {tgt_layers}; patch_pos={patch_pos}", flush=True)

    # (1) probe line: held-out-template clean accuracy per source layer
    probe_acc = [kfold_template(norm_rows(X[:, L, :]), y, _tmpl_ids(), len(ENTITIES), device)
                 for L in range(n_layer)]

    # (2) patchscope candidate-logit accuracy per source layer, per target layer
    ps = {T: [] for T in tgt_layers}
    floor = []
    for T in tgt_layers:
        ref = ref_norm_at(model, layers, target_ids, T, patch_pos, device)
        for L in range(n_layer):
            src = X[:, L, :]
            srcn = src / src.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
            pred = patch_readout(model, layers, target_ids, T, patch_pos, read_pos, srcn, cand, device)
            ps[T].append(float((pred == y).float().mean()))
        print(f"  target L{T}: patchscope acc range "
              f"{min(ps[T]):.2f}-{max(ps[T]):.2f}", flush=True)
    # random-vector floor at the best target layer
    bestT = max(tgt_layers, key=lambda T: sum(ps[T]) / len(ps[T]))
    ref = ref_norm_at(model, layers, target_ids, bestT, patch_pos, device)
    g = torch.Generator().manual_seed(0)
    for L in range(n_layer):
        rnd = torch.randn(len(y), X.shape[-1], generator=g)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True) * ref
        pred = patch_readout(model, layers, target_ids, bestT, patch_pos, read_pos, rnd, cand, device)
        floor.append(float((pred == y).float().mean()))

    # generation-based patchscope at best target layer (does it actually say it)
    gen_acc = []
    names = [n.lower() for n, _, _ in ENTITIES]
    ref = ref_norm_at(model, layers, target_ids, bestT, patch_pos, device)
    for L in range(n_layer):
        src = X[:, L, :]
        srcn = src / src.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
        outs = patch_generate(model, layers, target_ids, bestT, patch_pos, srcn, device)
        texts = [tok.decode(o).lower() for o in outs]
        gen_acc.append(sum(names[int(yi)] in txt for yi, txt in zip(y, texts)) / len(y))
    print(f"  best target L{bestT}: gen acc range {min(gen_acc):.2f}-{max(gen_acc):.2f}", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer, "chance": 1.0 / len(ENTITIES),
           "target_layers": tgt_layers, "best_target": bestT,
           "probe": probe_acc, "patchscope": {str(T): ps[T] for T in tgt_layers},
           "patchscope_random_floor": floor, "patchscope_generation": gen_acc}
    with open(os.path.join(args.out, "step2_qwen.json"), "w") as f:
        json.dump(res, f, indent=1)

    xs = range(n_layer)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(xs, probe_acc, "-o", ms=3, color=PALETTE[0], lw=2, label="linear probe (held-out-template)")
    for i, T in enumerate(tgt_layers):
        ax.plot(xs, ps[T], "--", color=PALETTE[1 + i % 3], alpha=0.9,
                label=f"patchscope logit, target L{T}")
    ax.plot(xs, gen_acc, "-.", color="#8a5a2c", label=f"patchscope generation, target L{bestT}")
    ax.plot(xs, floor, ":", color="0.5", label="random-vector patch (floor)")
    ax.axhline(res["chance"], color="0.7", lw=0.8, ls="--", label=f"chance ({res['chance']:.2f})")
    ax.axvspan(2 * n_layer // 3, n_layer - 1, color="0.93", zorder=0)
    ax.set_xlabel("source layer"); ax.set_ylabel("entity accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{args.hf_model}: same hidden state, read by probe vs patchscope\n"
                 "probe high deep + patchscope drops => patchscope is a weak readout, not info loss")
    ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, "step2_qwen.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}", flush=True)


def _tmpl_ids():
    # template index per example, matching build_source ordering
    return torch.tensor([ti for _ in ENTITIES for ti in range(N_TEMPLATES)])


if __name__ == "__main__":
    main()
