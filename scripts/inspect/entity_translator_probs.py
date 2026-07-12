"""Linear translator + per-choice probabilities + unseen-entity holdout.

Extends entity_translator.py:
  - the readout now returns the softmax prob over the 15 candidate entity names
    (not just argmax), so we record p(true), p(pred), and the full 15-vector for
    every readout.
  - two holdout regimes:
      template : original 3-fold CV over templates (all 15 entities seen)
      entity   : 5-fold CV over ENTITIES (train A on 12 entities, test on the 3
                 held-out entities' states) -- a real generalization test, since
                 the translator sees no entity labels, only h_src->h_ref pairs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect.entity_translator_probs \
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


def load_hf(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
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
def patch_probs(model, layers, target_ids, T, patch_pos, read_pos, src, cand, device):
    """Patch src (B,D) at (layer T, patch_pos); softmax over the len(cand)
    candidate name tokens -> (B, C)."""
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
    return torch.softmax(logits[:, cand.to(device)], dim=-1).cpu()  # (B, C)


def train_translator(Hs, Ht, device, steps=500):
    torch.manual_seed(0)
    A = nn.Linear(Hs.shape[1], Hs.shape[1]).to(device)
    opt = torch.optim.Adam(A.parameters(), lr=1e-3, weight_decay=1e-3)
    Hs, Ht = Hs.to(device), Ht.to(device)
    for _ in range(steps):
        loss = F.mse_loss(A(Hs), Ht)
        opt.zero_grad(); loss.backward(); opt.step()
    return A


def template_folds(tmpl, y):
    for start in range(0, N_TEMPLATES, FOLD):
        te = (tmpl >= start) & (tmpl < start + FOLD)
        yield ~te, te


def entity_folds(tmpl, y, k=3):
    n = len(ENTITIES)
    for start in range(0, n, k):
        te = (y >= start) & (y < start + k)
        yield ~te, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--ref-layer", type=int, default=10)
    ap.add_argument("--src-layers", type=int, nargs="+", default=[10, 20, 24, 28, 32])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/translator_probs")
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

    def probs_of(H):
        Hn = H / H.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
        return patch_probs(model, layers, target_ids, Lt, patch_pos, read_pos,
                           Hn, cand, device)

    def stats(P, yt):
        pred = P.argmax(1)
        ar = torch.arange(len(yt))
        return {"acc": float((pred == yt).float().mean()),
                "p_true": float(P[ar, yt].mean()),
                "p_pred": float(P.max(1).values.mean())}

    modes = {"template": template_folds, "entity": entity_folds}
    out = {"model": args.hf_model, "n_layer": n_layer, "ref_layer": Lt,
           "chance": 1.0 / len(ENTITIES),
           "entities": [n for n, _, _ in ENTITIES], "modes": {}}

    for mname, ffn in modes.items():
        folds = list(ffn(tmpl, y))
        rows = []
        for Ls in args.src_layers:
            Praw, Ptr, Pce, ys = [], [], [], []
            for trm, tem in folds:
                A = train_translator(X[trm, Ls, :], X[trm, Lt, :], device)
                with torch.no_grad():
                    HA = A(X[tem, Ls, :].to(device)).cpu()
                Praw.append(probs_of(X[tem, Ls, :]))
                Ptr.append(probs_of(HA))
                Pce.append(probs_of(X[tem, Lt, :]))
                ys.append(y[tem])
            Praw, Ptr, Pce, ys = (torch.cat(Praw), torch.cat(Ptr),
                                  torch.cat(Pce), torch.cat(ys))
            rows.append({
                "src": Ls,
                "raw": stats(Praw, ys),
                "translated": stats(Ptr, ys),
                "ceiling": stats(Pce, ys),
                "translated_probs": [{"true": int(t),
                                      "p": [round(float(v), 4) for v in row_]}
                                     for t, row_ in zip(ys.tolist(), Ptr)],
            })
            r = rows[-1]
            print(f"  [{mname}] src L{Ls:02d}->L{Lt}: "
                  f"raw acc {r['raw']['acc']:.2f} p_true {r['raw']['p_true']:.2f} | "
                  f"trans acc {r['translated']['acc']:.2f} p_true {r['translated']['p_true']:.2f} | "
                  f"ceil acc {r['ceiling']['acc']:.2f} p_true {r['ceiling']['p_true']:.2f}",
                  flush=True)
        out["modes"][mname] = rows

    slug = args.hf_model.replace("/", "_")
    with open(os.path.join(args.out, f"{slug}__translator_probs.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(args.out, f"{slug}__translator_probs.json"), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mname in zip(axes, ["template", "entity"]):
        rows = out["modes"][mname]
        xs = [r["src"] for r in rows]
        for key, col in [("raw", PALETTE[1]), ("translated", PALETTE[2]), ("ceiling", "0.5")]:
            ax.plot(xs, [r[key]["acc"] for r in rows], "-o", color=col, label=f"{key} acc")
            ax.plot(xs, [r[key]["p_true"] for r in rows], "--", color=col, alpha=0.6)
        ax.axhline(1.0 / len(ENTITIES), color="0.7", lw=0.8, ls=":")
        ax.set_title(f"{mname} holdout"); ax.set_xlabel(f"source layer (->L{Lt})")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0].set_ylabel("solid = accuracy, dashed = mean p(true)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.hf_model}: translator readability (accuracy & p(true))")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{slug}__translator_probs.png"), dpi=150)
    print("saved plot", flush=True)


if __name__ == "__main__":
    main()
