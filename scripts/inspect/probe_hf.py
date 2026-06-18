"""Run the three entity probes on a HuggingFace model (generalization check).

Ports the d24 probes (category / in-context identity / bound identity) to a
standard pretrained model so we can see whether the small-model conclusion
holds at scale and connects to the friend's Qwen/Pythia Diana finding. The
probe machinery (probe_all_layers) is model-agnostic; only feature capture is
HF-specific here.

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.probe_hf \
      --hf-model Qwen/Qwen3-8B-Base --out results/probe_hf
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import PALETTE
from scripts.inspect.probe_entity_category import probe_all_layers, ENTITIES as CAT
from scripts.inspect.probe_entity_identity import ENTITIES as ID_ENTS, CONTEXTS
from scripts.inspect.probe_bound_identity import ENTITIES as BOUND_ENTS, TEMPLATES


def load_hf(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise RuntimeError(f"unknown arch for {name}")
    return model, tok, layers, len(layers)


@torch.inference_mode()
def capture(model, layers, ids, device, n_layer):
    feats = [None] * n_layer
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, -1, :].detach().float().cpu()
        return hook

    for i, l in enumerate(layers):
        handles.append(l.register_forward_hook(mk(i)))
    try:
        model(torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(feats, 0)


def features(model, tok, layers, n_layer, device, phrase_label_pairs):
    X, y = [], []
    for phrase, lab in phrase_label_pairs:
        ids = tok(phrase, add_special_tokens=True)["input_ids"]
        X.append(capture(model, layers, ids, device, n_layer))
        y.append(lab)
    return torch.stack(X, 0), torch.tensor(y, dtype=torch.long)


def build_pairs():
    cat_keys = list(CAT.keys())
    cat = [(name, ci) for ci, k in enumerate(cat_keys) for name in CAT[k]]
    ident = [(ctx.format(e), ei) for ei, e in enumerate(ID_ENTS) for ctx in CONTEXTS]
    bound = [(tmpl.format(name=n, role=r), ei)
             for ei, (n, r, typ) in enumerate(BOUND_ENTS) for tmpl in TEMPLATES[typ]]
    return {"category": (cat, len(cat_keys)),
            "identity": (ident, len(ID_ENTS)),
            "bound": (bound, len(BOUND_ENTS))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/probe_hf")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    slug = args.hf_model.replace("/", "_")

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers", flush=True)
    pairs = build_pairs()
    out = {"model": args.hf_model, "n_layer": n_layer, "probes": {}}

    for name, (pl, n_classes) in pairs.items():
        chance = 1.0 / n_classes
        X, y = features(model, tok, layers, n_layer, device, pl)
        accs = probe_all_layers(X, y, n_layer, n_classes, device)
        deep = sum(accs[2 * n_layer // 3:]) / len(accs[2 * n_layer // 3:])
        out["probes"][name] = {"n_classes": n_classes, "chance": chance, "acc": accs}
        print(f"  [{name}] {n_classes}-way chance {chance:.3f} | "
              f"peak {max(accs):.2f}@L{accs.index(max(accs))} | "
              f"deep-third {deep:.2f} | last {accs[-1]:.2f}", flush=True)

    with open(os.path.join(args.out, f"{slug}__probes.json"), "w") as f:
        json.dump(out, f, indent=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    styles = {"category": ("category (8-way, bare name)", "-"),
              "identity": ("identity (15-way, name at end)", "--"),
              "bound": ("bound identity (15-way, generic last token)", "-")}
    for i, (name, rec) in enumerate(out["probes"].items()):
        lbl, ls = styles[name]
        ax.plot(range(n_layer), rec["acc"], marker="o", ms=2.5, ls=ls,
                color=PALETTE[i % len(PALETTE)], label=lbl)
        ax.axhline(rec["chance"], color=PALETTE[i % len(PALETTE)], lw=0.6, ls=":")
    ax.axvspan(2 * n_layer // 3, n_layer - 1, color="0.92", zorder=0)
    ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{args.hf_model}: entity probes across depth\n"
                 "dotted = chance; gray = deep third")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, f"{slug}__probes.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_png}", flush=True)


if __name__ == "__main__":
    main()
