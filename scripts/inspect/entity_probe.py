"""Per-layer linear probe: is the context-bound entity identity readable at
every layer?

Setup: 15 entities x 12 sentence templates. Every sentence names the entity
early and ends in a generic role noun shared within its category (all three
scientists end in "scientist"), so the last token's identity must be bound
from context. We capture the last token's hidden state at every layer and
train one linear classifier per layer to predict WHICH entity (15-way).

Checks reported per layer:
  clean        held-out-template k-fold (train on 8 templates, test on 4
               unseen ones; templates are shared across entities, so template
               is decorrelated from entity). PRIMARY metric.
  within-cat   same split, one category at a time (Einstein vs Newton ...).
  random-split per-example 70/30 split; entangles template with entity and
               under-estimates accuracy — shown for comparison.
  random-label labels shuffled; must sit at chance (leakage control).
  raw          clean split on un-normalized hidden states (norm control).

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect.entity_probe \
      --hf-model Qwen/Qwen3-8B-Base --label Qwen3-8B --out results/probe
  NANOCHAT_BASE_DIR=... CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect.entity_probe \
      --model-tag arch_d24_gpt_base_100B --label Attention --out results/probe
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

PALETTE = ["#1f77b4", "#e8a87c", "#2c8a8a", "#5aa75a"]

# (name, role noun, template family). Several entities share a role, so the
# 15-way identity target cannot be solved from the final token alone.
ENTITIES = [
    ("Einstein", "scientist", "person"), ("Newton", "scientist", "person"),
    ("Darwin", "scientist", "person"), ("Mozart", "composer", "person"),
    ("Beethoven", "composer", "person"), ("Shakespeare", "writer", "person"),
    ("Dickens", "writer", "person"), ("France", "country", "place"),
    ("Japan", "country", "place"), ("Paris", "city", "place"),
    ("Tokyo", "city", "place"), ("elephant", "animal", "animal"),
    ("tiger", "animal", "animal"), ("Google", "company", "company"),
    ("Apple", "company", "company"),
]

TEMPLATES = {
    "person": [
        "{name} was a celebrated {role}", "the famous {name}, a brilliant {role}",
        "many regard {name} as the greatest {role}",
        "{name} is remembered as an influential {role}",
        "a film about {name}, a legendary {role}", "{name} remains an iconic {role}",
        "history honors {name} as a pioneering {role}",
        "people admire {name}, a visionary {role}",
        "the legacy of {name}, a remarkable {role}", "{name} truly was a masterful {role}",
        "in school we studied {name}, a great {role}",
        "{name} is often called a world-class {role}",
    ],
    "place": [
        "{name} is a beautiful {role}", "I visited {name}, a wonderful {role}",
        "{name} is a fascinating {role}", "many travel to {name}, a vibrant {role}",
        "{name} remains a historic {role}", "a guide to {name}, a lovely {role}",
        "{name} is a popular {role}", "the charm of {name}, a unique {role}",
        "tourists love {name}, a stunning {role}", "{name} is a remarkable {role}",
        "we flew to {name}, an amazing {role}", "{name} is a well-known {role}",
    ],
    "animal": [
        "the {name} is a magnificent {role}", "I saw a {name}, a fascinating {role}",
        "the {name} is a powerful {role}", "a {name} is a remarkable {role}",
        "people fear the {name}, a wild {role}", "the {name} is an impressive {role}",
        "a documentary on the {name}, a majestic {role}",
        "the {name} is a graceful {role}", "the {name} is a strong {role}",
        "the {name} is a beautiful {role}", "the {name} is a rare {role}",
        "we watched the {name}, a wild {role}",
    ],
    "company": [
        "{name} is a major {role}", "{name} is a leading {role}",
        "{name} is a global {role}", "I work at {name}, a huge {role}",
        "{name} is an innovative {role}", "{name} is a powerful {role}",
        "{name} is a well-known {role}", "{name} is a successful {role}",
        "{name} is a giant {role}", "{name} is a famous {role}",
        "{name} is a profitable {role}", "{name} is an influential {role}",
    ],
}
N_TEMPLATES = 12
FOLD = 4  # held-out-template folds: 3 folds of 4 test templates


# ---------------------------------------------------------------- model I/O

def load_any(args, device):
    """Returns (model, block_list, n_layer, encode)."""
    if args.model_tag:
        from nanochat.checkpoint_manager import load_model
        model, tok, meta = load_model("base", device, phase="eval",
                                      model_tag=args.model_tag, step=args.step)
        model.eval()
        bos = tok.get_bos_token_id()
        return (model, model.transformer.h, meta["model_config"]["n_layer"],
                lambda p: [bos] + tok.encode(p))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16).to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        layers = model.transformer.h
    return (model, layers, len(layers),
            lambda p: tok(p, add_special_tokens=True)["input_ids"])


@torch.inference_mode()
def capture_last(model, layers, ids, device, n_layer):
    """Last-token hidden state after every block. Returns (n_layer, D) on cpu."""
    feats = [None] * n_layer
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, -1, :].detach().float().cpu()
        return hook

    for i, blk in enumerate(layers):
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        model(torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(feats, 0)


def build_features(model, layers, n_layer, encode, device):
    X, y, cat, tmpl = [], [], [], []
    cats = sorted(set(role for _, role, _ in ENTITIES))
    cat_idx = {c: i for i, c in enumerate(cats)}
    for ei, (name, role, typ) in enumerate(ENTITIES):
        for ti, t in enumerate(TEMPLATES[typ][:N_TEMPLATES]):
            X.append(capture_last(model, layers, encode(t.format(name=name, role=role)),
                                  device, n_layer))
            y.append(ei); cat.append(cat_idx[role]); tmpl.append(ti)
    return (torch.stack(X, 0), torch.tensor(y), torch.tensor(cat),
            torch.tensor(tmpl), cats)


# ---------------------------------------------------------------- probing

def norm_rows(X):
    return X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def train_probe(Xtr, ytr, Xte, yte, n_classes, device, steps=300):
    probe = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-2)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    for _ in range(steps):
        loss = F.cross_entropy(probe(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((probe(Xte).argmax(1) == yte).float().mean())


def per_class_split(y, n_classes, frac_train, seed):
    g = torch.Generator().manual_seed(seed)
    tr, te = [], []
    for c in range(n_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        k = int(round(frac_train * len(idx)))
        tr.append(perm[:k]); te.append(perm[k:])
    return torch.cat(tr), torch.cat(te)


def kfold_template(XL, y, tmpl, n_cls, device):
    accs = []
    for start in range(0, N_TEMPLATES, FOLD):
        te = (tmpl >= start) & (tmpl < start + FOLD)
        accs.append(train_probe(XL[~te], y[~te], XL[te], y[te], n_cls, device))
    return sum(accs) / len(accs)


def within_category(XL, y, cat, tmpl, cats, device):
    accs, chances = [], []
    for ci in range(len(cats)):
        idx = (cat == ci).nonzero(as_tuple=True)[0]
        ents = sorted(set(y[idx].tolist()))
        if len(ents) < 2:
            continue
        remap = {e: i for i, e in enumerate(ents)}
        yc = torch.tensor([remap[v.item()] for v in y[idx]])
        accs.append(kfold_template(XL[idx], yc, tmpl[idx], len(ents), device))
        chances += [1.0 / len(ents)] * len(idx)
    return sum(accs) / len(accs), sum(chances) / len(chances)


def random_split(XL, y, n_cls, device, seeds):
    a = [train_probe(*_sp(XL, y, n_cls, s), n_cls, device) for s in seeds]
    t = torch.tensor(a)
    return float(t.mean()), float(t.std())


def _sp(XL, y, n_cls, seed):
    tr, te = per_class_split(y, n_cls, 0.7, seed)
    return XL[tr], y[tr], XL[te], y[te]


def random_label(XL, y, n_cls, device, seeds):
    a = []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        ys = y[torch.randperm(len(y), generator=g)]
        a.append(train_probe(*_sp(XL, ys, n_cls, s), n_cls, device))
    return sum(a) / len(a)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-tag"); g.add_argument("--hf-model")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/probe")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    label = args.label or (args.model_tag or args.hf_model).replace("/", "_")

    model, layers, n_layer, encode = load_any(args, device)
    X, y, cat, tmpl, cats = build_features(model, layers, n_layer, encode, device)
    del model; torch.cuda.empty_cache()
    n_cls = len(ENTITIES)
    print(f"{label}: {n_layer} layers, {len(y)} examples, {n_cls} entities", flush=True)

    seeds = list(range(1000, 1010))
    res = {"label": label, "n_layer": n_layer, "chance": 1.0 / n_cls,
           "clean": [], "clean_raw": [], "within_cat": [], "within_cat_chance": None,
           "random_split": [], "random_split_std": [], "random_label": []}
    for L in range(n_layer):
        Xn, Xr = norm_rows(X[:, L, :]), X[:, L, :]
        res["clean"].append(kfold_template(Xn, y, tmpl, n_cls, device))
        res["clean_raw"].append(kfold_template(Xr, y, tmpl, n_cls, device))
        wc, wcc = within_category(Xn, y, cat, tmpl, cats, device)
        res["within_cat"].append(wc); res["within_cat_chance"] = wcc
        rm, rs = random_split(Xn, y, n_cls, device, seeds)
        res["random_split"].append(rm); res["random_split_std"].append(rs)
        res["random_label"].append(random_label(Xn, y, n_cls, device, seeds[:5]))
        print(f"  L{L:02d}  clean {res['clean'][-1]:.2f}  within-cat {wc:.2f}  "
              f"random-split {rm:.2f}  random-label {res['random_label'][-1]:.2f}",
              flush=True)

    with open(os.path.join(args.out, f"{label}__probe.json"), "w") as f:
        json.dump(res, f, indent=1)

    xs = range(n_layer)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, res["clean"], "-o", ms=3, color=PALETTE[0],
            label="held-out-template, 15-way (primary)")
    ax.plot(xs, res["clean_raw"], ":", color=PALETTE[0], alpha=0.6, label="raw h (control)")
    ax.plot(xs, res["within_cat"], "-s", ms=3, color=PALETTE[2], label="within-category")
    rmn, rsd = torch.tensor(res["random_split"]), torch.tensor(res["random_split_std"])
    ax.plot(xs, res["random_split"], "--", color=PALETTE[1],
            label="random split (template-confounded)")
    ax.fill_between(xs, (rmn - rsd).tolist(), (rmn + rsd).tolist(),
                    color=PALETTE[1], alpha=0.12)
    ax.plot(xs, res["random_label"], "-x", ms=3, color="0.5", label="random-label (floor)")
    ax.axhline(res["chance"], color="0.7", lw=0.8, ls="--",
               label=f"chance ({res['chance']:.2f})")
    ax.axhline(res["within_cat_chance"], color=PALETTE[2], lw=0.8, ls="--", alpha=0.5)
    ax.axvspan(2 * n_layer // 3, n_layer - 1, color="0.93", zorder=0)
    ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{label}: per-layer entity probe (gray = deep third)")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, f"{label}__probe.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
