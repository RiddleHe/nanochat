"""Per-layer linear probe for entity category: 'gone' vs 'unreadable'.

The category-patchscope (patchscope_category.py) reads a layer's hidden state
with the model's OWN frozen machinery and found that an entity's category stops
being decodable in deep layers (~L16+). That tells us the FIXED readout fails
there; it does not tell us whether the information is still present. This script
asks the complementary question with the strongest possible reader: train a
fresh linear probe at each layer to classify the entity's category from the
last-token hidden state.

    deep-layer probe accuracy stays HIGH  -> category info is still linearly
        present in deep layers; the patchscope "disappearance" was a readout
        limitation (info present, unreadable by the fixed probe).
    deep-layer probe accuracy DROPS too   -> even the best linear reader can't
        find it; the info is genuinely transformed/removed from the readable
        subspace, not just unreadable by patchscope.

Design notes:
  - Features are unit-normalized per row, so the probe reads DIRECTION not
    magnitude (deep residuals run hotter; we don't want norm to drive accuracy).
  - Linear probe (single Linear), L2 weight decay, averaged over several random
    train/test splits for a stable per-layer estimate. The cross-layer PROFILE
    is the result; absolute level matters less.
  - Run on the same two d24/100B models as the patchscope so the comparison is
    apples-to-apples.

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.probe_entity_category \
      --model-tags arch_d24_gpt_base_100B \
                   arch_d24_gpt_base_v_from_value_emb_learn_100B \
      --labels Attention BoV --out results/probe_category
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

from scripts.inspect.patchscope_few_shot import PALETTE


# 8 categories, ~20 clean entities each, no cross-category collisions
# (Tesla -> scientist only; Apple/Amazon -> company only; orange etc -> fruit).
ENTITIES = {
    "country": ["France", "Germany", "Japan", "Brazil", "Canada", "Italy", "Spain",
                "Mexico", "India", "Egypt", "Norway", "Greece", "Kenya", "Argentina",
                "Thailand", "Poland", "Sweden", "Vietnam", "Ireland", "Portugal"],
    "city": ["Paris", "London", "Tokyo", "Berlin", "Chicago", "Toronto", "Madrid",
             "Boston", "Seattle", "Vienna", "Amsterdam", "Dublin", "Mumbai", "Cairo",
             "Sydney", "Houston", "Miami", "Denver", "Atlanta", "Dallas"],
    "scientist": ["Einstein", "Newton", "Darwin", "Curie", "Tesla", "Galileo",
                  "Faraday", "Bohr", "Heisenberg", "Maxwell", "Pasteur", "Mendel",
                  "Hawking", "Feynman", "Planck", "Kepler", "Copernicus",
                  "Schrodinger", "Fermi", "Turing"],
    "writer": ["Shakespeare", "Dickens", "Tolstoy", "Hemingway", "Orwell", "Austen",
               "Twain", "Kafka", "Joyce", "Proust", "Dostoevsky", "Faulkner",
               "Steinbeck", "Fitzgerald", "Wilde", "Chekhov", "Hugo", "Dante",
               "Homer", "Melville"],
    "composer": ["Mozart", "Beethoven", "Bach", "Chopin", "Brahms", "Schubert",
                 "Wagner", "Vivaldi", "Handel", "Haydn", "Mahler", "Liszt",
                 "Tchaikovsky", "Verdi", "Debussy", "Ravel", "Puccini", "Dvorak",
                 "Grieg", "Schumann"],
    "animal": ["elephant", "tiger", "dolphin", "giraffe", "kangaroo", "penguin",
               "leopard", "rhinoceros", "hippopotamus", "zebra", "gorilla", "cheetah",
               "crocodile", "octopus", "eagle", "falcon", "salmon", "whale",
               "antelope", "buffalo"],
    "company": ["Google", "Apple", "Samsung", "Microsoft", "Amazon", "Toyota", "Sony",
                "Nike", "Intel", "Boeing", "Pfizer", "Disney", "Netflix", "Oracle",
                "Adobe", "Nvidia", "Honda", "Ford", "Walmart", "Spotify"],
    "fruit": ["banana", "orange", "mango", "pineapple", "strawberry", "blueberry",
              "raspberry", "watermelon", "peach", "apricot", "cherry", "grape",
              "lemon", "papaya", "kiwi", "plum", "pear", "pomegranate", "coconut",
              "avocado"],
}


def load_adapter(tag, step, device):
    from nanochat.checkpoint_manager import load_model
    model, tok, meta = load_model("base", device, phase="eval", model_tag=tag, step=step)
    model.eval()
    return model, tok, meta["model_config"]["n_layer"], tok.get_bos_token_id()


@torch.inference_mode()
def capture_last_token(model, ids, device, n_layer):
    """Last-token hidden state at every block. Returns (n_layer, D) float cpu."""
    feats = [None] * n_layer
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, -1, :].detach().float().cpu()
        return hook

    for i, b in enumerate(model.transformer.h):
        handles.append(b.register_forward_hook(mk(i)))
    try:
        model(torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(feats, dim=0)


def build_features(tag, step, device):
    model, tok, n_layer, bos = load_adapter(tag, step, device)
    cats = list(ENTITIES.keys())
    X = []  # list of (n_layer, D)
    y = []
    for ci, cat in enumerate(cats):
        for name in ENTITIES[cat]:
            ids = [bos] + tok.encode(name)
            X.append(capture_last_token(model, ids, device, n_layer))
            y.append(ci)
    del model
    torch.cuda.empty_cache()
    X = torch.stack(X, dim=0)          # (N, n_layer, D)
    y = torch.tensor(y, dtype=torch.long)
    return X, y, n_layer, cats


def train_probe(Xtr, ytr, Xte, yte, n_classes, device, steps=400):
    d = Xtr.shape[1]
    probe = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-2)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    for _ in range(steps):
        loss = F.cross_entropy(probe(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(Xte).argmax(1) == yte).float().mean().item()
    return acc


def per_class_split(y, n_classes, frac_train, seed):
    """Stratified split: frac_train of each class to train."""
    g = torch.Generator().manual_seed(seed)
    tr, te = [], []
    for c in range(n_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        k = int(round(frac_train * len(idx)))
        tr.append(perm[:k]); te.append(perm[k:])
    return torch.cat(tr), torch.cat(te)


def probe_all_layers(X, y, n_layer, n_classes, device, n_splits=5):
    # unit-normalize per (example, layer)
    Xn = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    accs = []
    for L in range(n_layer):
        XL = Xn[:, L, :]
        split_acc = []
        for s in range(n_splits):
            tr, te = per_class_split(y, n_classes, 0.7, seed=1000 + s)
            split_acc.append(train_probe(XL[tr], y[tr], XL[te], y[te], n_classes, device))
        accs.append(sum(split_acc) / len(split_acc))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/probe_category")
    args = ap.parse_args()
    labels = args.labels or args.model_tags
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    n_classes = len(ENTITIES)
    chance = 1.0 / n_classes
    results = {"chance": chance, "categories": list(ENTITIES.keys()),
               "n_entities": sum(len(v) for v in ENTITIES.values()), "models": {}}

    for tag, label in zip(args.model_tags, labels):
        print(f"\n=== {label} ({tag}) ===", flush=True)
        X, y, n_layer, cats = build_features(tag, args.step, device)
        accs = probe_all_layers(X, y, n_layer, n_classes, device)
        results["models"][label] = {"tag": tag, "n_layer": n_layer, "acc": accs}
        peak = max(accs); peak_L = accs.index(peak)
        deep = sum(accs[2 * n_layer // 3:]) / len(accs[2 * n_layer // 3:])
        print(f"  chance {chance:.2f} | peak {peak:.2f}@L{peak_L} | "
              f"deep-third mean {deep:.2f} | last layer {accs[-1]:.2f}", flush=True)
        for L, a in enumerate(accs):
            print(f"    L{L:02d} probe acc {a:.3f}", flush=True)

    with open(os.path.join(args.out, "probe_category.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, rec) in enumerate(results["models"].items()):
        ax.plot(range(rec["n_layer"]), rec["acc"], marker="o", ms=3,
                color=PALETTE[i % len(PALETTE)], label=label)
    ax.axhline(chance, color="0.6", lw=0.8, ls=":", label=f"chance ({chance:.2f})")
    nl = next(iter(results["models"].values()))["n_layer"]
    ax.axvspan(2 * nl // 3, nl - 1, color="0.9", zorder=0,
               label="patchscope 'gone' zone (deep third)")
    ax.set_xlabel("layer"); ax.set_ylabel("entity-category probe accuracy")
    ax.set_title("Can a trained probe read the entity category at each layer?\n"
                 "high in deep third = info present but unreadable by patchscope; "
                 "drop = genuinely gone")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, "probe_category.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_png}", flush=True)


if __name__ == "__main__":
    main()
