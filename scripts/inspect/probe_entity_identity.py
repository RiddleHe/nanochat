"""Fine-grained, in-context identity probe — closes two caveats of the
category probe (scripts/inspect/probe_entity_category.py).

Caveat 1 the category probe left open: it used BARE entity names, not entities
bound in context like the patchscope phrases. Here every entity is wrapped in
16 varied sentence CONTEXTS and we probe the last token of the in-context
phrase, so the representation is a contextually processed entity mention.

Caveat 2: the category probe read COARSE category (8 classes). Here the target
is the SPECIFIC entity (15-way, several entities share a category, so category
alone is useless) — a much finer, harder readout.

If a trained probe can still name WHICH specific entity from the deep-layer
hidden state, then specific identity (not just category) is linearly present
deep, in context — firmly pinning the patchscope "disappearance" as a readout
limitation rather than information loss.

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.probe_entity_identity \
      --model-tags arch_d24_gpt_base_100B \
                   arch_d24_gpt_base_v_from_value_emb_learn_100B \
      --labels Attention BoV --out results/probe_identity
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import PALETTE
from scripts.inspect.probe_entity_category import (
    load_adapter, capture_last_token, probe_all_layers)


# 15 specific entities; several share a category (3 scientists, 2 composers, 2
# writers, 2 countries, 2 cities, 2 animals, 2 companies) so the 15-way identity
# target cannot be solved by category alone.
ENTITIES = ["Einstein", "Newton", "Darwin", "Mozart", "Beethoven",
            "Shakespeare", "Dickens", "France", "Japan", "Paris", "Tokyo",
            "elephant", "tiger", "Google", "Apple"]

# 16 contexts; the entity is the final token of each phrase, but preceded by
# varied context so the probed representation is contextualized, not isolated.
CONTEXTS = [
    "{}",
    "the famous {}",
    "I was reading about {}",
    "a documentary about {}",
    "Have you ever heard of {}",
    "Yesterday we discussed {}",
    "My essay is about {}",
    "an article on {}",
    "speaking of {}",
    "the legacy of {}",
    "a biography of {}",
    "experts often mention {}",
    "the story of {}",
    "let me tell you about {}",
    "she is fascinated by {}",
    "everyone admires {}",
]


def build_features(tag, step, device):
    model, tok, n_layer, bos = load_adapter(tag, step, device)
    X, y = [], []
    for ei, ent in enumerate(ENTITIES):
        for ctx in CONTEXTS:
            ids = [bos] + tok.encode(ctx.format(ent))
            X.append(capture_last_token(model, ids, device, n_layer))
            y.append(ei)
    del model
    torch.cuda.empty_cache()
    return torch.stack(X, 0), torch.tensor(y, dtype=torch.long), n_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/probe_identity")
    args = ap.parse_args()
    labels = args.labels or args.model_tags
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    n_classes = len(ENTITIES)
    chance = 1.0 / n_classes
    results = {"chance": chance, "n_entities": n_classes,
               "n_contexts": len(CONTEXTS), "entities": ENTITIES, "models": {}}

    for tag, label in zip(args.model_tags, labels):
        print(f"\n=== {label} ({tag}) — {n_classes}-way identity, "
              f"{len(CONTEXTS)} contexts ===", flush=True)
        X, y, n_layer = build_features(tag, args.step, device)
        accs = probe_all_layers(X, y, n_layer, n_classes, device)
        results["models"][label] = {"tag": tag, "n_layer": n_layer, "acc": accs}
        peak = max(accs)
        deep = sum(accs[2 * n_layer // 3:]) / len(accs[2 * n_layer // 3:])
        print(f"  chance {chance:.3f} | peak {peak:.2f}@L{accs.index(peak)} | "
              f"deep-third mean {deep:.2f} | last layer {accs[-1]:.2f}", flush=True)
        for L, a in enumerate(accs):
            print(f"    L{L:02d} identity acc {a:.3f}", flush=True)

    with open(os.path.join(args.out, "probe_identity.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, rec) in enumerate(results["models"].items()):
        ax.plot(range(rec["n_layer"]), rec["acc"], marker="o", ms=3,
                color=PALETTE[i % len(PALETTE)], label=label)
    ax.axhline(chance, color="0.6", lw=0.8, ls=":", label=f"chance ({chance:.3f})")
    nl = next(iter(results["models"].values()))["n_layer"]
    ax.axvspan(2 * nl // 3, nl - 1, color="0.9", zorder=0,
               label="patchscope 'gone' zone (deep third)")
    ax.set_xlabel("layer"); ax.set_ylabel("specific-entity identity accuracy (15-way)")
    ax.set_title("Fine-grained, in-context identity probe\n"
                 "high in deep third = the SPECIFIC entity is still readable deep "
                 "(disappearance = readout artifact)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, "probe_identity.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_png}", flush=True)


if __name__ == "__main__":
    main()
