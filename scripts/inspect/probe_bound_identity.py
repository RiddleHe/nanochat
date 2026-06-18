"""Bound-identity probe — the faithful patchscope hard case.

probe_entity_identity.py put the entity name as the LAST token, so identity was
trivially present from layer 0 (~100% everywhere) — it tested deep retention
but not context BINDING. This version replicates the patchscope construction
exactly: each phrase names a specific entity EARLY, then ends in a GENERIC role
noun shared within its category (all 3 scientists end in "scientist", both
cities end in "city", ...). The probed last token is therefore generic; only
the bound context tells you WHICH entity.

Probe target = the specific 15-way entity. To distinguish Einstein from Newton
(both phrases end in "scientist"), the bound identity must have reached the
last token. Expected shape: low within-category identity in early layers (last
token still generic), rising as context binds. The question: does deep-layer
identity STAY high (info retained, patchscope disappearance = readout artifact)
or fall (info genuinely lost at the generic bound token)?

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.probe_bound_identity \
      --model-tags arch_d24_gpt_base_100B \
                   arch_d24_gpt_base_v_from_value_emb_learn_100B \
      --labels Attention BoV --out results/probe_bound
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


# (name, role-noun, template-type). Several entities share a role, so the
# shared final role token cannot reveal identity — only the bound context can.
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
        "the legacy of {name}, a remarkable {role}",
        "{name} truly was a masterful {role}",
    ],
    "place": [
        "{name} is a beautiful {role}", "I visited {name}, a wonderful {role}",
        "{name} is a fascinating {role}", "many travel to {name}, a vibrant {role}",
        "{name} remains a historic {role}", "a guide to {name}, a lovely {role}",
        "{name} is a popular {role}", "the charm of {name}, a unique {role}",
        "tourists love {name}, a stunning {role}", "{name} is a remarkable {role}",
    ],
    "animal": [
        "the {name} is a magnificent {role}", "I saw a {name}, a fascinating {role}",
        "the {name} is a powerful {role}", "a {name} is a remarkable {role}",
        "people fear the {name}, a wild {role}", "the {name} is an impressive {role}",
        "a documentary on the {name}, a majestic {role}",
        "the {name} is a graceful {role}", "the {name} is a strong {role}",
        "the {name} is a beautiful {role}",
    ],
    "company": [
        "{name} is a major {role}", "{name} is a leading {role}",
        "{name} is a global {role}", "I work at {name}, a huge {role}",
        "{name} is an innovative {role}", "{name} is a powerful {role}",
        "{name} is a well-known {role}", "{name} is a successful {role}",
        "{name} is a giant {role}", "{name} is a famous {role}",
    ],
}


def build_features(tag, step, device):
    model, tok, n_layer, bos = load_adapter(tag, step, device)
    X, y = [], []
    for ei, (name, role, typ) in enumerate(ENTITIES):
        for tmpl in TEMPLATES[typ]:
            phrase = tmpl.format(name=name, role=role)
            ids = [bos] + tok.encode(phrase)
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
    ap.add_argument("--out", default="results/probe_bound")
    args = ap.parse_args()
    labels = args.labels or args.model_tags
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    n_classes = len(ENTITIES)
    chance = 1.0 / n_classes
    results = {"chance": chance, "n_entities": n_classes,
               "entities": [e[0] for e in ENTITIES], "models": {}}

    for tag, label in zip(args.model_tags, labels):
        print(f"\n=== {label} ({tag}) — bound {n_classes}-way identity "
              f"(generic last token) ===", flush=True)
        X, y, n_layer = build_features(tag, args.step, device)
        accs = probe_all_layers(X, y, n_layer, n_classes, device)
        results["models"][label] = {"tag": tag, "n_layer": n_layer, "acc": accs}
        deep = sum(accs[2 * n_layer // 3:]) / len(accs[2 * n_layer // 3:])
        print(f"  chance {chance:.3f} | peak {max(accs):.2f}@L{accs.index(max(accs))} "
              f"| early L1 {accs[1]:.2f} | deep-third mean {deep:.2f} | "
              f"last {accs[-1]:.2f}", flush=True)
        for L, a in enumerate(accs):
            print(f"    L{L:02d} bound-identity acc {a:.3f}", flush=True)

    with open(os.path.join(args.out, "probe_bound.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, rec) in enumerate(results["models"].items()):
        ax.plot(range(rec["n_layer"]), rec["acc"], marker="o", ms=3,
                color=PALETTE[i % len(PALETTE)], label=label)
    ax.axhline(chance, color="0.6", lw=0.8, ls=":", label=f"chance ({chance:.3f})")
    nl = next(iter(results["models"].values()))["n_layer"]
    ax.axvspan(2 * nl // 3, nl - 1, color="0.9", zorder=0,
               label="patchscope 'gone' zone (deep third)")
    ax.set_xlabel("layer")
    ax.set_ylabel("bound identity accuracy (15-way, generic last token)")
    ax.set_title("Patchscope hard case: identity bound by context at a GENERIC token\n"
                 "high in deep third = bound specific identity survives deep "
                 "(disappearance = readout artifact)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out, "probe_bound.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_png}", flush=True)


if __name__ == "__main__":
    main()
