"""Entity interchange patching: causal interchange patching — at which layer does the model stop
USING an entity's information?

Setup. Pairs of sentences differ only in the entity name (same token length):

    clean     : "Everyone knows Einstein was a celebrated scientist. The scientist was"
    corrupted : "Everyone knows Newton  was a celebrated scientist. The scientist was"

For each layer L we run the corrupted sentence, but overwrite the name
position's CUMULATIVE hidden state (block output) at layer L with the clean
run's same-layer state, then let the remaining layers run. The next-token
prediction is read at the FINAL position. Sweeping L localizes where the
entity position is still being read: if patching at L flips the answer,
layers after L still consume that position; if nothing changes, they no
longer read it.

One run produces BOTH views from the same patched logits:
  entity_patching_recovery.png : normalized logit-diff recovery per layer
                        (1 = output fully flips to the clean entity)
  entity_patching_probs.png    : softmax P(clean entity) and P(corrupt entity) per
                        layer, two lines, with clean/corrupted-run reference
                        levels

Within-category ordered pairs (Einstein<->Newton, Paris<->Tokyo, ...) with
single-token names; pairs whose clean/corrupt logit-diffs are too close are
skipped. `--patch role` patches the role noun instead (documented confound:
the corrupt name remains readable in the prompt, so recovery ~0 everywhere).

Usage:
  python -m scripts.inspect.entity_patching --hf-model Qwen/Qwen3-8B-Base --device cpu
  # quick smoke: --max-pairs 1
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, TEAL, RED = "#5aa75a", "#2c8a8a", "#c0392b"

ENTITIES = [
    ("Einstein", "scientist"), ("Newton", "scientist"), ("Darwin", "scientist"),
    ("Mozart", "composer"), ("Beethoven", "composer"),
    ("Shakespeare", "writer"), ("Dickens", "writer"),
    ("France", "country"), ("Japan", "country"),
    ("Paris", "city"), ("Tokyo", "city"),
    ("Google", "company"), ("Apple", "company"),
]
TEMPLATE = "Everyone knows {name} was a celebrated {role}. The {role} was"


def load_hf(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        layers = model.transformer.h
    return model, tok, layers, len(layers)


def single_token(tok, name):
    return len(tok(" " + name, add_special_tokens=False)["input_ids"]) == 1


def within_pairs(tok):
    cats = {}
    for name, role in ENTITIES:
        if single_token(tok, name):
            cats.setdefault(role, []).append(name)
    return [(a, b, role) for role, names in cats.items()
            for a in names for b in names if a != b]


def find_pos(tok, ids, word):
    for i, t in enumerate(ids):
        if tok.decode([t]).strip() == word:
            return i
    raise RuntimeError(f"{word!r} not found in {[tok.decode([t]) for t in ids]}")


@torch.inference_mode()
def capture(model, layers, ids, pos, device, n_layer):
    """Forward once; per-layer cumulative hidden state at `pos` + final logits."""
    feats = [None] * n_layer
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, pos, :].detach().clone()
        return hook

    for i, l in enumerate(layers):
        handles.append(l.register_forward_hook(mk(i)))
    try:
        logits = model(torch.tensor([ids], device=device)).logits[0, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    return feats, logits


@torch.inference_mode()
def patched(model, layers, ids, L, pos, vec, device):
    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[0, pos, :] = vec.to(h.dtype)
        return (h,) + out[1:] if is_t else h

    handle = layers[L].register_forward_hook(hook)
    try:
        return model(torch.tensor([ids], device=device)).logits[0, -1, :].float()
    finally:
        handle.remove()


def run_topk_example(model, tok, layers, n_layer, device, out, template=TEMPLATE,
                     clean="Einstein", corrupt="Newton", role="scientist", k=10):
    """For one pair, dump the final position's top-k tokens+probs per patched
    layer (plus unpatched clean/corrupted reference rows) as json + a rendered
    table figure. Contextualizes the entity probabilities: what actually tops
    the distribution, and where the entity names rank."""
    ids_c = tok(template.format(name=clean, role=role), add_special_tokens=True)["input_ids"]
    ids_x = tok(template.format(name=corrupt, role=role), add_special_tokens=True)["input_ids"]
    pos = find_pos(tok, ids_c, clean)
    ct = tok(" " + clean, add_special_tokens=False)["input_ids"][0]
    xt = tok(" " + corrupt, add_special_tokens=False)["input_ids"][0]
    states, log_c = capture(model, layers, ids_c, pos, device, n_layer)
    _, log_x = capture(model, layers, ids_x, pos, device, n_layer)

    def topk(logits):
        sm = F.softmax(logits, -1)
        v, i = sm.topk(k)
        return [(tok.decode([int(t)]), float(p)) for p, t in zip(v, i)]

    rows = [("clean run", topk(log_c)), ("corrupt run", topk(log_x))]
    for L in range(n_layer):
        rows.append((f"patch L{L:02d}", topk(patched(model, layers, ids_x, L, pos,
                                                     states[L], device))))
        print(f"  {rows[-1][0]}: " + "  ".join(f"{t!r} {p:.1%}" for t, p in rows[-1][1][:5]),
              flush=True)

    with open(os.path.join(out, "entity_patching_topk.json"), "w") as f:
        json.dump({"clean": clean, "corrupt": corrupt, "role": role,
                   "rows": [{"row": r, "topk": tk} for r, tk in rows]}, f, indent=1)

    fig, ax = plt.subplots(figsize=(14, 0.62 + 0.30 * len(rows)))
    ax.axis("off")
    ax.set_xlim(0, 1 + k * 1.3)
    ax.set_ylim(-len(rows), 1.2)
    ax.text(0, 0.7, f"corrupted sentence patched with clean '{clean}' state; "
            f"final-position top-{k} (teal = {clean}, red = {corrupt})", fontsize=9)
    for r, (label, tk) in enumerate(rows):
        y = -r
        ax.text(0, y, label, fontsize=8, family="monospace", weight="bold")
        for j, (t, p) in enumerate(tk):
            color = ("#2c8a8a" if t.strip() == clean
                     else "#c0392b" if t.strip() == corrupt else "0.25")
            ax.text(1 + j * 1.3, y, f"{t.strip() or repr(t)} {p:.1%}",
                    fontsize=7.5, family="monospace", color=color,
                    weight="bold" if color != "0.25" else "normal")
    fig.tight_layout()
    p = os.path.join(out, "entity_patching_topk.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch", choices=["subject", "role"], default="subject")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--template", default=TEMPLATE,
                    help="sentence template with {name} and {role}")
    ap.add_argument("--topk-example", action="store_true",
                    help="run only the Einstein->Newton example and dump the "
                         "final position's top-10 tokens per patched layer")
    ap.add_argument("--out", default="results/entity_patching")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.topk_example:
        model, tok, layers, n_layer = load_hf(args.hf_model, device)
        print(f"{args.hf_model}: {n_layer} layers, top-k example mode", flush=True)
        run_topk_example(model, tok, layers, n_layer, device, args.out,
                         template=args.template)
        return

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    pairs = within_pairs(tok)[: args.max_pairs]
    print(f"{args.hf_model}: {n_layer} layers, {len(pairs)} ordered pairs, "
          f"patch={args.patch}", flush=True)

    recov = torch.zeros(n_layer)
    p_clean = torch.zeros(n_layer)
    p_corr = torch.zeros(n_layer)
    ref = {"clean_run_P_clean": 0.0, "corr_run_P_corr": 0.0}
    n_used = 0
    for clean, corrupt, role in pairs:
        ids_c = tok(args.template.format(name=clean, role=role), add_special_tokens=True)["input_ids"]
        ids_x = tok(args.template.format(name=corrupt, role=role), add_special_tokens=True)["input_ids"]
        if len(ids_c) != len(ids_x):
            continue
        pos = find_pos(tok, ids_c, role if args.patch == "role" else clean)
        ct = tok(" " + clean, add_special_tokens=False)["input_ids"][0]
        xt = tok(" " + corrupt, add_special_tokens=False)["input_ids"][0]
        states, log_c = capture(model, layers, ids_c, pos, device, n_layer)
        _, log_x = capture(model, layers, ids_x, pos, device, n_layer)
        clean_d = float(log_c[ct] - log_c[xt])
        corr_d = float(log_x[ct] - log_x[xt])
        denom = clean_d - corr_d
        if abs(denom) < 0.5:
            continue
        ref["clean_run_P_clean"] += float(F.softmax(log_c, -1)[ct])
        ref["corr_run_P_corr"] += float(F.softmax(log_x, -1)[xt])
        for L in range(n_layer):
            pl = patched(model, layers, ids_x, L, pos, states[L], device)
            recov[L] += (float(pl[ct] - pl[xt]) - corr_d) / denom
            sm = F.softmax(pl, -1)
            p_clean[L] += float(sm[ct]); p_corr[L] += float(sm[xt])
        n_used += 1
        print(f"  {clean}->{corrupt}: done", flush=True)
    assert n_used > 0, "no usable pairs"
    recov /= n_used; p_clean /= n_used; p_corr /= n_used
    for k in ref:
        ref[k] /= n_used
    print(f"used {n_used} pairs", flush=True)
    for L in range(0, n_layer, 3):
        print(f"  L{L:02d} recovery {recov[L]:.2f}  P_clean {p_clean[L]:.4f}  "
              f"P_corr {p_corr[L]:.4f}", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer, "n_pairs": n_used,
           "patch": args.patch, "recovery": recov.tolist(),
           "P_clean_entity": p_clean.tolist(), "P_corrupt_entity": p_corr.tolist(),
           "references": ref}
    with open(os.path.join(args.out, "entity_patching.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(n_layer), recov.tolist(), "-o", ms=3, color=GREEN)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel(f"patched layer ({args.patch}-token state, clean into corrupted)")
    ax.set_ylabel("logit-diff recovery toward clean entity (1 = fully flipped)")
    ax.set_title(f"{args.hf_model}: where is the entity position still read?")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "entity_patching_recovery.png"), dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    ax2.plot(range(n_layer), p_clean.tolist(), "-o", ms=3, color=TEAL,
             label="P(correct entity) — patched run")
    ax2.plot(range(n_layer), p_corr.tolist(), "-s", ms=3, color=RED,
             label="P(wrong entity) — patched run")
    ax2.axhline(ref["clean_run_P_clean"], color=TEAL, ls=":", lw=1,
                label="P(correct) in clean run (ref)")
    ax2.axhline(ref["corr_run_P_corr"], color=RED, ls=":", lw=1,
                label="P(wrong) in corrupted run (ref)")
    ax2.set_xlabel(f"patched layer ({args.patch}-token state, clean into corrupted)")
    ax2.set_ylabel("softmax probability at final position")
    ax2.set_title(f"{args.hf_model}: entity probabilities under interchange patching")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "entity_patching_probs.png"), dpi=150, bbox_inches="tight")
    print(f"saved {args.out}/entity_patching_recovery.png and entity_patching_probs.png", flush=True)


if __name__ == "__main__":
    main()
