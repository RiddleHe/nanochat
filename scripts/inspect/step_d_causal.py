"""Step D: causal restoration (activation patching) on Qwen3-8B.

A linear probe reading the entity does NOT prove the model itself uses that
information. This tests causal relevance. Clean and corrupted prompts differ
only in the entity:

    clean     : "Einstein was a celebrated scientist. The scientist was"
    corrupted : "Newton was a celebrated scientist. The scientist was"

The final position should predict the entity name. We run corrupted, but patch
the bound role-token ("scientist") hidden state at layer L with the CLEAN one,
and measure how far the output logit-difference

    logit(clean_entity) - logit(corrupt_entity)

moves from the corrupted baseline toward the clean value. Recovery fraction
1.0 = the patched layer-L role-token state fully drives the output; 0 = it has
no causal effect. Sweeping L tells WHERE the entity token's state is causally
read by the model's own computation.

Within-category single-token entity pairs (so the logit-diff is clean and the
two prompts have equal length). Qwen3-8B. Writes to results/step_d.
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import PALETTE
from scripts.inspect.probe_step1_robust import ENTITIES
from scripts.inspect.step2_probe_vs_patchscope import load_hf

TEMPLATE = "Everyone knows {name} was a celebrated {role}. The {role} was"


def single_token(tok, name):
    return len(tok(" " + name, add_special_tokens=False)["input_ids"]) == 1


def within_pairs(tok):
    cats = {}
    for name, role, _ in ENTITIES:
        if single_token(tok, name):
            cats.setdefault(role, []).append(name)
    pairs = []
    for role, names in cats.items():
        for a in names:
            for b in names:
                if a != b:
                    pairs.append((a, b, role))
    return pairs


def find_pos(tok, ids, word):
    """Index of the FIRST token whose decode strips to `word`."""
    for i, t in enumerate(ids):
        if tok.decode([t]).strip() == word:
            return i
    raise RuntimeError(f"{word!r} not found in {[tok.decode([t]) for t in ids]}")


@torch.inference_mode()
def capture_role_states(model, layers, ids, pos, device, n_layer):
    feats = [None] * n_layer
    hs = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h[0, pos, :].detach().clone()
        return hook

    for i, l in enumerate(layers):
        hs.append(l.register_forward_hook(mk(i)))
    try:
        logits = model(torch.tensor([ids], device=device)).logits[0, -1, :].float()
    finally:
        for h in hs:
            h.remove()
    return feats, logits


@torch.inference_mode()
def patched_logits(model, layers, ids, L, pos, vec, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/step_d")
    ap.add_argument("--max-pairs", type=int, default=24)
    ap.add_argument("--patch", choices=["role", "subject"], default="subject",
                    help="which token's state to patch clean->corrupted")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers", flush=True)
    pairs = within_pairs(tok)[: args.max_pairs]
    print(f"{len(pairs)} within-category ordered pairs", flush=True)

    recov = torch.zeros(n_layer)
    n_used = 0
    for clean, corrupt, role in pairs:
        ids_c = tok(TEMPLATE.format(name=clean, role=role), add_special_tokens=True)["input_ids"]
        ids_x = tok(TEMPLATE.format(name=corrupt, role=role), add_special_tokens=True)["input_ids"]
        if len(ids_c) != len(ids_x):
            continue
        # role token is shared text (same index in both); subject token is the
        # entity name itself (patch clean subject state into the corrupt subject
        # position, which sits at the same index since names are single-token).
        pos = find_pos(tok, ids_c, role if args.patch == "role" else clean)
        ct = tok(" " + clean, add_special_tokens=False)["input_ids"][0]
        xt = tok(" " + corrupt, add_special_tokens=False)["input_ids"][0]

        clean_states, clean_log = capture_role_states(model, layers, ids_c, pos, device, n_layer)
        _, corr_log = capture_role_states(model, layers, ids_x, pos, device, n_layer)
        clean_d = float(clean_log[ct] - clean_log[xt])
        corr_d = float(corr_log[ct] - corr_log[xt])
        denom = clean_d - corr_d
        if abs(denom) < 1e-3:
            continue
        for L in range(n_layer):
            pl = patched_logits(model, layers, ids_x, L, pos, clean_states[L], device)
            pd = float(pl[ct] - pl[xt])
            recov[L] += (pd - corr_d) / denom
        n_used += 1
    recov /= max(n_used, 1)
    print(f"used {n_used} pairs", flush=True)
    for L in range(0, n_layer, 3):
        print(f"  L{L:02d} recovery {recov[L]:.2f}", flush=True)
    deep = float(recov[2 * n_layer // 3:].mean())
    print(f"mid(L6-18) recovery {float(recov[6:19].mean()):.2f} | deep(L24+) recovery {deep:.2f}", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer, "n_pairs": n_used,
           "patch": args.patch, "recovery": recov.tolist()}
    with open(os.path.join(args.out, "step_d_qwen.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(n_layer), recov.tolist(), "-o", ms=3, color=PALETTE[3])
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvspan(2 * n_layer // 3, n_layer - 1, color="0.93", zorder=0)
    ax.set_xlabel(f"patched layer ({args.patch}-token state, clean into corrupted)")
    ax.set_ylabel("logit-diff recovery toward clean entity (1 = fully flipped)")
    ax.set_title(f"{args.hf_model}: causal effect of the {args.patch}-token state, by layer\n"
                 "high = layers after this point still read the entity position")
    ax.grid(alpha=0.3); fig.tight_layout()
    p = os.path.join(args.out, "step_d_causal.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
