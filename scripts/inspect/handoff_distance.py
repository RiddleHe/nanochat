"""E2: does the hand-off layer depend on the DISTANCE between the entity and
the readout token?

All hand-off evidence so far comes from an 11-token sentence where the entity
sits 8 tokens before the readout position, so any window >= 8 covers it. The
sliding-window hypothesis needs more: information from tokens far outside a
local window must ALSO be read into the readout position by the same ~2/3
depth. This script re-measures the two key patching curves with N filler
tokens inserted between the entity clause and the readout clause:

  "Everyone knows {name} was a celebrated {role}." + <N filler tokens>
  + " The {role} was"

Per layer L, clean(Einstein) -> corrupted(Newton) interchange patching at
  - the entity position   (green curve: readable while info still lives there)
  - the final position    (blue curve: readable once info has arrived there)
metric = logit-diff recovery (patched - corrupt) / (clean - corrupt).

Outcomes: hand-off layer independent of N -> deep-layer windowing is safe;
hand-off moves deeper with N -> deep layers DO long-range reads for distant
entities and the hypothesis fails at exactly the distances that matter.

Also records the clean-corrupt logit-diff (denominator): if the model can no
longer bind the entity across N tokens, recovery is noise and we say so.

Usage:
  python -m scripts.inspect.handoff_distance --hf-model Qwen/Qwen3-8B-Base
  # smoke: --max-pairs 1 --distances 0 100
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.step_d_causal import within_pairs

FILLER = ("The afternoon light settled slowly over the quiet valley, and a mild "
          "wind moved through the tall grass beside the river while distant "
          "clouds drifted across the pale sky. ")


def load(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok, model.model.layers, model.config.num_hidden_layers


def build_ids(tok, name, role, filler_ids, n_filler):
    pre = tok("Everyone knows", add_special_tokens=True)["input_ids"]
    span = tok(" " + name, add_special_tokens=False)["input_ids"]
    mid = tok(f" was a celebrated {role}.", add_special_tokens=False)["input_ids"]
    tail = tok(f" The {role} was", add_special_tokens=False)["input_ids"]
    ids = pre + span + mid + list(filler_ids[:n_filler]) + tail
    return ids, len(pre)  # entity position


@torch.inference_mode()
def capture(model, layers, ids, positions, device, n_layer):
    """Hidden state after every layer at each position, plus final logits."""
    feats = {p: [None] * n_layer for p in positions}
    hs = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            for p in positions:
                feats[p][i] = h[0, p, :].detach().clone()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--distances", type=int, nargs="*", default=[0, 100, 400, 800, 1400])
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--out", default="results/handoff_distance")
    args = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load(args.hf_model, device)
    pairs = within_pairs(tok)[: args.max_pairs]
    filler_ids = (tok(FILLER, add_special_tokens=False)["input_ids"]
                  * (max(args.distances) // 10 + 2))
    print(f"{args.hf_model}: {n_layer} layers, {len(pairs)} pairs, "
          f"distances {args.distances}", flush=True)

    results = {}
    for N in args.distances:
        rec_ent = torch.zeros(n_layer)
        rec_last = torch.zeros(n_layer)
        denom_sum, n_used = 0.0, 0
        for clean, corrupt, role in pairs:
            ids_c, pos_e = build_ids(tok, clean, role, filler_ids, N)
            ids_x, _ = build_ids(tok, corrupt, role, filler_ids, N)
            if len(ids_c) != len(ids_x):
                continue
            pos_last = len(ids_c) - 1
            ct = tok(" " + clean, add_special_tokens=False)["input_ids"][0]
            xt = tok(" " + corrupt, add_special_tokens=False)["input_ids"][0]
            states, log_c = capture(model, layers, ids_c, [pos_e, pos_last],
                                    device, n_layer)
            _, log_x = capture(model, layers, ids_x, [pos_e], device, n_layer)

            def ld(logits):
                return float(logits[ct] - logits[xt])

            denom = ld(log_c) - ld(log_x)
            if abs(denom) < 1e-6:
                continue
            denom_sum += denom
            for L in range(n_layer):
                for pos, vecs, acc in ((pos_e, states[pos_e], rec_ent),
                                       (pos_last, states[pos_last], rec_last)):
                    lp = patched(model, layers, ids_x, L, pos, vecs[L], device)
                    acc[L] += (ld(lp) - ld(log_x)) / denom
            n_used += 1
            print(f"  N={N} {clean}->{corrupt}: done", flush=True)
        rec_ent /= n_used
        rec_last /= n_used
        results[N] = {"n_pairs": n_used, "seq_len": len(ids_c),
                      "mean_denominator_logit_diff": denom_sum / n_used,
                      "recovery_entity_pos": rec_ent.tolist(),
                      "recovery_last_pos": rec_last.tolist()}
        print(f"N={N}: {n_used} pairs, seq {len(ids_c)}, "
              f"denom {denom_sum / n_used:.2f}", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer,
           "distances": args.distances, "per_distance": results}
    with open(os.path.join(args.out, "handoff_distance.json"), "w") as f:
        json.dump(res, f, indent=1)

    nd = len(args.distances)
    fig, axes = plt.subplots(1, nd, figsize=(4.2 * nd, 4.2), sharey=True)
    axes = [axes] if nd == 1 else list(axes)
    for ax, N in zip(axes, args.distances):
        r = results[N]
        ax.plot(range(n_layer), r["recovery_entity_pos"], "-o", ms=3,
                color="#2c8a2c", label="patch entity position")
        ax.plot(range(n_layer), r["recovery_last_pos"], "-s", ms=3,
                color="#1f77b4", label="patch final position")
        ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
        ax.set_title(f"N={N} filler (seq {r['seq_len']},\n"
                     f"denom {r['mean_denominator_logit_diff']:.1f})", fontsize=10)
        ax.set_xlabel("patched layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("logit-diff recovery")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.hf_model}: hand-off vs entity-readout distance "
                 f"({len(pairs)} pairs)")
    fig.tight_layout()
    p = os.path.join(args.out, "handoff_distance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
