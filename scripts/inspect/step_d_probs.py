"""Step D, probability view: per patched layer, the softmax PROBABILITY of the
correct (clean) entity vs the wrong (corrupt) entity at the final position.

Same interchange patching as step_d_causal (subject token, clean -> corrupted,
cumulative state, per layer), but instead of the normalized logit-diff recovery
we record full-vocab softmax probabilities, averaged over pairs:

  line 1: P(clean entity)   e.g. P(" Einstein") in the patched corrupted run
  line 2: P(corrupt entity) e.g. P(" Newton")

Reference bands: the same two probabilities in the unpatched clean run and the
unpatched corrupted run.

CPU-friendly (short sentences; loads fp32 on cpu).

Usage:
  python -m scripts.inspect.step_d_probs --hf-model Qwen/Qwen3-8B-Base --device cpu
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.step_d_causal import within_pairs, find_pos, TEMPLATE

PALETTE = ["#2c8a8a", "#c0392b", "#888888"]


def load(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok, model.model.layers, model.config.num_hidden_layers


@torch.inference_mode()
def capture(model, layers, ids, pos, device, n_layer):
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/step_d")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load(args.hf_model, device)
    pairs = within_pairs(tok)
    print(f"{args.hf_model}: {n_layer} layers, {len(pairs)} ordered pairs", flush=True)

    p_clean = torch.zeros(n_layer)   # patched run: P(clean entity)
    p_corr = torch.zeros(n_layer)    # patched run: P(corrupt entity)
    ref = {"clean_run_Pclean": 0.0, "clean_run_Pcorr": 0.0,
           "corr_run_Pclean": 0.0, "corr_run_Pcorr": 0.0}
    n_used = 0
    for clean, corrupt, role in pairs:
        ids_c = tok(TEMPLATE.format(name=clean, role=role), add_special_tokens=True)["input_ids"]
        ids_x = tok(TEMPLATE.format(name=corrupt, role=role), add_special_tokens=True)["input_ids"]
        if len(ids_c) != len(ids_x):
            continue
        pos = find_pos(tok, ids_c, clean)
        ct = tok(" " + clean, add_special_tokens=False)["input_ids"][0]
        xt = tok(" " + corrupt, add_special_tokens=False)["input_ids"][0]
        states, log_c = capture(model, layers, ids_c, pos, device, n_layer)
        _, log_x = capture(model, layers, ids_x, pos, device, n_layer)
        pc, px = F.softmax(log_c, -1), F.softmax(log_x, -1)
        ref["clean_run_Pclean"] += float(pc[ct]); ref["clean_run_Pcorr"] += float(pc[xt])
        ref["corr_run_Pclean"] += float(px[ct]); ref["corr_run_Pcorr"] += float(px[xt])
        for L in range(n_layer):
            pl = F.softmax(patched(model, layers, ids_x, L, pos, states[L], device), -1)
            p_clean[L] += float(pl[ct]); p_corr[L] += float(pl[xt])
        n_used += 1
        print(f"  {clean}->{corrupt}: done", flush=True)
    p_clean /= n_used; p_corr /= n_used
    for k in ref:
        ref[k] /= n_used
    print(f"used {n_used} pairs; refs: {ref}", flush=True)

    res = {"model": args.hf_model, "n_layer": n_layer, "n_pairs": n_used,
           "P_clean_entity": p_clean.tolist(), "P_corrupt_entity": p_corr.tolist(),
           "references": ref}
    with open(os.path.join(args.out, "step_d_probs.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(range(n_layer), p_clean.tolist(), "-o", ms=3, color=PALETTE[0],
            label="P(correct entity, e.g. Einstein) — patched run")
    ax.plot(range(n_layer), p_corr.tolist(), "-s", ms=3, color=PALETTE[1],
            label="P(wrong entity, e.g. Newton) — patched run")
    ax.axhline(ref["clean_run_Pclean"], color=PALETTE[0], ls=":", lw=1,
               label="P(correct) in clean run (upper ref)")
    ax.axhline(ref["corr_run_Pcorr"], color=PALETTE[1], ls=":", lw=1,
               label="P(wrong) in corrupted run (upper ref)")
    ax.set_xlabel("patched layer (subject token, clean into corrupted)")
    ax.set_ylabel("softmax probability at final position")
    ax.set_title(f"{args.hf_model}: entity probabilities under interchange patching\n"
                 f"({n_used} pairs averaged)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.out, "step_d_probs.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
