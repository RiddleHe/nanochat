"""Finding-1 phase-2 viability probe: can EARLY layers make use of a distant
token's DEEP (already-processed) state, injected at the readout position?

Motivation: distant reads naturally happen only in the last third of layers
(hand-off measurements). The chunk deep-KV design wants early layers to read
deep states of earlier tokens instead. Before training anything, test on the
FROZEN model whether early-injected processed content is usable at all.

Setup (entity task, N filler tokens, 16 within-category pairs):
  corrupt run = Newton sentence; donor = clean Einstein run.
  - ref_interchange : patch entity pos at L_early with donor entity-pos state
                      (known-good sanity, expect ~1.0 recovery)
  - inject_add      : readout pos at L_early += alpha * A(h_deep(entity, donor))
                      where A is a ridge linear map L_deep -> L_early trained on
                      wikitext hidden states (norm-matched before scaling)
  - inject_replace  : readout pos at L_early  = A(h_deep)
  - inject_add + deep window: same as inject_add, plus attention window W=256
                      from L_mask up (sink 4) so the natural late read of the
                      corrupt entity is suppressed — tests whether early
                      injection can SUBSTITUTE for the late read.
  - window only     : deficit reference for the substitution test.
Metric: logit-diff recovery toward the donor entity (0 = still corrupt, 1 = donor).

Verdict rule (STATE.md): early access viable if inject recovers a clearly
nonzero share (>=0.5 of the window-only deficit in the substitution test, or
>=0.3 recovery standalone). ~0 is confounded (frozen early layers never
learned to use such content) and defers to LCKV/RT trainability precedent.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect.early_injection --distance 800
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from scripts.inspect.handoff_distance import within_pairs, build_ids, capture, FILLER
from scripts.inspect.deep_window_mask import build_window_mask, install_hooks


@torch.inference_mode()
def run_with(model, layers, ids, device, hooks=()):
    handles = list(hooks)
    try:
        return model(torch.tensor([ids], device=device)).logits[0, -1, :].float()
    finally:
        for h in handles:
            h.remove()


def replace_hook(layers, L, pos, vec):
    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[0, pos, :] = vec.to(h.dtype)
        return (h,) + out[1:] if is_t else h
    return layers[L].register_forward_hook(hook)


def add_hook(layers, L, pos, vec, alpha):
    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        v = vec.to(h.dtype)
        v = v / v.norm() * h[0, pos, :].norm()  # norm-match, then scale
        h[0, pos, :] = h[0, pos, :] + alpha * v
        return (h,) + out[1:] if is_t else h
    return layers[L].register_forward_hook(hook)


@torch.inference_mode()
def fit_translator(model, tok, layers, n_layer, l_deep, l_early, device,
                   n_seq=20, seq_len=512, lam_rel=1e-2):
    """Ridge map A: h_deep -> h_early, fit on wikitext positions."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    Xs, Ys = [], []
    feats = {}

    def mk(name):
        def hook(_m, _inp, out):
            feats[name] = (out[0] if isinstance(out, tuple) else out)[0].float()
        return hook

    for k in range(n_seq):
        seq = ids[k * seq_len:(k + 1) * seq_len].to(device)[None]
        h1 = layers[l_deep].register_forward_hook(mk("deep"))
        h2 = layers[l_early].register_forward_hook(mk("early"))
        try:
            model(seq)
        finally:
            h1.remove(); h2.remove()
        Xs.append(feats["deep"].cpu()); Ys.append(feats["early"].cpu())
    X = torch.cat(Xs, 0).to(device)   # (N, d)
    Y = torch.cat(Ys, 0).to(device)
    XtX = X.T @ X
    lam = lam_rel * torch.diag(XtX).mean()
    W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=device), X.T @ Y)
    resid = (X @ W - Y).norm() / Y.norm()
    print(f"translator L{l_deep}->L{l_early}: {X.shape[0]} samples, "
          f"rel resid {resid:.3f}", flush=True)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--distance", type=int, default=800)
    ap.add_argument("--l-early", type=int, default=8)
    ap.add_argument("--l-deep", type=int, default=30)
    ap.add_argument("--l-mask", type=int, default=24)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--out", default="results/early_injection")
    args = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    layers = model.model.layers
    n_layer = model.config.num_hidden_layers

    W = fit_translator(model, tok, layers, n_layer, args.l_deep, args.l_early, device)

    pairs = within_pairs(tok)[: args.max_pairs]
    filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"] * (args.distance // 10 + 2)
    N = args.distance
    print(f"{len(pairs)} pairs, distance {N}", flush=True)

    conds = ["ref_interchange", "inject_replace", "window_only", "window_plus_inject"]
    conds += [f"inject_add_a{a}" for a in args.alphas]
    acc = {c: 0.0 for c in conds}
    n_used = 0
    mask = build_window_mask(0, 0, 0, device, model.dtype)  # placeholder rebuilt per T
    for clean, corrupt, role in pairs:
        ids_c, pos_e = build_ids(tok, clean, role, filler_ids, N)
        ids_x, _ = build_ids(tok, corrupt, role, filler_ids, N)
        if len(ids_c) != len(ids_x):
            continue
        T = len(ids_c)
        ct = tok(" " + clean, add_special_tokens=False)["input_ids"][0]
        xt = tok(" " + corrupt, add_special_tokens=False)["input_ids"][0]
        states, log_c = capture(model, layers, ids_c, [pos_e], device, n_layer)
        _, log_x = capture(model, layers, ids_x, [pos_e], device, n_layer)

        def ld(lg):
            return float(lg[ct] - lg[xt])

        denom = ld(log_c) - ld(log_x)
        if abs(denom) < 1e-6:
            continue

        h_deep = states[pos_e][args.l_deep].float()
        h_early_ref = states[pos_e][args.l_early]
        inj = (h_deep @ W).to(model.dtype)          # translated donor content
        wmask = build_window_mask(T, args.window, 4, device, model.dtype)

        def rec(logits):
            return (ld(logits) - ld(log_x)) / denom

        lg = run_with(model, layers, ids_x, device,
                      [replace_hook(layers, args.l_early, pos_e, h_early_ref)])
        acc["ref_interchange"] += rec(lg)
        lg = run_with(model, layers, ids_x, device,
                      [replace_hook(layers, args.l_early, T - 1, inj)])
        acc["inject_replace"] += rec(lg)
        for a in args.alphas:
            lg = run_with(model, layers, ids_x, device,
                          [add_hook(layers, args.l_early, T - 1, inj, a)])
            acc[f"inject_add_a{a}"] += rec(lg)
        lg = run_with(model, layers, ids_x, device,
                      install_hooks(model, args.l_mask, wmask))
        acc["window_only"] += rec(lg)
        lg = run_with(model, layers, ids_x, device,
                      install_hooks(model, args.l_mask, wmask)
                      + [add_hook(layers, args.l_early, T - 1, inj, 1.0)])
        acc["window_plus_inject"] += rec(lg)
        n_used += 1
        print(f"  {clean}->{corrupt}: done", flush=True)

    res = {"model": args.hf_model, "distance": N, "n_pairs": n_used,
           "l_early": args.l_early, "l_deep": args.l_deep,
           "l_mask": args.l_mask, "window": args.window,
           "recovery": {c: acc[c] / n_used for c in conds}}
    for c in conds:
        print(f"{c:24s} recovery {res['recovery'][c]:+.3f}", flush=True)
    with open(os.path.join(args.out, f"early_injection_N{N}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
