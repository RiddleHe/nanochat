"""Step 2b: ORIGINAL (free-generation) Patchscope + full source x target matrix.

Addresses the review points and the request for the ORIGINAL patchscope (let the
model GENERATE the answer, not just read one candidate logit):

  - point 1: print every entity's tokenization; the generation readout grades by
    name substring (tokenization-agnostic), and we cross-check the best target
    layer with a full-NAME log-probability readout (rigorous for the 1 multi-
    token entity, Beethoven).
  - point 2: print the tokens at patch_pos / read_pos.
  - point 3: full source-layer x target-layer MATRIX (candidate first-token
    readout) as a heatmap -> is a late-source hidden state unreadable at every
    target layer, or readable if patched into an earlier target layer (i.e. with
    enough downstream computation)?

Two readouts of the SAME patched state:
  (a) original patchscope = greedy free generation, grade if the entity name
      appears in the output;
  (b) candidate readout used only for the fast matrix.

Qwen3-8B only. Keeps the Step-2 outputs; writes new files to results/step2.

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.step2b_patchscope_matrix \
      --hf-model Qwen/Qwen3-8B-Base --out results/step2
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import PALETTE
from scripts.inspect.probe_step1_robust import ENTITIES
from scripts.inspect.step2_probe_vs_patchscope import (
    load_hf, build_source, target_positions, ref_norm_at,
    patch_readout, patch_generate, FEWSHOT)

GEN_TARGET_LAYERS_FRAC = [0.12, 0.28, 0.5]   # early / mid / late target layers for generation


def print_checks(tok):
    print("=== point 1: entity tokenizations (with leading space) ===")
    multi = []
    for name, _, _ in ENTITIES:
        ids = tok(" " + name, add_special_tokens=False)["input_ids"]
        tag = "SINGLE" if len(ids) == 1 else f"MULTI({len(ids)})"
        if len(ids) > 1:
            multi.append(name)
        print(f"    {name:12s} {tag:9s} {[tok.decode([i]) for i in ids]}", flush=True)
    print(f"  multi-token entities: {multi or 'none'}")
    ids = tok(FEWSHOT, add_special_tokens=True)["input_ids"]
    toks = [tok.decode([i]) for i in ids]
    print("=== point 2: target prompt tokens ===")
    print(f"    {list(enumerate(toks))}")
    return multi


def name_token_ids(tok):
    return [tok(" " + name, add_special_tokens=False)["input_ids"] for name, _, _ in ENTITIES]


@torch.inference_mode()
def fullname_logprob_readout(model, layers, target_ids, T, patch_pos, read_pos,
                             src, name_ids, device):
    """Rigorous readout: score each candidate by the summed log-prob of its FULL
    name (teacher-forced). One base forward covers every single-token candidate
    plus each multi-token candidate's first token; each multi-token candidate
    needs one extra forward for its remaining tokens. Returns pred idx (B,)."""
    B = src.shape[0]
    srcd = src.to(device)

    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[:, patch_pos, :] = srcd.to(h.dtype)
        return (h,) + out[1:] if is_t else h

    def run(seq):
        handle = layers[T].register_forward_hook(hook)
        try:
            return model(seq).logits.float()
        finally:
            handle.remove()

    base = torch.tensor([target_ids], device=device).expand(B, -1)
    base_lp = F.log_softmax(run(base)[:, read_pos, :], dim=-1)   # (B,V)
    scores = torch.zeros(B, len(name_ids), device=device)
    for c, toks in enumerate(name_ids):
        scores[:, c] = base_lp[:, toks[0]]
        if len(toks) > 1:
            seq = torch.cat([base, torch.tensor(toks[:-1], device=device)
                             .expand(B, -1)], dim=1)
            lg = F.log_softmax(run(seq), dim=-1)
            for j in range(1, len(toks)):
                scores[:, c] += lg[:, read_pos + j - 1, :][:, toks[j]]
    return scores.argmax(1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/step2")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers", flush=True)
    print_checks(tok)
    X, y = build_source(model, tok, layers, n_layer, device)
    target_ids, patch_pos, read_pos = target_positions(tok)
    name_ids = name_token_ids(tok)
    names = [n.lower() for n, _, _ in ENTITIES]
    chance = 1.0 / len(ENTITIES)

    ref = {T: ref_norm_at(model, layers, target_ids, T, patch_pos, device) for T in range(n_layer)}

    def nm(src, T):
        return src / src.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref[T]

    # ---- (A) ORIGINAL patchscope: free generation, grade by name substring ----
    gen_layers = sorted(set(int(f * n_layer) for f in GEN_TARGET_LAYERS_FRAC))
    gen = {T: [] for T in gen_layers}
    print(f"\n=== ORIGINAL patchscope (free generation), target layers {gen_layers} ===", flush=True)
    for T in gen_layers:
        for L in range(n_layer):
            outs = patch_generate(model, layers, target_ids, T, patch_pos, nm(X[:, L, :], T), device, n_new=6)
            texts = [tok.decode(o).lower() for o in outs]
            gen[T].append(sum(names[int(yi)] in t for yi, t in zip(y, texts)) / len(y))
        print(f"  target L{T}: gen-acc range {min(gen[T]):.2f}-{max(gen[T]):.2f}", flush=True)

    # ---- (B) full source x target matrix (candidate first-token readout) ----
    cand = torch.tensor([toks[0] for toks in name_ids])
    print("\n=== source x target matrix (candidate readout) ===", flush=True)
    mat = torch.zeros(n_layer, n_layer)
    for T in range(n_layer):
        for L in range(n_layer):
            pred = patch_readout(model, layers, target_ids, T, patch_pos, read_pos, nm(X[:, L, :], T), cand, device)
            mat[L, T] = float((pred == y).float().mean())
        if T % 6 == 0:
            print(f"  target L{T} done (col max {mat[:, T].max():.2f})", flush=True)

    # ---- (C) rigorous full-name-logprob check at the best target layer ----
    bestT = int(mat.max(dim=0).values.argmax())
    fn = []
    for L in range(n_layer):
        pred = fullname_logprob_readout(model, layers, target_ids, bestT, patch_pos, read_pos, nm(X[:, L, :], bestT), name_ids, device)
        fn.append(float((pred == y).float().mean()))
    print(f"\nbest target L{bestT}: first-token vs full-name (should match closely):", flush=True)
    print("  first-token:", [round(float(mat[L, bestT]), 2) for L in range(0, n_layer, 6)])
    print("  full-name  :", [round(fn[L], 2) for L in range(0, n_layer, 6)])

    res = {"model": args.hf_model, "n_layer": n_layer, "chance": chance,
           "gen_target_layers": gen_layers, "generation": {str(T): gen[T] for T in gen_layers},
           "matrix": mat.tolist(), "best_target": bestT,
           "fullname_logprob_bestT": fn}
    with open(os.path.join(args.out, "step2b_qwen.json"), "w") as f:
        json.dump(res, f, indent=1)

    # heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat.numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=chance, vmax=1.0)
    ax.set_xlabel("target layer (patch INTO)"); ax.set_ylabel("source layer (hidden state FROM)")
    ax.set_title(f"{args.hf_model}: patchscope entity accuracy\nsource x target (readout floor=chance {chance:.2f})")
    fig.colorbar(im, ax=ax, label="patchscope accuracy")
    fig.tight_layout()
    p1 = os.path.join(args.out, "step2b_matrix.png"); fig.savefig(p1, dpi=150, bbox_inches="tight")

    # generation line plot
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for i, T in enumerate(gen_layers):
        ax2.plot(range(n_layer), gen[T], "-o", ms=3, color=PALETTE[i % len(PALETTE)],
                 label=f"free-generation patchscope, target L{T}")
    ax2.axhline(chance, color="0.7", lw=0.8, ls="--", label=f"chance ({chance:.2f})")
    ax2.axvspan(2 * n_layer // 3, n_layer - 1, color="0.93", zorder=0)
    ax2.set_xlabel("source layer"); ax2.set_ylabel("generation accuracy (entity name appears)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"{args.hf_model}: ORIGINAL patchscope (free generation) across depth")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(args.out, "step2b_generation.png"); fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"saved {p1} and {p2}", flush=True)


if __name__ == "__main__":
    main()
