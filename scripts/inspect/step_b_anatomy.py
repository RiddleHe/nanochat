"""Anatomy of the translator A: WHAT does the late->mid format change consist of?

A linear map can only do three kinds of things: shift (translation by a mean
offset), scale (stretch/shrink per direction), rotate (turn the vector). This
script decomposes the learned translation and asks WHICH component actually
restores patchscope readability. For each source layer Ls (target fixed L10):

  variants patched into target L10 (held-out templates, candidate readout):
    raw          : h_s unchanged
    shift        : h_s + (mu_t - mu_s)            [mean offset only]
    shift+scale  : diagonal affine fit D*h_s + b  [per-dimension scaling]
    shift+rotate : subspace Procrustes rotation R(h_s - mu_s) + mu_t
                   [orthogonal = pure rotation, no scaling]
    full A       : the unconstrained linear translator
    true h_t     : ceiling

  geometry stats of A (per source layer):
    - singular value spectrum of A (identity-like? anisotropic?)
    - ||A - I||_F / ||I||_F  (how far from doing nothing)
    - stable rank of (A - I) (is the correction low-rank?)
    - mean angle between h and A(h) on held-out data (how much turning)
    - cosine( A(h)-h , h_t-h )  (does A move points toward the true target?)

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m scripts.inspect.step_b_anatomy \
      --hf-model Qwen/Qwen3-8B-Base --out results/step_bc
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
    load_hf, build_source, target_positions, ref_norm_at, patch_readout)
from scripts.inspect.step_root_cause_bc import train_translator, kfold_template_mask

TARGET = 10
SOURCES = [20, 24, 28, 32]


def fit_diag(Hs, Ht, device, steps=300):
    """Per-dimension affine: D*h + b (diagonal scale + shift)."""
    d = Hs.shape[1]
    D = torch.ones(d, device=device, requires_grad=True)
    b = torch.zeros(d, device=device, requires_grad=True)
    opt = torch.optim.Adam([D, b], lr=1e-2)
    Hs, Ht = Hs.to(device), Ht.to(device)
    for _ in range(steps):
        loss = F.mse_loss(Hs * D + b, Ht)
        opt.zero_grad(); loss.backward(); opt.step()
    return D.detach(), b.detach()


def fit_procrustes(Hs, Ht, k=100):
    """Subspace orthogonal Procrustes: rotation within the top-k PCA subspace of
    the centered train data, identity elsewhere. Returns (mu_s, mu_t, P, R)
    with map  x -> mu_t + (I - P^T P)(x - mu_s) + P^T R P (x - mu_s)."""
    mu_s, mu_t = Hs.mean(0), Ht.mean(0)
    Xs, Xt = Hs - mu_s, Ht - mu_t
    joint = torch.cat([Xs, Xt], 0)
    U, S, Vh = torch.linalg.svd(joint, full_matrices=False)
    P = Vh[:k]                                  # (k, d) subspace basis
    As, At = Xs @ P.T, Xt @ P.T                 # coords in subspace
    M = At.T @ As                               # (k, k)
    U2, _, V2h = torch.linalg.svd(M)
    R = U2 @ V2h                                # orthogonal (pure rotation)
    return mu_s, mu_t, P, R


def apply_procrustes(x, mu_s, mu_t, P, R):
    xc = x - mu_s
    sub = xc @ P.T
    rot = sub @ R.T
    return mu_t + xc - sub @ P + rot @ P


def stable_rank(M):
    s = torch.linalg.svdvals(M)
    return float((s ** 2).sum() / (s[0] ** 2).clamp(min=1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/step_bc")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers; target L{TARGET}", flush=True)
    X, y = build_source(model, tok, layers, n_layer, device)
    tmpl = torch.tensor([ti for _ in ENTITIES for ti in range(12)])
    target_ids, patch_pos, read_pos = target_positions(tok)
    cand = torch.tensor([tok(" " + n, add_special_tokens=False)["input_ids"][0]
                         for n, _, _ in ENTITIES])
    ref = ref_norm_at(model, layers, target_ids, TARGET, patch_pos, device)

    def acc(H, yte):
        Hn = H / H.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
        pred = patch_readout(model, layers, target_ids, TARGET, patch_pos,
                             read_pos, Hn, cand, device)
        return float((pred == yte).float().mean())

    variants = ["raw", "shift", "shift+scale", "shift+rotate", "full A", "true h_t"]
    results = {v: [] for v in variants}
    geom = []
    for Ls in SOURCES:
        accs = {v: [] for v in variants}
        g = {"sv": None, "dist_I": [], "stable_rank_AmI": [], "angle": [], "cos_toward": []}
        for trm, tem in kfold_template_mask(tmpl):
            Hs_tr, Ht_tr = X[trm, Ls, :], X[trm, TARGET, :]
            Hs_te, Ht_te, yte = X[tem, Ls, :], X[tem, TARGET, :], y[tem]

            accs["raw"].append(acc(Hs_te, yte))
            mu_s, mu_t = Hs_tr.mean(0), Ht_tr.mean(0)
            accs["shift"].append(acc(Hs_te + (mu_t - mu_s), yte))
            D, b = fit_diag(Hs_tr, Ht_tr, device)
            accs["shift+scale"].append(acc((Hs_te.to(device) * D + b).cpu(), yte))
            pr = fit_procrustes(Hs_tr, Ht_tr)
            accs["shift+rotate"].append(acc(apply_procrustes(Hs_te, *pr), yte))
            A = train_translator(Hs_tr, Ht_tr, device, steps=500)
            with torch.no_grad():
                HA = A(Hs_te.to(device)).cpu()
            accs["full A"].append(acc(HA, yte))
            accs["true h_t"].append(acc(Ht_te, yte))

            W = A.weight.detach().cpu()
            I = torch.eye(W.shape[0])
            g["dist_I"].append(float((W - I).norm() / I.norm()))
            g["stable_rank_AmI"].append(stable_rank(W - I))
            hn = F.normalize(Hs_te, dim=-1); an = F.normalize(HA, dim=-1)
            g["angle"].append(float(torch.rad2deg(torch.arccos(
                (hn * an).sum(-1).clamp(-1, 1))).mean()))
            g["cos_toward"].append(float(F.cosine_similarity(
                HA - Hs_te, Ht_te - Hs_te, dim=-1).mean()))
            if g["sv"] is None:
                g["sv"] = torch.linalg.svdvals(W)[:200].tolist()
        for v in variants:
            results[v].append(sum(accs[v]) / len(accs[v]))
        geom.append({k: (sum(vv) / len(vv) if isinstance(vv, list) and vv and
                         isinstance(vv[0], float) else vv) for k, vv in g.items()})
        print(f"  src L{Ls}: " + "  ".join(f"{v} {results[v][-1]:.2f}" for v in variants),
              flush=True)
        print(f"    geometry: |A-I|/|I| {geom[-1]['dist_I']:.2f}  "
              f"stable-rank(A-I) {geom[-1]['stable_rank_AmI']:.1f}  "
              f"angle(h,Ah) {geom[-1]['angle']:.1f}deg  "
              f"cos(A(h)-h, h_t-h) {geom[-1]['cos_toward']:.2f}", flush=True)

    with open(os.path.join(args.out, "step_b_anatomy.json"), "w") as f:
        json.dump({"model": args.hf_model, "target": TARGET, "sources": SOURCES,
                   "accuracy": results, "geometry": geom}, f, indent=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    xs = torch.arange(len(SOURCES))
    w = 0.14
    for i, v in enumerate(variants):
        ax.bar(xs + (i - 2.5) * w, results[v], w, label=v,
               color=PALETTE[i % len(PALETTE)] if v != "true h_t" else "0.6")
    ax.axhline(1 / len(ENTITIES), color="0.7", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([f"L{s}" for s in SOURCES])
    ax.set_xlabel(f"source layer (target L{TARGET})"); ax.set_ylabel("patchscope accuracy")
    ax.set_title("Which component of the translation does the work?")
    ax.legend(fontsize=7.5)
    for gi, Ls in enumerate(SOURCES):
        ax2.plot(geom[gi]["sv"], label=f"src L{Ls}",
                 color=PALETTE[gi % len(PALETTE)])
    ax2.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax2.set_xlabel("singular value index (top 200)"); ax2.set_ylabel("singular value of A")
    ax2.set_title("Spectrum of A (1.0 = identity-like)")
    ax2.legend(fontsize=8)
    fig.suptitle(f"{args.hf_model}: anatomy of the late->mid linear translation")
    fig.tight_layout()
    p = os.path.join(args.out, "step_b_anatomy.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
