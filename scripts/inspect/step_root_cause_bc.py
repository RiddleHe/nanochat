"""Root-cause Steps B and C on Qwen3-8B.

Step C (direction alignment): a linear probe reads the entity at every layer,
but Patchscope needs the model to OUTPUT the entity name. Are these the same
direction? For each layer L we take the probe's entity-separating direction
(w_i - w_j) and the model's output/unembedding direction (U_i - U_j) and measure
their cosine. If mid layers align but late layers stay classifiable yet
mis-aligned, then late-layer entity info is present but NOT on the
'say-the-name' output direction -> explains probe-yes / patchscope-no.

Step B (linear translator): if a late-layer state is unreadable by Patchscope,
is it just a linear map away from a readable one? Train a linear A mapping a
late source layer's hidden state toward a good reference layer, then
Patchscope-read A(h). If accuracy recovers (e.g. 0.2 -> 0.6), the info is not
lost, only in a readout-incompatible coordinate system (translatable).

Qwen3-8B only. Writes to results/step_bc.
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
from scripts.inspect.probe_step1_robust import ENTITIES
from scripts.inspect.step2_probe_vs_patchscope import (
    load_hf, build_source, target_positions, ref_norm_at, patch_readout)


def norm_rows(X):
    return X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def within_pairs():
    cats = {}
    for i, (_, role, _) in enumerate(ENTITIES):
        cats.setdefault(role, []).append(i)
    pairs = []
    for ids in cats.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs.append((ids[a], ids[b]))
    return pairs


def probe_weights(XL, y, n_cls, device, steps=400):
    d = XL.shape[1]
    probe = nn.Linear(d, n_cls).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-2)
    XL, y = XL.to(device), y.to(device)
    for _ in range(steps):
        loss = F.cross_entropy(probe(XL), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return probe.weight.detach().float().cpu()   # (n_cls, d)


def step_c(X, y, U, device, n_layer):
    """cosine( probe(w_i-w_j), unembed(U_i-U_j) ) averaged over within-cat pairs."""
    pairs = within_pairs()
    Un = U / U.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    out = []
    for L in range(n_layer):
        W = probe_weights(norm_rows(X[:, L, :]), y, len(ENTITIES), device)
        cs = []
        for i, j in pairs:
            pd = W[i] - W[j]; vd = Un[i] - Un[j]
            cs.append(float(F.cosine_similarity(pd[None], vd[None]).abs()))
        out.append(sum(cs) / len(cs))
    return out


def kfold_template_mask(tmpl, n_t=12, fold=4):
    for start in range(0, n_t, fold):
        te = (tmpl >= start) & (tmpl < start + fold)
        yield ~te, te


def train_translator(Hs, Ht, device, steps=500):
    """Linear A: Hs -> Ht (MSE), weight-decayed."""
    d = Hs.shape[1]
    A = nn.Linear(d, d).to(device)
    opt = torch.optim.Adam(A.parameters(), lr=1e-3, weight_decay=1e-3)
    Hs, Ht = Hs.to(device), Ht.to(device)
    for _ in range(steps):
        loss = F.mse_loss(A(Hs), Ht)
        opt.zero_grad(); loss.backward(); opt.step()
    return A


def ps_acc(model, layers, target_ids, T, patch_pos, read_pos, H, y, cand, ref, device):
    Hn = H / H.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
    pred = patch_readout(model, layers, target_ids, T, patch_pos, read_pos, Hn, cand, device)
    return float((pred == y).float().mean())


def step_b(model, layers, X, y, tmpl, device, n_layer, Lt, src_layers):
    tok_ids, patch_pos, read_pos = target_positions_cached
    cand = cand_cached
    ref = ref_norm_at(model, layers, tok_ids, Lt, patch_pos, device)
    rows = []
    for Ls in src_layers:
        raw, trans, upper = [], [], []
        for trm, tem in kfold_template_mask(tmpl):
            A = train_translator(X[trm, Ls, :], X[trm, Lt, :], device)
            with torch.no_grad():
                Atest = A(X[tem, Ls, :].to(device)).cpu()
            raw.append(ps_acc(model, layers, tok_ids, Lt, patch_pos, read_pos, X[tem, Ls, :], y[tem], cand, ref, device))
            trans.append(ps_acc(model, layers, tok_ids, Lt, patch_pos, read_pos, Atest, y[tem], cand, ref, device))
            upper.append(ps_acc(model, layers, tok_ids, Lt, patch_pos, read_pos, X[tem, Lt, :], y[tem], cand, ref, device))
        rows.append({"src": Ls, "raw": sum(raw) / len(raw),
                     "translated": sum(trans) / len(trans), "upper_bound_trueLt": sum(upper) / len(upper)})
        print(f"  Step B  src L{Ls} -> target L{Lt}: raw {rows[-1]['raw']:.2f} | "
              f"translated {rows[-1]['translated']:.2f} | true-Lt upper {rows[-1]['upper_bound_trueLt']:.2f}",
              flush=True)
    return rows


def main():
    global target_positions_cached, cand_cached
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/step_bc")
    ap.add_argument("--ref-layer", type=int, default=10)
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, tok, layers, n_layer = load_hf(args.hf_model, device)
    print(f"{args.hf_model}: {n_layer} layers", flush=True)
    X, y = build_source(model, tok, layers, n_layer, device)
    tmpl = torch.tensor([ti for _ in ENTITIES for ti in range(12)])
    target_positions_cached = target_positions(tok)
    name_first = [tok(" " + n, add_special_tokens=False)["input_ids"][0] for n, _, _ in ENTITIES]
    cand_cached = torch.tensor(name_first)
    U = model.get_output_embeddings().weight.detach()[cand_cached].float().cpu()  # (15, d)

    print("=== Step C: probe direction vs vocab/output direction (cosine) ===", flush=True)
    align = step_c(X, y, U, device, n_layer)
    for L in range(0, n_layer, 3):
        print(f"  L{L:02d} |cos| {align[L]:.3f}", flush=True)

    print(f"\n=== Step B: linear translator -> target L{args.ref_layer} ===", flush=True)
    src_layers = [args.ref_layer, 20, 24, 28, 32]
    rows = step_b(model, layers, X, y, tmpl, device, n_layer, args.ref_layer, src_layers)

    res = {"model": args.hf_model, "n_layer": n_layer, "ref_layer": args.ref_layer,
           "direction_cosine": align, "translator": rows}
    with open(os.path.join(args.out, "step_bc_qwen.json"), "w") as f:
        json.dump(res, f, indent=1)

    # Step C plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(n_layer), align, "-o", ms=3, color=PALETTE[2])
    ax.axvspan(2 * n_layer // 3, n_layer - 1, color="0.93", zorder=0)
    ax.set_xlabel("layer"); ax.set_ylabel("|cosine| probe-dir vs output-dir")
    ax.set_title(f"{args.hf_model}: does the entity direction align with the 'say-the-name' direction?\n"
                 "high mid + drop late => info present but off the output direction")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "step_c_direction.png"), dpi=150, bbox_inches="tight")

    # Step B plot
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    xs = [r["src"] for r in rows]
    ax2.plot(xs, [r["raw"] for r in rows], "-o", color=PALETTE[1], label="raw late state, patchscope")
    ax2.plot(xs, [r["translated"] for r in rows], "-s", color=PALETTE[2], label="translated (linear A), patchscope")
    ax2.plot(xs, [r["upper_bound_trueLt"] for r in rows], "--", color="0.5",
             label=f"upper bound (true L{args.ref_layer} state)")
    ax2.set_xlabel(f"source layer (translated toward L{args.ref_layer})")
    ax2.set_ylabel("patchscope accuracy"); ax2.set_ylim(0, 1.02)
    ax2.set_title(f"{args.hf_model}: can a linear map make a late state readable by patchscope?")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "step_b_translator.png"), dpi=150, bbox_inches="tight")
    print("saved step_c_direction.png and step_b_translator.png", flush=True)


if __name__ == "__main__":
    main()
