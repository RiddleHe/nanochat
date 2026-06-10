"""Tuned lens (Belrose et al., 2303.08112) for nanochat checkpoints.

Robustness check for the logit-lens "late crystallization" finding. The logit
lens reads every layer's residual with the FINAL layer's unembedding
(norm + lm_head); a model whose intermediate residuals live in a different
basis/scale (BoV's deep residuals run ~2x hotter than the attention
baseline's) could look "not yet decided" when the information is present but
unreadable by that fixed ruler. The tuned lens removes this objection: each
layer L gets its own affine translator W_L (identity-init Linear(d, d)),
trained so that lm_head(norm(W_L h_L)) predicts the next token as well as
possible. Per-layer readability is then measured with each layer's OWN best
linear ruler.

Pre-registered decision rule (set before running):
  - If, with tuned readout, BoV still needs ~5+ more layers than Attention to
    reach the same next-token accuracy, the finding stands: BoV's deep layers
    are genuinely still computing the answer.
  - If BoV's L14-16 residuals become as readable as Attention's, the
    logit-lens finding was a readout artifact.

The model is frozen throughout; only the 24 translators train (identical
data stream, steps, and lr for both models).

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.inspect.tuned_lens \
        --model-tags arch_d24_gpt_base_100B \
                     arch_d24_gpt_base_v_from_value_emb_learn_100B \
        --labels Attention BoV --out-dir results/patchscopes_small
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.model.gpt import norm
from scripts.inspect.patchscope_few_shot import PALETTE
from scripts.inspect.patchscope_raw_completion import (
    SOURCE_SETS, ENTITIES, EXPECTED_NEXT_TOKEN, first_token_id)


def capture_all_residuals(model, x):
    """One forward; returns list of 24 (B, T, D) residual tensors, detached."""
    feats = [None] * len(model.transformer.h)
    handles = []

    def mk(i):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i] = h.detach()
        return hook

    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        with torch.inference_mode():
            _ = model(x)
    finally:
        for h in handles:
            h.remove()
    # inference_mode tensors can't join autograd graphs; shallow-clone out.
    return [f.clone() for f in feats]


def lens_logits(model, translator, h, vocab_size):
    """Translator -> frozen norm -> frozen lm_head -> slice padding."""
    z = translator(h.float())
    z = norm(z)
    logits = model.lm_head(z)[..., :vocab_size]
    return logits


def run_model(tag, label, args, device):
    model, tokenizer, meta = load_model("base", device, phase="eval",
                                        model_tag=tag, step=args.step)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = meta["model_config"]
    n_layer, d, vocab = cfg["n_layer"], cfg["n_embd"], cfg["vocab_size"]

    translators = torch.nn.ModuleList(
        [torch.nn.Linear(d, d, bias=True) for _ in range(n_layer)]).to(device)
    for t in translators:
        with torch.no_grad():
            t.weight.copy_(torch.eye(d))
            t.bias.zero_()
    opt = torch.optim.Adam(translators.parameters(), lr=args.lr)

    # ---- train (identical stream for every model: same split/batch/steps) ----
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.batch_size, cfg["sequence_len"], split="train",
        device=device)
    print(f"\n=== [{label}] training {n_layer} translators "
          f"({args.train_steps} steps x {args.batch_size}x{cfg['sequence_len']} tok) ===",
          flush=True)
    for step in range(args.train_steps):
        x, y = next(train_loader)
        feats = capture_all_residuals(model, x)
        total = 0.0
        for L in range(n_layer):
            logits = lens_logits(model, translators[L], feats[L], vocab)
            loss = F.cross_entropy(logits.reshape(-1, vocab).float(),
                                   y.reshape(-1))
            loss.backward()
            total += float(loss)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.train_steps} mean-CE "
                  f"{total/n_layer:.4f}", flush=True)

    # ---- eval: per-layer CE + top-1 accuracy on held-out val tokens ----
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.batch_size, cfg["sequence_len"], split="val",
        device=device)
    ce = torch.zeros(n_layer)
    acc = torch.zeros(n_layer)
    final_ce, final_acc, n_tok = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(args.val_steps):
            x, y = next(val_loader)
            feats = capture_all_residuals(model, x)
            with torch.inference_mode():
                flogits = model(x)
            fl = flogits.reshape(-1, flogits.size(-1)).float()
            yy = y.reshape(-1)
            final_ce += float(F.cross_entropy(fl, yy, reduction="sum"))
            final_acc += float((fl.argmax(-1) == yy).sum())
            n_tok += yy.numel()
            for L in range(n_layer):
                logits = lens_logits(model, translators[L], feats[L], vocab)
                ll = logits.reshape(-1, vocab).float()
                ce[L] += float(F.cross_entropy(ll, yy, reduction="sum"))
                acc[L] += float((ll.argmax(-1) == yy).sum())
    ce, acc = (ce / n_tok), (acc / n_tok)
    final_ce, final_acc = final_ce / n_tok, final_acc / n_tok
    print(f"  [{label}] final-layer CE {final_ce:.4f} acc {final_acc:.4f}")
    for L in range(n_layer):
        print(f"  [{label}] tuned L{L:02d}  CE {ce[L]:.4f}  acc {acc[L]:.4f}",
              flush=True)

    # ---- entity prompts: per-layer tuned top-1 vs expected next token ----
    ent = {}
    with torch.no_grad():
        for key in ENTITIES:
            src = SOURCE_SETS["alt"][key]
            exp_id, _ = first_token_id_nanochat(tokenizer, EXPECTED_NEXT_TOKEN[key])
            ids = [tokenizer.get_bos_token_id()] + tokenizer.encode(src)
            x = torch.tensor([ids], device=device)
            feats = capture_all_residuals(model, x)
            hits = []
            for L in range(n_layer):
                logits = lens_logits(model, translators[L],
                                     feats[L][:, -1:, :], vocab)
                hits.append(int(int(logits[0, -1].argmax()) == exp_id))
            ent[key] = hits
            print(f"  [{label}] {key:10s}: "
                  + "".join("#" if h else "." for h in hits), flush=True)

    rec = {"model": tag, "label": label, "n_layer": n_layer,
           "train_steps": args.train_steps, "lr": args.lr,
           "ce": ce.tolist(), "acc": acc.tolist(),
           "final_ce": final_ce, "final_acc": final_acc,
           "entity_hits": ent}
    del model, translators
    torch.cuda.empty_cache()
    return rec


def first_token_id_nanochat(tokenizer, text):
    ids = tokenizer.encode(text)
    return ids[0], tokenizer.decode([ids[0]])


def plot(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
    for i, rec in enumerate(recs):
        color = PALETTE[i % len(PALETTE)]
        xs = range(rec["n_layer"])
        axA.plot(xs, rec["ce"], marker="o", ms=3, color=color,
                 label=rec["label"])
        axA.axhline(rec["final_ce"], color=color, lw=0.8, ls=":")
        axB.plot(xs, rec["acc"], marker="o", ms=3, color=color,
                 label=rec["label"])
        axB.axhline(rec["final_acc"], color=color, lw=0.8, ls=":")
    axA.set_xlabel("layer"); axA.set_ylabel("next-token CE (tuned lens)")
    axA.set_title("Lower = more readable prediction"); axA.legend(); axA.grid(alpha=0.3)
    axB.set_xlabel("layer"); axB.set_ylabel("next-token top-1 accuracy")
    axB.set_title("Dotted = final-layer reference"); axB.legend(); axB.grid(alpha=0.3)
    fig.suptitle("Tuned lens: how much of the final prediction is already "
                 "readable at each layer (per-layer affine readout, trained "
                 "identically for both models)")
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
    print(f"[plot] {out_base}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--val-steps", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/patchscopes_small")
    args = ap.parse_args()

    labels = args.labels or args.model_tags
    assert len(labels) == len(args.model_tags)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    recs = []
    for tag, label in zip(args.model_tags, labels):
        rec = run_model(tag, label, args, device)
        with open(os.path.join(args.out_dir, f"{tag}__tuned_lens.json"), "w") as f:
            json.dump(rec, f, indent=1)
        recs.append(rec)
    plot(recs, os.path.join(args.out_dir, "tuned_lens_compare"))


if __name__ == "__main__":
    main()
