"""
What is the learned per-layer gate actually doing?

The gate rescales each layer's contribution: x = x + gate * f(x), with
gate = 1 + 0.5*tanh(w . norm(x)), so it lives in [0.5, 1.5]. gate = 1 is
exactly the baseline; below 1 the layer contributes less than a normal
transformer layer would, above 1 it contributes more.

The histogram figure shows shallow layers sitting below 1 and deep layers
above 1. That shape has (at least) two very different explanations:

  (a) BORING -- the gate is effectively a per-layer constant, i.e. a learned
      depth-wise rescaling. Every token in a layer gets nearly the same value.
      Nothing to do with content. (Would also be consistent with the known
      Pre-LN "deep layers under-contribute" pathology.)
  (b) INTERESTING -- the gate is genuinely per-token routing: within one layer
      different tokens get meaningfully different values, tracking something
      about what that token needs.

This script separates them WITHOUT any training, by decomposing the variance
of the gate into a between-layer part and a within-layer (token-to-token) part.
If the within-layer share is tiny, (a) wins and the per-token story is dead.

Also reports, per layer: mean, sd, and percentiles, so the eyeballed histogram
gets replaced by numbers.

Usage:
  NANOCHAT_BASE_DIR=/local-ssd/$USER CUDA_VISIBLE_DEVICES=1 \
    python -m scripts.inspect.gate_stats \
      --model-tag arch_d24_skip_ahead_dense_tanh_l2_lr1e3 --batches 8
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device-batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", default="results/gate_stats")
    args = ap.parse_args()

    device = torch.device("cuda")  # build_model inspects device.type
    from nanochat.checkpoint_manager import load_model
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

    model, tok, meta = load_model("base", device, phase="eval",
                                  model_tag=args.model_tag, step=args.step)
    model.eval()
    cfg = meta["model_config"]
    n_layer = cfg["n_layer"]
    assert cfg.get("skip_ahead_mode", "none") != "none", \
        f"{args.model_tag} has no gates (skip_ahead_mode=none)"
    print(f"{args.model_tag}: n_layer={n_layer} gate_type={cfg.get('skip_gate_type')} "
          f"source={cfg.get('skip_gate_source')} val_bpb={meta.get('val_bpb')}")

    # Hook every gate Linear and turn its logit into the actual gate value.
    # The mapping depends on skip_gate_type -- hardcoding the tanh form silently
    # produces wrong numbers for the sqrt gate, whose range is [0.1, inf) rather
    # than [0.5, 1.5].
    gtype = cfg.get("skip_gate_type", "tanh")

    def logit_to_gate(z):
        z = z.float()
        if gtype == "tanh":
            return 1.0 + 0.5 * torch.tanh(z)
        if gtype == "sigmoid":
            return 2.0 * torch.sigmoid(z)
        if gtype == "sqrt":
            # 0.1 + 0.9*(z + sqrt(1+z^2)); negative branch rationalised as in the
            # model code to avoid cancellation
            root = torch.sqrt(1.0 + z.square())
            mult = torch.where(z >= 0.0, z + root, 1.0 / (root - z))
            return 0.1 + 0.9 * mult
        raise ValueError(gtype)

    captured = [[] for _ in range(n_layer)]

    def mk(i):
        def hook(_m, _inp, out):
            captured[i].append(logit_to_gate(out).squeeze(-1).flatten().cpu())
        return hook

    handles = [model.skip_gates[i].register_forward_hook(mk(i)) for i in range(n_layer)]

    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.device_batch_size, args.seq_len, split="val", device=device)
    with torch.inference_mode():
        for b in range(args.batches):
            inputs, _ = next(loader)
            model(inputs)
            print(f"  batch {b+1}/{args.batches}", end="\r")
    for h in handles:
        h.remove()
    print()

    gates = torch.stack([torch.cat(c) for c in captured])   # (n_layer, N)
    n_tok = gates.shape[1]

    # Variance decomposition. Total variance over (layer, token) splits into
    # variance of the per-layer means (between) + mean of the per-layer
    # variances (within). The within share is the whole question.
    per_layer_mean = gates.mean(dim=1)
    between = per_layer_mean.var(unbiased=False).item()
    within = gates.var(dim=1, unbiased=False).mean().item()
    within_share = within / (between + within)

    print(f"\ntokens per layer: {n_tok:,}")
    print(f"between-layer variance : {between:.6f}")
    print(f"within-layer  variance : {within:.6f}")
    verdict = ("mostly a per-layer constant -- the per-token story is weak"
               if within_share < 0.15 else
               "substantial per-token variation -- NOT just a depth-wise rescaling")
    print(f"WITHIN-LAYER SHARE     : {within_share:.1%}   <- {verdict}")
    print("   (this rules out / in the 'per-layer constant' explanation; it does NOT\n"
          "    show the per-token variation is meaningful -- for that, shuffle the gate\n"
          "    values within each layer and see whether quality drops.)")

    print("\nlayer   mean     sd      p5     p25     p50     p75     p95")
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    rows = []
    for i in range(n_layer):
        g = gates[i]
        p = torch.quantile(g[torch.randperm(g.numel())[:200_000]], qs)
        rows.append(dict(layer=i + 1, mean=g.mean().item(), sd=g.std().item(),
                         p5=p[0].item(), p25=p[1].item(), p50=p[2].item(),
                         p75=p[3].item(), p95=p[4].item()))
        r = rows[-1]
        print(f"  {r['layer']:2d}   {r['mean']:.4f}  {r['sd']:.4f}  "
              f"{r['p5']:.4f}  {r['p25']:.4f}  {r['p50']:.4f}  {r['p75']:.4f}  {r['p95']:.4f}")

    below = [r["layer"] for r in rows if r["mean"] < 1.0]
    above = [r["layer"] for r in rows if r["mean"] >= 1.0]
    print(f"\nlayers with mean gate < 1 (contribute LESS than baseline): {below}")
    print(f"layers with mean gate >= 1 (contribute MORE than baseline): {above}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.model_tag}__gate_stats.json")
    json.dump(dict(model_tag=args.model_tag, n_layer=n_layer, n_tokens=n_tok,
                   val_bpb=meta.get("val_bpb"),
                   between_layer_var=between, within_layer_var=within,
                   within_layer_share=within_share, per_layer=rows),
              open(path, "w"), indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
