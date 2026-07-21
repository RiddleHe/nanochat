"""E2 on nanochat checkpoints: hand-off layer vs entity-readout distance.

Port of handoff_distance.py (Qwen3-8B) to nanochat models, reusing the
multi-token name-span handling of step_d_nanochat.py. N filler tokens are
inserted between the entity clause and " The {role} was"; per layer we patch
(a) the name span and (b) the final position, clean -> corrupted, and measure
logit-diff recovery at the final position.

Extra wrinkle worth watching: these models were TRAINED with the SSSL window
pattern (S = 512-token window), so for N >= ~500 the entity is invisible to
the S layers at the readout position — any long-range read must happen inside
the L layers (every 4th layer). If the distant-N curves show steps at exactly
those layers, that is a strong internal validation of the method.

Usage:
  NANOCHAT_BASE_DIR=/local-ssd/mh3897 CUDA_VISIBLE_DEVICES=1 python -m \
      scripts.inspect.handoff_distance_nanochat --model-tag arch_d24_gpt_base_100B --label d24
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITIES = [
    ("Einstein", "scientist"), ("Newton", "scientist"), ("Darwin", "scientist"),
    ("Mozart", "composer"), ("Beethoven", "composer"),
    ("Shakespeare", "writer"), ("Dickens", "writer"),
    ("France", "country"), ("Japan", "country"),
    ("Paris", "city"), ("Tokyo", "city"),
    ("Google", "company"), ("Apple", "company"),
]
PREFIX = "Everyone knows"


def load_nanochat(tag, step, device):
    from nanochat.checkpoint_manager import load_model
    model, tok, meta = load_model("base", device, phase="eval", model_tag=tag, step=step)
    model.eval()
    return model, tok, meta["model_config"]["n_layer"], tok.get_bos_token_id()

FILLER = ("The afternoon light settled slowly over the quiet valley, and a mild "
          "wind moved through the tall grass beside the river while distant "
          "clouds drifted across the pale sky. ")


def build_ids(tok, bos, name, role, filler_ids, n_filler):
    pre = [bos] + tok.encode(PREFIX)
    span = tok.encode(" " + name)
    mid = tok.encode(f" was a celebrated {role}.")
    tail = tok.encode(f" The {role} was")
    ids = pre + span + mid + list(filler_ids[:n_filler]) + tail
    return ids, len(pre), len(pre) + len(span)


@torch.inference_mode()
def run_capture(model, ids, s0, s1, device, n_layer):
    """Capture name-span AND final-position hidden states per layer + logits."""
    span_f = [None] * n_layer
    last_f = [None] * n_layer
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            span_f[i] = h[0, s0:s1, :].detach().clone()
            last_f[i] = h[0, -1:, :].detach().clone()
        return hook

    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    return span_f, last_f, logits


@torch.inference_mode()
def run_patched(model, ids, L, p0, p1, states, device):
    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = (out[0] if is_t else out).clone()
        h[0, p0:p1, :] = states.to(h.dtype)
        return (h,) + out[1:] if is_t else h

    handle = model.transformer.h[L].register_forward_hook(hook)
    try:
        return model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1, :].float()
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--distances", type=int, nargs="*", default=[0, 100, 400, 800, 1400])
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--out", default="results/handoff_distance")
    args = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    label = args.label or args.model_tag

    model, tok, n_layer, bos = load_nanochat(args.model_tag, args.step, device)
    filler_ids = tok.encode(FILLER) * (max(args.distances) // 10 + 2)

    cats = {}
    for name, role in ENTITIES:
        cats.setdefault(role, []).append(name)
    pairs = []
    for role, names in cats.items():
        for a in names:
            for b in names:
                if a != b and len(tok.encode(" " + a)) == len(tok.encode(" " + b)):
                    pairs.append((a, b, role))
    pairs = pairs[: args.max_pairs]
    print(f"{label}: {n_layer} layers, {len(pairs)} pairs, "
          f"distances {args.distances}", flush=True)

    results = {}
    for N in args.distances:
        rec_ent = torch.zeros(n_layer)
        rec_last = torch.zeros(n_layer)
        denom_sum, n_used = 0.0, 0
        for clean, corrupt, role in pairs:
            ids_c, s0, s1 = build_ids(tok, bos, clean, role, filler_ids, N)
            ids_x, _, _ = build_ids(tok, bos, corrupt, role, filler_ids, N)
            if len(ids_c) != len(ids_x):
                continue
            T = len(ids_c)
            ct = tok.encode(" " + clean)[0]
            xt = tok.encode(" " + corrupt)[0]
            span_c, last_c, log_c = run_capture(model, ids_c, s0, s1, device, n_layer)
            _, _, log_x = run_capture(model, ids_x, s0, s1, device, n_layer)

            def ld(logits):
                return float(logits[ct] - logits[xt])

            denom = ld(log_c) - ld(log_x)
            if abs(denom) < 0.5:  # model can't separate this pair at this distance
                continue
            denom_sum += denom
            for L in range(n_layer):
                for p0, p1, vecs, acc in ((s0, s1, span_c[L], rec_ent),
                                          (T - 1, T, last_c[L], rec_last)):
                    lp = run_patched(model, ids_x, L, p0, p1, vecs, device)
                    acc[L] += (ld(lp) - ld(log_x)) / denom
            n_used += 1
        if n_used == 0:
            print(f"N={N}: NO usable pairs (binding lost at this distance)", flush=True)
            results[N] = {"n_pairs": 0}
            continue
        rec_ent /= n_used
        rec_last /= n_used
        results[N] = {"n_pairs": n_used, "seq_len": len(ids_c),
                      "mean_denominator_logit_diff": denom_sum / n_used,
                      "recovery_entity_pos": rec_ent.tolist(),
                      "recovery_last_pos": rec_last.tolist()}
        print(f"N={N}: {n_used}/{len(pairs)} pairs, seq {len(ids_c)}, "
              f"denom {denom_sum / n_used:.2f}", flush=True)

    res = {"model": args.model_tag, "label": label, "n_layer": n_layer,
           "distances": args.distances, "per_distance": results}
    with open(os.path.join(args.out, f"{label}__handoff_distance.json"), "w") as f:
        json.dump(res, f, indent=1)

    shown = [N for N in args.distances if results[N]["n_pairs"] > 0]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4.2), sharey=True)
    axes = [axes] if len(shown) == 1 else list(axes)
    for ax, N in zip(axes, shown):
        r = results[N]
        ax.plot(range(n_layer), r["recovery_entity_pos"], "-o", ms=3,
                color="#2c8a2c", label="patch entity span")
        ax.plot(range(n_layer), r["recovery_last_pos"], "-s", ms=3,
                color="#1f77b4", label="patch final position")
        for Lf in range(3, n_layer, 4):  # SSSL: every 4th layer is full-context
            ax.axvline(Lf, color="0.85", lw=0.8, zorder=0)
        ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
        ax.set_title(f"N={N} (pairs {r['n_pairs']}, denom "
                     f"{r['mean_denominator_logit_diff']:.1f})", fontsize=10)
        ax.set_xlabel("patched layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("logit-diff recovery")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label}: hand-off vs distance (grey verticals = trained "
                 f"full-context layers, SSSL)")
    fig.tight_layout()
    p = os.path.join(args.out, f"{label}__handoff_distance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
