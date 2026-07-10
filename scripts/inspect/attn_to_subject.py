"""Raw attention check: where does the final token LOOK, layer by layer?

Feed the Step-D sentences unmodified ("Everyone knows Einstein was a celebrated
scientist. The scientist was") and record, at every layer, the attention of the
LAST token ("was") over all previous positions (averaged over heads and over
sentences). If early/mid layers put attention mass on the subject name and late
layers stop, that is direct, assumption-free corroboration of the causal
boundary (Step D). Note attention is correlational (a head can attend without
copying anything useful), so this complements, not replaces, the patching.

Runs on CPU fine (short sentences). Requires eager attention for weights.

Usage:
  python -m scripts.inspect.attn_to_subject --hf-model Qwen/Qwen3-8B-Base --device cpu
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.step_d_nanochat import ENTITIES, PREFIX, SUFFIX

PALETTE = ["#1f77b4", "#e8a87c", "#2c8a8a", "#5aa75a", "#888888"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boundary", type=int, default=22,
                    help="measured stop-reading boundary to overlay")
    ap.add_argument("--out", default="results/boundary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager").to(device).eval()
    n_layer = model.config.num_hidden_layers
    print(f"{args.hf_model}: {n_layer} layers on {device}", flush=True)

    # masses per layer, averaged over sentences: name span / last token itself /
    # first token (attention sink) / everything else
    cats = {"subject name": [], "subject name (max head)": [],
            "last token (itself)": [], "first token (sink)": [], "rest": []}
    per_sent = []
    example = None
    for name, role in ENTITIES:
        pre = tok(PREFIX, add_special_tokens=True)["input_ids"]
        span = tok(" " + name, add_special_tokens=False)["input_ids"]
        suf = tok(SUFFIX.format(role=role), add_special_tokens=False)["input_ids"]
        ids = pre + span + suf
        s0, s1 = len(pre), len(pre) + len(span)
        with torch.inference_mode():
            out = model(torch.tensor([ids], device=device), output_attentions=True)
        rows, name_max = [], []  # head-mean rows; per-layer max-over-heads on the name
        for att in out.attentions:
            h = att[0, :, -1, :].float()          # (heads, T)
            rows.append(h.mean(0))
            name_max.append(h[:, s0:s1].sum(1).max())
        A = torch.stack(rows, 0)  # (n_layer, T)
        T = A.shape[1]
        name_mean = A[:, s0:s1].sum(1)
        per_sent.append({
            "subject name": name_mean,
            "subject name (max head)": torch.stack(name_max),
            "last token (itself)": A[:, T - 1],
            "first token (sink)": A[:, 0],
            "rest": 1.0 - (name_mean + A[:, T - 1] + A[:, 0]),
        })
        if example is None and name == "Einstein":
            example = (A, [tok.decode([i]) for i in ids])
        print(f"  {name}: done", flush=True)

    for k in cats:
        cats[k] = torch.stack([d[k] for d in per_sent], 0).mean(0)

    res = {"model": args.hf_model, "n_layer": n_layer, "boundary": args.boundary,
           "mass": {k: v.tolist() for k, v in cats.items()}}
    with open(os.path.join(args.out, "attn_to_subject.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.8),
                                  gridspec_kw={"width_ratios": [1, 1.2]})
    for i, (k, v) in enumerate(cats.items()):
        ax.plot(range(n_layer), v.tolist(), "-o", ms=3, color=PALETTE[i], label=k)
    ax.axvline(args.boundary, color="r", ls="--", lw=1,
               label=f"causal boundary L{args.boundary}")
    ax.set_xlabel("layer"); ax.set_ylabel("attention mass of the final token")
    ax.set_title("Where does the final token look? (avg over heads & sentences)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    A, toks = example
    im = ax2.imshow(A.numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax2.set_xticks(range(len(toks)))
    ax2.set_xticklabels(toks, rotation=90, fontsize=7)
    ax2.set_ylabel("layer"); ax2.set_title("example: Einstein sentence (last-token attention)")
    ax2.axhline(args.boundary, color="r", ls="--", lw=1)
    fig.colorbar(im, ax=ax2, fraction=0.03)
    fig.tight_layout()
    p = os.path.join(args.out, "attn_to_subject.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
