"""Patchscope with a forced-choice category readout — usable on small models.

The few-shot patchscope (scripts/inspect/patchscope_few_shot.py) grades a free
generation ("x: <20 generated tokens>") with substring criteria. That readout
needs a model strong enough to write coherent entity descriptions; a d24
nanochat base model mostly can't, so every layer grades 0 and the probe is
blind. This variant keeps the patching method identical (capture the residual
at the last source-prompt token at every source layer L, REPLACE the residual
of the `x` placeholder at one fixed target layer) but swaps the readout for a
single-token forced choice:

    target prompt:  "Africa: continent\nLeonardo DiCaprio: actor\n
                     Samsung: company\nx:"
    readout:        next-token logits at the ":" position, restricted to the
                    first tokens of the candidate category words
                    (" princess", " king", " boxer", ...).
    hit:            the correct category outranks the other candidates.

A layer's residual "still remembers the entity" if patching it makes the model
pick the right category. No generation, so model weakness in writing doesn't
confound the measurement; chance level is 1/len(ENTITIES).

Calibration mode runs the same cloze with the real entity text in place of
`x` (no patching). An entity is only a valid probe if the model gets it right
with the full text visible — that's the probe's ceiling.

Caveat for BoV-style models (v_from_value_emb): at target layers >= 2n/3 the
value at the `x` position comes from the *token id* of `x`, not from the
patched residual, so the patched content reaches the readout only through
Q/K routing at deep layers and through values at layers T+1..2n/3-1. The
within-model curve across source layers is unaffected (the readout stack is
constant); absolute levels across models should be compared with this in mind.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.inspect.patchscope_category \
        --model-tags arch_d24_gpt_base_100B \
                     arch_d24_gpt_base_v_from_value_emb_learn_100B \
        --labels Attention BoV \
        --target-layers 2 4 6 8 \
        --out-dir results/patchscopes_small
"""
import argparse
import json
import os
import glob

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import PALETTE, _style_axes


# Entity phrases: the last token is (mostly) generic, so the entity identity
# must have been bound into the final-position residual by attention.
# Categories are graded by their FIRST token under the nanochat tokenizer.
ENTITIES = ["diana", "alexander", "ali", "jurassic", "ww2", "usa", "einstein"]
SOURCE_PROMPTS = {
    "diana":     "Diana, princess of Wales",
    "alexander": "Alexander the Great",
    "ali":       "Muhammad Ali",
    "jurassic":  "Jurassic Park",
    "ww2":       "World War II",
    "usa":       "the United States of America",
    "einstein":  "Albert Einstein",
}
CATEGORIES = {
    "diana":     " princess",
    "alexander": " king",
    "ali":       " boxer",
    "jurassic":  " movie",
    "ww2":       " war",
    "usa":       " country",
    "einstein":  " scientist",
}
ENTITY_TITLES = {
    "diana":     "Diana, princess of Wales",
    "alexander": "Alexander the Great",
    "ali":       "Muhammad Ali",
    "jurassic":  "Jurassic Park",
    "ww2":       "World War II",
    "usa":       "United States of America",
    "einstein":  "Albert Einstein",
}
# Few-shot prefix that primes the "entity: category" format. Its categories
# must not collide with any test category (asserted below).
FEWSHOT_PREFIX = "Africa: continent\nLeonardo DiCaprio: actor\nSamsung: company\n"


def load_adapter(model_tag, step, device):
    from nanochat.checkpoint_manager import load_model
    model, tokenizer, meta = load_model("base", device, phase="eval",
                                        model_tag=model_tag, step=step)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "blocks": model.transformer.h,
        "n_layer": meta["model_config"]["n_layer"],
        "bos": tokenizer.get_bos_token_id(),
        "name": model_tag,
    }


def build_target_ids(adapter, entity_text=None):
    """BOS + few-shot prefix + (entity text | 'x') + ':'. Returns (ids, patch_pos).
    patch_pos is the position of the LAST token of the slot (the 'x', or the
    final entity token in calibration) — i.e. where the source residual goes.
    The readout position is always the final ':' token."""
    tok = adapter["tokenizer"]
    ids = [adapter["bos"]] + tok.encode(FEWSHOT_PREFIX)
    slot = tok.encode(entity_text if entity_text is not None else "x")
    ids += slot
    patch_pos = len(ids) - 1
    ids += tok.encode(":")
    return ids, patch_pos


@torch.inference_mode()
def forward_logits(adapter, ids, patch=None):
    """One forward; returns float logits at the last position. If patch is
    (target_layer, pos, vec), replace the residual output of that block at pos."""
    device = next(adapter["model"].parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    handle = None
    if patch is not None:
        t_layer, pos, vec = patch

        def hook(_mod, _inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            h = h.clone()
            h[:, pos, :] = vec.to(h.dtype)
            return (h,) + out[1:] if is_tuple else h

        handle = adapter["blocks"][t_layer].register_forward_hook(hook)
    try:
        logits = adapter["model"](x)
    finally:
        if handle is not None:
            handle.remove()
    return logits[0, -1, :].float()


@torch.inference_mode()
def capture_source_residuals(adapter, src_ids, src_pos):
    """One forward on the source prompt; residual-stream output of every block
    at src_pos. Blocks return (x, v_init), so out[0] is the residual."""
    residuals = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            residuals[idx] = h[0, src_pos, :].detach().clone()
        return hook

    for i, blk in enumerate(adapter["blocks"]):
        handles.append(blk.register_forward_hook(make_hook(i)))
    try:
        device = next(adapter["model"].parameters()).device
        x = torch.tensor([src_ids], dtype=torch.long, device=device)
        _ = adapter["model"](x)
    finally:
        for h in handles:
            h.remove()
    return residuals


def candidate_ids(tokenizer):
    """First token id of each category word. Must be pairwise distinct."""
    cands = {}
    for key, cat in CATEGORIES.items():
        ids = tokenizer.encode(cat)
        cands[key] = ids[0]
    assert len(set(cands.values())) == len(cands), \
        f"candidate first-tokens collide: { {k: tokenizer.decode([v]) for k, v in cands.items()} }"
    fewshot_cats = [" continent", " actor", " company"]
    for fc in fewshot_cats:
        assert tokenizer.encode(fc)[0] not in cands.values(), \
            f"few-shot category {fc!r} collides with a test category"
    return cands


def restricted_choice(logits, cands, correct_key):
    """Forced choice among candidate first-tokens. Returns (hit, prob, ranking)."""
    keys = list(cands.keys())
    scores = torch.tensor([logits[cands[k]] for k in keys])
    probs = torch.softmax(scores, dim=0)
    order = [keys[i] for i in torch.argsort(scores, descending=True)]
    hit = int(order[0] == correct_key)
    prob = float(probs[keys.index(correct_key)])
    return hit, prob, order


@torch.inference_mode()
def reference_norm(adapter, ids, t_layer, pos):
    """Residual norm at (t_layer, pos) in an unpatched forward of `ids` —
    the in-distribution scale a patched vector should have there."""
    box = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        box["n"] = h[0, pos, :].norm()

    handle = adapter["blocks"][t_layer].register_forward_hook(hook)
    try:
        device = next(adapter["model"].parameters()).device
        _ = adapter["model"](torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        handle.remove()
    return box["n"]


def run_model(adapter, target_layers, norm_match=False, topk_show=5):
    tok = adapter["tokenizer"]
    n_layer = adapter["n_layer"]
    cands = candidate_ids(tok)
    rec = {"model": adapter["name"], "n_layer": n_layer,
           "fewshot_prefix": FEWSHOT_PREFIX, "norm_match": norm_match,
           "categories": CATEGORIES, "calibration": {}, "sweep": {}}

    # ---- calibration: full entity text in the slot, no patching ----
    print(f"\n=== [{adapter['name']}] calibration (full text, no patch) ===")
    for key in ENTITIES:
        ids, _ = build_target_ids(adapter, entity_text=SOURCE_PROMPTS[key])
        logits = forward_logits(adapter, ids)
        hit, prob, order = restricted_choice(logits, cands, key)
        top = torch.topk(logits, k=topk_show)
        top_str = [tok.decode([int(i)]) for i in top.indices]
        rec["calibration"][key] = {"hit": hit, "prob": prob, "order": order,
                                   "top_unrestricted": top_str}
        mark = "OK " if hit else "MISS"
        print(f"  [{mark}] {key:10s} -> {order[0]:10s} p={prob:.2f}  top5={top_str}")

    # ---- source residual capture + patched sweep ----
    for key in ENTITIES:
        src_text = SOURCE_PROMPTS[key]
        src_ids = [adapter["bos"]] + tok.encode(src_text)
        src_pos = len(src_ids) - 1
        last_tok = tok.decode([src_ids[-1]])
        residuals = capture_source_residuals(adapter, src_ids, src_pos)
        for T in target_layers:
            ids, patch_pos = build_target_ids(adapter, entity_text=None)
            ref_n = reference_norm(adapter, ids, T, patch_pos) if norm_match else None
            curve = []
            for L in range(n_layer):
                vec = residuals[L]
                if norm_match:
                    vec = vec * (ref_n / vec.norm().clamp(min=1e-6))
                logits = forward_logits(adapter, ids,
                                        patch=(T, patch_pos, vec))
                hit, prob, order = restricted_choice(logits, cands, key)
                curve.append({"src_layer": L, "hit": hit, "prob": prob,
                              "top_choice": order[0]})
            rec["sweep"].setdefault(str(T), {})[key] = curve
            hits = "".join("#" if c["hit"] else "." for c in curve)
            print(f"  [{adapter['name']}] tgt L{T:02d} {key:10s} "
                  f"(last src tok {last_tok!r}): {hits}")
    return rec


# ----------------------------------------------------------------------------
# Plotting: one figure per target layer; rows = models, cols = entities.
# Solid step = binary hit; faint line = restricted-softmax prob of correct.
# ----------------------------------------------------------------------------

def regenerate_plot(out_dir, target_layer, labels_by_slug, suffix=""):
    paths = sorted(glob.glob(os.path.join(out_dir, f"*__category{suffix}.json")))
    recs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        if str(target_layer) in r["sweep"]:
            recs.append(r)
    if not recs:
        print(f"[plot] nothing for target layer {target_layer}")
        return None

    n_rows, n_cols = len(recs), len(ENTITIES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols + 1.0, 1.6 + 1.4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    for r, rec in enumerate(recs):
        color = PALETTE[r % len(PALETTE)]
        label = labels_by_slug.get(rec["model"], rec["model"])
        for c, key in enumerate(ENTITIES):
            ax = axes[r, c]
            curve = rec["sweep"][str(target_layer)][key]
            xs = [pt["src_layer"] for pt in curve]
            hits = [pt["hit"] for pt in curve]
            probs = [pt["prob"] for pt in curve]
            calib_ok = rec["calibration"][key]["hit"] == 1
            ax.plot(xs, probs, color=color, linewidth=0.9, alpha=0.35, zorder=2)
            ax.plot(xs, hits, color=color, linewidth=1.2,
                    drawstyle="steps-mid", zorder=3)
            ax.scatter([x for x, h in zip(xs, hits) if h],
                       [1] * sum(hits), color=color, s=8, zorder=4)
            ax.axhline(1.0 / len(ENTITIES), color="0.75", lw=0.6, ls=":")
            ax.set_ylim(-0.15, 1.15)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["no", "yes"], fontsize=8)
            ax.set_xlim(-1, max(xs) + 1)
            ax.set_xticks([0, 8, 16, 23])
            ax.tick_params(axis="x", labelsize=8)
            _style_axes(ax)
            if not calib_ok:
                ax.set_facecolor("#f2f2f2")  # model fails even with full text
            if r == 0:
                ax.set_title(f"{ENTITY_TITLES[key]}\n(-> {CATEGORIES[key]!r})",
                             fontsize=8.5, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("source layer", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)

    nm = ", norm-matched patch" if suffix else ""
    fig.suptitle(
        f"Category patchscope (forced choice) — fixed target layer L{target_layer}{nm}; "
        f"step = hit, faint = P(correct | candidates), gray panel = fails calibration",
        fontsize=9.5, y=0.99)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.16,
                        wspace=0.45, hspace=0.55)
    out_base = os.path.join(out_dir,
                            f"patchscope_category_tgt{target_layer}{suffix}_grid")
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    print(f"[plot] {out_base}.png  ({n_rows} models x {n_cols} entities)")
    return out_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--target-layers", type=int, nargs="+", default=[4])
    ap.add_argument("--norm-match", action="store_true",
                    help="Rescale each patched vector to the norm the target "
                         "model natively has at (target_layer, x position). "
                         "Controls for residual-norm growth across depth "
                         "(BoV's deep residuals run ~2x hotter than the "
                         "attention baseline's).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/patchscopes_small")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    labels = args.labels or args.model_tags
    assert len(labels) == len(args.model_tags)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    suffix = "_normmatch" if args.norm_match else ""
    labels_by_slug = {}
    for tag, label in zip(args.model_tags, labels):
        adapter = load_adapter(tag, args.step, device)
        labels_by_slug[tag] = label
        for T in args.target_layers:
            assert 0 <= T < adapter["n_layer"]
        rec = run_model(adapter, args.target_layers, norm_match=args.norm_match)
        rec["label"] = label
        out_path = os.path.join(args.out_dir, f"{tag}__category{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"  -> {out_path}")
        del adapter
        torch.cuda.empty_cache()

    if not args.no_plot:
        # Pick up labels from the JSONs for models run previously.
        for p in glob.glob(os.path.join(args.out_dir, f"*__category{suffix}.json")):
            with open(p) as f:
                r = json.load(f)
            labels_by_slug.setdefault(r["model"], r.get("label", r["model"]))
        for T in args.target_layers:
            regenerate_plot(args.out_dir, T, labels_by_slug, suffix=suffix)


if __name__ == "__main__":
    main()
