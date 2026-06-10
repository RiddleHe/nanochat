"""Patchscope — raw-completion variant (logit lens on source residuals).

For each source layer L=0..N-1, take the residual stream at the last token of
the source prompt and project it directly through the model's own
`final_norm + lm_head`. Check whether the top-1 predicted next token equals the
entity-binding token: " Wales" for "Diana, the princess of", " Great" for
"Alexander the", etc.

Companion to `scripts/inspect/patchscope_few_shot.py`. Where the few-shot
variant asks "does this residual carry enough info for downstream target-prompt
layers to integrate the few-shot context?", this raw-completion variant asks
"does this residual already encode the next entity-binding token on its own?".
A layer can pass raw-completion but fail few-shot (the info is there but the
integration with the target doesn't take), or pass few-shot but fail raw
completion (the residual isn't itself a finished prediction, but it's
compatible with the target stack finishing the job).

After the sweep, regenerates `results/patchscopes/logit_lens_alt_grid.{pdf,png}`
containing every model whose `__logit_lens__alt__` outputs are present.

Usage:
    CUDA_VISIBLE_DEVICES=N uv run python -m scripts.inspect.patchscope_raw_completion \\
        --hf-model EleutherAI/pythia-12b
    CUDA_VISIBLE_DEVICES=N uv run python -m scripts.inspect.patchscope_raw_completion \\
        --hf-model Qwen/Qwen3-8B-Base
"""
import argparse
import os
import re
import glob

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.inspect.patchscope_few_shot import (
    SOURCE_SETS, ENTITIES, ENTITY_TITLES, PALETTE, _style_axes,
    _order_and_label,
)


# The next token we'd want each layer's residual to predict for the alt prompt.
# These are tokenizer-agnostic strings; per-tokenizer encoding is done at runtime.
EXPECTED_NEXT_TOKEN = {
    "diana":     " Wales",
    "alexander": " Great",
    "ali":       " Ali",
    "jurassic":  " Park",
    "nyc":       " York",
}


# ----------------------------------------------------------------------------
# HF backend: load + locate (blocks, final_norm, lm_head).
# ----------------------------------------------------------------------------

def _load_hf(hf_model, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(hf_model, dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    if hasattr(model, "gpt_neox"):  # Pythia / GPT-NeoX
        blocks = model.gpt_neox.layers
        final_norm = model.gpt_neox.final_layer_norm
        lm_head = model.embed_out
        embed_in = model.gpt_neox.embed_in
    elif hasattr(model, "model") and hasattr(model.model, "layers"):  # Llama/Qwen
        blocks = model.model.layers
        final_norm = model.model.norm
        lm_head = model.lm_head
        embed_in = model.model.embed_tokens
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2/J
        blocks = model.transformer.h
        final_norm = model.transformer.ln_f
        lm_head = model.lm_head
        embed_in = model.transformer.wte
    else:
        raise RuntimeError(f"Don't know the unembed path for {hf_model}")

    n_layer = len(blocks)
    return {
        "model": model, "tokenizer": tok, "blocks": blocks,
        "final_norm": final_norm, "lm_head": lm_head, "embed_in": embed_in,
        "n_layer": n_layer, "name": hf_model.replace("/", "_"),
    }


class _NanochatTok:
    """Duck-types the two HF tokenizer methods this script uses."""
    def __init__(self, tok):
        self._tok = tok

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text)

    def decode(self, ids, skip_special_tokens=False):
        return self._tok.decode(ids)


def _load_nanochat(model_tag, step, device):
    """nanochat checkpoint adapter. Unembed path mirrors GPTBase.forward:
    functional RMSNorm + lm_head sliced to the real vocab (softcap is
    monotonic, so it cannot change top-k order and is skipped)."""
    from nanochat.checkpoint_manager import load_model
    from nanochat.model.gpt import norm
    model, tok, meta = load_model("base", device, phase="eval",
                                  model_tag=model_tag, step=step)
    model.eval()
    vocab_size = meta["model_config"]["vocab_size"]
    return {
        "model": model, "tokenizer": _NanochatTok(tok),
        "blocks": model.transformer.h,
        "final_norm": norm,
        "lm_head": lambda x: model.lm_head(x)[..., :vocab_size],
        "embed_in": model.transformer.wte,
        "n_layer": meta["model_config"]["n_layer"],
        "name": model_tag,
        "prepend_bos": tok.get_bos_token_id(),
    }


# ----------------------------------------------------------------------------
# Logit-lens core
# ----------------------------------------------------------------------------

@torch.inference_mode()
def capture_layer_residuals(adapter, src_ids, src_pos):
    """Run one forward; capture residual at src_pos after every block."""
    residuals = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            residuals[idx] = x[0, src_pos, :].detach().clone()
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


@torch.inference_mode()
def unembed_top1(adapter, residual_vec, lens="logit"):
    """Map residual to top-k token IDs under one of two lenses.

    lens='logit': final_norm + lm_head, top tokens by predicted next-token logit
                  (i.e., "what token does the model think comes next?").
    lens='embed': cosine similarity to the input token-embedding matrix
                  (i.e., "what token does the model think this position IS?",
                   nearest-neighbour in embedding space).
    Returns (top1_id, top1_score, top5_ids)."""
    x = residual_vec.unsqueeze(0)  # (1, d_model)
    if lens == "logit":
        x = adapter["final_norm"](x)
        scores = adapter["lm_head"](x).squeeze(0).float()  # (vocab,)
    elif lens == "embed":
        E = adapter["embed_in"].weight.float()  # (V, d_model)
        r = residual_vec.float()
        r_n = r / (r.norm() + 1e-8)
        E_n = E / (E.norm(dim=-1, keepdim=True) + 1e-8)
        scores = E_n @ r_n  # (V,) cosine similarities
    else:
        raise ValueError(f"unknown lens: {lens}")
    top5 = torch.topk(scores, k=5)
    return int(top5.indices[0]), float(top5.values[0]), [int(i) for i in top5.indices]


def first_token_id(tokenizer, text):
    """Get the FIRST token id of `text` under this tokenizer (for the
    expected-next-token check)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Empty encoding for {text!r}")
    return ids[0], tokenizer.decode([ids[0]])


def run_one_entity(adapter, ent_key, source, expected_text=None, lens="logit"):
    """If expected_text is None (canonical set), hit is always 0 and exp_* are None.
    lens is passed through to unembed_top1."""
    if expected_text is not None:
        exp_id, exp_str = first_token_id(adapter["tokenizer"], expected_text)
    else:
        exp_id, exp_str = None, None

    src_ids = adapter["tokenizer"].encode(source, add_special_tokens=False)
    if adapter.get("prepend_bos") is not None:
        src_ids = [adapter["prepend_bos"]] + src_ids
    src_pos = len(src_ids) - 1

    residuals = capture_layer_residuals(adapter, src_ids, src_pos)

    results = []  # (layer, hit?, top1_str, top5_strs)
    for L in range(adapter["n_layer"]):
        top1_id, _, top5_ids = unembed_top1(adapter, residuals[L], lens=lens)
        top1_str = adapter["tokenizer"].decode([top1_id])
        top5_strs = [adapter["tokenizer"].decode([i]) for i in top5_ids]
        hit = int(top1_id == exp_id) if exp_id is not None else 0
        results.append((L, hit, top1_str, top5_strs))
    return results, exp_id, exp_str


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def _load_run_file(path):
    rows = []
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            head, tail = line.split("\t", 1)
            if head.startswith("L"):
                # Lxx\thit\ttop1_str\ttop5_csv
                parts = tail.split("\t", 2)
                hit = int(parts[0])
                rows.append((int(head[1:]), hit, parts[1] if len(parts) > 1 else "",
                             parts[2] if len(parts) > 2 else ""))
            else:
                meta[head] = tail
    return meta, rows


def regenerate_plot(out_dir):
    pattern = os.path.join(out_dir, "*__logit_lens__alt__*.txt")
    all_files = glob.glob(pattern)
    seen = {}
    for path in all_files:
        base = os.path.basename(path)
        m = re.match(r"^(.+?)__logit_lens__alt__(.+)\.txt$", base)
        if not m:
            continue
        seen.setdefault(m.group(1), set()).add(m.group(2))
    complete = [m for m, ents in seen.items()
                if all(e in ents for e in ENTITIES)]
    if not complete:
        print("[plot] no complete model runs found")
        return None
    models, labels = _order_and_label(complete)

    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"

    n_rows, n_cols = len(models), len(ENTITIES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9.0, 1.6 + 1.4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for r, model_slug in enumerate(models):
        color = PALETTE[r % len(PALETTE)]
        for c, ent_key in enumerate(ENTITIES):
            ax = axes[r, c]
            _, rows = _load_run_file(os.path.join(
                out_dir, f"{model_slug}__logit_lens__alt__{ent_key}.txt"))
            xs = [L for L, _, _, _ in rows]
            ys = [h for _, h, _, _ in rows]
            ax.plot(xs, ys, color=color, linewidth=1.2,
                    drawstyle="steps-mid", zorder=3)
            ax.scatter([L for L, h, _, _ in rows if h],
                       [1] * sum(ys), color=color, s=8, zorder=4)
            ax.set_ylim(-0.15, 1.15)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["no", "yes"], fontsize=8)
            ax.set_xlim(-1, max(xs) + 1)
            ax.set_xticks([0, 10, 20, 30] if max(xs) >= 30 else [0, 10, 20])
            ax.tick_params(axis="x", labelsize=8)
            _style_axes(ax)
            if r == 0:
                title = ENTITY_TITLES[ent_key]
                src = SOURCE_SETS["alt"][ent_key]
                expected = EXPECTED_NEXT_TOKEN[ent_key]
                ax.set_title(f"{title}\n({src!r} → {expected!r})",
                             fontsize=9, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("source layer", fontsize=9)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=9)

    fig.suptitle("Logit lens — alt source prompts, top-1 next-token match",
                 fontsize=10, y=0.98)
    plt.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.16,
                        wspace=0.45, hspace=0.55)

    out_base = os.path.join(out_dir, "logit_lens_alt_grid")
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=200)
    plt.close(fig)
    print(f"[plot] {out_base}.pdf  ({n_rows} models × {n_cols} entities)")
    return out_base


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-model", help="HF causal-LM name.")
    src.add_argument("--model-tag", help="nanochat checkpoint tag.")
    ap.add_argument("--step", type=int, default=None,
                    help="nanochat step (last if omitted)")
    ap.add_argument("--source-set", choices=list(SOURCE_SETS), default="alt",
                    help="alt = incomplete-sentence prompts (last source token "
                         "is a connective; expected next is the entity-binding "
                         "word, hit ✓/✗ shown). canonical = entity-name prompts "
                         "(last source token IS the entity name; no expected "
                         "next, just top-5 dump).")
    ap.add_argument("--lens", choices=["logit", "embed"], default="logit",
                    help="logit = project residual through final_norm + lm_head "
                         "(predicted next token). embed = cosine similarity to "
                         "the input token-embedding matrix (nearest tokens to "
                         "this residual in embedding space — \"what does the "
                         "model think this position IS?\").")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="./results/patchscopes")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    adapter = (_load_hf(args.hf_model, device) if args.hf_model
               else _load_nanochat(args.model_tag, args.step, device))
    print(f"Model: {adapter['name']}  n_layer={adapter['n_layer']}  "
          f"source_set={args.source_set}  lens={args.lens}")
    os.makedirs(args.out_dir, exist_ok=True)

    # File tag: keep the old "logit_lens" name when --lens=logit to preserve
    # backward compat; use "embed_lens" otherwise.
    lens_tag = "logit_lens" if args.lens == "logit" else "embed_lens"

    sources = SOURCE_SETS[args.source_set]
    for ent_key in ENTITIES:
        source = sources[ent_key]
        expected = EXPECTED_NEXT_TOKEN[ent_key] if args.source_set == "alt" else None
        head = f"{ent_key} : {source!r}"
        if expected:
            head += f" → {expected!r}"
        print(f"\n===== {head} =====")
        results, exp_id, exp_str = run_one_entity(adapter, ent_key, source,
                                                   expected, lens=args.lens)
        for L, hit, top1, top5 in results:
            marker = ("✓" if hit else "·") if expected else "."
            print(f"  L{L:02d} {marker}  top1={top1!r:18s}  top5={top5}")
        out_path = os.path.join(
            args.out_dir,
            f"{adapter['name']}__{lens_tag}__{args.source_set}__{ent_key}.txt")
        with open(out_path, "w") as f:
            f.write(f"model\t{adapter['name']}\n")
            f.write(f"entity_key\t{ent_key}\n")
            f.write(f"source\t{source}\n")
            f.write(f"source_set\t{args.source_set}\n")
            f.write(f"lens\t{args.lens}\n")
            f.write(f"expected_text\t{expected if expected else ''}\n")
            f.write(f"expected_id\t{exp_id if exp_id is not None else ''}\n")
            exp_repr = repr(exp_str) if exp_str is not None else ""
            f.write(f"expected_str_decoded\t{exp_repr}\n")
            for L, hit, top1, top5 in results:
                top5_str = " | ".join(repr(t) for t in top5)
                f.write(f"L{L:02d}\t{hit}\t{top1}\t{top5_str}\n")
        print(f"  -> {out_path}")

    # The line-grid plot only makes sense for alt + logit (where there's an
    # expected next token). Skip otherwise.
    if not args.no_plot and args.source_set == "alt" and args.lens == "logit":
        print()
        regenerate_plot(args.out_dir)


if __name__ == "__main__":
    main()
