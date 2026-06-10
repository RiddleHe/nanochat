"""Patchscopes (Ghandeharioun et al., 2024) for nanochat or any HF causal LM.

For every source layer L=0..N-1, capture the residual stream at the last
source-prompt token and inject it into the `x` placeholder slot of a few-shot
description target prompt at a *fixed* target layer (default L=6). Every source
layer thus gets the same downstream stack to integrate the few-shot context.

Two source-prompt sets:
  --source-set canonical : the patchscopes-paper set (e.g. "Diana, princess of
                           Wales"). Last token is generic (e.g. " Wales"),
                           which puts the burden on the model to bind the
                           multi-token entity into the final-position residual.
  --source-set alt       : context-primed next-token-prediction prompts (e.g.
                           "Diana, the princess of"). The last-token residual
                           is doing predictive work for the entity, so it
                           carries integrated context already, with less burden
                           on the binding mechanism.

Reference: PAIR-code/interpretability/patchscopes/code/entity_processing.ipynb
(uses EleutherAI/pythia-12b on the canonical set).

After the sweep, regenerates a single comparison figure that includes every
model whose `__tgt{N}__{set}__` output files are present.
"""
import argparse
import os
import re
import glob

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_DEFAULT = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer electronics corporation, "
    "x"
)

# Each entity has a short key, a display title (for plot column headers), and
# one source prompt per source-set. The binary-decode criteria are keyed by
# the same short key — they apply to whichever source-set was used.
ENTITIES = ["diana", "alexander", "ali", "jurassic", "nyc"]
ENTITY_TITLES = {
    "diana": "Diana, princess of Wales",
    "alexander": "Alexander the Great",
    "ali": "Muhammad Ali",
    "jurassic": "Jurassic Park",
    "nyc": "New York City",
}
SOURCE_SETS = {
    "canonical": {
        "diana": "Diana, princess of Wales",
        "alexander": "Alexander the Great",
        "ali": "Muhammad Ali",
        "jurassic": "Jurassic Park",
        "nyc": "New York City",
    },
    "alt": {
        # Context-primed: last token is doing next-token prediction for the entity.
        "diana": "Diana, the princess of",
        "alexander": "Alexander the",
        "ali": "the boxer Muhammad",
        "jurassic": "the film Jurassic",
        "nyc": "Manhattan, New",
    },
}

# Substring criteria for binary entity-decoded judgment.
# 'pos': any match → ✓. 'neg': any match → force ✗ (wrong-attribution hallucinations).
CRITERIA = {
    "diana": {
        "pos": [
            "diana", "princess of wales", "princess of cornwall", "british royal",
            "royal family", "prince charles", "prince william", "prince harry",
            "duke of cambridge", "duchess of cornwall", "wife of prince",
        ],
        "neg": [],
    },
    "alexander": {
        "pos": [
            "macedon", "macedonia", "macedonian", "greek king", "greek general",
            "ancient greek", "hellenistic", "323 bc", "356 bc",
        ],
        "neg": [],
    },
    "ali": {"pos": ["boxer", "boxing", "boxers"], "neg": []},
    "jurassic": {
        "pos": [
            "spielberg", "crichton", "science fiction film", "sci-fi film",
            "novel by", "1993 ", "1994 ", "lost world", "dinosaur",
            "american film", "american movie", "theme park",
        ],
        "neg": ["michael bay", "paul w. s. anderson", "paul verhoeven",
                "neill blomkamp", "alex garland"],
    },
    "nyc": {
        "pos": [
            "new york", "nyc", "city in the united states",
            "city in the state of new york", "u.s. state", "us state",
        ],
        "neg": [],
    },
}

# Skill §3 palette: blue, peach, teal, green.
PALETTE = ["#1f77b4", "#e8a87c", "#2c8a8a", "#5aa75a"]

# Preferred row order for multi-model plots (most recent / largest first),
# and clean display names for y-axis labels.
MODEL_ORDER = [
    "Qwen_Qwen3-8B-Base",
    "Qwen_Qwen3-4B-Base",
    "Qwen_Qwen3-0.6B-Base",
    "EleutherAI_pythia-12b",
]
MODEL_DISPLAY = {
    "Qwen_Qwen3-8B-Base":    "Qwen3-8B-Base",
    "Qwen_Qwen3-4B-Base":    "Qwen3-4B-Base",
    "Qwen_Qwen3-0.6B-Base":  "Qwen3-0.6B-Base",
    "EleutherAI_pythia-12b": "Pythia-12B",
}


def _order_and_label(available_slugs):
    """Return (ordered_slugs, display_labels). Known slugs come first in
    MODEL_ORDER; unknown slugs follow in alphabetical order using their slug."""
    known = [m for m in MODEL_ORDER if m in available_slugs]
    extras = sorted(m for m in available_slugs if m not in set(MODEL_ORDER))
    ordered = known + extras
    labels = [MODEL_DISPLAY.get(s, s) for s in ordered]
    return ordered, labels


# ----------------------------------------------------------------------------
# Backend adapters — both nanochat and HF expose the residual stream as out[0]
# of a per-block forward, so the hook is uniform; backends only differ in
# load + tokenize + generate.
# ----------------------------------------------------------------------------

def _load_nanochat(model_tag, step, device):
    from nanochat.checkpoint_manager import load_model
    model, tokenizer, meta = load_model("base", device, phase="eval",
                                        model_tag=model_tag, step=step)
    n_layer = meta["model_config"]["n_layer"]

    def encode(text): return tokenizer.encode(text)
    def decode(ids): return tokenizer.decode(ids)

    @torch.inference_mode()
    def forward_once(ids):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _ = model(x)

    def generate_tokens(ids, max_tokens):
        out = []
        for tok in model.generate(list(ids), max_tokens=max_tokens, temperature=0.0):
            out.append(tok)
        return out

    return {
        "model": model, "blocks": model.transformer.h, "n_layer": n_layer,
        "encode": encode, "decode": decode,
        "forward_once": forward_once, "generate_tokens": generate_tokens,
        "name": model_tag,
    }


def _load_hf(hf_model, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(hf_model, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        blocks = model.gpt_neox.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    else:
        raise RuntimeError(f"Don't know where the blocks live for {hf_model}")
    n_layer = len(blocks)

    def encode(text): return tok.encode(text, add_special_tokens=False)
    def decode(ids): return tok.decode(ids, skip_special_tokens=False)

    @torch.inference_mode()
    def forward_once(ids):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _ = model(x)

    @torch.inference_mode()
    def generate_tokens(ids, max_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
        out = model.generate(
            x, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=eos if eos is not None else 0,
        )
        return out[0, x.shape[1]:].tolist()

    return {
        "model": model, "blocks": blocks, "n_layer": n_layer,
        "encode": encode, "decode": decode,
        "forward_once": forward_once, "generate_tokens": generate_tokens,
        "name": hf_model.replace("/", "_"),
    }


# ----------------------------------------------------------------------------
# Patchscopes core
# ----------------------------------------------------------------------------

def capture_source_hiddens(adapter, src_ids, src_pos):
    """One forward on the source prompt; capture residual-stream output of each
    block at src_pos. Both backends return (x, ...) so out[0] is residual."""
    src_hiddens = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            src_hiddens[idx] = tensor_out[0, src_pos, :].detach().clone()
        return hook

    for i, blk in enumerate(adapter["blocks"]):
        handles.append(blk.register_forward_hook(make_hook(i)))
    try:
        adapter["forward_once"](src_ids)
    finally:
        for h in handles:
            h.remove()
    return src_hiddens


def capture_source_block_contributions(adapter, src_ids, src_pos):
    """One forward on the source prompt; capture each block's *contribution* at
    src_pos — i.e., block-output residual minus block-input residual. This is
    the delta this block alone adds to the residual stream, without the
    carry-through from earlier layers."""
    block_inputs = {}
    block_outputs = {}
    handles = []

    def make_pre(idx):
        def pre(_mod, args):
            hs = args[0]
            block_inputs[idx] = hs[0, src_pos, :].detach().clone()
        return pre

    def make_post(idx):
        def post(_mod, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            block_outputs[idx] = tensor_out[0, src_pos, :].detach().clone()
        return post

    for i, blk in enumerate(adapter["blocks"]):
        handles.append(blk.register_forward_pre_hook(make_pre(i)))
        handles.append(blk.register_forward_hook(make_post(i)))
    try:
        adapter["forward_once"](src_ids)
    finally:
        for h in handles:
            h.remove()

    return {i: block_outputs[i] - block_inputs[i]
            for i in range(adapter["n_layer"])}


def patched_generate(adapter, tgt_ids, target_layer, tgt_pos, src_vec,
                     max_tokens, op="replace"):
    """Hook on block `target_layer`: at the prefill step (input length covers
    tgt_pos), modify the residual at position tgt_pos using `src_vec` and `op`.
      op='replace': overwrite (standard patchscope; works when src_vec is
                    itself a residual stream at the right scale).
      op='add':     add src_vec on top of the existing residual (steering-style;
                    use when src_vec is a small-norm delta like a block
                    contribution — replacing would erase target context).
    Decode steps under KV cache have input length 1 (< tgt_pos), so the hook
    is a no-op there — the patched info propagates via cached K/V at
    downstream layers."""
    def patch_hook(_mod, _inp, out):
        is_tuple = isinstance(out, tuple)
        x = out[0] if is_tuple else out
        if x.shape[1] <= tgt_pos:
            return out
        new_x = x.clone()
        if op == "replace":
            new_x[:, tgt_pos, :] = src_vec.to(new_x.dtype)
        elif op == "add":
            new_x[:, tgt_pos, :] = new_x[:, tgt_pos, :] + src_vec.to(new_x.dtype)
        else:
            raise ValueError(f"unknown op: {op}")
        return (new_x,) + out[1:] if is_tuple else new_x

    handle = adapter["blocks"][target_layer].register_forward_hook(patch_hook)
    try:
        gen = adapter["generate_tokens"](tgt_ids, max_tokens)
        return adapter["decode"](gen)
    finally:
        handle.remove()


def run_one_source(adapter, source, target, target_layer, max_tokens,
                   inject_mode="residual"):
    """inject_mode='residual' (existing): extract residual-stream output at
        each src layer and REPLACE target residual at tgt_layer with it.
    inject_mode='block': extract each block's contribution (output - input)
        at the src last token, ADD it on top of target residual at tgt_layer."""
    src_ids = adapter["encode"](source)
    tgt_ids = adapter["encode"](target)
    assert len(src_ids) >= 1, f"Source must tokenize to >=1 token: {source!r}"
    assert len(tgt_ids) >= 2, f"Target needs >=2 tokens; got {len(tgt_ids)}"
    if len(src_ids) < 2:  # nanochat smear-gate needs T>1
        src_ids = [src_ids[0]] + src_ids
    src_pos = len(src_ids) - 1
    tgt_pos = len(tgt_ids) - 1

    if inject_mode == "residual":
        src_vecs = capture_source_hiddens(adapter, src_ids, src_pos)
    elif inject_mode == "block":
        src_vecs = capture_source_block_contributions(adapter, src_ids, src_pos)
    else:
        raise ValueError(f"unknown inject_mode: {inject_mode}")
    op = "replace"  # standard patchscope semantics, both modes

    results = []
    for src_layer in range(adapter["n_layer"]):
        text = patched_generate(adapter, tgt_ids, target_layer, tgt_pos,
                                src_vecs[src_layer], max_tokens, op=op)
        results.append((src_layer, text.lstrip().split("\n")[0]))
    return results


# ----------------------------------------------------------------------------
# Plotting — finds every model in the results dir with matching target_layer
# AND source-set, produces N-row × 5-col binary line-chart grid.
# ----------------------------------------------------------------------------

def _grade(text, criteria):
    t = text.lower()
    if any(n in t for n in criteria["neg"]):
        return 0
    return 1 if any(p in t for p in criteria["pos"]) else 0


def _style_axes(ax):
    """Skill §3: top/bottom spines only, faint h-grid."""
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, alpha=0.3, color="grey")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", length=2)
    ax.tick_params(axis="y", which="both", length=0)


def _load_run_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S") and "\t" in line:
                tag, text = line.split("\t", 1)
                rows.append((int(tag.split("_")[0][1:]), text))
    return rows


def regenerate_plot(out_dir, target_layer, source_set, inject_mode="residual"):
    """Find every `*__tgt{N}__{set}[__{mode}]__{ent}.txt` and build one
    multi-row figure. For inject_mode='residual' we keep the legacy filename
    (no mode segment); for 'block' the segment is included."""
    if inject_mode == "residual":
        regex = rf"^(.+?)__tgt{target_layer}__{source_set}__(.+)\.txt$"
        pattern = os.path.join(out_dir, f"*__tgt{target_layer}__{source_set}__*.txt")
    else:
        regex = (rf"^(.+?)__tgt{target_layer}__{source_set}__"
                 rf"{inject_mode}__(.+)\.txt$")
        pattern = os.path.join(
            out_dir,
            f"*__tgt{target_layer}__{source_set}__{inject_mode}__*.txt")
    all_files = glob.glob(pattern)
    models_seen = {}
    for path in all_files:
        base = os.path.basename(path)
        m = re.match(regex, base)
        if not m:
            continue
        model_slug, ent_key = m.group(1), m.group(2)
        # Residual-mode pattern is ambiguous: a block-mode file looks like
        # "<model>__tgt6__canonical__block__diana.txt" and would match with
        # ent_key='block__diana'. Filter so ent_key must be a known entity.
        if ent_key not in ENTITIES:
            continue
        models_seen.setdefault(model_slug, set()).add(ent_key)
    complete = [m for m, ents in models_seen.items()
                if all(e in ents for e in ENTITIES)]
    if not complete:
        print(f"[plot] no complete model runs found for "
              f"target_layer={target_layer}, source_set={source_set}")
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
            if inject_mode == "residual":
                fname = f"{model_slug}__tgt{target_layer}__{source_set}__{ent_key}.txt"
            else:
                fname = (f"{model_slug}__tgt{target_layer}__{source_set}__"
                         f"{inject_mode}__{ent_key}.txt")
            rows = _load_run_file(os.path.join(out_dir, fname))
            xs = [x for x, _ in rows]
            ys = [_grade(t, CRITERIA[ent_key]) for _, t in rows]
            ax.plot(xs, ys, color=color, linewidth=1.2,
                    drawstyle="steps-mid", zorder=3)
            hit_xs = [x for x, y in zip(xs, ys) if y == 1]
            ax.scatter(hit_xs, [1] * len(hit_xs), color=color, s=8, zorder=4)
            ax.set_ylim(-0.15, 1.15)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["no", "yes"], fontsize=8)
            ax.set_xlim(-1, max(xs) + 1)
            ax.set_xticks([0, 10, 20, 30] if max(xs) >= 30 else [0, 10, 20])
            ax.tick_params(axis="x", labelsize=8)
            _style_axes(ax)
            if r == 0:
                title = ENTITY_TITLES[ent_key]
                src = SOURCE_SETS[source_set][ent_key]
                ax.set_title(f"{title}\n({src!r})", fontsize=9, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("source layer", fontsize=9)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=9)

    fig.suptitle(
        f"Patchscopes — fixed target layer L{target_layer}, "
        f"source set: {source_set}, inject: {inject_mode}",
        fontsize=10, y=0.98)
    plt.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.16,
                        wspace=0.45, hspace=0.55)

    suffix = "" if inject_mode == "residual" else f"_{inject_mode}"
    out_base = os.path.join(
        out_dir,
        f"patchscopes_tgt{target_layer}_{source_set}{suffix}_grid")
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
    src.add_argument("--model-tag", help="nanochat checkpoint tag")
    src.add_argument("--hf-model", help="HF causal-LM, e.g. EleutherAI/pythia-12b")
    ap.add_argument("--step", type=int, default=None,
                    help="nanochat step (last if omitted)")
    ap.add_argument("--target-layer", type=int, default=6,
                    help="Target block to inject into (default 6).")
    ap.add_argument("--source-set", choices=list(SOURCE_SETS), default="canonical",
                    help="Which set of source prompts to use.")
    ap.add_argument("--inject-mode", choices=["residual", "block"],
                    default="residual",
                    help="In both modes we REPLACE the target residual at "
                         "the x position. They differ in WHAT source vector we "
                         "use: "
                         "residual = each source layer's residual-stream "
                         "OUTPUT (the running residual after that block, "
                         "carries integrated context — standard patchscope). "
                         "block = each source block's CONTRIBUTION "
                         "(output - input at the source last token), the "
                         "delta that block alone added to the residual stream.")
    ap.add_argument("--target", default=TARGET_DEFAULT,
                    help="Few-shot description prompt; `x` placeholder is replaced.")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="./results/patchscopes")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip regenerating the comparison figure at the end.")
    args = ap.parse_args()

    device = torch.device(args.device)
    adapter = (_load_hf(args.hf_model, device) if args.hf_model
               else _load_nanochat(args.model_tag, args.step, device))
    n_layer = adapter["n_layer"]
    assert 0 <= args.target_layer < n_layer, \
        f"--target-layer {args.target_layer} out of range for n_layer={n_layer}"

    sources = SOURCE_SETS[args.source_set]
    print(f"Model: {adapter['name']}  n_layer={n_layer}  "
          f"target_layer={args.target_layer}  source_set={args.source_set}  "
          f"inject_mode={args.inject_mode}")
    print(f"Target prompt: {args.target!r}")
    os.makedirs(args.out_dir, exist_ok=True)

    # File naming: keep legacy name for residual mode (no extra segment); add
    # an inject-mode segment for any other mode.
    mode_seg = "" if args.inject_mode == "residual" else f"{args.inject_mode}__"
    for ent_key in ENTITIES:
        source = sources[ent_key]
        print(f"\n===== Source [{ent_key}]: {source!r} =====")
        results = run_one_source(adapter, source, args.target,
                                 args.target_layer, args.max_tokens,
                                 inject_mode=args.inject_mode)
        for src_layer, text in results:
            print(f"  S{src_layer:02d}->T{args.target_layer:02d}: {text}")
        out_path = os.path.join(
            args.out_dir,
            f"{adapter['name']}__tgt{args.target_layer}__"
            f"{args.source_set}__{mode_seg}{ent_key}.txt")
        with open(out_path, "w") as f:
            f.write(f"model\t{adapter['name']}\n")
            f.write(f"entity_key\t{ent_key}\n")
            f.write(f"source\t{source}\n")
            f.write(f"source_set\t{args.source_set}\n")
            f.write(f"inject_mode\t{args.inject_mode}\n")
            f.write(f"target\t{args.target}\n")
            f.write(f"target_layer\t{args.target_layer}\n")
            for src_layer, text in results:
                f.write(f"S{src_layer:02d}_T{args.target_layer:02d}\t{text}\n")
        print(f"  -> {out_path}")

    if not args.no_plot:
        print()
        regenerate_plot(args.out_dir, args.target_layer, args.source_set,
                        inject_mode=args.inject_mode)


if __name__ == "__main__":
    main()
