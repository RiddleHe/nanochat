"""Windowed target-layer patchscope for target-suffix prompts."""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect.patchscope_few_shot_target_suffix import (  # noqa: E402
    CRITERIA,
    ENTITIES,
    ENTITY_TITLES,
    SOURCE_SETS,
    TARGET_DEFAULT,
    TARGET_PLACEHOLDER_DEFAULT,
    _grade,
    _load_hf,
    _load_nanochat,
    build_target_with_suffix,
    capture_source_hiddens,
)


CMAP = ListedColormap(["#f7f9fc", "#0f766e"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], CMAP.N)

STRICT_TERMS = {
    "diana": ["diana", "princess of wales"],
    "alexander": ["alexander the great", "alexander"],
    "ali": ["muhammad ali"],
    "jurassic": ["jurassic park"],
    "nyc": ["new york city", "nyc"],
}
CATEGORY_TERMS = {
    "diana": [
        "british royal", "royal family", "prince charles", "prince william",
        "prince harry", "duke of cambridge", "duchess of cornwall",
        "wife of prince",
    ],
    "alexander": [
        "macedon", "macedonia", "macedonian", "greek king", "greek general",
        "ancient greek", "hellenistic", "323 bc", "356 bc",
    ],
    "ali": ["boxer", "boxing", "boxers", "heavyweight", "wbc", "wba", "ibf"],
    "jurassic": [
        "spielberg", "crichton", "science fiction film", "sci-fi film",
        "novel by", "1993 ", "1994 ", "lost world", "dinosaur",
        "american film", "american movie", "theme park",
    ],
    "nyc": [
        "city in the united states", "city in the state of new york",
        "u.s. state", "us state", "manhattan", "brooklyn",
    ],
}


def parse_window_starts(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_windows(n_layer, size, stride, starts_text):
    starts = parse_window_starts(starts_text) if starts_text else list(range(0, n_layer, stride))
    windows = []
    for start in starts:
        end = start + size - 1
        if start < 0 or start >= n_layer or end >= n_layer:
            raise ValueError(
                f"target window {start}-{end} out of range for n_layer={n_layer}")
        windows.append((start, end))
    return windows


def patched_generate_window(adapter, tgt_ids, target_window, tgt_pos, src_vec,
                            max_tokens, normalize=False):
    handles = []

    def make_hook():
        def patch_hook(_mod, _inp, out):
            is_tuple = isinstance(out, tuple)
            x = out[0] if is_tuple else out
            if x.shape[1] <= tgt_pos:
                return out
            new_x = x.clone()
            v = src_vec.to(new_x.dtype)
            if normalize:
                orig_norm = new_x[:, tgt_pos, :].norm(dim=-1, keepdim=True)
                v = v * (orig_norm / (v.norm() + 1e-8))
            new_x[:, tgt_pos, :] = v
            return (new_x,) + out[1:] if is_tuple else new_x
        return patch_hook

    try:
        for layer in range(target_window[0], target_window[1] + 1):
            handles.append(adapter["blocks"][layer].register_forward_hook(make_hook()))
        gen = adapter["generate_tokens"](tgt_ids, max_tokens)
        return adapter["decode"](gen)
    finally:
        for handle in handles:
            handle.remove()


def run_one_source(adapter, source, target, target_placeholder, target_suffix,
                   target_query, windows, max_tokens, normalize=False):
    target_before_x, target_with_x, target_full = build_target_with_suffix(
        target, target_placeholder, target_suffix, target_query)
    src_ids = adapter["encode"](source)
    target_with_x_ids = adapter["encode"](target_with_x)
    tgt_ids = adapter["encode"](target_full)
    assert len(src_ids) >= 1, f"Source must tokenize to >=1 token: {source!r}"
    assert len(target_with_x_ids) >= 1, (
        f"Target with placeholder must tokenize to >=1 token: {target_with_x!r}")
    assert len(target_with_x_ids) <= len(tgt_ids), (
        "Target-with-placeholder tokenization is longer than full target; check suffix/query")
    assert len(tgt_ids) >= 2, f"Target needs >=2 tokens; got {len(tgt_ids)}"
    if len(src_ids) < 2:
        src_ids = [src_ids[0]] + src_ids
    src_pos = len(src_ids) - 1
    tgt_pos = len(target_with_x_ids) - 1
    src_vecs = capture_source_hiddens(adapter, src_ids, src_pos)

    rows = []
    for src_layer in range(adapter["n_layer"]):
        for start, end in windows:
            text = patched_generate_window(
                adapter, tgt_ids, (start, end), tgt_pos, src_vecs[src_layer],
                max_tokens, normalize=normalize)
            rows.append({
                "source_layer": src_layer,
                "target_window_start": start,
                "target_window_end": end,
                "generated_text": text.lstrip().split("\n")[0],
            })
    return rows, {
        "target_before_x": target_before_x,
        "target_with_x": target_with_x,
        "target_full": target_full,
        "target_suffix": target_suffix,
        "target_query": target_query,
        "target_placeholder": target_placeholder,
        "tgt_pos": tgt_pos,
        "tgt_token_id": tgt_ids[tgt_pos],
        "tgt_token": adapter["decode"]([tgt_ids[tgt_pos]]),
        "target_token_count": len(tgt_ids),
        "target_with_x_token_count": len(target_with_x_ids),
    }


def strict_hit(ent, text):
    t = text.lower()
    return int(any(term in t for term in STRICT_TERMS[ent]))


def category_hit(ent, text):
    t = text.lower()
    return int(any(term in t for term in CATEGORY_TERMS[ent]))


def write_entity_file(path, metadata, rows):
    with open(path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}\t{value}\n")
        for row in rows:
            f.write(
                f"S{row['source_layer']:02d}_W"
                f"{row['target_window_start']:02d}-{row['target_window_end']:02d}"
                f"\t{row['generated_text']}\n")


def plot_entity(folder, ent, frame_name, windows, rows):
    n_source = max(row["source_layer"] for row in rows) + 1
    mat = np.zeros((n_source, len(windows)), dtype=float)
    for row in rows:
        win_idx = windows.index((row["target_window_start"], row["target_window_end"]))
        mat[row["source_layer"], win_idx] = _grade(row["generated_text"], CRITERIA[ent])

    hits = int(mat.sum())
    plots_dir = folder / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")
    im = ax.imshow(mat, origin="lower", aspect="auto", interpolation="nearest",
                   cmap=CMAP, norm=NORM)
    ax.set_xlabel("target window")
    ax.set_ylabel("source layer")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"{s}-{e}" for s, e in windows], rotation=30, ha="right")
    ax.set_yticks(range(0, n_source, 5))
    ax.set_title(f"{frame_name}\n{ENTITY_TITLES[ent]} | hits {hits}/{mat.size}",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["miss", "hit"])
    fig.tight_layout()

    png = plots_dir / f"{ent}__window_heatmap.png"
    pdf = plots_dir / f"{ent}__window_heatmap.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def write_summaries(folder, all_rows, windows):
    summary_path = folder / "window_heatmap_summary.csv"
    scoring_path = folder / "window_strict_vs_category_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["entity", "target_window", "hits", "total"])
        writer.writeheader()
        for ent in ENTITIES:
            for start, end in windows:
                subset = [
                    r for r in all_rows[ent]
                    if r["target_window_start"] == start and r["target_window_end"] == end
                ]
                writer.writerow({
                    "entity": ent,
                    "target_window": f"{start}-{end}",
                    "hits": sum(_grade(r["generated_text"], CRITERIA[ent]) for r in subset),
                    "total": len(subset),
                })

    with scoring_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "entity", "target_window", "strict_entity_name_hits",
                "category_description_hits", "original_criteria_hits", "total",
            ],
        )
        writer.writeheader()
        for ent in ENTITIES:
            for start, end in windows:
                subset = [
                    r for r in all_rows[ent]
                    if r["target_window_start"] == start and r["target_window_end"] == end
                ]
                writer.writerow({
                    "entity": ent,
                    "target_window": f"{start}-{end}",
                    "strict_entity_name_hits": sum(strict_hit(ent, r["generated_text"])
                                                   for r in subset),
                    "category_description_hits": sum(category_hit(ent, r["generated_text"])
                                                     for r in subset),
                    "original_criteria_hits": sum(_grade(r["generated_text"], CRITERIA[ent])
                                                  for r in subset),
                    "total": len(subset),
                })
    return summary_path, scoring_path


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-tag")
    src.add_argument("--hf-model")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--target-window-size", type=int, default=6)
    ap.add_argument("--target-window-stride", type=int, default=6)
    ap.add_argument("--target-window-starts", default="")
    ap.add_argument("--target-suffix", default=" refers to")
    ap.add_argument("--target-query", default="")
    ap.add_argument("--target-placeholder", default=TARGET_PLACEHOLDER_DEFAULT)
    ap.add_argument("--source-set", choices=list(SOURCE_SETS), default="canonical")
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    adapter = (_load_hf(args.hf_model, device) if args.hf_model
               else _load_nanochat(args.model_tag, args.step, device))
    windows = build_windows(
        adapter["n_layer"], args.target_window_size, args.target_window_stride,
        args.target_window_starts)
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    print(f"Model: {adapter['name']}  n_layer={adapter['n_layer']}")
    print(f"Target suffix: {args.target_suffix!r}")
    print(f"Target query: {args.target_query!r}")
    print(f"Normalize: {args.normalize}")
    print(f"Target windows: {windows}")
    print(f"Max tokens: {args.max_tokens}")

    sources = SOURCE_SETS[args.source_set]
    all_rows = {}
    for ent_key in ENTITIES:
        source = sources[ent_key]
        print(f"\n===== Source [{ent_key}]: {source!r} =====")
        rows, target_info = run_one_source(
            adapter, source, args.target, args.target_placeholder,
            args.target_suffix, args.target_query, windows, args.max_tokens,
            normalize=args.normalize)
        all_rows[ent_key] = rows
        metadata = {
            "model": adapter["name"],
            "entity_key": ent_key,
            "source": source,
            "source_set": args.source_set,
            "normalize": args.normalize,
            "target_window_size": args.target_window_size,
            "target_window_stride": args.target_window_stride,
            "target_windows": ",".join(f"{s}-{e}" for s, e in windows),
            "max_tokens": args.max_tokens,
            **target_info,
        }
        out_path = out_dir / (
            f"{adapter['name']}__tgtWIN{args.target_window_size}"
            f"__{args.source_set}__{'norm__' if args.normalize else ''}{ent_key}.txt")
        write_entity_file(out_path, metadata, rows)
        print(f"  target_full: {target_info['target_full']!r}")
        print(f"  chosen tgt_pos: {target_info['tgt_pos']}")
        print(f"  decoded target token at tgt_pos: {target_info['tgt_token']!r}")
        print(f"  -> {out_path} ({len(rows)} rows)")
        for start, end in windows:
            print(f"  window {start:02d}-{end:02d} done")

    for ent_key, rows in all_rows.items():
        png, pdf = plot_entity(out_dir, ent_key, "x refers to, 6-layer windows",
                               windows, rows)
        print(f"[plot] {ent_key}: {png} {pdf}")
    summary_path, scoring_path = write_summaries(out_dir, all_rows, windows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {scoring_path}")


if __name__ == "__main__":
    main()
