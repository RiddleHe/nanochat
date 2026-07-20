"""No-patch prompt-only control for a shared patchscope target prompt."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect.patchscope_few_shot_target_suffix import (
    CRITERIA,
    ENTITIES,
    TARGET_DEFAULT,
    TARGET_PLACEHOLDER_DEFAULT,
    _grade,
    _load_hf,
    _load_nanochat,
    build_target_with_suffix,
)


def run_control(adapter, args):
    _, target_with_x, target_full = build_target_with_suffix(
        args.target,
        args.target_placeholder,
        args.target_suffix,
        args.target_query,
    )
    target_with_x_ids = adapter["encode"](target_with_x)
    target_ids = adapter["encode"](target_full)
    tgt_pos = len(target_with_x_ids) - 1
    tgt_token = adapter["decode"]([target_ids[tgt_pos]])

    gen_ids = adapter["generate_tokens"](target_ids, args.max_tokens)
    generated_text = adapter["decode"](gen_ids).lstrip().split("\n")[0]
    scores = {ent: _grade(generated_text, CRITERIA[ent]) for ent in ENTITIES}

    row = {
        "prompt": target_full,
        "generated_text": generated_text,
        "model": adapter["name"],
        "max_tokens": args.max_tokens,
        "target_suffix": args.target_suffix,
        "target_query": args.target_query,
        "target_placeholder": args.target_placeholder,
        "target_with_x": target_with_x,
        "tgt_pos": tgt_pos,
        "tgt_token_id": target_ids[tgt_pos],
        "tgt_token": tgt_token,
        "target_token_count": len(target_ids),
        "target_with_x_token_count": len(target_with_x_ids),
        "prompt_only": True,
    }
    for ent, score in scores.items():
        row[f"score_{ent}"] = score

    print(f"prompt: {target_full!r}")
    print(f"tgt_pos: {tgt_pos}  tgt_token: {tgt_token!r}")
    print(f"generated_text: {generated_text!r}")
    print(f"prompt_only_scores: {scores}")
    return row


def write_outputs(row, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "no_patch_target_prompt.json")
    csv_path = os.path.join(out_dir, "no_patch_target_prompt.csv")
    txt_path = os.path.join(out_dir, "no_patch_target_prompt.txt")

    with open(json_path, "w") as f:
        json.dump(row, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with open(txt_path, "w") as f:
        for key, value in row.items():
            f.write(f"{key}\t{value}\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--target-placeholder", default=TARGET_PLACEHOLDER_DEFAULT)
    ap.add_argument("--target-suffix", default="")
    ap.add_argument("--target-query", default="")
    args = ap.parse_args()

    if bool(args.hf_model) == bool(args.model_tag):
        raise SystemExit("Specify exactly one of --hf-model or --model-tag")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.hf_model:
        adapter = _load_hf(args.hf_model, device)
    else:
        adapter = _load_nanochat(args.model_tag, args.step, device)

    row = run_control(adapter, args)
    write_outputs(row, args.out_dir)


if __name__ == "__main__":
    main()
