"""No-patch prompt-only control for entity-specific target suffix prompts."""
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
    SOURCE_SETS,
    TARGET_DEFAULT,
    TARGET_PLACEHOLDER_DEFAULT,
    _grade,
    _load_hf,
    _load_nanochat,
    build_target_with_suffix,
)


ENTITY_SPECIFIC_TARGETS = {
    "diana": {
        "target_suffix": " was known for her public life",
        "target_query": ". This person was",
    },
    "alexander": {
        "target_suffix": " was known for his military conquests",
        "target_query": ". This person was",
    },
    "ali": {
        "target_suffix": " was known for his boxing career",
        "target_query": ". This person was",
    },
    "jurassic": {
        "target_suffix": " is known as a popular film about dinosaurs",
        "target_query": ". This work was",
    },
    "nyc": {
        "target_suffix": " is known as a major city in the United States",
        "target_query": ". This place was",
    },
}


def run_control(adapter, args):
    rows = []
    for ent_key in ENTITIES:
        entity_phrase = SOURCE_SETS[args.source_set][ent_key]
        spec = ENTITY_SPECIFIC_TARGETS[ent_key]
        _, target_with_x, target_full = build_target_with_suffix(
            args.target,
            args.target_placeholder,
            spec["target_suffix"],
            spec["target_query"],
        )
        target_with_x_ids = adapter["encode"](target_with_x)
        target_ids = adapter["encode"](target_full)
        tgt_pos = len(target_with_x_ids) - 1
        tgt_token = adapter["decode"]([target_ids[tgt_pos]])

        gen_ids = adapter["generate_tokens"](target_ids, args.max_tokens)
        generated_text = adapter["decode"](gen_ids).lstrip().split("\n")[0]
        hit = _grade(generated_text, CRITERIA[ent_key])

        row = {
            "entity_key": ent_key,
            "entity_phrase": entity_phrase,
            "target_full": target_full,
            "generated_text": generated_text,
            "max_tokens": args.max_tokens,
            "model": adapter["name"],
            "hit": hit,
            "target_suffix": spec["target_suffix"],
            "target_query": spec["target_query"],
            "target_placeholder": args.target_placeholder,
            "target_with_x": target_with_x,
            "tgt_pos": tgt_pos,
            "tgt_token_id": target_ids[tgt_pos],
            "tgt_token": tgt_token,
            "target_token_count": len(target_ids),
            "target_with_x_token_count": len(target_with_x_ids),
        }
        rows.append(row)
        print(f"[{ent_key}] {entity_phrase!r}")
        print(f"  target_full: {target_full!r}")
        print(f"  tgt_pos: {tgt_pos}  tgt_token: {tgt_token!r}")
        print(f"  generated_text: {generated_text!r}")
        print(f"  hit: {hit}")
    return rows


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "no_patch_entity_specific_target.json")
    csv_path = os.path.join(out_dir, "no_patch_entity_specific_target.csv")
    txt_path = os.path.join(out_dir, "no_patch_entity_specific_target.txt")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w") as f:
        for row in rows:
            f.write(f"entity_key\t{row['entity_key']}\n")
            f.write(f"entity_phrase\t{row['entity_phrase']}\n")
            f.write(f"model\t{row['model']}\n")
            f.write(f"max_tokens\t{row['max_tokens']}\n")
            f.write(f"hit\t{row['hit']}\n")
            f.write(f"target_full\t{row['target_full']}\n")
            f.write(f"generated_text\t{row['generated_text']}\n\n")

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
    ap.add_argument("--source-set", choices=sorted(SOURCE_SETS), default="canonical")
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--target-placeholder", default=TARGET_PLACEHOLDER_DEFAULT)
    args = ap.parse_args()

    if bool(args.hf_model) == bool(args.model_tag):
        raise SystemExit("Specify exactly one of --hf-model or --model-tag")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.hf_model:
        adapter = _load_hf(args.hf_model, device)
    else:
        adapter = _load_nanochat(args.model_tag, args.step, device)

    rows = run_control(adapter, args)
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
