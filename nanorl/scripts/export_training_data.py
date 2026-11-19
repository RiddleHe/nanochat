"""Export HuggingFace datasets to nanorl JSONL (RLExample schema).

Usage
-----
Export DAPO-Math-17k-Processed (English + Chinese, default config):

    python export_training_data.py dapo -o dapo_math.jsonl

Export only the English subset:

    python export_training_data.py dapo -o dapo_math_en.jsonl --config en
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer in \\boxed{}."
)


def export_dapo_math(
    output: Path,
    dataset_id: str = "open-r1/DAPO-Math-17k-Processed",
    config_name: str = "all",
    split: str = "train",
) -> int:
    logger.info(f"{dataset_id=}, {config_name=}, {split=}")
    ds = load_dataset(dataset_id, config_name, split=split)

    records: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        row: Mapping[str, Any]
        prompt_text = (row.get("prompt") or "").strip()
        reward_model = row.get("reward_model") or {}
        top_extra = row.get("extra_info") or {}
        nested_extra = reward_model.get("extra_info") or {}
        raw_id = top_extra.get("index") or nested_extra.get("index")
        example_id = f"dapo_math/{raw_id}" if raw_id and str(raw_id).strip() else f"dapo_math/row_{i}"

        ground_truth = ((row.get("solution") or "").strip()
                        or (reward_model.get("ground_truth") or "").strip())

        if not prompt_text or not ground_truth:
            logger.warning(f"Skipping row {i} ({example_id}): empty prompt or ground_truth")
            continue

        meta: dict[str, Any] = {"hf_config": config_name}
        for key in ("data_source", "ability"):
            val = row.get(key)
            if val is not None:
                meta[key] = val

        records.append({
            "id": example_id,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "meta": meta,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"{output=}, num_lines={len(records)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export HuggingFace datasets to nanorl JSONL.",
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    dapo = subparsers.add_parser("dapo", help="Export open-r1/DAPO-Math-17k-Processed.")
    dapo.add_argument("-o", "--output", required=True, type=Path, help="Output .jsonl path.")
    dapo.add_argument("--dataset-id", default="open-r1/DAPO-Math-17k-Processed", help="HF dataset id.")
    dapo.add_argument("--config", default="all", help="Dataset config (default: all).")
    dapo.add_argument("--split", default="train", help="Split name (default: train).")

    args = parser.parse_args(argv)
    if args.dataset == "dapo":
        return export_dapo_math(args.output, args.dataset_id, args.config, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
