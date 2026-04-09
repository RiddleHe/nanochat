"""
Prepare the rStar-Coder dataset into the canonical JSONL format consumed
by nanochat's RL pipeline (nanochat/rl_data.py).

Reads from HF: RiddleHe/rStar-Coder (preprocessed: test cases capped,
schema simplified, JSON blobs pre-parsed into proper columns).

Writes to: <base_dir>/data/rl/rstar_seed_train.jsonl

Usage:
    python -m scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B
    python -m scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B --limit 10
"""

from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer
from datasets import load_dataset

from nanochat.common import get_base_dir


def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1]
                    os.environ["HF_TOKEN"] = token
                    return token
    return None


def build_prompt(question: str, starter_code: str, tokenizer) -> str:
    user_content = question.strip()
    if starter_code:
        user_content += (
            "\n\nUse this starter code:\n```python\n"
            + starter_code.strip()
            + "\n```"
        )
    user_content += "\n\nProvide your complete solution as a single Python code block."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare rStar-Coder for RL")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HF tokenizer name (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N rows (for smoke testing)")
    parser.add_argument("--time-limit", type=float, default=4.0,
                        help="Per-test time limit in seconds")
    parser.add_argument("--memory-limit-mb", type=int, default=256,
                        help="Per-test memory limit in MB")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path")
    parser.add_argument("--dataset", type=str, default="RiddleHe/rStar-Coder",
                        help="HF dataset repo (default: RiddleHe/rStar-Coder)")
    args = parser.parse_args()

    hf_token = load_hf_token()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=hf_token)

    # Load preprocessed dataset (small enough to load directly)
    split = "train"
    if args.limit:
        split = f"train[:{args.limit}]"
    print(f"Loading {args.dataset} split={split} ...")
    ds = load_dataset(args.dataset, split=split, token=hf_token)
    print(f"Loaded {len(ds)} rows")

    # Output path
    if args.output:
        out_path = args.output
    else:
        base = get_base_dir()
        out_dir = os.path.join(base, "data", "rl")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "rstar_seed_train.jsonl")

    n_call_based = 0
    n_stdin_stdout = 0

    with open(out_path, "w") as f:
        for row in ds:
            kind = row["kind"]
            starter_code = row["starter_code"] or ""
            inputs = row["inputs"]    # already a list[str]
            outputs = row["outputs"]  # already a list[str]

            # For call-based, parse each element from JSON string to native
            if kind == "code_call_based":
                inputs_parsed = [json.loads(s) for s in inputs]
                outputs_parsed = [json.loads(s) for s in outputs]
                n_call_based += 1
            else:
                inputs_parsed = inputs
                outputs_parsed = outputs
                n_stdin_stdout += 1

            prompt_str = build_prompt(row["question"], starter_code, tokenizer)

            payload = {
                "inputs": inputs_parsed,
                "outputs": outputs_parsed,
                "time_limit_s": args.time_limit,
                "memory_limit_mb": args.memory_limit_mb,
            }
            if kind == "code_call_based":
                payload["fn_name"] = row["func_name"]
                if starter_code:
                    payload["starter_code"] = starter_code

            jsonl_row = {
                "id": f"rstar/{row['question_id']}",
                "prompt": prompt_str,
                "kind": kind,
                "payload": payload,
                "meta": {
                    "source": "rstar_seed",
                    "n_tests": row["n_tests"],
                },
            }
            f.write(json.dumps(jsonl_row) + "\n")

    n_emitted = n_call_based + n_stdin_stdout
    print(f"\n--- Summary ---")
    print(f"emitted {n_emitted} rows to {out_path}")
    print(f"  call_based:   {n_call_based}")
    print(f"  stdin_stdout: {n_stdin_stdout}")


if __name__ == "__main__":
    main()
