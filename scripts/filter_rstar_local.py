"""
Filter the official microsoft/rStar-Coder seed_testcase split into a local
JSONL cache that can later be converted into nanochat's canonical RL JSONL.

This script is intentionally local-only:
- streams from the official HF dataset
- filters / subsamples tests
- writes simplified rows to bucket files
- shuffles within buckets and concatenates them to one final JSONL

It does NOT write parquet or push anything to HF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time

from datasets import load_dataset


MIN_TESTS = 5
MAX_TESTS = 20


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _stable_bucket(question_id: str, num_buckets: int) -> int:
    digest = hashlib.md5(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_buckets


def _stable_rng(question_id: str) -> random.Random:
    digest = hashlib.md5(question_id.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _classify_kind(row: dict) -> str:
    starter_code = (row.get("starter_code") or "").strip()
    func_name = (row.get("func_name") or "").strip()
    if starter_code or func_name:
        return "code_call_based"
    return "code_stdin_stdout"


def _open_bucket_writers(bucket_dir: str, num_buckets: int) -> list:
    os.makedirs(bucket_dir, exist_ok=True)
    writers = []
    for i in range(num_buckets):
        path = os.path.join(bucket_dir, f"bucket_{i:04d}.jsonl")
        writers.append(open(path, "w"))
    return writers


def _close_writers(writers: list) -> None:
    for w in writers:
        w.close()


def stream_filter_to_buckets(
    output_path: str,
    bucket_dir: str,
    num_buckets: int,
    seed: int,
) -> dict:
    ds = load_dataset(
        "microsoft/rStar-Coder",
        "seed_testcase",
        split="train",
        streaming=True,
    )

    writers = _open_bucket_writers(bucket_dir, num_buckets)
    stats = {
        "seen": 0,
        "kept": 0,
        "skipped_parse": 0,
        "skipped_too_few": 0,
        "subsampled": 0,
        "call_based": 0,
        "stdin_stdout": 0,
    }
    t0 = time.time()

    try:
        for row in ds:
            stats["seen"] += 1
            qid = row.get("question_id", f"row_{stats['seen']}")

            try:
                inputs = json.loads(row["inputs"])
                outputs = json.loads(row["outputs"])
            except (json.JSONDecodeError, TypeError) as exc:
                stats["skipped_parse"] += 1
                if stats["skipped_parse"] <= 10:
                    log(f"  SKIP(parse) {qid}: {exc}")
                continue

            n = len(inputs)
            if n != len(outputs):
                stats["skipped_parse"] += 1
                continue
            if n <= MIN_TESTS:
                stats["skipped_too_few"] += 1
                continue

            if n > MAX_TESTS:
                stats["subsampled"] += 1
                rng = _stable_rng(qid)
                idx = sorted(rng.sample(range(n), MAX_TESTS))
                inputs = [inputs[i] for i in idx]
                outputs = [outputs[i] for i in idx]

            inputs = [s if isinstance(s, str) else json.dumps(s) for s in inputs]
            outputs = [s if isinstance(s, str) else json.dumps(s) for s in outputs]

            kind = _classify_kind(row)
            if kind == "code_call_based":
                stats["call_based"] += 1
            else:
                stats["stdin_stdout"] += 1

            out_row = {
                "question_id": qid,
                "question": row["question"],
                "kind": kind,
                "starter_code": row.get("starter_code", "") or "",
                "func_name": row.get("func_name", "") or "",
                "class_name": row.get("class_name", "") or "",
                "inputs": inputs,
                "outputs": outputs,
                "n_tests": len(inputs),
            }

            bucket = _stable_bucket(qid, num_buckets)
            writers[bucket].write(json.dumps(out_row) + "\n")
            stats["kept"] += 1

            if stats["seen"] % 100 == 0:
                elapsed = time.time() - t0
                log(
                    f"  [{elapsed:.0f}s] processed {stats['seen']} rows, kept {stats['kept']}, "
                    f"skipped(parse={stats['skipped_parse']}, few={stats['skipped_too_few']}), "
                    f"subsampled={stats['subsampled']}, "
                    f"call={stats['call_based']}, stdio={stats['stdin_stdout']}"
                )
    finally:
        _close_writers(writers)

    bucket_ids = list(range(num_buckets))
    random.Random(seed).shuffle(bucket_ids)

    with open(output_path, "w") as out_f:
        emitted = 0
        for bucket in bucket_ids:
            bucket_path = os.path.join(bucket_dir, f"bucket_{bucket:04d}.jsonl")
            if not os.path.exists(bucket_path) or os.path.getsize(bucket_path) == 0:
                continue
            with open(bucket_path) as f:
                lines = f.readlines()
            random.Random(seed ^ bucket).shuffle(lines)
            out_f.writelines(lines)
            emitted += len(lines)
            log(f"  merged bucket {bucket:04d}, emitted {emitted} rows so far")

    stats["emitted"] = stats["kept"]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter official rStar seed_testcase into local JSONL")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Final local filtered JSONL path",
    )
    parser.add_argument(
        "--bucket-dir",
        type=str,
        required=True,
        help="Temporary directory for bucketed intermediate files",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=512,
        help="Number of bucket files used for coarse shuffle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used to shuffle bucket order and lines within each bucket",
    )
    parser.add_argument(
        "--keep-buckets",
        action="store_true",
        help="Keep intermediate bucket files after writing the final JSONL",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if os.path.exists(args.bucket_dir):
        raise FileExistsError(f"Refusing to reuse existing bucket dir: {args.bucket_dir}")

    log("Streaming official microsoft/rStar-Coder seed_testcase ...")
    stats = stream_filter_to_buckets(
        output_path=args.output,
        bucket_dir=args.bucket_dir,
        num_buckets=args.num_buckets,
        seed=args.seed,
    )

    if not args.keep_buckets:
        shutil.rmtree(args.bucket_dir)
        log(f"Removed bucket dir: {args.bucket_dir}")

    log("\n--- Filter stats ---")
    log(f"seen:             {stats['seen']}")
    log(f"skipped(parse):   {stats['skipped_parse']}")
    log(f"skipped(<=5):     {stats['skipped_too_few']}")
    log(f"subsampled(>20):  {stats['subsampled']}")
    log(f"kept:             {stats['kept']}")
    log(f"call_based:       {stats['call_based']}")
    log(f"stdin_stdout:     {stats['stdin_stdout']}")
    log(f"output:           {args.output}")


if __name__ == "__main__":
    main()
