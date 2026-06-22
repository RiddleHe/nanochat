"""Merge multiple probe shards into one dir.

Concatenates grads.npy + meta.jsonl, then re-buckets entropy GLOBALLY over the
combined distribution (so bin labels are consistent across shards).

Usage:
  python merge_shards.py OUT_DIR SHARD_DIR1 SHARD_DIR2 ...
"""
import json
import os
import sys
import numpy as np


def bucket_of(edges, value):
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if lo <= value <= hi:
                return i
        elif lo <= value < hi:
            return i
    return len(edges) - 2


def main():
    if len(sys.argv) < 3:
        print("usage: merge_shards.py OUT_DIR SHARD_DIR1 SHARD_DIR2 ...")
        sys.exit(1)
    out_dir = sys.argv[1]
    shards = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)

    all_grads = []
    all_meta = []
    manifests = []
    for d in shards:
        all_grads.append(np.load(os.path.join(d, "grads.npy")))
        with open(os.path.join(d, "meta.jsonl")) as f:
            shard_meta = [json.loads(l) for l in f]
        # Namespace prompt_idx per-shard so collisions across seeds can't merge
        # two different prompts. Use shard index as a tag.
        shard_tag = len(manifests)
        for m in shard_meta:
            m["prompt_idx"] = (shard_tag, int(m["prompt_idx"]))
        all_meta.extend(shard_meta)
        manifests.append(json.load(open(os.path.join(d, "manifest.json"))))

    grads = np.concatenate(all_grads, axis=0)
    all_ents = np.array([m["entropy"] for m in all_meta], dtype=np.float64)
    n_bins = manifests[0]["bins"]
    edges = np.quantile(all_ents, np.linspace(0.0, 1.0, n_bins + 1)).tolist()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    for i, m in enumerate(all_meta):
        m["bin"] = bucket_of(edges, m["entropy"])
        m["row"] = i

    np.save(os.path.join(out_dir, "grads.npy"), grads)
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m) + "\n")
    ref = manifests[0].copy()
    ref["entropy_edges"] = edges
    ref["num_positions"] = len(all_meta)
    ref["num_rollouts"] = sum(m.get("num_rollouts", 0) for m in manifests)
    ref["num_prompts"] = sum(m.get("num_prompts", 0) for m in manifests)
    ref["num_correct_rollouts"] = sum(m.get("num_correct_rollouts", 0) for m in manifests)
    ref["shards"] = shards
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(ref, f, indent=2)
    print(f"[merge] {len(shards)} shards -> {grads.shape} positions, {out_dir}")


if __name__ == "__main__":
    main()
