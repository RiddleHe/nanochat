"""Scan the local ClimbMix pretraining shards for the 5 patchscopes entities.

For each entity we count, across every document in every train shard:
  - mentions: total number of (case-insensitive) substring hits, summed over
    all surface-form variants for that entity;
  - docs: number of documents that contain at least one variant.

Purpose: confirm whether the entities probed in
`scripts/inspect/patchscope_few_shot.py` (Diana / Alexander / Ali / Jurassic
Park / NYC) actually occur in the data the nanochat models were trained on, and
how frequently. Pure provenance check — no model involved.
"""
import os
import glob
import json
import time
from multiprocessing import Pool

import pyarrow.parquet as pq

DATA_DIR = "/local-ssd/mh3897/base_data_climbmix"
VAL_SHARD = "shard_06542.parquet"  # pinned validation shard, excluded from train aggregate
OUT_JSON = "./results/patchscopes/climbmix_entity_counts.json"

# Surface-form variants per entity. The first form is the "canonical" phrase the
# patchscope source prompt uses; the rest are common alternates. All matched
# case-insensitively as plain substrings.
ENTITY_VARIANTS = {
    "diana": [
        "diana, princess of wales", "princess diana", "princess of wales",
        "lady diana", "diana spencer",
    ],
    "alexander": ["alexander the great"],
    "ali": ["muhammad ali"],
    "jurassic": ["jurassic park"],
    "nyc": ["new york city", "new york"],
}
ENTITIES = list(ENTITY_VARIANTS)


def scan_shard(path):
    """Return per-entity {mentions, docs}, plus n_docs and total_chars."""
    mentions = {e: 0 for e in ENTITIES}
    docs = {e: 0 for e in ENTITIES}
    per_variant = {v: 0 for e in ENTITIES for v in ENTITY_VARIANTS[e]}
    n_docs = 0
    total_chars = 0
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg).column("text").to_pylist()
        for t in texts:
            n_docs += 1
            total_chars += len(t)
            tl = t.lower()
            for e in ENTITIES:
                hit = False
                for v in ENTITY_VARIANTS[e]:
                    c = tl.count(v)
                    if c:
                        mentions[e] += c
                        per_variant[v] += c
                        hit = True
                if hit:
                    docs[e] += 1
    return {
        "path": os.path.basename(path),
        "mentions": mentions,
        "docs": docs,
        "per_variant": per_variant,
        "n_docs": n_docs,
        "total_chars": total_chars,
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    train_files = [f for f in files if os.path.basename(f) != VAL_SHARD]
    print(f"shards total={len(files)} train={len(train_files)} "
          f"(val={VAL_SHARD} excluded)")

    t0 = time.time()
    agg_mentions = {e: 0 for e in ENTITIES}
    agg_docs = {e: 0 for e in ENTITIES}
    agg_variant = {v: 0 for e in ENTITIES for v in ENTITY_VARIANTS[e]}
    n_docs = 0
    total_chars = 0
    done = 0

    with Pool(processes=64) as pool:
        for r in pool.imap_unordered(scan_shard, train_files):
            for e in ENTITIES:
                agg_mentions[e] += r["mentions"][e]
                agg_docs[e] += r["docs"][e]
            for v, c in r["per_variant"].items():
                agg_variant[v] += c
            n_docs += r["n_docs"]
            total_chars += r["total_chars"]
            done += 1
            if done % 100 == 0 or done == len(train_files):
                dt = time.time() - t0
                print(f"[{done}/{len(train_files)}] {dt:.0f}s "
                      f"docs={n_docs:,}", flush=True)

    # ClimbMix tokenizes at ~ chars/4.2 for English text; report a rough token
    # estimate so frequencies can be read as "per-N-tokens".
    est_tokens = total_chars / 4.2
    out = {
        "n_train_shards": len(train_files),
        "n_docs": n_docs,
        "total_chars": total_chars,
        "est_tokens": est_tokens,
        "entity_variants": ENTITY_VARIANTS,
        "mentions": agg_mentions,
        "docs": agg_docs,
        "doc_fraction": {e: agg_docs[e] / n_docs for e in ENTITIES},
        "per_variant_mentions": agg_variant,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n==== RESULTS ====")
    print(f"docs={n_docs:,}  est_tokens={est_tokens/1e9:.1f}B")
    for e in ENTITIES:
        print(f"  {e:10s} docs={agg_docs[e]:>9,} "
              f"({100*agg_docs[e]/n_docs:.4f}% of docs)  "
              f"mentions={agg_mentions[e]:>10,}")
    print("  per-variant mentions:")
    for v, c in agg_variant.items():
        print(f"    {v:32s} {c:>10,}")
    print(f"\nwrote {OUT_JSON}  ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
