"""Export unique completion strings for manual classification.

Follows qwen_entity_relay_fixed_window.md: the automatic name-presence fields are
rough diagnostics; the reported result must come from human classification of each
unique string into patched_only / original_only / both / neither, decided WITHOUT
seeing the layer or span. This writes the strings alone (shuffled, deduplicated,
with occurrence counts), plus a mapping file so labels can be joined back.
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conditions", default="relay,stage2_direct,stage2_block,relay_block")
    ap.add_argument("--exclude-templates", default="dialogue")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run); conds = set(args.conditions.split(","))
    excl = set(args.exclude_templates.split(",")) if args.exclude_templates else set()
    rows = [json.loads(l) for l in (run / "rows.jsonl").open()]
    rows = [r for r in rows if r["condition"] in conds and r["template_name"] not in excl]

    counts = Counter(r["completion"] for r in rows)
    ctx = defaultdict(set)
    for r in rows:
        ctx[r["completion"]].add((r["donor"], r["recipient"]))
    uniq = sorted(counts)
    random.Random(0).shuffle(uniq)
    out = Path(args.out or run / "manual_labeling")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "to_label.jsonl").open("w") as f:
        for i, s in enumerate(uniq):
            f.write(json.dumps({
                "id": i, "completion": s, "n_rows": counts[s],
                "candidate_names": sorted({n for pair in ctx[s] for n in pair}),
                "label": "", "_options": "patched_only|original_only|both|neither",
            }, ensure_ascii=False) + "\n")
    (out / "README.txt").write_text(
        "Fill the 'label' field of every row in to_label.jsonl with one of:\n"
        "  patched_only | original_only | both | neither\n"
        "Decide from the string alone. 'candidate_names' lists the donor/recipient names\n"
        "that appeared in rows with this string (donor vs recipient order is NOT given,\n"
        "so the layer/condition cannot bias the decision).\n"
        f"{len(uniq)} unique strings cover {sum(counts.values())} rows.\n"
        "Then run: python -m scripts.inspect.export_relay_labeling --run <run> --apply\n")
    print(f"wrote {out/'to_label.jsonl'}: {len(uniq)} unique strings covering {sum(counts.values())} rows")
    top = counts.most_common(12)
    print("\nmost frequent strings (for a quick sense of the workload):")
    for s, n in top:
        print(f"  {n:5d}x  {s!r}")


if __name__ == "__main__":
    main()
