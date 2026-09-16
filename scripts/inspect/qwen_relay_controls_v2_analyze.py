"""Read the two v2 controls and state whether the two-stage claims survive them.

C1 block_only vs stage2_block / relay_block: how much of the blocked-condition
   donor flips is rescue rather than the model losing its own answer.
C2 mismatch_transfer vs stage1_transfer: is the transfer score entity-specific?
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(p):
    return [json.loads(l) for l in Path(p).open()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--exclude-templates", default="dialogue")
    args = ap.parse_args()
    excl = set(args.exclude_templates.split(",")) if args.exclude_templates else set()
    ctl = [r for r in load(Path(args.controls) / "rows.jsonl") if r["template_name"] not in excl]
    main_rows = [r for r in load(Path(args.main) / "rows.jsonl") if r["template_name"] not in excl]

    # ---- C1
    bo_keep = defaultdict(list)   # recipient answer survives block alone
    bo_donor = defaultdict(list)  # donor appears with no patch at all (should be ~0)
    for r in ctl:
        if r["condition"] == "block_only":
            bo_keep[r["relay_layer_t"]].append(r["output_category"] == "expected_only")
            bo_donor[r["relay_layer_t"]].append(r["output_category"] == "alternative_only")
    s2, s2b, rl, rlb = (defaultdict(list) for _ in range(4))
    for r in main_rows:
        ok = r.get("output_category") == "expected_only"
        {"stage2_direct": s2, "stage2_block": s2b,
         "relay": rl, "relay_block": rlb}.get(r["condition"], {}).setdefault(
            r.get("relay_layer_t") if r["condition"].startswith("stage2") else r.get("source_layer_s"), []).append(ok)

    print("=== C1  block_only baseline (no patch at all) ===")
    print(f"{'T':>4} {'recip kept':>11} {'donor appears':>14} | {'s2_direct':>10} {'s2_block':>9} | {'relay':>7} {'relay_blk':>10}")
    for T in sorted(bo_keep):
        def g(d): return f"{np.mean(d[T]):.2f}" if T in d and d[T] else "  -  "
        print(f"{T:4d} {np.mean(bo_keep[T]):11.2f} {np.mean(bo_donor[T]):14.2f} | "
              f"{g(s2):>10} {g(s2b):>9} | {g(rl):>7} {g(rlb):>10}")
    late = [T for T in bo_keep if T >= 28]
    mid = [T for T in bo_keep if 8 <= T <= 27]
    if late:
        bo_late = float(np.mean([np.mean(bo_donor[T]) for T in late]))
        s2b_late = float(np.mean([np.mean(s2b[T]) for T in late if T in s2b and s2b[T]]))
        print(f"\nlate T (>=28): donor appears WITHOUT patch = {bo_late:.2f}; "
              f"with patch + block = {s2b_late:.2f}  -> rescue margin {s2b_late - bo_late:+.2f}")
        print("  reading: margin large => the blocked-condition flips are genuine rescue, "
              "not an artifact of blocking")
    if mid:
        print(f"mid T (8-27): recipient answer kept under block alone = "
              f"{float(np.mean([np.mean(bo_keep[T]) for T in mid])):.2f} "
              "(low => blocking alone already destroys the answer; mid-T conclusions rest on "
              "stage-1/natural-divergence, not on this control)")

    # ---- C2
    mm = defaultdict(list)
    for r in ctl:
        if r["condition"] == "mismatch_transfer":
            mm[r["source_layer_s"]].append(r["transfer_proj"])
    tr = defaultdict(list)
    for r in main_rows:
        if r["condition"] == "stage1_transfer" and r["relay_layer_t"] - r["source_layer_s"] == 4:
            tr[r["source_layer_s"]].append(r["transfer_proj"])
    print("\n=== C2  transfer score: matched donor vs unrelated donor (same axis) ===")
    print(f"{'S':>4} {'matched':>9} {'mismatched':>12} {'gap':>8}")
    for S in sorted(mm):
        if S in tr:
            a, b = float(np.mean(tr[S])), float(np.mean(mm[S]))
            print(f"{S:4d} {a:9.3f} {b:12.3f} {a-b:8.3f}")
    band = [S for S in mm if 17 <= S <= 22]
    if band:
        a = float(np.mean([np.mean(tr[S]) for S in band if S in tr]))
        b = float(np.mean([np.mean(mm[S]) for S in band]))
        print(f"\nS=17-22 band: matched={a:.3f} mismatched={b:.3f} -> "
              + ("SPECIFIC (metric tracks this donor's identity)" if a - b >= 0.3
                 else "NOT SPECIFIC -- score largely reflects generic perturbation; stage-1 claim weakens"))


if __name__ == "__main__":
    main()
