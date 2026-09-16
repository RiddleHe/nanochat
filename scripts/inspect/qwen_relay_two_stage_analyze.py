"""Summarize + plot a qwen_relay_two_stage run.

Outputs (in the run dir):
  summary.json          per-S / per-T aggregates + pre-registered verdict
  two_stage_curves.png  stage-1 transfer(S), stage-2 survival(T), relay flip(S), predicted product
  stage1_plane.png      2D heatmap of stage-1 transfer over (S, T)
Usage: python -m scripts.inspect.qwen_relay_two_stage_analyze --run results/relay_two_stage
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED, GRID = "#202124", "#e8eaed", "#9aa0a6", "#3c4043"
BLUE, YELLOW, PURPLE = "#8ab4f8", "#fdd663", "#c58af9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exclude-templates", default="dialogue")
    args = ap.parse_args()
    run = Path(args.run)
    excl = set(args.exclude_templates.split(",")) if args.exclude_templates else set()
    rows = [json.loads(l) for l in (run / "rows.jsonl").open()]
    rows = [r for r in rows if r["template_name"] not in excl]
    meta = json.loads((run / "metadata.json").read_text())
    L = meta["n_layers"]; w = meta["relay_width"]

    base_ok = defaultdict(int); base_n = defaultdict(int)
    for r in rows:
        if r["condition"] in ("donor_baseline", "recipient_baseline"):
            base_n[r["condition"]] += 1
            base_ok[r["condition"]] += r["output_category"] == "expected_only"

    s1 = defaultdict(list); s1rec = defaultdict(list); plane = defaultdict(list)
    for r in rows:
        if r["condition"] == "stage1_transfer":
            plane[(r["source_layer_s"], r["relay_layer_t"])].append(r["transfer_proj"])
            if r["relay_layer_t"] - r["source_layer_s"] == w:
                s1[r["source_layer_s"]].append(r["transfer_proj"])
                if r["ll_recovery"] is not None:
                    s1rec[r["source_layer_s"]].append(r["ll_recovery"])
    s2 = defaultdict(list)
    for r in rows:
        if r["condition"] == "stage2_direct":
            s2[r["relay_layer_t"]].append(r["output_category"] == "expected_only")
    relay = defaultdict(list)
    for r in rows:
        if r["condition"] == "relay":
            relay[r["source_layer_s"]].append(r["output_category"] == "expected_only")

    S_ax = sorted(s1); T_ax = sorted(s2)
    s1m = np.array([np.mean(s1[s]) for s in S_ax])
    s1r = np.array([np.mean(s1rec[s]) if s1rec[s] else np.nan for s in S_ax])
    s2m = np.array([np.mean(s2[t]) for t in T_ax])
    rl = np.array([np.mean(relay[s]) for s in S_ax])
    def s2_at(t):  # nearest measured T when stage 2 was run on a subset of layers
        tt = min(T_ax, key=lambda x: abs(x - t))
        return np.mean(s2[tt])
    pred = np.array([np.clip(np.mean(s1[s]), 0, 1) * s2_at(s + w) for s in S_ax])

    mid = [s for s in S_ax if 8 <= s <= 20]
    late = [s for s in S_ax if s >= L - 8]
    s1_mid = float(np.mean([np.mean(s1[s]) for s in mid])); s1_late = float(np.mean([np.mean(s1[s]) for s in late]))
    s2_mid = float(np.mean([np.mean(s2[t]) for t in T_ax if 8 <= t <= 20]))
    s1_mid_sub = s1_mid >= 0.3
    s2_mid_high = s2_mid >= 0.6
    verdict = {(False, True): "A: late transfer real; overwrite irrelevant",
               (False, False): "B: late transfer real AND overwrite exists",
               (True, True): "C: contradiction -> inspect pass-3 transplant",
               (True, False): "D: late peak is an overwrite artifact"}[(s1_mid_sub, s2_mid_high)]

    summary = {"n_cases": base_n["donor_baseline"], "baseline_ok": dict(base_ok), "baseline_n": dict(base_n),
               "stage1_transfer_by_S": {s: float(np.mean(s1[s])) for s in S_ax},
               "stage1_llrecovery_by_S": {s: (float(np.mean(s1rec[s])) if s1rec[s] else None) for s in S_ax},
               "stage2_survival_by_T": {t: float(np.mean(s2[t])) for t in T_ax},
               "relay_flip_by_S": {s: float(np.mean(relay[s])) for s in S_ax},
               "predicted_relay_by_S": {s: float(p) for s, p in zip(S_ax, pred)},
               "stage1_mid_mean(S8-20)": s1_mid, "stage1_late_mean": s1_late,
               "stage2_mid_mean(T8-20)": s2_mid, "verdict": verdict}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"cases={summary['n_cases']}  baselines ok: {dict(base_ok)} / {dict(base_n)}")
    print(f"{'S':>3} {'stage1_transfer':>16} {'ll_recovery':>12} {'relay_flip':>11} {'pred(s1*s2)':>12}")
    for s, a, b, c, d in zip(S_ax, s1m, s1r, rl, pred):
        print(f"{s:3d} {a:16.3f} {b:12.3f} {c:11.2f} {d:12.2f}")
    print(f"\n{'T':>3} {'stage2_survival':>16}")
    for t, v in zip(T_ax, s2m):
        print(f"{t:3d} {v:16.2f}")
    print(f"\nstage1 mid(S8-20)={s1_mid:.3f} late={s1_late:.3f} | stage2 mid(T8-20)={s2_mid:.2f}")
    print("VERDICT:", verdict)

    # curves
    fig, ax = plt.subplots(figsize=(11, 6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(S_ax, s1m, "-", color=BLUE, lw=2.4, label="stage 1: transfer score at T=S+%d (no generation)" % w)
    ax.plot(T_ax, s2m, "-", color=YELLOW, lw=2.4, label="stage 2: direct donor-state survival vs T")
    ax.plot(S_ax, rl, "-", color=PURPLE, lw=2.4, label="relay flip rate (original readout) vs S")
    ax.plot(S_ax, pred, "--", color=PURPLE, lw=1.6, alpha=0.7, label="predicted relay = stage1 x stage2")
    ax.set_xlabel("layer (S for stage 1 / relay; T for stage 2)", color=MUTED); ax.set_ylabel("score / rate", color=MUTED)
    ax.set_ylim(-0.1, 1.05); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for t in leg.get_texts(): t.set_color(FG)
    fig.suptitle("Relay decomposed: where does transfer happen vs. can a signal survive?", color=FG, x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.93, f"Qwen3-8B-Base · {summary['n_cases']} cases · verdict: {verdict}", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(run / "two_stage_curves.png", dpi=180, facecolor=BG); plt.close(fig)

    # plane
    M = np.full((L, L), np.nan)
    for (s, t), v in plane.items(): M[s, t] = np.mean(v)
    fig, ax = plt.subplots(figsize=(7.5, 6.5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    im = ax.imshow(M.T, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("source layer S", color=MUTED); ax.set_ylabel("relay layer T", color=MUTED)
    ax.tick_params(colors=MUTED, length=0); cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(colors=MUTED)
    cb.set_label("stage-1 transfer score", color=MUTED)
    fig.suptitle("Stage-1 transfer over the full (S, T) plane", color=FG, x=0.07, ha="left")
    fig.tight_layout(); fig.savefig(run / "stage1_plane.png", dpi=180, facecolor=BG); plt.close(fig)
    print("wrote", run / "two_stage_curves.png", run / "stage1_plane.png")


if __name__ == "__main__":
    main()
