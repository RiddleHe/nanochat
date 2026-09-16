"""Report figures for the two-stage relay decomposition."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = Path("results/relay_two_stage"); CTL = Path("results/relay_controls_v2")
OUT = Path("results/report_figures"); OUT.mkdir(parents=True, exist_ok=True)
BG, FG, MUTED, GRID = "#1b1f23", "#e8eaed", "#9aa0a6", "#343b42"
BLUE, YELLOW, PURPLE, GREEN, RED = "#8ab4f8", "#fdd663", "#c58af9", "#81c995", "#f28b82"

def load(p, excl="dialogue"):
    return [r for r in (json.loads(l) for l in (p/"rows.jsonl").open()) if r["template_name"] != excl]
main, ctl = load(MAIN), load(CTL)

nat, s2, rl, s1, mm, bo = (defaultdict(list) for _ in range(6))
for r in main:
    c = r["condition"]
    if c == "stage1_transfer":
        if "clean_diff_rel" in r: nat[r["relay_layer_t"]].append(r["clean_diff_rel"])
        if r["relay_layer_t"] - r["source_layer_s"] == 4: s1[r["source_layer_s"]].append(r["transfer_proj"])
    elif c == "stage2_direct": s2[r["relay_layer_t"]].append(r["output_category"] == "expected_only")
    elif c == "relay": rl[r["source_layer_s"]].append(r["output_category"] == "expected_only")
for r in ctl:
    if r["condition"] == "mismatch_transfer": mm[r["source_layer_s"]].append(r["transfer_proj"])
    elif r["condition"] == "block_only": bo[r["relay_layer_t"]].append(r["output_category"] == "expected_only")
m = lambda d: (sorted(d), [float(np.mean(d[k])) for k in sorted(d)])

def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED, fontsize=11); ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)

# FIG 1: injection-free evidence for the timeline
fig, ax = plt.subplots(figsize=(11, 5.6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
x, y = m(nat); ax.plot(x, np.array(y)/max(y), "-", color=GREEN, lw=2.6, label="A. Natural divergence of the readout state (no intervention at all), normalized")
x, y = m(bo); ax.plot(x, y, "-o", color=BLUE, lw=2.4, ms=5, label="B. Recipient answer survives when the readout's entity access is blocked from layer T+1 (ablation only)")
xs = sorted(set(s1) & set(mm)); gap = [float(np.mean(s1[s]) - np.mean(mm[s])) for s in xs]
# curve C is measured at T = S + 4, so plot it on the same "layer where the readout
# state is inspected" axis as A and B, not on S.
# the transfer score divides by |d_T - r_T|, which is near zero before L~21;
# that region is not measurable, so it is drawn faint and excluded from the claim.
xsT = [s + 4 for s in xs]
ok = [(t, g) for t, g in zip(xsT, gap) if t >= 21]
no = [(t, g) for t, g in zip(xsT, gap) if t <= 21]
ax.plot([t for t, _ in no], [g for _, g in no], "-", color=YELLOW, lw=1.2, alpha=0.25)
ax.plot([t for t, _ in ok], [g for _, g in ok], "-", color=YELLOW, lw=2.4, label="C. Entity-specific transfer into the readout (matched minus unrelated donor), no generation")
ax.text(10, 0.055, "not measurable\n(denominator ~ 0)", color=YELLOW, alpha=0.6, fontsize=9, ha="center")
ax.axvspan(22, 24, color=YELLOW, alpha=0.13); ax.axvspan(31, 33, color=GREEN, alpha=0.13)
ax.text(23, 0.745, "first read\nL22-24", ha="center", color=YELLOW, fontsize=10.5, weight="bold")
ax.text(32, 0.30, "answer fixed\nL31-33", ha="center", color=GREEN, fontsize=10.5, weight="bold")
ax.set_ylim(-0.05, 1.08); ax.set_xlim(0, 35)
style(ax, "layer at which the readout state is inspected", "normalized score / rate")
leg = ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
for t in leg.get_texts(): t.set_color(FG)
fig.suptitle("Figure 1  Three injection-free measurements agree on the same timeline", color=FG, x=0.06, ha="left", y=0.99, fontsize=13.5)
fig.text(0.06, 0.94, "None of these passes a transplanted state through later layers, so residual-norm growth cannot affect them.", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0.16, 1, 0.92)); fig.savefig(OUT/"fig1_injection_free.png", dpi=180, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# FIG 2: the two-stage decomposition explains the relay peak
fig, ax = plt.subplots(figsize=(11, 5.6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
# every curve is placed on the relay's own S axis (window S -> S+4), so the
# three are directly comparable at the same x: relay(S) ~ stage1(S) x stage2(S+4).
x, y = m(s2); ax.plot([t - 4 for t in x], y, "-", color=YELLOW, lw=2.6, label="Stage 2: can a complete donor readout state survive to the output?  (measured at T = S+4)")
x, y = m(rl); ax.plot(x, y, "-", color=PURPLE, lw=2.6, label="Original relay: output flips to the donor")
xs2 = sorted(set(s1) & set(mm)); g2 = [min(1.0, max(0.0, float(np.mean(s1[s]) - np.mean(mm[s])))) for s in xs2]
okv = [(s_, g) for s_, g in zip(xs2, g2) if s_ + 4 >= 21]
nov = [(s_, g) for s_, g in zip(xs2, g2) if s_ + 4 <= 21]
ax.plot([a for a, _ in nov], [b for _, b in nov], "--", color=BLUE, lw=1.1, alpha=0.25)
ax.plot([a for a, _ in okv], [b for _, b in okv], "--", color=BLUE, lw=2.0, label="Stage 1: entity-specific transfer into the readout")
ax.axvspan(18, 20, color=YELLOW, alpha=0.10); ax.axvspan(27, 29, color=GREEN, alpha=0.10)
ax.text(19, 0.50, "read happens\ninside window", ha="center", color=YELLOW, fontsize=9)
ax.text(28, 0.97, "window ends after\nanswer is fixed", ha="center", color=GREEN, fontsize=9)
ax.annotate("relay works only here: the window\ncovers the read AND ends after\nthe answer is fixed",
            xy=(30, 0.82), xytext=(17.5, 0.62), color=PURPLE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))
ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 35)
style(ax, "relay source layer S   (window S -> S+4)", "rate / score")
leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
for t in leg.get_texts(): t.set_color(FG)
fig.suptitle("Figure 2  Decomposition: transfer is early, the answer is fixed late", color=FG, x=0.06, ha="left", y=0.98, fontsize=13.5)
fig.text(0.06, 0.925, "The relay's late-layer peak follows the survival curve, not the transfer curve.", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT/"fig2_decomposition.png", dpi=180, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# FIG 3: specificity control
fig, ax = plt.subplots(figsize=(10, 5.2)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
xs = sorted(set(s1) & set(mm))
ax.plot(xs, [np.mean(s1[s]) for s in xs], "-", color=BLUE, lw=2.6, label="matched donor")
ax.plot(xs, [np.mean(mm[s]) for s in xs], "-", color=RED, lw=2.4, label="unrelated donor (control)")
ax.fill_between(xs, [np.mean(mm[s]) for s in xs], [np.mean(s1[s]) for s in xs], color=YELLOW, alpha=0.25, label="entity-specific component")
ax.set_xlim(0, 32); ax.set_ylim(-0.02, 0.9)
style(ax, "source layer S (T = S + 4)", "projection onto the donor axis")
leg = ax.legend(frameon=False, fontsize=10, loc="upper left")
for t in leg.get_texts(): t.set_color(FG)
fig.suptitle("Figure 3  Specificity control: half of the raw transfer score is generic", color=FG, x=0.06, ha="left", y=0.98, fontsize=13.5)
fig.text(0.06, 0.925, "Both curves rise at the same layers, so the timing holds; only the magnitude reading changes.", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT/"fig3_specificity.png", dpi=180, facecolor=BG, bbox_inches="tight"); plt.close(fig)
print("wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
