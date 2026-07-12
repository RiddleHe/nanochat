import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/translator_probs/Qwen_Qwen3-8B-Base__translator_probs.json"))
rows = d["modes"]["template"]
xs = [r["src"] for r in rows]

plt.rcParams["font.family"] = "STIXGeneral"

# ---- plot one: accuracy vs source layer ----
fig, ax = plt.subplots(figsize=(6.3, 4.0))
ax.plot(xs, [r["ceiling"]["acc"] for r in rows], "--", color="0.5", marker="o", label="L10")
ax.plot(xs, [r["translated"]["acc"] for r in rows], "-", color="#2c8a8a", marker="o", label="translated")
ax.plot(xs, [r["raw"]["acc"] for r in rows], "-", color="#e8a87c", marker="o", label="raw")
ax.set_xlabel("layer"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.0)
ax.set_xticks(xs); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("results/translator_probs/plot_one_acc.png", dpi=200)
print("wrote plot_one_acc.png")

# ---- plot two: top-1 vs top-2 mean prob (translated) ----
top1, top2 = [], []
for r in rows:
    P = torch.tensor([e["p"] for e in r["translated_probs"]])
    t2 = P.topk(2, 1).values
    top1.append(float(t2[:, 0].mean())); top2.append(float(t2[:, 1].mean()))
fig, ax = plt.subplots(figsize=(6.3, 4.0))
X = np.arange(len(xs)); w = 0.38
ax.bar(X - w / 2, top1, w, color="#2c8a8a", label="top-1 token (correct)")
ax.bar(X + w / 2, top2, w, color="#e8a87c", label="top-2 token")
ax.set_xlabel("layer"); ax.set_ylabel("prob"); ax.set_ylim(0, 1.0)
ax.set_xticks(X); ax.set_xticklabels(xs); ax.grid(alpha=0.3, axis="y"); ax.legend()
fig.tight_layout()
fig.savefig("results/translator_probs/plot_two_top12.png", dpi=200)
print("wrote plot_two_top12.png")
print("top1:", [round(v,3) for v in top1], "top2:", [round(v,3) for v in top2])
