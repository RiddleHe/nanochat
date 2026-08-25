"""Plot heatmaps for completed full-matrix patchscope output files."""
import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect.patchscope_few_shot_target_suffix import (  # noqa: E402
    CRITERIA,
    ENTITIES,
    ENTITY_TITLES,
    _grade,
)


ROW_RE = re.compile(r"^S(\d+)_T(\d+)\t(.*)$")
CMAP = ListedColormap(["#f7f9fc", "#0f766e"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], CMAP.N)


def entity_from_path(path):
    stem = path.stem
    for ent in ENTITIES:
        if stem.endswith(f"__{ent}"):
            return ent
    return None


def parse_rows(path, ent):
    rows = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        src = int(m.group(1))
        tgt = int(m.group(2))
        text = m.group(3)
        rows.append((src, tgt, text, _grade(text, CRITERIA[ent])))
    return rows


def plot_one(folder, plots_dir, frame_name, ent, rows):
    n_source = max(src for src, _, _, _ in rows) + 1
    n_target = max(tgt for _, tgt, _, _ in rows) + 1
    mat = np.zeros((n_source, n_target), dtype=float)
    for src, tgt, _, score in rows:
        mat[src, tgt] = score

    hits = int(mat.sum())
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")
    im = ax.imshow(mat, origin="lower", aspect="auto", interpolation="nearest",
                   cmap=CMAP, norm=NORM)
    ax.set_xlabel("target layer")
    ax.set_ylabel("source layer")
    ax.set_xticks(range(0, n_target, 5))
    ax.set_yticks(range(0, n_source, 5))
    ax.set_title(f"{frame_name}\n{ENTITY_TITLES[ent]} | hits {hits}/{mat.size}",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["miss", "hit"])
    fig.tight_layout()

    png = plots_dir / f"{ent}__heatmap.png"
    pdf = plots_dir / f"{ent}__heatmap.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return hits, mat.size, png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--frame-name", required=True)
    args = ap.parse_args()

    folder = Path(args.folder)
    plots_dir = folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    summary_rows = []
    for path in sorted(folder.glob("*.txt")):
        if path.name == "command.txt":
            continue
        ent = entity_from_path(path)
        if ent is None:
            continue
        rows = parse_rows(path, ent)
        if len(rows) != 1296:
            raise RuntimeError(f"{path} has {len(rows)} rows; expected 1296")
        hits, total, png, pdf = plot_one(folder, plots_dir, args.frame_name, ent, rows)
        summary_rows.append({
            "entity": ent,
            "entity_name": ENTITY_TITLES[ent],
            "hits": hits,
            "total": total,
            "png": str(png),
            "pdf": str(pdf),
        })
        print(f"{ent}: {hits}/{total} -> {png}")

    if len(summary_rows) != len(ENTITIES):
        raise RuntimeError(f"found {len(summary_rows)} entity files; expected {len(ENTITIES)}")

    summary_path = folder / "heatmap_hit_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["entity", "entity_name", "hits", "total", "png", "pdf"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
