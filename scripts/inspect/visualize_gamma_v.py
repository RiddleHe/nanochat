"""Visualize the learned value-scaling coefficient gamma_v across target layers.

For variants trained with `learn_init_coeffs=True`, each target layer carries a
scalar `gamma_v` (init 1.0) that scales the routed value vector:
    v_from_v1:  v = gamma_v * (cached layer-0 v)
    v_from_x0:  v = gamma_v * c_v(norm(x0))
This inspector loads the d12 checkpoints for the v_from_v1_learn and
v_from_x0_learn variants, reads gamma_v from each target layer, and plots both
as curves over layer index (they track each other closely, hence one panel).

Usage:
    CUDA_VISIBLE_DEVICES=2 NANOCHAT_BASE_DIR=/local-ssd/mh3897 \\
        uv run python -m scripts.inspect.visualize_gamma_v
"""
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import build_model

# Times-metric serif for all text including math (STIX matches Times)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# (checkpoint tag, legend label, (fill, line) colors)
#   V_1  : v_from_v1_learn   -> v = gamma_v * (cached layer-0 v)
#   x_0  : v_from_x0_learn   -> v = gamma_v * c_v(norm(x0))   (layer-0 input)
#   x_i  : v_scale_learn     -> v = gamma_v * c_v(x_i)        (current layer input, control)
VARIANTS = [
    ('arch_d12_gpt_base_v_from_v1_learn', r'$\mathbf{V}_1$', ('#C7DDEB', '#5b8fb0')),
    ('arch_d12_gpt_base_v_from_x0_learn', r'$\mathbf{x}_0$', ('#FBE0CF', '#d18f5e')),
    ('arch_d12_gpt_base_v_scale_learn',   r'$\mathbf{x}_i$', ('#B9EBEA', '#4ea8a4')),
]


def find_last_step(ckpt_dir: Path) -> int:
    return max(int(re.search(r"_(\d+)\.pt$", str(p)).group(1))
               for p in ckpt_dir.glob("model_*.pt"))


def gamma_v_by_layer(ckpt_dir: Path, device):
    step = find_last_step(ckpt_dir)
    m, _, _ = build_model(str(ckpt_dir), step=step, device=device, phase='eval')
    layers, gammas = [], []
    for i, blk in enumerate(m.transformer.h):
        if getattr(blk, 'is_target', False) and hasattr(blk.attn, 'gamma_v'):
            layers.append(i)
            gammas.append(blk.attn.gamma_v.item())
    return layers, gammas


def main():
    base_dir = get_base_dir()
    ckpt_root = Path(base_dir) / "base_checkpoints"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    all_layers = set()
    for tag, label, (fill, line) in VARIANTS:
        layers, gammas = gamma_v_by_layer(ckpt_root / tag, device)
        all_layers.update(layers)
        print(f'{label}: ' + '  '.join(f'L{l}={g:.3f}' for l, g in zip(layers, gammas)))
        ax.plot(layers, gammas, '-o', color=line, markerfacecolor=fill,
                markeredgecolor=line, markersize=9, linewidth=2.4,
                markeredgewidth=1.4, label=label, zorder=3)

    ax.axhline(1.0, color='#999999', linestyle='--', linewidth=1.0, alpha=0.8, zorder=2)
    ax.set_xticks(sorted(all_layers))  # integer layer indices only (no 8.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('coefficient value', fontsize=12)
    ax.tick_params(axis='both', length=0, labelsize=10)

    ax.spines['top'].set_visible(True);    ax.spines['top'].set_linewidth(1.4);    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_linewidth(1.4); ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(False);  ax.spines['right'].set_visible(False)

    ax.grid(axis='y', color='#dddddd', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=13, frameon=False, loc='upper left')

    plt.tight_layout()
    out_dir = Path('/hdd/mh3897/nanochat/results')
    out_dir.mkdir(exist_ok=True)
    png = out_dir / 'd12_gamma_v_v1_x0.png'
    pdf = out_dir / 'd12_gamma_v_v1_x0.pdf'
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    print(f'Saved {png}')
    print(f'Saved {pdf}')


if __name__ == '__main__':
    main()
