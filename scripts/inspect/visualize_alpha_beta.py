"""Visualize the learned blend coefficients (alpha_v, beta_v) per target layer
for the additive value-residual variants, d12.

    add_init_v_learn:     v = alpha_v * c_v(x_i)  +  beta_v * V_1   (V_1 = cached layer-0 v)
    add_init_res_v_learn: v = alpha_v * c_v(x_i)  +  beta_v * c_v(norm(x_0))

alpha_v (init 1.0) scales the standard current-layer value x_i; beta_v (init 0.0)
scales the injected init-value source. Two vertically stacked panels, squashed
for a double-column layout.

Usage:
    CUDA_VISIBLE_DEVICES=2 NANOCHAT_BASE_DIR=/local-ssd/mh3897 \\
        uv run python -m scripts.inspect.visualize_alpha_beta
"""
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import build_model

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# consistent color semantics across all gamma/alpha-beta plots:
C_XI = ('#B9EBEA', '#4ea8a4')  # x_i  (teal)  -> alpha_v, the standard current value
C_V1 = ('#C7DDEB', '#5b8fb0')  # V_1  (blue)  -> beta_v for add_init_v
C_X0 = ('#FBE0CF', '#d18f5e')  # x_0  (peach) -> beta_v for add_init_res_v

# (tag, panel title, beta-source label, beta-source color)
PANELS = [
    ('arch_d12_gpt_base_add_init_v_learn',     r'(a) $\mathbf{x}_i + \mathbf{V}_1$', r'$\mathbf{V}_1$', C_V1),
    ('arch_d12_gpt_base_add_init_res_v_learn', r'(b) $\mathbf{x}_i + \mathbf{x}_0$', r'$\mathbf{x}_0$', C_X0),
]


def find_last_step(ckpt_dir: Path) -> int:
    return max(int(re.search(r"_(\d+)\.pt$", str(p)).group(1))
               for p in ckpt_dir.glob("model_*.pt"))


def alpha_beta_by_layer(ckpt_dir: Path, device):
    step = find_last_step(ckpt_dir)
    m, _, _ = build_model(str(ckpt_dir), step=step, device=device, phase='eval')
    layers, alphas, betas = [], [], []
    for i, blk in enumerate(m.transformer.h):
        if getattr(blk, 'is_target', False) and hasattr(blk.attn, 'alpha_v'):
            layers.append(i)
            alphas.append(blk.attn.alpha_v.item())
            betas.append(blk.attn.beta_v.item())
    return layers, alphas, betas


def style_axes(ax, xticks):
    ax.axhline(0.0, color='#999999', linestyle='--', linewidth=1.0, alpha=0.8, zorder=2)
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', length=0, labelsize=9)
    ax.spines['top'].set_visible(True);    ax.spines['top'].set_linewidth(1.4);    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_linewidth(1.4); ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(False);  ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    ckpt_root = Path(get_base_dir()) / "base_checkpoints"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 4.0), sharex=True)
    for ax, (tag, title, beta_label, beta_color) in zip(axes, PANELS):
        layers, alphas, betas = alpha_beta_by_layer(ckpt_root / tag, device)
        print(f'{title}: ' + '  '.join(f'L{l}:a={a:.2f},b={b:.2f}'
                                       for l, a, b in zip(layers, alphas, betas)))
        ax.plot(layers, alphas, '-o', color=C_XI[1], markerfacecolor=C_XI[0],
                markeredgecolor=C_XI[1], markersize=7, linewidth=2.2,
                markeredgewidth=1.2, label=r'$\mathbf{x}_i$', zorder=3)
        ax.plot(layers, betas, '-o', color=beta_color[1], markerfacecolor=beta_color[0],
                markeredgecolor=beta_color[1], markersize=7, linewidth=2.2,
                markeredgewidth=1.2, label=beta_label, zorder=3)
        style_axes(ax, layers)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_ylabel('coefficient value', fontsize=10)
        ax.legend(fontsize=11, frameon=False, loc='best', ncol=2)

    axes[-1].set_xlabel('Layer Index', fontsize=11)
    plt.tight_layout()
    out_dir = Path('/hdd/mh3897/nanochat/results')
    png = out_dir / 'd12_alpha_beta.png'
    pdf = out_dir / 'd12_alpha_beta.pdf'
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    print(f'Saved {png}')
    print(f'Saved {pdf}')


if __name__ == '__main__':
    main()
