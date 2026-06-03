"""Learned value-routing coefficients per target layer, d12, 1x3 panel.

(a) single vector : v = c * (source), one scalar per layer (gamma_v)
      V_1 = v_from_v1_learn, x_0 = v_from_x0_learn, x_i = v_scale_learn
(b) x_i + V_1     : v = alpha_v * c_v(x_i) + beta_v * V_1   (add_init_v_learn)
(c) x_i + x_0     : v = alpha_v * c_v(x_i) + beta_v * c_v(norm(x_0)) (add_init_res_v_learn)

Color semantics are consistent across panels: x_i = teal, V_1 = blue, x_0 = peach.
Panels (b) and (c) share y-limits for direct comparison.

Usage:
    CUDA_VISIBLE_DEVICES=2 NANOCHAT_BASE_DIR=/local-ssd/mh3897 \\
        uv run python -m scripts.inspect.visualize_value_coeffs
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
# Render at final ACL size (figure width = \textwidth = 455.24 pt = 6.32 in) so
# width=\textwidth in LaTeX is 1:1 with no scaling. Fonts set directly to target pt.
mpl.rcParams.update({
    "axes.titlesize":  10,   # panel (a)/(b)/(c) — matches caption
    "axes.labelsize":   9,   # x/y labels
    "legend.fontsize":  9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})

C_XI = ('#B9EBEA', '#4ea8a4')  # x_i  teal
C_V1 = ('#C7DDEB', '#5b8fb0')  # V_1  blue
C_X0 = ('#FBE0CF', '#d18f5e')  # x_0  peach


def find_last_step(d: Path) -> int:
    return max(int(re.search(r"_(\d+)\.pt$", str(p)).group(1)) for p in d.glob("model_*.pt"))


def read_attr(ckpt_root, tag, attr, device):
    d = ckpt_root / tag
    m, _, _ = build_model(str(d), step=find_last_step(d), device=device, phase='eval')
    layers, vals = [], []
    for i, blk in enumerate(m.transformer.h):
        if getattr(blk, 'is_target', False) and hasattr(blk.attn, attr):
            layers.append(i); vals.append(getattr(blk.attn, attr).item())
    return layers, vals


def curve(ax, x, y, color, label):
    ax.plot(x, y, '-o', color=color[1], markerfacecolor=color[0], markeredgecolor=color[1],
            markersize=4, linewidth=1.3, markeredgewidth=0.8, label=label, zorder=3)


def style(ax, xticks):
    ax.axhline(0.0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)
    ax.set_xticks(xticks)
    ax.set_xlabel('Layer Index')
    ax.tick_params(axis='both', length=0)
    ax.spines['top'].set_visible(True);    ax.spines['top'].set_linewidth(0.9);    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_linewidth(0.9); ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(False);  ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left')


def main():
    root = Path(get_base_dir()) / "base_checkpoints"
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # figure width = ACL \textwidth = 455.24 pt = 6.32 in
    fig, axes = plt.subplots(1, 3, figsize=(6.32, 2.0))

    # (a) single vector: gamma_v for v1 / x0 / x_i
    ax = axes[0]
    lx, v1 = read_attr(root, 'arch_d12_gpt_base_v_from_v1_learn', 'gamma_v', dev)
    _,  x0 = read_attr(root, 'arch_d12_gpt_base_v_from_x0_learn', 'gamma_v', dev)
    _,  xi = read_attr(root, 'arch_d12_gpt_base_v_scale_learn',   'gamma_v', dev)
    curve(ax, lx, v1, C_V1, r'$\mathbf{V}_1$')
    curve(ax, lx, x0, C_X0, r'$\mathbf{x}_0\mathbf{W}_V$')
    curve(ax, lx, xi, C_XI, r'$\mathbf{V}$')
    ax.set_title('(a) substitutive', pad=4)
    ax.set_ylabel('coefficient value')
    style(ax, lx)

    # (b) x_i + V_1 : add_init_v_learn
    ax = axes[1]
    lb, a_b = read_attr(root, 'arch_d12_gpt_base_add_init_v_learn', 'alpha_v', dev)
    _,  b_b = read_attr(root, 'arch_d12_gpt_base_add_init_v_learn', 'beta_v', dev)
    curve(ax, lb, a_b, C_XI, r'$\mathbf{V}$')
    curve(ax, lb, b_b, C_V1, r'$\mathbf{V}_1$')
    ax.set_title(r'(b) additive: $\mathbf{V} + \mathbf{V}_1$', pad=4)
    style(ax, lb)

    # (c) x_i + x_0 : add_init_res_v_learn
    ax = axes[2]
    lc, a_c = read_attr(root, 'arch_d12_gpt_base_add_init_res_v_learn', 'alpha_v', dev)
    _,  b_c = read_attr(root, 'arch_d12_gpt_base_add_init_res_v_learn', 'beta_v', dev)
    curve(ax, lc, a_c, C_XI, r'$\mathbf{V}$')
    curve(ax, lc, b_c, C_X0, r'$\mathbf{x}_0\mathbf{W}_V$')
    ax.set_title(r'(c) additive: $\mathbf{V} + \mathbf{x}_0\mathbf{W}_V$', pad=4)
    style(ax, lc)

    # shared y-limits for (b) and (c)
    allbc = a_b + b_b + a_c + b_c
    pad = 0.08 * (max(allbc) - min(allbc))
    ylim_bc = (min(allbc) - pad, max(allbc) + pad)
    axes[1].set_ylim(ylim_bc)
    axes[2].set_ylim(ylim_bc)

    plt.tight_layout(pad=0.5)
    out = Path('/hdd/mh3897/nanochat/results')
    # No bbox_inches='tight' — keep the canvas exactly 6.32 in wide so
    # width=\textwidth in LaTeX renders 1:1 with the target font pt.
    for ext in ('png', 'pdf'):
        p = out / f'd12_value_coeffs.{ext}'
        plt.savefig(p, dpi=200)
        print(f'Saved {p}')


if __name__ == '__main__':
    main()
