# Exclusive Self-Attention (XSA)

**Paper:** https://arxiv.org/abs/2603.09078
**Authors:** Shuangfei Zhai (Apple)

## Core Idea

After standard attention computes `y_i = sum_j(a_{ij} * v_j)`, XSA subtracts the component of `y_i` that lies along `v_i` (the token's own value vector):

```
z_i = y_i - (y_i^T v_i / ||v_i||^2) * v_i
```

This is a vector rejection — the output is guaranteed orthogonal to `v_i`. The motivation: trained transformers show high cosine similarity between attention output `y_i` and self-value `v_i` (the "attention similarity bias"), meaning attention wastes capacity re-encoding information the token already has. Since `v_i`'s information is available via the residual connection anyway, forcing attention to only provide *contextual* (non-self) information promotes cleaner division of labor.

## Implementation

Two lines after standard SDPA:

```python
Vn = F.normalize(V, dim=-1)
Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
```

Applied selectively to last N layers only (default 4) since the bias increases with depth.

## Results

0.7B, 1.3B, 2.7B models on FineWeb-100BT. Gains scale with model size (+0.26 to +1.36 avg benchmark points) and sequence length. Negligible overhead.
