# The Curse of Depth in Large Language Models

**Paper:** https://arxiv.org/abs/2502.05795
**Authors:** Wenfang Sun, Xinyuan Song, Pengxiang Li et al.

## Core Finding

Pre-LN transformers suffer from "curse of depth": deep layers become increasingly redundant. The hidden state changes less and less per layer (angular distance → 0), and pruning deep layers barely hurts performance. Post-LN models don't have this problem but are harder to train.

## Key Metric: Angular Distance

Measures directional change between layer inputs at layer l and l+n:

```
d(x^l, x^{l+n}) = (1/pi) * arccos(cos_sim(x^l, x^{l+n}))
```

- Small angular distance = layers are redundant (near-identity transformations)
- Ideally each layer should introduce meaningful representational shifts

## Solution: LayerNorm Scaling

Scale the LayerNorm output by 1/sqrt(depth) to control variance growth, preventing deep layers from collapsing into identity mappings.

## Relevance to Our Experiments

AttnRes should show different angular distance patterns than standard GPT:
- Standard GPT (Pre-LN): expect decreasing angular distance in deep layers
- AttnRes: by selectively aggregating prior outputs instead of accumulating, may maintain more diverse representations across depth

Visualizing angular distance for both models would show whether AttnRes mitigates the curse of depth.
