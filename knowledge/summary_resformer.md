# Value Residual Learning (ResFormer)

**Paper:** https://arxiv.org/abs/2410.17897
**Authors:** Zcchill et al.
**Code:** https://github.com/Zcchill/Value-Residual-Learning

## Core Idea

Token-level information gets diluted in deep layers despite hidden state residuals. ResFormer adds a second residual pathway on the **value vectors**: mix the first layer's value V_1 into every subsequent layer's value before attention.

```
V'_n = lambda_1 * V_1 + lambda_2 * V_n
```

V_1 = H_0 @ W_V^1 contains pure token-level information (linear projection of embeddings). Mixing it in ensures token identity survives into deep layers.

## Variants

- **Identity-ResFormer**: lambda_1 = lambda_2 = 0.5 (simplest, already effective)
- **Constant-ResFormer**: fixed lambdas, optimal at lambda_1=2
- **Learnable-ResFormer**: trainable lambdas, init 0.5
- **Sparse-ResFormer**: only apply to certain layers (deepest ones benefit most)
- **SVFormer**: extreme — reuse V_1 entirely, skip v_proj in layers 2+

## Implementation

```python
if self.layer_idx == 0:
    formal_layer_values.append(value_states)  # cache V_1
else:
    value_states = 0.5 * formal_layer_values[0] + 0.5 * value_states  # mix
```

Key: V_1 is mixed in *before* attention, so it shares the same attention weights A_n. Ablations show only value residuals help; Q, K, and attention output residuals hurt.

## Results

82M-1.6B models on SlimPajama. 468M model matches baseline loss with 16% fewer params and 20% less data. Downstream: +1.7 points avg accuracy. Gains scale with depth and model size.

## Nanochat Connection

nanochat already implements value embeddings (ResFormer-style) via `value_embeds` with learned gating per head — a more sophisticated version of this idea.
