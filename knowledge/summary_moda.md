# Mixture-of-Depths Attention (MoDA)

**Paper:** https://arxiv.org/abs/2603.15619
**Authors:** Lianghui Zhu, Yuxin Fang et al. (ByteDance Seed / HUST)
**Code:** https://github.com/hustvl/MoDA

## Core Idea

Each attention head jointly attends to **sequence KV** (standard causal attention) AND **depth KV** (KV pairs from the same token position across all preceding layers) in a single unified softmax.

Standard attention: token i attends to tokens j<=i at the current layer.
MoDA: token i attends to tokens j<=i at current layer PLUS its own KV from layers 0..l-1 (depth dimension).

```
O = softmax([Q @ K_seq^T, Q @ K_depth^T] / sqrt(d)) @ [V_seq; V_depth]
```

The depth KV pairs come from both attention layers AND FFN layers (via lightweight KV projections on FFN outputs).

## How it relates to ResFormer / XSA / AttnRes

MoDA is conceptually different from all three:
- **ResFormer**: interpolates V_1 into current V (fixed mixing, same attention weights)
- **XSA**: removes self-value from attention output (post-processing)
- **AttnRes**: replaces residual connections with softmax over depth (modifies layer inputs)
- **MoDA**: extends the attention KV space to include depth history (modifies what attention can see)

MoDA doesn't replace residual connections — it adds depth retrieval ON TOP of standard residuals. Each layer still does `x = x + attn(norm(x))` but the attention itself can now look at prior layers' KV pairs.

## Key Design: "Read, Operate, Write"

- **Read**: Current hidden state + depth KV cache from prior layers
- **Operate**: Unified attention over sequence + depth KV in one softmax
- **Write**: Append current layer's KV to depth cache; standard residual for hidden state

## Results (1.5B, 400B tokens, OLMo2 recipe)

- Average perplexity improvement: 0.2 across 10 validation benchmarks
- Average downstream improvement: +2.11% on 10 tasks
- FLOPs overhead: 3.7%
- Hardware-efficient kernel: 97.3% of FlashAttention-2 efficiency at 64K

## Ablation highlights

1. Depth attention alone (without sequence attention) improves 0.11 C4 val PPL with only 0.12% extra FLOPs
2. Adding FFN KV to depth cache further improves 0.27 C4 val PPL
3. Post-norm works better than pre-norm with MoDA
4. Attention visualization shows MoDA reduces attention sink behavior

## Complexity

O(TL^2 D) FLOPs — same as depth attention, much cheaper than depth dense O(TL^2 D^2). The L^2 factor means cost grows quadratically with depth but linearly with width.

## Nanochat Connection

MoDA is more invasive than AttnRes or XSA — it requires modifying the attention kernel to accept depth KV alongside sequence KV. The unified softmax means it can't be naively implemented on top of standard Flash Attention. However, a simplified version could cache per-layer KV and concatenate them with sequence KV before attention, at the cost of longer effective sequence length.

The core insight (let attention look at prior layers' KV) is related to AttnRes (let layer inputs aggregate prior outputs) but mechanistically different: MoDA operates WITHIN attention (expanding what it can see), while AttnRes operates OUTSIDE attention (changing what it receives).
