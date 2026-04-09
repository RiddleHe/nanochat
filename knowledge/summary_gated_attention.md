# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

**Paper:** https://arxiv.org/abs/2505.06708
**Authors:** Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, et al. (Qwen Team, Alibaba Group)

## Core Idea

A simple modification to standard softmax attention: apply a **head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA) output**. This is the paper's best variant (called G1):

```
Y' = Y * sigmoid(X @ W_theta)
```

where Y is the SDPA output and X is the pre-norm hidden state input to attention.

## Key Findings

### 1. Gating Position Matters
- **G1 (after SDPA)**: Best overall. Elementwise sigmoid, head-specific. PPL 5.761 vs 6.026 baseline on 15B MoE.
- **G2 (after value projection)**: Second best. PPL 5.820.
- **G3 (after key), G4 (after query), G5 (after output projection)**: Minimal or no improvement.

### 2. Why G1 and G2 Work: Non-linearity
- In attention, `W_v @ W_o` collapses into a single low-rank linear mapping.
- Gating at G1 or G2 injects non-linearity between W_v and W_o, increasing expressiveness.
- G5 (after W_o) doesn't help because it doesn't break the W_v*W_o low-rank bottleneck.

### 3. Why G1 > G2: Sparsity + Query-Dependence
- G1 gate scores are highly sparse (mean ~0.116), concentrated near 0.
- G2 gate scores are less sparse (mean ~0.221).
- G1 is query-dependent (gate computed from query token's hidden state). G2 is key/value-dependent.
- **Query-dependent sparse gating filters out context irrelevant to the current query**.

### 4. Attention Sink Elimination
- Standard attention allocates ~46.7% of scores to the first token (attention sink).
- G1 gating reduces this to ~4.8%.
- Attention sinks arise because softmax weights must sum to 1; tokens "dump" unused attention onto initial tokens.
- Sparse gating provides an alternative mechanism to suppress irrelevant information, removing the need for sinks.

### 5. Training Stability
- Gating dramatically reduces loss spikes during training.
- Enables higher learning rates (2x) and larger batch sizes.
- Reduces massive activations in hidden states (from ~1053 to ~94 max activation).

### 6. Context Length Extension
- Attention-sink-free models extrapolate much better to longer contexts.
- With YaRN extension to 128k: gated model scores 58.82 vs baseline 31.65 on RULER.

## Design Choices That Matter
- **Head-specific > head-shared**: Different heads should have independent gates.
- **Multiplicative > additive**: Element-wise multiplication preferred over addition.
- **Sigmoid > SiLU**: Sigmoid's [0,1] range gives sparser scores.
- **Input-dependent > learned-static**: Fixed gates lose the sparsity benefit.

## Parameter Cost
- Headwise gating: ~1.6M params for 15B model (nearly free, still very effective).
- Elementwise gating: ~201M params (same as adding 16 query heads, but much more effective).

## Relevance to Nanochat

### Direct Application: Gate the Attention Residual
The attention residual mechanism (`gpt_attn_res.py`) uses softmax over the layer dimension to aggregate prior sublayer outputs. This softmax has the same sum-to-1 constraint as sequence-dimension attention, meaning:
1. It could develop "layer sinks" analogous to attention sinks.
2. Adding a gate after the depth-wise softmax aggregation could introduce beneficial sparsity and non-linearity.

### Key Consideration: The Gate Input
The paper strongly emphasizes that **query-dependent** gating matters. In the attention residual:
- The pseudo-queries `w_l` are static (not input-dependent).
- The aggregated output `h = sum(alpha * v)` IS input-dependent.
- Using `h` itself (or the pre-norm hidden state) to compute the gate would add the missing input-dependent modulation.

### Practical Notes
- Nanochat already has several gating mechanisms (smear_gate, ve_gate). Adding another small gate is consistent with the codebase style.
- For the AttnRes use case, a lightweight gate (small projection or headwise scalar) may suffice since the depth dimension is small (2*n_layer+1 values vs thousands of sequence positions).
