# Kimi Delta Attention (KDA)

**Paper:** "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692)
**Code:** https://github.com/MoonshotAI/Kimi-Linear, https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct

## Forget Gate Architecture

KDA's state update uses a forget gate (alpha) with low-rank down/up projections:

```
alpha_t = exp(-exp(A_log) * softplus(W_up(W_down(x_t)) + dt_bias))
```

- **Down**: `Linear(hidden_size, head_dim, bias=False)` — compresses to head_dim (128)
- **Up**: `Linear(head_dim, num_heads * head_dim, bias=False)` — expands back
- **A_log**: per-head learnable scale, init as `log(Uniform(1, 16))`
- **dt_bias**: per-dim bias, Mamba-style init (inverse softplus of small exponentials)

The gate output is negative (log-decay), so `alpha = exp(gate)` is in [0, 1].

## Output Gate

Separate gate for the output, also low-rank:
```
o_t = W_o * (sigmoid(W_g_up(W_g_down(x))) * RMSNorm(KDA_output))
```

## Initialization

**Both down and up projections: Normal(0, 0.02)** — standard init, NOT zero.

```python
def _init_weights(self, module):
    std = 0.02  # config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
```

A_log and dt_bias have specialized Mamba-style inits and are excluded from weight decay.

## Key Takeaway for Our Experiments

KDA uses random normal init for gate projections, not zero. This aligns with the gated attention paper (also normal init) and differs from our gated AttnRes discussion. The difference: KDA's gate output passes through `softplus` then `exp(-...)`, not raw `sigmoid`. At init with Normal(0, 0.02) inputs, `softplus(~0) ≈ 0.69`, so `alpha ≈ exp(-exp(A_log) * 0.69)` — the A_log init (1-16 range) determines the initial decay rate, not the projection init. The projections just need to be non-zero for gradient flow; the actual gate behavior is controlled by A_log and dt_bias.
