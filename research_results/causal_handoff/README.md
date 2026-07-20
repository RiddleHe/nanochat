# Qwen3-8B-Base causal entity position handoff

Mode: `merged-3gpu`

Model: `Qwen/Qwen3-8B-Base`

Prompt template: `Everyone knows {name} was a celebrated {role}. The {role} was`

Tokenizer: model tokenizer with `add_special_tokens=True` for prompts, matching PR #15.

Metric: PR #15 logit-difference metric `m(x) = logit(donor entity) - logit(recipient entity)` using leading-space single-token entity IDs.

Layer hook: cumulative residual state at output of transformer block; layer 0 is output of block 0

Valid ordered pairs: 16

Skipped pairs: 0

No donor activation normalization is applied.
