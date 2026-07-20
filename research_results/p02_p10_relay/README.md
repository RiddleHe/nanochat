# Qwen3-8B-Base P02-to-P10 causal relay

This experiment asks when information introduced at the entity-name position becomes causally usable at the final next-token readout position.

Model: `Qwen/Qwen3-8B-Base`. Prompt: `Everyone knows {entity} was a celebrated {role}. The {role} was`. Prompts use the model tokenizer with `add_special_tokens=True`. All 16 ordered pairs from PR #15 validated exactly against `/ssd/mh3897/patchscope_results/qwen3_8b_base_causal_entity_position_handoff_20260713_181400/pair_metadata.csv`; skipped pairs: 0.

P02 and P10 are descriptive labels verified from tokenization for every pair. P02 is the entity-name span. P10 is the final `was`, whose residual state supplies the next-token logits. Source and relay position lists are stored in every row.

At source block output `s`, donor P02 replaces recipient P02. The resulting hybrid run supplies P10 or all strictly post-subject states at block output `t`; those states are relayed into a fresh recipient sequence. `t=s` is included and `t<s` is undefined and was never run. The direct clean-donor P10 condition is an upper-bound control, not the primary relay result.

Metric: `m(x) = logit(donor entity) - logit(recipient entity)`. Recovery: `(m(intervention) - m(recipient)) / (m(donor) - m(recipient))`. Values are not clipped. Plot annotations use 0 for no donor recovery and 1 for full donor recovery.

Layer hook: cumulative residual state at output of transformer block; layer 0 is output of block 0. No donor activation normalization was applied. The experiment calls no text-generation API and records only next-token logits.

`relay_summary.csv` contains means and descriptive 95% cluster-bootstrap intervals from 2000 resamples. The 8 unordered entity pairs are clusters and both reciprocal directions are retained together when a cluster is sampled. These intervals are descriptive because reciprocal directions are dependent.

Files include raw relay, source end-to-end, direct/identity/baseline control rows, merged metadata, validation, per-pair checkpoints within shards, and PNG/PDF plots under `plots/`.
