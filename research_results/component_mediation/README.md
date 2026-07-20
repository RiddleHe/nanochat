# Qwen3-8B component-level causal relay

Exactly three experiments were run on `Qwen/Qwen3-8B-Base`: attention-output mediation, matched MLP intervention, and the predefined head/source-position scan at layers 23–26. The dataset is exactly the completed relay's 16 ordered directions in eight reciprocal clusters. P02 is the entity token and P10 is the final `was`. The hybrid trajectory patches donor P02 into recipient P02 at block-0 output.

Recovery is `(intervention margin - recipient margin) / (donor margin - recipient margin)` and is never clipped. Confidence intervals use a fixed-seed (20260715) cluster bootstrap over eight unordered reciprocal clusters; both directions are resampled together. Raw CSVs retain target logits, denominator, direction, cluster, layer, position, component, head/source position, intervention, vector norms/cosines, token IDs, and labels.

Validation status: **PASS**. See `validation_report.json` and `reference_curve_comparison.csv`. Total summed per-pair worker runtime: 316.4 seconds (workers ran concurrently).

Key files: raw experiment CSVs, `mlp_effects.csv`, causal-effect CSVs, `pair_direction_summary.csv`, `aggregate_*.csv`, `top_heads.csv`, `quantitative_findings.json`, plots in `plots/`, and `scientific_summary.md`.
