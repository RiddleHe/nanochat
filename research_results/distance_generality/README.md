# Qwen3-8B-Base distance generality

Two controlled experiments test absolute-position shift (prefix filler) and subject-to-readout distance (gap filler) while preserving the original prompt semantics. The same three natural filler families and exact token blocks are used in matched prefix/gap conditions. Actual `P_subject`, `P_readout`, filler positions, and token counts are recorded for every prompt.

The dataset authority is `/ssd/mh3897/patchscope_results/qwen3_8b_base_causal_entity_position_handoff_20260713_181400` and its completed relay metadata. Recovery and hook definitions are unchanged. Full token-by-layer matrices were produced for target added lengths 0, 32, and 128 in every family; grouped scans cover all five lengths. Attention-output sufficiency/necessity at `P_readout` covers every layer and condition.

Validation: **PASS**. See `validation_report.md` and `validation_report.json`. Scientific results are in `scientific_summary.md`; raw/aggregate CSVs, prompt metadata, worker checkpoints/logs, commands, GPU assignments, code snapshots/diffs, and seven requested plot families are retained here.

The first smoke attempt is preserved in `smoke_validation_failed_batch_controls/`. It exposed a BF16 mixed-batch control offset; the corrected singleton identity/oracle controls passed exactly before either full worker was launched.
