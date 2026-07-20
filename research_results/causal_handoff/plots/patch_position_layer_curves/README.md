# Exact patch-position layer curves

These plots use the merged `token_position_results.csv` from the
Qwen3-8B-Base causal entity-position handoff experiment.

## Prompt positions

The prompt is:

`Everyone knows {entity} was a celebrated {role}. The {role} was`

| Position | Label |
| --- | --- |
| P00 | `Everyone`, before the entity |
| P01 | `knows`, before the entity |
| P02 | entity name, e.g. `Newton` in the recipient or `Einstein` in the donor |
| P03 | `was` immediately after the entity |
| P04 | `a` |
| P05 | `celebrated` |
| P06 | first role token, e.g. the first `scientist` |
| P07 | period (`.`) |
| P08 | `The` |
| P09 | second role token, e.g. the second `scientist` |
| P10 | final `was`; this position supplies the next-token logits |

## Outputs

- `all_patch_positions_by_layer_small_multiples.*`: all 11 exact positions on
  a shared axis.
- `key_patch_positions_by_layer.*`: subject, role mentions, and final-token
  curves together.
- `patch_position_XX_*.png`: one curve per exact position, with a 95% pair
  bootstrap interval.
- `patch_position_curve_summary.csv`: plotted means, standard deviations,
  bootstrap intervals, and sample counts.
- `plot_patch_position_layer_curves.py`: reproducible plotting script.

## Main descriptive pattern

- P02 (the subject entity) carries almost all recoverable causal influence in
  early and middle layers, then falls sharply around layers 23-26.
- P10 (the final input token) begins rising around layer 21, is comparable to
  P02 at layers 24-25, overtakes it at layer 26, and reaches full recovery at
  layer 35.
- Other exact single-token positions peak below 0.05 mean recovery. This means
  the strong late effect of patching all post-subject positions together is
  dominated by the final token when positions are tested individually.

For example, in the Einstein-into-Newton direction, patching P02 replaces the
hidden state at the visible `Newton` position with the same-layer state from
the `Einstein` prompt. Patching P09 replaces the contextual hidden state of the
second `scientist` token; it does not change that visible token text.

The confidence bands bootstrap the 16 ordered pairs and are descriptive. The
ordered directions are not fully independent because some reuse the same
entities.
