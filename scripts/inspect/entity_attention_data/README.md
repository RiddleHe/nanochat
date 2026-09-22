# Entity-attention experiments: commands and outputs

These scripts ask Qwen3-8B-Base to recover a name from a prompt, remove selected
queries' access to that name, and record the generated answer or attention heatmap.

## Setup

Run from the repository root with Python 3.10, PyTorch 2.10.0, Transformers
4.57.3, and Matplotlib 3.10.8. Point `MODEL_PATH` to a downloaded
`Qwen/Qwen3-8B-Base` snapshot at revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Select an available GPU through
`CUDA_VISIBLE_DEVICES`.

```bash
MODEL_PATH=/path/to/Qwen3-8B-Base/snapshot
CORE=scripts/inspect/qwen_entity_attention_ablation.py
DATA=scripts/inspect/entity_attention_data
```

Template IDs: **0** = Direct Fact, **1** = Person in List, **2** = Friend,
**3** = Visitor Register. `diverse100` contains 100 names, giving 400 cases
across all four templates. Generation is greedy, with at most 12 new tokens.
Use a fresh output directory for each run.

Here, **block** means removing only the name token's weighted value contribution;
attention probabilities are not zeroed or renormalized. Intermediate queries are
prompt positions after the name and before the final prompt token. All blocked
conditions below also block generated queries from reading the name in every layer.

## Generate answers

Every command here writes `generations.jsonl` (prompts, token IDs, complete saved
continuations, and intervention details) and `metadata.json` (run settings).

**Ordinary baseline — 400 answers without any intervention.**

```bash
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --ordinary-baseline-only --out-dir results/ordinary
```

**Block intermediate access — 400 answers with only the final prompt query
allowed to read the name directly.**

```bash
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --block-intermediate-prompt-entity --baseline-only --out-dir results/intermediate_blocked
```

**Scan layers — Direct Fact answers after blocking final-prompt access in each
sliding span of 1, 2, 3, 4, 6, or 8 layers.** Intermediate access stays blocked.
The output includes the exact span for each answer and 100 bottleneck-control rows.

```bash
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0 --block-intermediate-prompt-entity --widths 1,2,3,4,6,8 --out-dir results/layer_scan
```

To test whether late access alone is sufficient, replace `--widths 1,2,3,4,6,8`
with `--widths 20 --start-layers 0` and use a new output directory. This blocks
L0–19 and leaves L20–35 available to the final prompt query. Layers are zero-indexed.

**Restore late entity contributions — 200 Person-in-List/Friend answers with
intermediate access still blocked.** At L23–35, add the positive per-head deficit
`max(ordinary_attention - live_attention, 0) * entity_value`. Each output row also
contains the ordinary/live coefficients and restoration trace. This is a full-cohort
run; it does not select only previously failed cases.

```bash
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 1,2 --block-intermediate-prompt-entity --restore-entity-attention-start-layer 23 --out-dir results/late_restore
```

**Block final-prompt access — 400 intervened answers while intermediate queries
retain access to the name.** The file also includes 400 control answers with only
generated-query access blocked; these controls differ from the ordinary baseline.

```bash
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --widths 36 --start-layers 0 --out-dir results/final_blocked
```

## Compare correct and wrong descriptions

**Correct descriptions:** insert `NAME (DESCRIPTION)`, such as
`Einstein (a physicist)`. The first command writes 400 ordinary answers; the
second writes 400 final-prompt-blocked answers plus 400 generated-query-blocked controls.

```bash
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --entity-descriptions-json "$DATA/entity_descriptions.json" --ordinary-baseline-only --out-dir results/description_ordinary
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --entity-descriptions-json "$DATA/entity_descriptions.json" --widths 36 --start-layers 0 --out-dir results/description_final_blocked
```

**Wrong descriptions:** use the same final-prompt block and replace only the
description. This writes 400 answers, `metadata.json`, and `validation.json`,
which checks that the name token and tokens outside the description are unchanged.

```bash
python scripts/inspect/run_wrong_description.py --model "$MODEL_PATH" --cases "$DATA/cases.json" --manifest "$DATA/manifest.json" --outdir results/wrong_description
```

The inputs are `entity_descriptions.json` (100 correct descriptions),
`manifest.json` (100 fixed wrong-description assignments reused across templates),
and `cases.json` (400 original correct-description cases for paired token checks).
The wrong-description driver imports the main ablation script.

## Capture attention heatmaps

Compare ordinary inference with intermediate access blocked. The current profiler
supports **Direct Fact and Visitor Register only**, with 100 entities and 36 layers.

```bash
python scripts/inspect/qwen_prompt_attention_profile.py --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,3 --out-dir results/attention_profiles
```

Outputs:

- `generations.jsonl`: 400 answers (2 templates × 100 names × 2 conditions).
- `metadata.json`: run settings and measurement definitions.
- `attention_profiles.pt`: per-head attention and contribution data.
- `attention_summary.json`: aggregated measurements.
- `*_attention_comparison.png`: ordinary, blocked, and difference heatmaps.
- `*_value_contribution_comparison.png`: corresponding value-contribution plots.

Heatmap rows are layers; columns are prompt tokens. Color is the final prompt
query's attention, maximized across heads and then averaged across names.

For a quick CPU check without a checkpoint:

```bash
CUDA_VISIBLE_DEVICES= python scripts/inspect/qwen_prompt_attention_profile.py --self-test
```

## Reading the results

Review complete saved continuations before scoring name recovery. The automatic
`entity_in_completion` field is a diagnostic, not a reviewed semantic score.
An answer missing at the 12-token limit is a failure within that saved horizon.

Restoration and layer scanning currently run separately; a combined restored
layer scan is not implemented. The profiler also needs generalization before
running other templates or model depths.
