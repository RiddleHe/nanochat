# Entity-attention paper experiments

Run commands from the repository root. These are the existing Qwen3-8B-Base
experiments, not the proposed ten-template/four-model production suite.

## Files and paper sections

| File | Role |
|---|---|
| `scripts/inspect/qwen_entity_attention_ablation.py` | Sections 1, 3, 4, 4.2: shared entity-value ablation, generation, positive-only restoration, and description insertion |
| `scripts/inspect/qwen_prompt_attention_profile.py` | Section 3: ordinary/bottleneck final-query attention tensors, heatmaps, and projected value contributions |
| `scripts/inspect/run_wrong_description.py` | Section 4.2: paired wrong-description validation and generation; imports the shared engine |
| `scripts/inspect/entity_attention_data/entity_descriptions.json` | Frozen correct indefinite descriptions for 100 entities |
| `scripts/inspect/entity_attention_data/cases.json` | 400 original correct-description cases with tokenization and saved continuations, used by the wrong-description driver |
| `scripts/inspect/entity_attention_data/manifest.json` | Frozen wrong-description assignments and assignment/review metadata |

The current profiler explicitly requires templates 0 and 3, exactly 100 entities,
and a 36-layer model. It needs generalization for Person-in-List/Friend and the
expanded paper suite. The main engine's template IDs 0/1/2/3 are Direct Fact,
Person in List, Friend, and Visitor Register. Other registered tasks are not
part of this four-template name-recovery cohort.

The restoration-plus-layer-scan workflow for Section 1 is not implemented:
the main CLI runs restoration separately and rejects overlapping restored and
ablated layers. Section 3 mechanism dissection and cross-family validation also
remain future work. The separate signed-description `run_restoration.py` is
outside this paper-script package.

## Environment and checkpoint

Recorded environment: Python 3.10, PyTorch 2.10.0+cu128, Transformers 4.57.3,
Matplotlib 3.10.8; BF16 native SDPA, greedy generation, 12 new tokens maximum.
Use these versions for historical reproduction; the repository's broad
Transformers dependency alone does not pin this environment.

Download `Qwen/Qwen3-8B-Base` at revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4` and set `MODEL_PATH` to that local
snapshot. Set `CUDA_VISIBLE_DEVICES` to an available GPU. The wrong-description
driver uses logical `cuda:0`. In the examples, `python` means the interpreter in
the experiment environment. Choose fresh output directories for every run.

```bash
MODEL_PATH=/path/to/pinned/Qwen3-8B-Base/snapshot
CORE=scripts/inspect/qwen_entity_attention_ablation.py
DATA=scripts/inspect/entity_attention_data
```

## Commands

```bash
# Ordinary baseline: no entity-value restrictions.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --ordinary-baseline-only --out-dir results/paper_ordinary

# Bottleneck control: intermediate and generated queries blocked, final prompt free.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --block-intermediate-prompt-entity --baseline-only --out-dir results/paper_bottleneck

# Section 1: direct-fact sliding final-query layer ablations under the bottleneck.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0 --block-intermediate-prompt-entity --widths 1,2,3,4,6,8 --out-dir results/paper_spans

# Section 1: early entity access blocked, L20-35 available to the final prompt query.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0 --block-intermediate-prompt-entity --widths 20 --start-layers 0 --out-dir results/paper_early_block

# Section 3: the currently supported direct/visitor attention profiles.
python scripts/inspect/qwen_prompt_attention_profile.py --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,3 --out-dir results/paper_profiles

# Section 4: full-cohort L23-35 positive-only rescue of person-in-list and friend.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 1,2 --block-intermediate-prompt-entity --restore-entity-attention-start-layer 23 --out-dir results/paper_rescue

# Section 4.2: final prompt and generated queries blocked; intermediate queries native.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --widths 36 --start-layers 0 --out-dir results/paper_necessity

# Correct-description ordinary baseline and final-query ablation.
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --entity-descriptions-json "$DATA/entity_descriptions.json" --ordinary-baseline-only --out-dir results/paper_description_ordinary
python "$CORE" --model "$MODEL_PATH" --entity-set diverse100 --template-ids 0,1,2,3 --entity-descriptions-json "$DATA/entity_descriptions.json" --widths 36 --start-layers 0 --out-dir results/paper_description_necessity

# Frozen wrong-description experiment, including paired tokenization checks.
python scripts/inspect/run_wrong_description.py --model "$MODEL_PATH" --cases "$DATA/cases.json" --manifest "$DATA/manifest.json" --outdir results/paper_wrong_description

# CPU synthetic profiler validation; no checkpoint or GPU required.
CUDA_VISIBLE_DEVICES= python scripts/inspect/qwen_prompt_attention_profile.py --self-test
```

The full-cohort rescue command above reruns all 200 cases. Historical bare-name
rescue results reran only 94 failures; their composite totals are not a
full-cohort restored evaluation. Non-ordinary baseline rows emitted by an
ablation run are generated-query-blocked controls, not ordinary inference.

## Interpretation and output review

Interventions remove only the entity's weighted value contribution on selected
query rows. They do not zero or renormalize softmax attention. Generated queries
are blocked in every intervention condition. Positive-only restoration adds
`max(ordinary_attention - live_attention, 0) * entity_value` per head at the
final prompt query; the effective coefficient sum can exceed one.

The profiler saves per-head tensors and plots the mean across entities of the
maximum across query heads, separately for each layer/token. Attention profiles
are measurements; generation supplies behavioral evidence.

Save and review complete continuations before semantic scoring. Existing
`entity_in_completion` fields are automatic diagnostics, not scientific labels.
Preserve reviewer identity, manual decisions, raw-row mapping, and termination
reason. The primary description metric is supplied-name recovery within the
saved horizon, not intended-celebrity identity. This package contains inference
scripts and frozen inputs; it does not include the complete historical output
review archive or claim that the proposed production experiments are finished.

Wrong-description assignments use seed 20260912 and occupation-mismatch review.
They are not a strict permutation: eight `a public figure` entries became
`a mathematician`. Per-entity description token lengths match in 66 cases;
17 are one token longer and 17 one token shorter. The frozen manifest records
these facts. The driver validates that tokens outside the description and the
entity token are unchanged.

## Frozen input provenance

Source runs: `qwen_entity_descriptions_20260911_162742` and
`qwen_wrong_description_control_20260912`. SHA-256:

```text
entity_descriptions.json ba3cdcbfddfd74d5192f7ec05e44dabb532fd373f4c62bf55d1138f06bf2837f
cases.json 4ad5404bf05a9c05c36722524b15c5384d282c3b22d95c1c90e905aa9c03c67a
manifest.json c5138ed724ed6ebe9889233bfbfcc53ec18c5cce2d2748548453e33d259f6908
```
