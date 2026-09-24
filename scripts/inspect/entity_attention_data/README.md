# Entity-attention inputs

For experiments 01–10, use **entities100_balanced12.json**. It is the current ordered cohort of 100 people: 50 one-token and 50 two-token names under the recorded Qwen tokenizer and five templates. Preserve order, because array index defines entity_id.

Template IDs: 0 direct_fact, 1 person_in_list, 2 friend, 3 visitor_register, 7 name_badge. The main script treats every token in the inserted name as the entity span. It excludes any appended description from that span.

The self-contained private [experiment repository](https://github.com/RiddleHe/entity-attention-results) contains the runnable code, canonical identity map, exact templates, description mappings, cohort configs, audits, plotting scripts and saved results. Its root README is the complete handoff; no second repository is needed to run those experiments.

~~~bash
python scripts/inspect/qwen_entity_attention_ablation.py \
  --model /path/to/Qwen3-8B-Base/snapshot \
  --entities-json scripts/inspect/entity_attention_data/entities100_balanced12.json \
  --template-ids 0,1,2,3,7 \
  --ordinary-baseline-only --out-dir /path/to/fresh/ordinary
~~~

Outputs include raw generations.jsonl and metadata.json. Always review complete continuations for semantic categories; entity_in_completion is only a diagnostic.

Other files in this directory support historical cohorts and experiments. In particular, full_names100.json, the built-in diverse100/original10 sets, and the earlier 400-case wrong-description inputs are not the current balanced cohort. The five-template description experiments now use the main script plus the current mapping JSON; the legacy wrong-description driver is not required.

Blocking removes selected weighted entity values without renormalizing other attention probabilities. Signed restoration uses ordinary-minus-current coefficients independently per head and entity token, retaining negative corrections. Attention profiling is handled separately by qwen_prompt_attention_profile.py.
