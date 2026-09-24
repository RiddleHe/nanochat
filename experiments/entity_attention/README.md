# Portable entity-attention experiments 01–10

The full handoff and saved results are in the private [entity-attention-results repository](https://github.com/RiddleHe/entity-attention-results). Its root README documents protocols, exact cohorts, commands, outputs, categories and interpretation.

From this repository root:

~~~bash
pip install -r experiments/entity_attention/requirements.txt
python experiments/entity_attention/run.py 10 --results ../entity-attention-results --out /path/to/fresh/run --dry-run
python experiments/entity_attention/verify_results.py ../entity-attention-results
~~~

Remove --dry-run to run on the device selected through CUDA_VISIBLE_DEVICES; the default model is the pinned Qwen3-8B-Base revision. --model accepts an existing snapshot. Output directories must be new. The runner uses the maintained main/profiler implementations, with no machine-specific paths or fixed GPU assignment.

data/entities100.json is the current ordered 50-one-token/50-two-token cohort; template IDs are 0,1,2,3,7. configs/01.json through 10.json preserve the exact per-template entity selections. 06–09 are all 500 pairs; 10 deliberately uses the 446 survivors from 01. Do not substitute the older full_names100.json or main-script default entity pool.

analysis/ contains CPU plotting commands for saved results. New inference produces raw outputs with annotation_status=not_reviewed, never automatic semantic judgments.

The raw results, attention tensors, annotations and historical source snapshots stay in the private repository. This public package contains experiment inputs, code, tests and reproduction instructions only.
