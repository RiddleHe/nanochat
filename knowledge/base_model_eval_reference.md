# Evaluating Pretrained (Base) LLMs on Academic Benchmarks

Reference doc for how researchers evaluate **non-instruction-tuned** language models on MMLU, GSM8K, ARC-Challenge, and similar benchmarks. Grounded in actual harness code, task configs, and papers.

---

## Core Evaluation Framework

The de facto standard is **EleutherAI's lm-evaluation-harness** (`lm-eval`).

- **Repo:** https://github.com/EleutherAI/lm-evaluation-harness
- **Paper:** Biderman et al. (2024), "Lessons from the Trenches on Reproducible Evaluation of Language Models" — https://arxiv.org/abs/2405.14782
- Used as the backend for HuggingFace's Open LLM Leaderboard
- Task definitions live in YAML configs under `lm_eval/tasks/`

### Two fundamental output types

| Output Type | How it works | Used by |
|---|---|---|
| `multiple_choice` (loglikelihood) | Model scores each candidate continuation; highest log-prob wins. No generation. | MMLU, ARC-Challenge, HellaSwag, WinoGrande |
| `generate_until` | Model generates text greedily until stop token; answer extracted via regex/filter pipeline. | GSM8K, MATH, IFEval, BBH, HumanEval |

For **base models**, log-likelihood scoring is strongly preferred for MC tasks because base models aren't trained to output "A"/"B"/"C"/"D" — they're completion engines. Log-likelihood sidesteps the formatting problem entirely.

---

## MMLU (Massive Multitask Language Understanding)

- **Paper:** Hendrycks et al. (2020), "Measuring Massive Multitask Language Understanding" — https://arxiv.org/abs/2009.03300
- **Dataset:** https://huggingface.co/datasets/cais/mmlu
- **Task configs:** `lm_eval/tasks/mmlu/default/`
- **57 subjects**, 4-choice MC, ~14k test questions

### Evaluation method for base models

- **`output_type: multiple_choice`** — log-likelihood scoring, NOT generation
- **`num_fewshot: 5`** (standard; drawn from the dev split, 5 per subject)
- Prompt template (from harness default):
  ```
  The following are multiple choice questions (with answers) about {subject}.

  {5 few-shot examples}

  {question}
  A. {choice_a}
  B. {choice_b}
  C. {choice_c}
  D. {choice_d}
  Answer:
  ```
- For each of the 4 choices, the model computes `log P(continuation | prompt)`
- The choice with the highest (possibly normalized) log-prob is selected
- Metrics: `acc` (unnormalized) and `acc_norm` (byte-length normalized)
- Aggregation: micro-average over all samples (Hendrycks original) vs. macro-average over subjects (some papers differ here)

### MMLU variants in the harness

| Task name | Method | Notes |
|---|---|---|
| `mmlu` (default) | loglikelihood MC | Standard, used by Open LLM Leaderboard v1 |
| `mmlu_flan_n_shot_generative` | generate_until | Model generates answer letter; exact match |
| `mmlu_flan_n_shot_loglikelihood` | loglikelihood | FLAN-style prompts |
| `mmlu_flan_cot_fewshot` | generate_until | Chain-of-thought |

### Known pitfalls

- MMLU scores vary **up to 30%** across implementations (lm-eval vs. HELM vs. original code) due to prompt differences, normalization, and scoring method
- Reference: https://huggingface.co/blog/open-llm-leaderboard-mmlu
- Prompt sensitivity: 4-5% score variance across prompt templates on original MMLU
- ~6.5% of MMLU questions contain ground-truth errors (Gema et al., 2024)
- Data contamination is a major concern — the dataset is public and widely scraped

---

## ARC-Challenge (AI2 Reasoning Challenge)

- **Paper:** Clark et al. (2018), "Think you have Solved Question Answering? Try ARC" — https://arxiv.org/abs/1803.05457
- **Dataset:** https://huggingface.co/datasets/allenai/ai2_arc
- **Task config:** `lm_eval/tasks/arc/arc_challenge.yaml`
- Variable number of choices (3-5), science questions

### Evaluation method for base models

- **`output_type: multiple_choice`** — log-likelihood scoring
- **`num_fewshot: 25`** (standard in the harness)
- Metrics: `acc` and `acc_norm`
- For ARC-Challenge, `acc_norm` (byte-length normalized) is the standard reported metric

### Normalization methods (critical detail)

From the EleutherAI blog on MC normalization (https://blog.eleuther.ai/multiple-choice-normalization/):

1. **Unnormalized (`acc`):** `score_i = Σ log P(x_j | x_{0:j})` over continuation tokens. Biased toward shorter answers.

2. **Byte-length normalized (`acc_norm`):** Divides total log-likelihood by character/byte length of the completion. Tokenization-agnostic. This is what the harness reports as `acc_norm`.

3. **Token-length normalized:** Divides by number of tokens. NOT tokenization-agnostic, so the harness does NOT use this (GPT-3 paper used it for most tasks).

4. **Unconditional likelihood normalized:** `score_i = Σ [log P(x_j | context) - log P(x_j)]`. Measures how much the prompt *increases* the probability vs. unconditional. GPT-3 (Brown et al., 2020) used this specifically for ARC, OpenBookQA, and RACE. The harness does NOT use this by default — source of discrepancies with GPT-3 reported numbers.

---

## GSM8K (Grade School Math 8K)

- **Paper:** Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems" — https://arxiv.org/abs/2110.14168
- **Dataset:** https://huggingface.co/datasets/openai/gsm8k
- **Task configs:** `lm_eval/tasks/gsm8k/gsm8k.yaml` and `gsm8k-cot.yaml`
- 1,319 test problems, each with a chain-of-thought solution ending in `#### <number>`

### Evaluation method for base models

- **`output_type: generate_until`** — actual text generation, NOT log-likelihood
- **`num_fewshot: 5`** (default) or **8** (CoT variant)
- **`temperature: 0.0, do_sample: false`** — greedy decoding
- Stop strings: `["Question:", "</s>", "<|im_end|>"]`
- **Answer extraction pipeline:**
  1. `strict-match` filter: regex `#### (\-?[0-9\.\,]+)` to extract the final answer
  2. `flexible-extract` filter: fallback regex grabbing the last number
  3. Strip commas, dollar signs, trailing periods
  4. `exact_match` against ground truth

### From the actual YAML config

```yaml
dataset_path: openai/gsm8k
output_type: generate_until
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
```

### Why GSM8K is harder for base models

- The model must actually **generate** well-formatted CoT reasoning and produce `#### <answer>`
- Few-shot examples do heavy lifting to steer the base model into the right output format
- No log-likelihood shortcut — if the model can't write coherent math steps, it fails
- Self-consistency variant: sample N times at temperature > 0, majority-vote extracted answers

---

## Running evaluations

### Basic command (local HF model)

```bash
# Install
pip install lm-eval[hf]

# MMLU (5-shot, log-likelihood)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path results/

# ARC-Challenge (25-shot, log-likelihood)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size auto

# GSM8K (5-shot, generative)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size auto
```

### With vLLM (faster)

```bash
pip install lm-eval[vllm]

lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=auto,gpu_memory_utilization=0.8 \
  --tasks mmlu,arc_challenge,gsm8k \
  --batch_size auto \
  --output_path results/
```

### Important flags

- `--num_fewshot N` — number of in-context examples
- `--batch_size auto` — auto-detect optimal batch size
- `--limit N` — run on N samples only (for debugging)
- `--log_samples` — save per-sample predictions
- `--output_path` — save JSON results
- `--apply_chat_template` — for instruct models only, NOT base models

### API-served models

```bash
lm_eval --model local-completions \
  --model_args model=my-model,base_url=http://localhost:8000/v1/completions \
  --tasks gsm8k \
  --batch_size 16
```

**Caveat:** API endpoints typically don't expose logprobs, so you're limited to generative tasks (GSM8K, IFEval, BBH) and cannot run loglikelihood-based tasks (default MMLU, ARC, HellaSwag) unless the API returns logprobs.

---

## Interpreting results

### Output format

The harness outputs JSON with per-task results:

```json
{
  "results": {
    "mmlu_abstract_algebra": {
      "acc,none": 0.35,
      "acc_stderr,none": 0.048,
      "acc_norm,none": 0.38,
      "acc_norm_stderr,none": 0.049
    }
  }
}
```

### Which metric to report

| Benchmark | Standard metric | Notes |
|---|---|---|
| MMLU | `acc` (micro-averaged) | Some papers use macro-average over subjects |
| ARC-Challenge | `acc_norm` | Length-normalized; `acc` also reported |
| HellaSwag | `acc_norm` | Length-normalized |
| GSM8K | `exact_match` (strict-match) | Also reports flexible-extract |
| WinoGrande | `acc` | |

### Standard error

The harness reports SE of the mean: `SE = sqrt(p * (1-p) / n)` (Bernoulli SE). For MMLU with ~14k questions, this gives SE ≈ 0.4%. For GSM8K with ~1.3k questions, SE ≈ 1.3%. Always report these when comparing models.

---

## Key references

| Reference | URL |
|---|---|
| lm-evaluation-harness repo | https://github.com/EleutherAI/lm-evaluation-harness |
| Harness paper (Biderman et al., 2024) | https://arxiv.org/abs/2405.14782 |
| MC normalization blog | https://blog.eleuther.ai/multiple-choice-normalization/ |
| HF MMLU discrepancy analysis | https://huggingface.co/blog/open-llm-leaderboard-mmlu |
| MMLU paper (Hendrycks et al., 2020) | https://arxiv.org/abs/2009.03300 |
| ARC paper (Clark et al., 2018) | https://arxiv.org/abs/1803.05457 |
| GSM8K paper (Cobbe et al., 2021) | https://arxiv.org/abs/2110.14168 |
| MMLU-Pro paper (TIGER-Lab) | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| MMLU-Pro-NoMath (logprobs discussion) | https://huggingface.co/blog/sam-paech/mmlu-pro-nomath |
| HELM MMLU reproduction | https://crfm.stanford.edu/2024/05/01/helm-mmlu.html |
| GPT-3 paper (Brown et al., 2020) | https://arxiv.org/abs/2005.14165 |
| Task YAML configs (GSM8K) | https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml |
| New task guide | https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md |

---

## Common gotchas

1. **Never compare MMLU numbers across different implementations** (lm-eval vs. HELM vs. original code) without checking prompt templates and scoring method
2. **`acc` vs `acc_norm`** — always specify which you're reporting; `acc_norm` is generally more reliable for MC tasks with variable-length answers
3. **Base vs. instruct model eval** — do NOT use `--apply_chat_template` for base models; do NOT use `--fewshot_as_multiturn` for base models
4. **Log-likelihood scoring gives base models a boost** over generative scoring because it constrains the output space — this is a feature, not a bug, for measuring knowledge
5. **GSM8K on base models** depends heavily on few-shot formatting — the model needs to learn the `#### <number>` pattern from the examples
6. **Data contamination** — MMLU is public and widely scraped; consider MMLU-Pro or MMLU-CF for more reliable differentiation
7. **Aggregation** — Hendrycks et al. use micro-average (over all samples); some papers use macro-average (over subjects). These differ.
