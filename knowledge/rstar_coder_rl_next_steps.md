# rStar-Coder RL pipeline — next steps

Continuation plan for finishing the data pipeline. Companion to
`rstar_coder_rl_pipeline.md`, which covers dataset facts and design
decisions; this file is the action list.

---

## Step 0 — Verify schema before committing (10 min)

On the new machine, **before** writing any prep code:

```python
from datasets import load_dataset
ds = load_dataset("microsoft/rStar-Coder", "seed_testcase", split="train[:5]")
print("columns:", ds.column_names)
for k, v in ds[0].items():
    print(f"{k}: {type(v).__name__} → {repr(v)[:400]}")
```

Confirm:

- `inputs`/`outputs` are lists of JSON-encoded strings.
- `starter_code` shape (empty string vs missing for stdio mode).
- Whether `is_synthesized` and `test_case_type` are per-test sequences or
  scalars.
- Whether there is a `difficulty` column. The paper mentions difficulty,
  but it has not been confirmed in the released schema.

Also load a few different `question_id`s to see at least one stdio problem
and one call-based problem.

If anything diverges from the assumed schema in
`knowledge/rstar_coder_rl_pipeline.md`, update that doc **first**, then
proceed.

---

## Step 1 — Write `scripts/prepare_rstar.py` (~120 lines)

Single script. Output `<base_dir>/data/rl/rstar_seed_train.jsonl`, plus
`..._test.jsonl` if a test split exists (otherwise emit only train and
let later eval do its own random holdout).

CLI:

```
--tokenizer Qwen/Qwen3-0.6B   # required; templates per family
--limit N                     # optional; for smoke testing on a small subset
--human-tests-only            # filter out synthesized tests
--time-limit 4.0
--memory-limit-mb 256
```

Per-row processing:

1. `inputs_parsed = [json.loads(s) for s in row["inputs"]]`, same for
   `outputs`. Wrap in try/except — drop the row on failure, increment a
   `malformed` counter.
2. Drop rows where `len(inputs_parsed) == 0` or lengths mismatch — increment
   a counter.
3. Detect mode:
   `kind = "code_call_based" if row["starter_code"].strip() else "code_stdin_stdout"`.
4. For call-based: extract `fn_name` from `starter_code` with
   `re.search(r"def\s+(\w+)\s*\(", starter)`. If no match, drop and count.
5. (Optional) If `--human-tests-only`, filter `inputs_parsed` and
   `outputs_parsed` by `is_synthesized[i] == False`. Drop the row if zero
   tests remain.
6. Build the prompt:

   ```python
   user_content = row["question"].strip()
   if starter_code:
       user_content += (
           "\n\nUse this starter code:\n```python\n"
           + starter_code.strip()
           + "\n```"
       )
   user_content += "\n\nProvide your complete solution as a single Python code block."
   prompt_str = tokenizer.apply_chat_template(
       [{"role": "user", "content": user_content}],
       tokenize=False,
       add_generation_prompt=True,
   )
   ```

7. Build the JSONL row matching the schema in
   `knowledge/rstar_coder_rl_pipeline.md`. `meta.difficulty` only if a
   difficulty field exists in the source row.
8. `f.write(json.dumps(row) + "\n")`.

End with a counter dump:

```
loaded N rows
dropped M (malformed=..., empty_tests=..., missing_fn_name=..., zero_human_tests=...)
emitted K rows to <path>
  - call_based: ..
  - stdin_stdout: ..
```

---

## Step 2 — Smoke test the prep script with `--limit 50`

Run with a tiny limit, then `head -1 <path> | python -m json.tool` to
confirm shape, plus:

```python
from nanochat.rl_data import JSONLRLDataset, verify
ds = JSONLRLDataset("<path>")
print(len(ds), ds[0].kind, list(ds[0].payload.keys()))
# pick a call-based and a stdio example
# hand-craft a correct response and verify reward == 1.0
```

This validates that the schema written by prep is the schema the loader
expects, end-to-end, before doing the full multi-GB run.

---

## Step 3 — Full prep run

Drop `--limit`. Expect a multi-GB download plus a few minutes of parsing.
Confirm the final counters look sane (drop rate ≤10%, both modes present
in reasonable proportions).

---

## Step 4 — End-to-end RL smoke test on real data

Tiny model + tiny step shape:

```
python -m scripts.rl_train \
  --model Qwen/Qwen3-0.6B \
  --task rstar_seed \
  --num-steps 5 \
  --prompts-per-step 4 \
  --num-samples 4 \
  --reward-workers 4 \
  --k-tests 5
```

This will exercise: vLLM load, dataset load, loader, rollout generation,
reward worker pool against real model output, `prepare_batch`,
`get_logprobs`, loss, backward, weight sync. **Expect failures** — this is
the first time the whole pipeline meets real data.

Likely failure modes to watch for:

- **vLLM weight sync API mismatch.** If `collective_rpc` isn't on the `LLM`
  object, our fallback path runs but only handles TP=1.
- **Schema mismatch in payload fields.** Hopefully caught at Step 2 already.
- **All-zero rewards on every prompt.** Means the base model can't solve
  any of these even partially. Mitigation: filter the JSONL down to the
  easiest difficulty bucket, or smaller test counts so partial credit
  shows up sooner.
- **Sandbox timeouts dominating step time.** If most rollouts produce
  infinite-loop code, every test waits the full timeout. Drop
  `--time-limit` to 2 s or lower.
- **Reward worker pool deadlock or pickle errors.** `_score_one` and
  `RLExample` are picklable, but if any payload field accidentally contains
  non-picklable data (numpy scalars from HF, etc.), the spawn workers will
  die. Cast everything to Python natives at prep time.

---

## Step 5 — Production-ish run + outstanding RL features

Once Step 4 produces non-degenerate gradients, add:

- **Group-degenerate filtering** in the training loop: if all `num_samples`
  for a prompt got the same reward, drop that group from the loss.
  ~10 lines.
- **Real eval pass.** Currently `--eval-every` just prints
  "Evaluating...". A reasonable shape: every N steps, sample K problems
  from `rstar_seed_test.jsonl` (or a held-out slice of train), generate
  one rollout each greedy, score with **all** tests, log pass rate to
  wandb.
- **DDP rollout fan-out** if you go multi-rank: either give every rank its
  own colocated vLLM (simplest, what OpenRLHF does for small models) or
  have rank 0 gather prompts and scatter rollouts.
- **Better prompt format** if pass rate is too low: add a one-shot example,
  system prompt, or chain-of-thought scaffold. This is dataset-side and
  lives in the prep script.

---

## Stretch / nice-to-have, not blocking

- Stratified test subsampling at prep time (keep all human tests + a budget
  per complexity bucket from synthesized) instead of uniform random
  subsample at verify time.
- Sandbox CPU rlimit as defense in depth.
- `info` aggregation to wandb tables: pass rate, mean failure type, slowest
  example per step.
- Vendoring our own dedup against the eval set (HumanEval/MBPP) to avoid
  contamination.

---

## Summary

The two pieces of true work are **Step 0 (verify schema)** and **Step 1
(write prep script)**. Everything after that is running things and
iterating on what breaks. The companion file
`rstar_coder_rl_pipeline.md` should be enough for a fresh session to resume
without re-asking any of the questions worked through earlier.
