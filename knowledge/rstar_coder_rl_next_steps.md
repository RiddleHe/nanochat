# rStar-Coder RL pipeline — next steps

Continuation plan for finishing the data pipeline. Companion to
`rstar_coder_rl_pipeline.md`, which covers dataset facts and design
decisions; this file is the action list.

---

## Step 0 — Schema ✅ (2026-04-09)

Verified via HF datasets-server API. See `rstar_coder_rl_pipeline.md`
for the authoritative schema. Key facts:

```
seed_testcase columns:
  question_id      : string
  question         : string
  starter_code     : string
  inputs           : large_string  (JSON list)
  outputs          : large_string  (JSON list)
  is_synthesized   : large_string  (JSON list)
  test_case_type   : large_string  (JSON list)
  func_name        : string
  class_name       : string
```

- ~7.8k rows (HF viewer shows first 317 due to 5GB preview cap)
- No `difficulty` column
- `func_name`/`class_name` provided directly (no regex extraction needed)

---

## Step 1 — Write `scripts/prepare_rstar.py` ✅ DONE (2026-04-09)

Script written. Handles both schema corrections (func_name column, no
difficulty, JSON large_string parsing). See script for details.

CLI:
```
python -m scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B
python -m scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B --limit 10
python -m scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B --human-tests-only
```

---

## Step 2 — Smoke test the prep script with `--limit 10`

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

Drop `--limit`. Expect a ~2.3 GB download plus a few minutes of parsing.
Confirm the final counters look sane (drop rate ≤10%, both modes present
in reasonable proportions).

**Note:** ~7.8k problems total (not 317 — the HF viewer was truncated
by the 5GB preview cap). Good diversity for RL.

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
  any of these even partially. Mitigation: use smaller test counts so
  partial credit shows up sooner.
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

- **Prepare `synthetic_rl_testcase`** (~464k rows) for even more
  problem diversity beyond the ~7.8k seed problems.
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

Steps 0 and 1 are done. The prep script (`scripts/prepare_rstar.py`) is
written and handles the actual schema. Next: smoke test (Step 2), then
full run (Step 3), then end-to-end RL test (Step 4).
