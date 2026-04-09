# rStar-Coder + nanochat RL data pipeline

Reference document for the RL training data path. Covers what the rStar-Coder
dataset is, what we use from it, and the design decisions baked into the
pipeline modules under `nanochat/rl_*` and `scripts/rl_train.py`. Implementation
state and outstanding work are tracked in the conversation, not here.

---

## The dataset: microsoft/rStar-Coder

- Publisher: Microsoft Research, accompanying paper "rStar-Coder: Scaling
  Competitive Code Reasoning with a Large-Scale Verified Dataset" (arXiv
  2505.21297, May 2025).
- HF: `microsoft/rStar-Coder`. License: MIT for the dataset.
- Total release: ~473 GB across five configs. Mostly synthetic; the seed
  portion is a small fraction of that.

### The five configs

| Config                  | Origin                                | Purpose          |
| ----------------------- | ------------------------------------- | ---------------- |
| `seed_sft`              | human-written problems + oracle solutions | SFT          |
| `seed_testcase`         | tests for the seed problems           | RL / verification |
| `synthetic_sft`         | LLM-rewritten variants                | SFT              |
| `synthetic_rl`          | LLM-rewritten problems for RL         | RL               |
| `synthetic_rl_testcase` | tests for the synthetic RL problems   | RL               |

**For nanochat RL we only need `seed_testcase`.** It already contains the
problem statement, the starter code (if any), and the test inputs/outputs —
which is everything verification needs. `seed_sft` is the SFT-shaped view
that adds reference solutions; we ignore reference solutions in pure RL, so
loading `seed_sft` would be wasted bandwidth. The synthetic configs are an
order of magnitude larger (most of the 473 GB) and again unnecessary for the
seed-only RL run.

### Seed problem provenance

Problems come from 11 competitive programming sources, deduplicated:

> AIZU, AtCoder, CodeChef, CodeWars, GeeksForGeeks, HackerEarth, HackerRank,
> LeetCode, Codeforces, IOI (2002–2023), USACO (2011–2023)

The paper reports **37,754 unique expert-written problems** retained after
dedup and after filtering out problems with no reference solution (down from
57,215 raw collected). Each problem has at least one oracle solution that
the rStar pipeline used to generate and validate the expanded test sets.

**Actual released counts** (verified 2026-04-09 via HF dashboard):
- `seed_testcase`: **~7.8k rows** (not 37k — only problems with test cases)
- `seed_sft`: ~592k rows (multiple solutions per problem)
- `synthetic_rl_testcase`: **~464k rows**
- `synthetic_rl`: ~398k rows
- `synthetic_sft`: ~398k rows

Note: the HF dataset viewer and datasets-server `/size` API only show
the first 5GB of data for large configs. `seed_testcase` shows "317 rows"
in the viewer because each row is ~15MB (massive test suites), but
`load_dataset` downloads the full parquet and returns all ~7.8k rows.

### `seed_testcase` schema (verified 2026-04-09)

Confirmed via HF datasets-server API and dashboard inspection.

```
seed_testcase columns:
  question_id      : string                    # join key
  question         : string                    # problem statement
  starter_code     : string                    # function signature for call-based; "" for stdio
  inputs           : large_string              # JSON-encoded list of test inputs (single blob)
  outputs          : large_string              # JSON-encoded list of expected outputs (single blob)
  is_synthesized   : large_string              # JSON-encoded list of per-test booleans
  test_case_type   : large_string              # JSON-encoded list of per-test category tags
  func_name        : string                    # function name for call-based; "" for stdio
  class_name       : string                    # class name (e.g. "Solution") if applicable; "" otherwise
```

Important quirks:

1. **`inputs`, `outputs`, `is_synthesized`, `test_case_type` are each a
   single `large_string`** — one JSON blob per row, not a sequence of
   separate strings. `json.loads(row["inputs"])` yields a list. For
   call-based problems, each element of that list is itself a JSON-encoded
   string (arg list / return value) that needs a second `json.loads`. For
   stdio, elements are plain strings (stdin/stdout text). Expect some rows
   to have malformed payloads — drop at prep time, log the count.
2. **Both problem modes coexist in one table**, distinguished by whether
   `starter_code` is non-empty:
   - **Call-based** (starter_code present): the model writes a function with
     that signature. Each `inputs[i]` (after json.loads) is a list of
     positional arguments. Each `outputs[i]` is the expected return value.
   - **Standard I/O** (starter_code empty): the model writes a complete
     program. Each `inputs[i]` is a string piped to the program's stdin.
     Each `outputs[i]` is the expected stdout string.
3. **`func_name` and `class_name` columns** provide the function/class name
   directly — no need to regex-extract from `starter_code`. For stdio
   problems both are empty strings.
4. **LeetCode-style class wrapper.** Many call-based problems use the
   LeetCode convention `class Solution:\n    def fn_name(self, ...)`. Models
   imitate this. The sandbox driver must handle both `fn = ns[fn_name]` AND
   `fn = Solution().fn_name` — already implemented in `nanochat/rl_sandbox.py`.
5. **Test cases are graded by complexity scale**, with input sizes spanning
   10⁰ to 10⁵. The `test_case_type` field tags this. Useful for stratified
   subsampling: keep all human tests + a budget-capped subset of synthesized
   tests per scale bucket.
6. **`is_synthesized`** lets you separate human-written from rStar-generated
   tests. For a clean run you can train only on the human-authored ones.
7. **No `difficulty` column exists** in this config. The paper mentions
   difficulty levels but they are not included in the released schema.

### Download cost: a real friction point

- The `seed_testcase` parquet is **multi-GB** (rows are ~15MB each due to
  massive test suites). `load_dataset(..., split="train[:5]")` still
  downloads the full parquet, because HF datasets does not support
  byte-range row-group reads.
- `streaming=True` does not help — it still has to fetch the parquet bytes.
- The HF datasets-server REST API
  (`https://datasets-server.huggingface.co/rows?...`) returns sample rows
  over HTTP without any parquet download — use this for schema inspection
  in environments where the multi-GB download is impractical. Note: the
  `/size` endpoint and the HF viewer both cap at 5GB, so they report
  only ~317 rows; the actual dataset has ~7.8k rows.
- For real prep work, plan for a multi-GB one-time download and run on a
  machine with good bandwidth and disk.

---

## Pipeline modules in nanochat/

Implemented and tested:

- **`nanochat/rl_loss.py`** — `grpo_loss`, `dapo_loss`, `reinforce_loss`,
  `ALGORITHMS` registry. Each loss takes `(logprobs, old_logprobs, rewards, **kw)`
  and computes its own advantages internally.

- **`nanochat/rl_rollout.py`** — `get_logprobs`, `generate_rollouts`,
  `prepare_batch`, `vllm_weight_sync`. The weight sync uses vLLM's
  `collective_rpc` (modern path, TP>1 capable) with a fallback to direct
  driver-worker access for older vLLM. Hands the HF `state_dict` to vLLM's
  own `model.load_weights`, which handles HF→vLLM name mapping (fused QKV,
  fused gate_up_proj, MoE expert packing). Calls `torch.cuda.empty_cache()`
  after sync.

- **`nanochat/rl_sandbox.py`** — bounded execution of model-generated Python.
  Subprocess per test, `os.setsid` + `RLIMIT_AS` + `RLIMIT_NPROC` via
  `preexec_fn`, `subprocess.communicate(timeout=...)`, `os.killpg` on
  timeout. Two modes share one transport:
  - `code_call_based`: JSON driver via `python -I -c`, stdin carries
    `{code, fn_name, args}`, driver writes one JSON line `{ok: ...}` or
    `{err: ...}`. Driver suppresses user prints inside the exec so they
    can't pollute the result line. Falls back to `Solution().fn_name` if
    the function isn't at module level.
  - `code_stdin_stdout`: writes user code to a temp file in a fresh tempdir,
    runs `python -I solution.py` with the test stdin piped in.
  - Returns `TestResult(passed: bool, detail: str, duration_s: float)`. No
    outcome enum — training only consumes `passed`. `detail` is a free-form
    string carrying whatever the failure looked like, for human investigation.
  - Output normalization: per-line `rstrip` + drop trailing blank lines.
  - Threading note: `preexec_fn` is not safe from a multi-threaded parent.
    Call from process pools or single-threaded code only.

- **`nanochat/rl_data.py`** — JSONL loader, distributed loader, verifier,
  reward worker pool.
  - `RLExample` dataclass: `id, prompt, kind, payload, meta`.
  - `JSONLRLDataset(path, difficulty_filter=None)`: loads the canonical
    JSONL into memory. Optional difficulty filter at load time (drops rows
    entirely; doesn't sample at training time).
  - `build_rl_dataset(name, split, difficulty_filter)`: resolves to
    `<base_dir>/data/rl/<name>_<split>.jsonl` via `nanochat.common.get_base_dir()`.
  - `distributed_rl_loader(dataset, prompts_per_step, world_size, rank,
    seed, resume_state)`: yields `(list[RLExample], state_dict)`. Each rank
    gets a disjoint slice of the same epoch-shuffled order. State is
    `{epoch, cursor}`; resume reproduces the exact stream. Drops partial
    epoch tails so step shape stays constant. Asserts
    `prompts_per_step % world_size == 0` and `prompts_per_step <= len(dataset)`.
  - `extract_code(response)`: regex on ```` ```python ``` ```` fenced blocks,
    takes the **last** match (models often emit "let me think...```python
    bad``` actually ```python good```"), falls back to whole response.
  - `verify(example, response, step, k_tests)`: top-level dispatch on
    `example.kind`. Currently handles both code kinds via `verify_code`.
    Add new kinds (e.g. `python_asserts`, `math_boxed`) as new elif branches
    + sibling helpers.
  - `verify_code`: subsamples up to `k_tests` deterministically by
    `(example.id, step)` so all rollouts in one step see the same test
    subset (rewards are then comparable for group-relative advantages).
    Reward = passed / executed. `info` carries `{passed, total, n_total,
    first_failure}`.
  - `RewardWorkerPool(num_workers, k_tests)`: parallel reward computation
    via `multiprocessing.get_context("spawn").Pool`. **Spawn, not fork** —
    fork from a CUDA-initialized parent corrupts the child. `num_workers=0`
    runs synchronously in the parent for debugging.
  - `_score_one`: pool worker entry point. Top-level for picklability.
    On exception, logs trace to stderr identifying the example id, then
    re-raises. Loud failure, not silent zeros.

- **`scripts/rl_train.py`** — orchestration only. CLI → compute_init →
  HF model + DDP wrap → vLLM (rank 0) → dataset/loader/rewarder/loss →
  optimizer → loop → save. The loop is:
  ```
  examples   = next(loader)
  rollouts   = generate_rollouts(vllm_engine, ..., prompt_texts)
  responses  = [r["response"] for r in rollouts]
  expanded   = [examples[i // num_samples] for i in range(len(rollouts))]
  rewards, _ = rewarder.score(expanded, responses, step=step)
  batch      = prepare_batch(rollouts, rewards, ...)
  old_lp     = get_logprobs(raw_model, ...)
  for micro_batch:
      loss = loss_fn(logprobs, old_lp, rewards, ...) / n_microbatches
      loss.backward()
  optimizer.step()
  vllm_weight_sync(vllm_engine, raw_model)
  ```
  CLI: `--task rstar_seed`, `--difficulty`, `--reward-workers`, `--k-tests`,
  `--algorithm` (grpo/dapo/reinforce), plus standard model/optim/runtime args.

---

## Canonical JSONL contract

The on-disk file produced by `scripts/prepare_rstar.py` and consumed by
`JSONLRLDataset`. One row per line:

```json
{
  "id": "rstar/<question_id>",
  "prompt": "<chat-templated prompt string>",
  "kind": "code_call_based" | "code_stdin_stdout",
  "payload": {
    "fn_name": "twoSum",                  // call-based only; null/absent for stdio
    "starter_code": "class Solution:...", // optional; included for prep traceability
    "inputs":  [...],                     // call-based: list[list], stdio: list[str]
    "outputs": [...],                     // call-based: list[Any], stdio: list[str]
    "time_limit_s": 4.0,
    "memory_limit_mb": 256
  },
  "meta": {
    "source": "rstar_seed",
    "n_tests": 12
  }
}
```

Path convention: `<NANOCHAT_BASE_DIR or ~/.cache/nanochat>/data/rl/rstar_seed_train.jsonl`
and `..._test.jsonl`.

The `payload` shape is per-`kind`. The dataset loader doesn't care; the
verifier dispatching on `kind` knows what fields to expect. To add a new
verifier kind, add a JSONL row with that `kind`, add an elif in
`nanochat.rl_data.verify`, write a sibling helper.

---

## Design decisions, with rationale

These were debated in the conversation; recording them here so they don't
get re-litigated.

### Two-script split (prep vs train), not one
Prep runs once per (dataset × tokenizer-family). Training runs hundreds of
times. Coupling them means every experiment pays the parse-and-template
cost; mixing failure modes means a malformed test row blows up a multi-hour
training run. The JSONL artifact is `head | jq`-able, version-pinnable, and
mirrors how `scripts/base_train.py` consumes pre-tokenized shards.

### Canonical JSONL + verifier registry, not per-dataset Dataset subclasses
The four-field schema (`id, prompt, kind, payload, meta`) absorbs every
real coding RL dataset by varying `payload` per `kind`. Adding APPS, MBPP,
MATH later = a new prep script + (maybe) a new verifier. Zero changes to
the loader or training script.

### Reward = fraction of tests passed, not all-or-nothing
On hard problems with a small base model, all-or-nothing rewards produce
mostly zeros, which kills GRPO's group-relative gradient (zero variance →
zero update). Fraction-of-tests-passed is dense, mirrors what DeepCoder /
OpenRLHF default to. Acceptable concession: model can game by passing
easy edge cases.

### Test subsampling at verify time, all tests stored in JSONL
JSONL stays the source of truth. Subsampling is `k_tests` deterministic on
`(example.id, step)` so all rollouts in a step see the same subset (their
rewards are comparable for the group-relative advantage). Periodic eval
runs the full test suite — implementation TODO.

### Chat templating at prep time, not training time
Each JSONL is per-tokenizer-family. Prep script takes `--tokenizer` and
calls `tokenizer.apply_chat_template` once. Loader becomes trivial; no
template logic at runtime; matches how the policy was instruction-tuned.

### `spawn` multiprocessing context, not `fork`
Forking from a CUDA-initialized process corrupts the child. The reward
worker pool pays Python startup cost per spawn, but workers are reused
across the run.

### vLLM in-process, not HTTP server
HTTP server requires NCCL setup or restart-to-sync between trainer and
inference. In-process vLLM allows direct weight sync via `collective_rpc`
into the worker's `model.load_weights`. This is what TRL/OpenRLHF/verl do
when training and inference colocate on the same GPU. HTTP server is for
physically separate inference clusters.

### LeetCode `class Solution` fallback in the sandbox
Many rStar problems have starter code shaped as `class Solution:\n
    def fn_name(self, ...)`. Models imitate this. The call-based driver
tries `ns[fn_name]` first, then `Solution().fn_name`, before raising.

### Two outcomes (`passed: bool` + free-form `detail: str`), not an enum
Training only consumes `passed`. Categorizing failures into
timeout/oom/runtime/compile is for human investigation, which can grep
`detail` strings. Add categories only when training (or a logged metric)
would behave differently based on them.

### No `RLDataset` ABC, no `VERIFIERS` registry, no DDP rollout fan-out
Each is a deferred decision: add when there are real entries / real
multi-rank runs. Currently one concrete dataset class, one verify
function with an if/elif, single-rank rollout. The training script crashes
loudly on multi-rank because vLLM is only loaded on rank 0; that's
intentional until someone needs DDP RL.

---

## Key references

- Paper: [arXiv 2505.21297](https://arxiv.org/abs/2505.21297)
- Dataset: [microsoft/rStar-Coder on HF](https://huggingface.co/datasets/microsoft/rStar-Coder)
- HTML version of paper: <https://arxiv.org/html/2505.21297v1>
- Verl-formatted mirror (synthetic, not seed): <https://huggingface.co/datasets/sungyub/rstar-coder-verl>
- HF datasets-server REST (for schema inspection without download):
  `https://datasets-server.huggingface.co/rows?dataset=microsoft%2FrStar-Coder&config=seed_testcase&split=train&offset=0&length=2`
