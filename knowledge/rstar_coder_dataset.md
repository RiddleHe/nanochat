# rStar-Coder dataset notes

Reference document for the `microsoft/rStar-Coder` dataset, which is the
source for our RL training data. Covers what the dataset is, which parts
we use, and the schema / canonical JSONL contract the prep script targets.

---

## The dataset: microsoft/rStar-Coder

- Publisher: Microsoft Research, accompanying paper "rStar-Coder: Scaling
  Competitive Code Reasoning with a Large-Scale Verified Dataset"
  (arXiv 2505.21297).
- HF: `microsoft/rStar-Coder`. License: MIT for the dataset.

### The five configs

| Config                  | Origin                                    | Purpose            |
| ----------------------- | ----------------------------------------- | ------------------ |
| `seed_sft`              | human-written problems + oracle solutions | SFT                |
| `seed_testcase`         | tests for the seed problems               | RL / verification  |
| `synthetic_sft`         | LLM-rewritten variants                    | SFT                |
| `synthetic_rl`          | LLM-rewritten problems for RL             | RL                 |
| `synthetic_rl_testcase` | tests for the synthetic RL problems       | RL                 |

**For nanochat RL we primarily use `seed_testcase`.** It contains the
problem statement, starter code (if any), and test inputs/outputs —
everything verification needs. `seed_sft` adds reference solutions which
pure RL ignores, so loading it is wasted bandwidth. The synthetic configs
are much larger and optional for extra problem diversity.

### Seed problem provenance

Problems come from competitive programming sources, deduplicated:

> AIZU, AtCoder, CodeChef, CodeWars, GeeksForGeeks, HackerEarth, HackerRank,
> LeetCode, Codeforces, IOI (2002–2023), USACO (2011–2023)

### `seed_testcase` schema

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
   - **Call-based** (starter_code present): the model writes a function
     with that signature. Each `inputs[i]` (after json.loads) is a list
     of positional arguments. Each `outputs[i]` is the expected return
     value.
   - **Standard I/O** (starter_code empty): the model writes a complete
     program. Each `inputs[i]` is a string piped to the program's stdin.
     Each `outputs[i]` is the expected stdout string.
3. **`func_name` and `class_name` columns** provide the function/class
   name directly — no regex extraction from `starter_code` needed. For
   stdio problems both are empty strings.
4. **LeetCode-style class wrapper.** Many call-based problems use the
   LeetCode convention `class Solution:\n    def fn_name(self, ...)`.
   Models imitate this. The sandbox driver must handle both
   `fn = ns[fn_name]` AND `fn = Solution().fn_name` — already implemented
   in `nanochat/rl_sandbox.py`.
5. **Test cases are graded by complexity scale**, with input sizes
   spanning a wide range. The `test_case_type` field tags this. Useful
   for stratified subsampling: keep all human tests + a budget-capped
   subset of synthesized tests per scale bucket.
6. **`is_synthesized`** separates human-written from rStar-generated
   tests. For a clean run you can train only on the human-authored ones.
7. **No `difficulty` column exists** in this config. The paper mentions
   difficulty levels but they are not in the released schema.

### Download cost

- The `seed_testcase` parquet is large (rows are heavy due to massive
  test suites). `load_dataset(..., split="train[:5]")` still downloads
  the full parquet, because HF datasets does not support byte-range
  row-group reads.
- `streaming=True` does not help — it still has to fetch the parquet bytes.
- The HF datasets-server REST API
  (`https://datasets-server.huggingface.co/rows?...`) returns sample rows
  over HTTP without any parquet download — use this for schema inspection
  in environments where the full download is impractical.
- Plan for a multi-GB one-time download on a machine with good bandwidth
  and disk.

---

## Canonical JSONL contract

The on-disk file produced by `scripts/prepare_rstar.py` and consumed by
`nanochat/rl_data.py`. One row per line:

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
    "source": "rstar_seed"
  }
}
```

The `payload` shape is per-`kind`. The dataset loader doesn't care; the
verifier dispatching on `kind` knows what fields to expect. To add a new
verifier kind, add a JSONL row with that `kind`, add an elif in
`nanochat.rl_data.verify`, write a sibling helper.

---

## Design decisions, with rationale

These were debated when the pipeline was first designed; recording here
so they don't get re-litigated.

### Two-script split (prep vs train), not one
Prep runs once per (dataset × tokenizer-family). Training runs many
times. Coupling them means every experiment pays the parse-and-template
cost; mixing failure modes means a malformed test row blows up a
multi-hour training run. The JSONL artifact is `head | jq`-able,
version-pinnable, and mirrors how `scripts/base_train.py` consumes
pre-tokenized shards.

### Canonical JSONL + verifier registry, not per-dataset Dataset subclasses
The four-field schema (`id, prompt, kind, payload, meta`) absorbs every
real coding RL dataset by varying `payload` per `kind`. Adding APPS,
MBPP, MATH later = a new prep script + (maybe) a new verifier. Zero
changes to the loader or training script.

### Reward = fraction of tests passed, not all-or-nothing
On hard problems with a small base model, all-or-nothing rewards produce
mostly zeros, which kills GRPO's group-relative gradient (zero variance
→ zero update). Fraction-of-tests-passed is dense, mirrors what
DeepCoder / OpenRLHF default to. Acceptable concession: model can game
by passing easy edge cases.

### Test subsampling at verify time, all tests stored in JSONL
JSONL stays the source of truth. Subsampling is `k_tests` deterministic
on `(example.id, step)` so all rollouts in a step see the same subset
(their rewards are comparable for the group-relative advantage).
Periodic eval runs the full test suite.

### Chat templating at prep time, not training time
Each JSONL is per-tokenizer-family. Prep script takes `--tokenizer` and
calls `tokenizer.apply_chat_template` once. Loader becomes trivial; no
template logic at runtime; matches how the policy was instruction-tuned.

### LeetCode `class Solution` fallback in the sandbox
Many rStar problems have starter code shaped as
`class Solution:\n    def fn_name(self, ...)`. Models imitate this. The
call-based driver tries `ns[fn_name]` first, then `Solution().fn_name`,
before raising.

### Two outcomes (`passed: bool` + free-form `detail: str`), not an enum
Training only consumes `passed`. Categorizing failures into
timeout/oom/runtime/compile is for human investigation, which can grep
`detail` strings. Add categories only when training (or a logged metric)
would behave differently based on them.

---

## Key references

- Paper: [arXiv 2505.21297](https://arxiv.org/abs/2505.21297)
- Dataset: [microsoft/rStar-Coder on HF](https://huggingface.co/datasets/microsoft/rStar-Coder)
- HF datasets-server REST (for schema inspection without download):
  `https://datasets-server.huggingface.co/rows?dataset=microsoft%2FrStar-Coder&config=seed_testcase&split=train&offset=0&length=2`
