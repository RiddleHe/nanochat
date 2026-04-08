"""
RL dataset, distributed loader, verifier, and reward worker pool.

Data flow:

    JSONL on disk  ──►  JSONLRLDataset  ──►  distributed_rl_loader
                                                      │
                                                      ▼
                          rollouts ──► RewardWorkerPool.score ──► (rewards, infos)
                                                │
                                                ▼ per (example, response)
                                              verify
                                                │
                                                ▼
                                          verify_code
                                                │
                                                ▼
                                       run_test  (rl_sandbox)

The on-disk JSONL is the canonical contract. Each row:

    {
      "id": "rstar/1234",
      "prompt": "<chat-templated prompt>",
      "kind": "code_call_based" | "code_stdin_stdout",
      "payload": { ... per-kind ... },
      "meta":    { "source": "...", "difficulty": "...", ... }
    }

For the two code kinds, payload carries inputs/outputs/limits and (call-based
only) fn_name. New verifier kinds (e.g. python_asserts, math_boxed) get added
to `verify` with an elif branch and a sibling verify_* helper.
"""

from __future__ import annotations

import os
import re
import sys
import json
import random
import traceback
from dataclasses import dataclass, field
from typing import Any

from nanochat.common import get_base_dir
from nanochat.rl_sandbox import run_test


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------

@dataclass
class RLExample:
    id: str
    prompt: str
    kind: str
    payload: dict
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class JSONLRLDataset:
    """Loads a JSONL of RLExamples into memory.

    rStar seed is ~37k rows × a few KB each = roughly 100MB resident. Cheap.
    `difficulty_filter` lets you keep only one bucket without re-prepping.
    """

    def __init__(self, path: str, difficulty_filter: str | None = None):
        self.path = path
        self.examples: list[RLExample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if difficulty_filter is not None:
                    if row.get("meta", {}).get("difficulty") != difficulty_filter:
                        continue
                self.examples.append(RLExample(
                    id=row["id"],
                    prompt=row["prompt"],
                    kind=row["kind"],
                    payload=row["payload"],
                    meta=row.get("meta", {}),
                ))
        if not self.examples:
            raise RuntimeError(f"JSONLRLDataset loaded zero examples from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> RLExample:
        return self.examples[i]


def build_rl_dataset(name: str, split: str = "train",
                     difficulty_filter: str | None = None) -> JSONLRLDataset:
    """Resolve a dataset name to its on-disk JSONL and load it.

    Convention: <base_dir>/data/rl/<name>_<split>.jsonl
    Prep scripts (scripts/prepare_*.py) are responsible for producing the file.
    """
    base = get_base_dir()
    path = os.path.join(base, "data", "rl", f"{name}_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL dataset not found: {path}\n"
            f"Run the corresponding prep script (e.g. scripts/prepare_{name}.py) first."
        )
    return JSONLRLDataset(path, difficulty_filter=difficulty_filter)


# ----------------------------------------------------------------------------
# Distributed loader
# ----------------------------------------------------------------------------

def distributed_rl_loader(
    dataset: JSONLRLDataset,
    prompts_per_step: int,
    world_size: int,
    rank: int,
    seed: int = 0,
    resume_state: dict | None = None,
):
    """Yield (list[RLExample], state_dict) per step.

    Each step pulls `prompts_per_step` examples from a deterministic
    epoch-shuffled order, then slices the rank's disjoint share. State is
    `{epoch, cursor}` so a checkpoint resumes to the same prompt stream.
    """
    n = len(dataset)
    assert prompts_per_step % world_size == 0, (
        f"prompts_per_step ({prompts_per_step}) must be divisible by "
        f"world_size ({world_size})"
    )
    assert prompts_per_step <= n, (
        f"prompts_per_step ({prompts_per_step}) > dataset size ({n})"
    )
    per_rank = prompts_per_step // world_size

    def _epoch_order(epoch_idx: int) -> list[int]:
        rng = random.Random(seed * 1_000_003 + epoch_idx)
        order = list(range(n))
        rng.shuffle(order)
        return order

    if resume_state is not None:
        epoch = resume_state["epoch"]
        cursor = resume_state["cursor"]
    else:
        epoch = 0
        cursor = 0
    order = _epoch_order(epoch)

    while True:
        # Roll over to a new epoch when we'd run off the end. Drop the partial
        # tail of the current epoch — keeps step shape constant.
        if cursor + prompts_per_step > n:
            epoch += 1
            cursor = 0
            order = _epoch_order(epoch)

        step_idx = order[cursor:cursor + prompts_per_step]
        rank_idx = step_idx[rank * per_rank:(rank + 1) * per_rank]
        examples = [dataset[i] for i in rank_idx]
        cursor += prompts_per_step
        yield examples, {"epoch": epoch, "cursor": cursor}


# ----------------------------------------------------------------------------
# Code extraction + verifier
# ----------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response: str) -> str:
    """Pull a Python code block out of a model response.

    Models often emit "let me think...```python <bad>``` actually ...```python
    <good>```", so we take the *last* fenced block. Falls back to the whole
    response if no fence is present.
    """
    matches = _CODE_FENCE_RE.findall(response)
    if matches:
        return matches[-1].strip()
    return response.strip()


def verify_code(example: RLExample, response: str, step: int,
                k_tests: int) -> tuple[float, dict]:
    """Score a code response against the example's test suite.

    Reward = passed / executed. We deterministically subsample up to k_tests
    per (example.id, step) so all rollouts in the same step see the same
    subset (so their rewards are comparable for group-relative advantages).
    """
    code = extract_code(response)
    payload = example.payload
    inputs = payload["inputs"]
    outputs = payload["outputs"]
    n_total = len(inputs)
    if n_total == 0:
        return 0.0, {"passed": 0, "total": 0, "n_total": 0, "first_failure": "no_tests"}

    if n_total > k_tests:
        rng = random.Random(hash((example.id, step)) & 0xffffffff)
        idx = sorted(rng.sample(range(n_total), k_tests))
    else:
        idx = list(range(n_total))

    fn_name = payload.get("fn_name")
    time_limit_s = payload.get("time_limit_s", 4.0)
    memory_limit_mb = payload.get("memory_limit_mb", 256)

    passed = 0
    first_failure = ""
    for i in idx:
        if example.kind == "code_call_based":
            test = {"args": inputs[i], "expected": outputs[i]}
        elif example.kind == "code_stdin_stdout":
            test = {"stdin": inputs[i], "expected": outputs[i]}
        else:
            raise ValueError(f"verify_code given unsupported kind: {example.kind!r}")

        result = run_test(
            example.kind, code, test,
            fn_name=fn_name,
            time_limit_s=time_limit_s,
            memory_limit_mb=memory_limit_mb,
        )
        if result.passed:
            passed += 1
        elif not first_failure:
            first_failure = result.detail

    reward = passed / len(idx)
    info = {
        "passed": passed,
        "total": len(idx),
        "n_total": n_total,
        "first_failure": first_failure[:200],
    }
    return reward, info


def verify(example: RLExample, response: str, step: int,
           k_tests: int = 10) -> tuple[float, dict]:
    """Top-level verifier dispatch on example.kind.

    Add new kinds with an elif branch + sibling verify_* helper.
    """
    if example.kind in ("code_call_based", "code_stdin_stdout"):
        return verify_code(example, response, step, k_tests)
    raise ValueError(f"unknown verifier kind: {example.kind!r}")


# ----------------------------------------------------------------------------
# Reward worker pool
# ----------------------------------------------------------------------------

def _score_one(args: tuple) -> tuple[float, dict]:
    """Pool worker entry point. Top-level so it's picklable for spawn workers."""
    example, response, step, k_tests = args
    try:
        return verify(example, response, step=step, k_tests=k_tests)
    except Exception:
        sys.stderr.write(f"[rl_data] verify failed for example id={example.id!r}:\n")
        sys.stderr.write(traceback.format_exc())
        raise


class RewardWorkerPool:
    """Parallel reward computation across a batch of (example, response) pairs.

    `num_workers=0` runs synchronously in the parent — useful for debugging
    or for tiny smoke tests where mp overhead dominates. Otherwise we spawn
    a `num_workers`-sized process pool with the spawn context (NOT fork),
    because the trainer process holds an initialized CUDA context and forking
    from a CUDA-initialized parent corrupts the child.
    """

    def __init__(self, num_workers: int = 0, k_tests: int = 10):
        self.num_workers = num_workers
        self.k_tests = k_tests
        self._pool = None
        if num_workers > 0:
            from multiprocessing import get_context
            ctx = get_context("spawn")
            self._pool = ctx.Pool(num_workers)

    def score(self, examples: list[RLExample], responses: list[str],
              step: int) -> tuple[list[float], list[dict]]:
        assert len(examples) == len(responses)
        jobs = [(ex, resp, step, self.k_tests)
                for ex, resp in zip(examples, responses)]
        if self._pool is None:
            results = [_score_one(j) for j in jobs]
        else:
            results = self._pool.map(_score_one, jobs)
        rewards = [r for r, _ in results]
        infos = [i for _, i in results]
        return rewards, infos

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
