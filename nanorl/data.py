"""
RL dataset, distributed loader, math verifier, and reward worker pool.

Data flow:

    JSONL on disk  ──►  JSONLRLDataset  ──►  distributed_rl_loader
                                                      │
                                                      ▼
                          rollouts ──► RewardWorkerPool.score ──► (rewards, infos)
                                                │
                                                ▼ per (example, response)
                                           verify_math
                                                │
                                                ▼
                              last \\boxed{...} extraction + string compare

JSONL row schema:

    {
      "id":           "dapo_math/<uuid>",
      "prompt":       "<fully rendered prompt, including \\boxed{} instruction>",
      "ground_truth": "<answer string>",
      "meta":         { ... }
    }
"""

from __future__ import annotations

import os
import sys
import json
import random
import traceback
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------

@dataclass
class RLExample:
    id: str
    prompt: str
    ground_truth: str
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class JSONLRLDataset:
    """Loads a JSONL of RLExamples into memory."""

    def __init__(self, path: str):
        self.path = path
        self.examples: list[RLExample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.examples.append(RLExample(
                    id=row["id"],
                    prompt=row["prompt"],
                    ground_truth=row["ground_truth"],
                    meta=row.get("meta", {}),
                ))
        if not self.examples:
            raise RuntimeError(f"JSONLRLDataset loaded zero examples from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> RLExample:
        return self.examples[i]


RL_DATASET_PATH = os.environ.get("RL_DATASET_PATH", "/local-ssd/mh3897/data/rl/dapo_math_17k.jsonl")


def build_rl_dataset() -> JSONLRLDataset:
    if not os.path.exists(RL_DATASET_PATH):
        raise FileNotFoundError(f"RL dataset not found on disk: {RL_DATASET_PATH}")
    return JSONLRLDataset(RL_DATASET_PATH)


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
    """Yield (list[RLExample], state_dict) per step."""
    n = len(dataset)
    assert prompts_per_step % world_size == 0
    assert prompts_per_step <= n
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
# Math verifier
# ----------------------------------------------------------------------------

def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last brace-balanced \\boxed{...} in text, or None."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _canon(s: str) -> str:
    """Minimal normalization: strip whitespace and common LaTeX padding."""
    s = s.strip()
    # Collapse internal whitespace (LaTeX tolerance)
    return " ".join(s.split())


def verify_math(example: RLExample, response: str, step: int) -> tuple[float, dict]:
    """Score a math response by extracting the last \\boxed{...} and comparing to ground truth."""
    pred = extract_last_boxed(response)
    matched = pred is not None and _canon(pred) == _canon(example.ground_truth)
    return (1.0 if matched else 0.0), {
        "pred": (pred or "")[:200],
        "gt": example.ground_truth[:200],
        "matched": matched,
    }


# ----------------------------------------------------------------------------
# Reward shaping
# ----------------------------------------------------------------------------

# DAPO overlong reward shaping: linear penalty from 0 -> -PENALTY_FACTOR as the
# response length grows from (1 - BUFFER_RATIO) * max_new_tokens to max_new_tokens.
# Matches verl workers/reward_manager/dapo.py with buffer_len auto-derived.
_OVERLONG_BUFFER_RATIO = 0.25
_OVERLONG_PENALTY_FACTOR = 1.0


def apply_overlong_shaping(rewards: list[float], response_lens: list[int],
                           max_new_tokens: int) -> list[float]:
    buffer_len = int(_OVERLONG_BUFFER_RATIO * max_new_tokens)
    expected_len = max_new_tokens - buffer_len
    shaped = list(rewards)
    for i, n in enumerate(response_lens):
        exceed = n - expected_len
        shaped[i] += min(-exceed / buffer_len * _OVERLONG_PENALTY_FACTOR, 0.0)
    return shaped


# ----------------------------------------------------------------------------
# Reward worker pool
# ----------------------------------------------------------------------------

def _score_one(args: tuple) -> tuple[float, dict]:
    example, response, step = args
    try:
        return verify_math(example, response, step=step)
    except Exception:
        sys.stderr.write(f"[rl_data] verify failed for example id={example.id!r}:\n")
        sys.stderr.write(traceback.format_exc())
        raise


class RewardWorkerPool:
    """Parallel reward computation across a batch of (example, response) pairs.

    num_workers=0 runs synchronously in the parent. Otherwise we spawn a
    num_workers-sized process pool with 'spawn' context (NOT fork) because
    the trainer parent holds a CUDA context.
    """

    def __init__(self, num_workers: int = 0):
        self.num_workers = num_workers
        self._pool = None
        if num_workers > 0:
            from multiprocessing import get_context
            ctx = get_context("spawn")
            self._pool = ctx.Pool(num_workers)

    def score(self, examples: list[RLExample], responses: list[str],
              step: int) -> tuple[list[float], list[dict]]:
        assert len(examples) == len(responses)
        jobs = [(ex, resp, step) for ex, resp in zip(examples, responses)]
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
