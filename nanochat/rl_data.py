"""
RL dataset + reward interface.

PLACEHOLDER. The shapes below are the agreed contract for the upcoming
implementation; concrete datasets (GSM8K, MATH, DeepCoder, TACO) and the
sandboxed reward worker pool will land in a follow-up.

Design notes (see discussion in scripts/rl_train.py refactor):
  - An RLExample carries everything needed to (a) prompt the policy and
    (b) verify a response. For math that's a gold answer; for code that's
    a list of test cases + resource limits.
  - RLDataset.verify(example, response) -> (reward, info) is the single
    entry point for reward computation. Math verifiers run in-process;
    code verifiers must dispatch into a sandboxed subprocess pool.
  - distributed_rl_loader yields a disjoint slice of prompts per rank per
    step, and is resumable via a state_dict (mirrors the pretraining
    dataloader contract in nanochat/dataloader.py).
  - RewardWorkerPool owns the sandbox process pool and is shared across
    steps so we don't pay fork cost per rollout.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RLExample:
    """A single RL training example."""
    prompt: str
    prompt_messages: list = field(default_factory=list)  # chat-format messages, if applicable
    verifier: dict = field(default_factory=dict)         # gold answer / test cases / etc.
    reward_kind: Literal["math", "code", "format"] = "math"
    meta: dict = field(default_factory=dict)             # difficulty, source, problem id


class RLDataset:
    """Abstract base class for an RL training dataset.

    Subclasses must implement __len__, __getitem__, and verify.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> RLExample:
        raise NotImplementedError

    def verify(self, example: RLExample, response: str) -> tuple[float, dict]:
        """Score a single (example, response) pair.

        Returns (reward, info) where info carries diagnostics for logging
        (e.g. which tests failed, did it timeout, did the parser fire).
        """
        raise NotImplementedError


def build_rl_dataset(name: str, split: str = "train") -> RLDataset:
    """Factory for RL datasets. Concrete impls land in follow-up."""
    raise NotImplementedError(f"RL dataset '{name}' not yet implemented")


def distributed_rl_loader(dataset: RLDataset, prompts_per_step: int,
                          world_size: int, rank: int, seed: int = 0,
                          resume_state: dict | None = None):
    """Yield disjoint slices of prompts per rank per step.

    Each call returns (list[RLExample], state_dict). state_dict captures
    epoch + index so a checkpoint can resume to the exact same prompt stream.
    """
    raise NotImplementedError


class RewardWorkerPool:
    """Shared (multi-)process pool for reward computation.

    Math examples are scored in-process (cheap regex / sympy).
    Code examples dispatch into sandboxed subprocesses with rlimit-based
    time/memory caps.
    """

    def __init__(self, dataset: RLDataset, num_workers: int = 0):
        self.dataset = dataset
        self.num_workers = num_workers

    def score(self, examples: list[RLExample], rollouts: list[dict]
              ) -> tuple[list[float], list[dict]]:
        """Score a flat list of rollouts. Returns (rewards, infos)."""
        raise NotImplementedError

    def close(self):
        pass
