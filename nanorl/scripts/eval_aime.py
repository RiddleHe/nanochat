from __future__ import annotations

import json
from typing import Any

import wandb  # log() uses the globally active run initialized in train.py
from datasets import load_dataset

from nanochat.common import print0
from nanorl.data import extract_last_boxed
from nanorl.rollout import generate_rollouts_remote


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful competition math solver. "
    "Think step by step before answering. "
    "Then provide the final answer inside \\boxed{...}."
)


def _canon(s: str) -> str:
    return " ".join(str(s).strip().split())


def check_answer(pred: str | None, answer: int | float | str) -> bool:
    if pred is None:
        return False
    pred_s = _canon(pred)
    ans_s = _canon(str(answer))
    if pred_s == ans_s:
        return True
    try:
        return float(pred_s) == float(ans_s)
    except (ValueError, TypeError):
        return False


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{problem}"


def run_eval(
    rollout_worker_url: str,
    tokenizer,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on AIME 2025 using the already weight-synced rollout worker."""
    problems: list[dict] = list(load_dataset("MathArena/aime_2025", split="train"))
    prompts = [build_prompt(p["problem"], tokenizer) for p in problems]

    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    # rollouts is flat: eval_k responses per problem in order
    per_problem = []
    for i, prob in enumerate(problems):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        preds = [extract_last_boxed(r["response"]) for r in batch]
        n_correct = sum(check_answer(p, prob["answer"]) for p in preds)
        per_problem.append(
            {
                "problem_idx": prob["problem_idx"],
                "n_correct": n_correct,
                "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
            }
        )

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {
        f"eval/pass@{eval_k}": overall,
    }

    print0(f"[eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(f"  problem {r['problem_idx']:02d}: {r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}")

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
