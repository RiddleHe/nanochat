# Research: connect each experiment to a decision

## Ideas and scope

Keep the main research question in STATE. For each experiment explain what competing
explanations it separates, and what changes after a positive, negative, or inconclusive
result. Replication and baseline validation can be valuable without novelty. If no
outcome changes a decision, clarify why the work is needed.

When asked for a new direction or after a decisive result, consider a small shortlist
of alternatives. For each record the observation, closest verified prior work,
precise difference, falsifiable prediction, cheapest discriminating test, and cost.
Search primary sources before claiming novelty; distinguish a cited finding from
your extension. Keep unselected ideas with a reopening condition outside startup
context. Do not substitute a large literature search for a testable question.

## Before execution

Write a reproducible plan proportional to the experiment:

- Intervention, baseline, competing explanation, and invariants. Vary the intended
  factor; identify remaining confounds. Match the quantity relevant to the claim:
  equal steps, parameters, tokens, FLOPs, and wall time are different comparisons.
- Data identity, selection, split, order, evaluation unit, and metric denominator.
  Keep selection/tuning evidence separate from confirmation where warranted.
- Code/configuration/dependency versions, seed, optimizer state, checkpoint source,
  restore checks, resource scope, and expected outputs. Freeze the run manifest.
- A cheap correctness check and an external numerical reference where available.
  Check important parameter regimes, not only a convenient example. Passing
  software tests does not establish the scientific hypothesis.
- An existing authorized resource limit, numerical failure checks, stopping rule,
  and decision for each outcome. Unknown costs stay unknown until estimated;
  a skill or task card cannot create permission to launch costly jobs.

Run a small preflight before a long run when it can catch meaningful failures.
Check existing processes/locks and shared resources before launching. Use deterministic
status collection when available; do not repeatedly wake agents to stare at logs.
Preserve failure records. Fixes get a versioned plan and retain the original failure;
do not silently tune settings to hide an anomaly.

## Analysis and claims

Verify completeness, configuration/data identity, finite values, expected controls,
and original artifacts before interpreting summaries. Plot relevant comparisons
with units, denominators, direction of improvement, and readable labels. Inspect
the rendered plots, not just plotting code.

For each important claim keep an evidence row in the task record:

| Claim and status | Source / exact comparison | Uncertainty / limitation | Decision |
| --- | --- | --- | --- |
| Observed, computed, inferred, or proposed | Run/file/version and metric | What the evidence does not establish | Continue, revise, stop, or test |

Distinguish evaluation-sample uncertainty from training-run variability. Use paired
comparisons when observations are paired; document exclusions and influential
observations. Do not interpret a single seed, one checkpoint, an exploratory screen,
or an agent vote as generality. Show important negative or contradictory results.
For paper writing, every central claim must survive comparison with the actual
figure/table and protocol, not merely read persuasively.

## When stuck

If attempts repeat the same explanation without new evidence, stop that repetition.
A practical trigger is two uninformative cycles; it is not a universal stopping law.
Use a fresh evidence-first review (see collaboration.md), then compare proposed
tests with the failed-attempt record. Distinguish implementation failure from a
negative scientific result. Record whether to continue, revise the mechanism, stop
the branch, or change question, with evidence and a reopening condition.
