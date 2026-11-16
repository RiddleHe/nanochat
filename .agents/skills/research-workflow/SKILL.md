---
name: research-workflow
description: Set up or maintain a research runbook, compact project memory, experiment plans, and evidence-based handoffs. Use for sustained research, experiment design, result interpretation, or recovery after context loss; also supports explicitly requested workflows for other multi-session work.
---

# Research workflow

Keep the research question, evidence, and next decision connected across sessions.
This is an operating procedure, not an autonomous experiment launcher, prompt
marketplace, or guarantee of scientific novelty.

## Start with the smallest useful context

Follow the latest user request and applicable project instructions. For an ongoing
project, read its existing short state and task handoff; use its index to retrieve
relevant evidence. Do not migrate established memory merely to match this layout.
If asked to initialize a workflow and none exists, run:

```bash
python3 <skill-directory>/scripts/context.py --project <project-directory> init
```

Use actual paths in place of the two path arguments. Initialization creates only
missing files under `.research/`, excludes that directory from Git through a local
ignore file, and does not change training code or agent configuration. Adapt the
records to the real project. Merge a short recovery pointer into the project's
agent instructions when setting up the workflow; preserve existing rules. Name
STATE, CURRENT_TASK, and INDEX; do not paste the whole runbook.

Read [memory.md](references/memory.md) when setting up or recovering memory. Do not
load every reference for every task. Simple questions and edits need no task card,
delegation, or experimental protocol.

## Turn a request into an executable task

The built-in prompt step is **request → task specification**, not another mandatory
LLM call. Preserve the user's question, scope, and authorization. State the intended
deliverable, available evidence, success checks, unknowns, and next action. Record
a short task card for substantial work using [task-card.md](assets/task-card.md).
Infer routine details; ask only when a missing answer changes the work. Never
invent data, a budget, approval, or a preferred conclusion to fill a template.

- Research design, interpretation, or new direction: [research.md](references/research.md).
- Delegation or a fresh perspective: [collaboration.md](references/collaboration.md).
- Rationale and external examples: [cases.md](references/cases.md).
- Ordinary multi-session work: use state, task, evidence, and handoff only;
  do not force hypotheses or GPU procedures onto unrelated tasks.

## Finish a meaningful unit of work

Save the artifact and its verification, then update the task handoff and index.
Separate what was observed, computed, inferred, and proposed. Preserve negative
results, failed attempts, uncertainty, and the next decision. Keep large logs,
papers, and result tables out of startup memory; link to their exact locations.

Recovery is instruction-driven: this skill installs no compaction hook, scheduler,
background agent, or automatic cloud service. Before resuming a remote job, recheck
live status and artifacts. A stale note is neither permission nor proof it runs.
Do not launch, restart, publish, change accounts, or incur costs beyond the user's
existing authorization. Credentials belong in the host's secret store, not memory.
