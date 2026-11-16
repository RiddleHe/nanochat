# Memory and recovery

Keep one authoritative location for each fact. Reuse existing project records;
the default for a new project is the Git-ignored `.research/` directory:

| Record | Purpose | When to read/update |
| --- | --- | --- |
| STATE.md | Main question, constraints, current focus | Start; update when focus or constraints change |
| CURRENT_TASK.md | Current request, scope, completed actions, next action | Start/recovery; update at milestones and handoff |
| INDEX.md | Routes to tasks, runs, evidence, archived decisions | When locating detail or adding artifacts |
| RUNBOOK.md | Project procedures and quality checks | Read/update only relevant sections |
| tasks/<id>.md | Task card, decisions, failed approaches, evidence | Only the relevant task |
| runs/<id>/ | Manifests, logs, results, plots or pointers to them | When verifying an experimental claim |

The helper creates the four top-level records, not empty run histories. Create
task/run records as needed. Artifacts may stay elsewhere; INDEX can point to them.
Do not collect secrets or entire chat transcripts here. Share code and intentionally
selected scientific evidence separately. Local memory is not a backup; use an
appropriate private backup for irreplaceable records.

## Startup and compaction

1. Reconcile the latest user instruction with STATE and CURRENT_TASK. A handoff
   can be stale; it cannot override the current request or higher-priority rules.
2. Retrieve the relevant INDEX entry and task, then only source sections,
   configuration fields, or result rows needed. Avoid reloading the whole archive.
3. Verify claims against code/data/results. Recheck process and artifact status
   before reporting a remote experiment as currently running or complete.
4. Continue from the last verified action; do not repeat launches or writes just
   because the conversation was compacted. Record missing context as unknown.

Aim to keep STATE + CURRENT_TASK around 500 words combined. This is a local target,
not a tokenizer guarantee. Move detail into task records before it grows. Preserve
qualifiers that change conclusions. A concise startup does not limit how much raw
evidence a difficult analysis may need.

## Selective retrieval helper

From the project root (adjust the skill path if installed elsewhere):

```bash
python3 .agents/skills/research-workflow/scripts/context.py resume
python3 .agents/skills/research-workflow/scripts/context.py index
python3 .agents/skills/research-workflow/scripts/context.py read .research/RUNBOOK.md --heading "Validation"
python3 .agents/skills/research-workflow/scripts/context.py read runs/example/status.json --key status
```

`--project` selects another root. `--max-chars` sets an explicit output cap
(default 12,000 characters, not tokens). Oversized output fails visibly instead of
silently dropping information; narrow the selection. `read` accepts project-local
paths, including symlinks only if their resolved target remains inside the project.
It is a retrieval tool, not a summarizer or vector database.

## Writing and correction

Update at task start, approved design changes, launch, important results/failure,
handoff, and completion. Do not record every unchanged poll. Give a timestamp and
source for status and claims. Before interruption, save completed actions and the
next safe action whenever possible; there is no guaranteed pre-compaction callback.

The parent writes shared state; workers write assigned artifacts. Preserve earlier
evidence, mark superseded conclusions with the correcting source, and change the
short current summary. Store concise evidence-backed reasons, not private internal
reasoning transcripts. Include deferred options and conditions for revisiting them.
