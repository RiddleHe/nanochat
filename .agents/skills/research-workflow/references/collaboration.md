# Bounded collaboration

Use the parent for the research question, task allocation, integration, and final
claims. Delegate separable searches, implementation, or independent checks when
available and useful. Simple work can remain single-agent. Model choice and worker
count depend on capabilities, task coupling, cost, and available slots; this skill
does not hard-code a vendor, model, or large permanent team.

Give each worker a bounded contract: question, source inputs, owned output files,
invariants, acceptance check, relevant prior failures, and a stopping/budget limit
when one exists. Require artifacts plus a short report of evidence, limitations,
and next action. Do not inherit the entire chat by default. Keep one writer per
artifact; only the parent merges findings into shared state. A new worker reads
the task packet and needed evidence, not the previous worker's whole conversation.

For a fresh perspective, initially give the reviewer the scientific question,
protocol, and raw evidence without the parent's favored explanation. Ask for the
strongest explanation and a test distinguishing it from alternatives. Then reveal
prior attempts before choosing new work. This reduces anchoring; it does not make
agents statistically independent or their agreement scientific evidence.

For consequential conclusions, verify sources and checks rather than counting
supporting reviewers. Reuse a worker for a closely related bounded follow-up; use
a fresh context for a distinct question or independent audit. No subagent support?
Use separate written task packets and evidence checks, label self-review honestly,
and obtain human review when needed. Do not claim an independent review happened.

Use scheduler/process status and artifact timestamps for long jobs. Monitor within
the host's supported execution model; scripts do not magically wake a sleeping
agent. Notify on a material result, failure, completion, or needed action. Do not
restart a failed job or continue an open-ended search beyond authorized scope.
