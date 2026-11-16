# Cases behind the design

Sources checked 2026-11-15. These are useful examples, not a ranking of the best
frameworks or proof this skill improves publication acceptance. The adaptations
below are design choices; no framework code or service is installed.

- [Long-running Claude for scientific computing](https://www.anthropic.com/research/long-running-claude):
  a numerical solver project uses progress/failure notes and comparison against a
  reference solver. Its coupled pipeline motivates sequential diagnosis with
  selective delegation. Borrow reference checks and explicit coverage limits;
  matching some outputs does not validate all operating regimes.
- [Google AI co-scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/):
  generates, critiques, ranks, and refines hypotheses under a scientist's objective,
  with experimental validation in selected biomedical applications. Borrow candidate
  comparison and discriminating tests; agent rankings alone are not validation,
  and biomedical demonstrations do not establish performance in ML research.
- [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2):
  explores experiments using an experiment manager and tree search. Its own README
  cautions that broader exploration can have lower success than a strong v1
  template. Borrow bounded exploration; keep proven baselines instead of always
  replacing focused work with a larger autonomous search.
- [Agent Laboratory](https://arxiv.org/html/2501.04227v2):
  studies research assistants and human feedback. Automated reviews scored papers
  more highly than human reviews in its evaluation. Borrow human decision points
  and external evidence checks; self-review is not proof of paper quality.

The prompt-library alternative, [prompts.chat](https://github.com/f/prompts.chat),
is an optional source of phrasing examples. This skill does not import it or send
requests through a rewriting service. Its task-specification step preserves intent
and makes evidence and acceptance checks explicit; elaborate wording alone does
not establish better research outcomes.

[Codex skills documentation](https://developers.openai.com/codex/skills/) describes
repository skill discovery and progressive loading. Other hosts can read SKILL.md
and its references directly, but discovery, tools, delegation, and permissions
depend on that host. This package cannot configure every agent automatically.
