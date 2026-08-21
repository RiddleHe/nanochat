# Qwen fixed-window relay: analyze a run

`qwen_entity_relay_fixed_window.py` runs the intervention and saves raw generations. Analyze `generations.jsonl` only after the run finishes. Its name-presence fields are rough diagnostics, not the final result.

## Procedure

1. Check completeness. With the default five templates and ten pairs, expect 100 baseline rows and 50 relay rows for every `(width, source_layer_s, relay_layer_t)` span (5,050 relay rows total for widths 1, 2, and 4). Inspect every baseline and report any that does not answer with its own entity; do not silently remove rows.

2. Make a list of the unique relay `completion` strings. First read each string without its template, entity pair, layer, or span. Decide which name or names it actually supplies as answers. Do not label anything patched or original yet.

3. Manually check every unusual string: full names, multiple names, lists, verbose answers, repetitions, dialogue loops, distractors, or no clear answer. Examples:

   - `Messi\nAnswer: Ronaldo` supplies both Messi and Ronaldo.
   - `Lincoln\nSpeaker A: I met Kennedy...` supplies Lincoln; Kennedy is in the next generated dialogue turn.
   - `Speaker A: I met Kennedy...` with no answer before the restarted dialogue supplies no answer.

   If a string is genuinely ambiguous when viewed alone, flag it and then inspect its original prompt context. Do not use layer or span information to decide its meaning.

4. Map each unique-string decision back to every relay row containing that string. Compare the supplied answer names with that row's `donor` (patched entity) and `recipient` (original entity), and assign exactly one result:

   - `patched_only`
   - `original_only`
   - `both`
   - `neither`

   Count all original rows. Never use the unique strings themselves as the denominator.

5. For each span, divide each of the four counts by its 50 rows. For a layer plot, a span contains layer `L` when `source_layer_s < L <= relay_layer_t`. Width 1 uses its single span per layer; for widths 2 and 4, use the mean across all spans containing the layer.

6. Plot:

   - `patched_only` versus layer for widths 1, 2, and 4;
   - `original_only`, `both`, and `neither` versus layer for width 4.

Before reporting, verify that every relay row was assigned once, every span has 50 rows, and the four counts for each span sum to 50. Save the unique-string decisions with the derived results so the manual interpretation remains inspectable. Do not report a binary `correct` score or use `accuracy.json`.

## Suggested instruction to an agent

> Read `scripts/inspect/qwen_entity_relay_fixed_window.md`, inspect the completed run's `generations.jsonl`, classify the unique answers using that procedure, and produce the four rates and requested plots. Show me ambiguous strings before deciding them.
