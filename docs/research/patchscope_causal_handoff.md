# Patchscope and causal entity handoff

This branch collects the source used to study where entity-dependent evidence is readable and how it moves to a next-token readout. Patchscope target-prompt injection and same-prompt causal patching are separate experiment families; conclusions from one are not treated as conclusions from the other. Raw result bundles, model artifacts, caches, and logs are intentionally excluded.

## Patchscope experiments

1. **All-layer matrices and normalization.** `patchscope_few_shot.py` scans every source layer and, with `--target-layer -1`, every target layer. Each matrix cell replaces the target placeholder residual with the selected source residual and grades the continuation. `--normalize` preserves the source direction while matching its norm to the target residual being replaced. Raw-completion lenses provide a separate final-norm/unembedding readout and are not interchangeable with target-prompt Patchscope success.

2. **Source-context and target-frame controls.** Source-context variants add controlled prefix or suffix context around the source entity while preserving and recording the patched entity position. Target-suffix variants keep the patched placeholder earlier in the target sequence and start generation only after a shared or entity-specific suffix/query. Together these distinguish changes in source integration from changes caused by the target frame.

3. **No-patch leakage controls.** Shared and entity-specific target prompts are also generated with no activation patch. Any entity hit in these controls is prompt leakage rather than evidence transferred by the intervention.

4. **Multi-layer windows.** Window scripts patch the same source vector across consecutive target layers and record the exact window. These tests ask whether a one-layer intervention is overwritten and whether persistence across a short downstream interval improves readout.

5. **Layer translators and adapters.** The linear translator learns a held-out mapping between ordinary source-layer and readout-layer residuals, then compares translated vectors with raw, oracle, random-norm, and wrong-entity controls. The adapter search extends this to constrained affine/low-rank candidates with train/validation splits. Reconstruction quality and Patchscope behavior are reported separately; a good reconstruction score alone is not evidence of causal entity transfer.

These scripts support controlled measurement. A successful Patchscope decode means the patched state is compatible with the chosen target computation; it does not by itself identify the model's native causal route.

## Same-prompt causal handoff

6. **Entity-position handoff.** `causal_entity_position_handoff.py` patches unmodified donor block-output residuals into the same token positions of a recipient prompt and measures unclipped recovery of the donor-versus-recipient next-token logit margin. Grouped and token-level scans compare the subject span, intervening positions, and final readout across all 36 blocks. Identity, unrelated-position, and all-position oracle conditions validate intervention execution.

7. **Donor-exposure limitation.** The original subject/readout crossing curves use a different clean-donor state at each patch depth. Later donor states have undergone more donor-conditioned computation, so the effective donor exposure changes with layer. The crossing is descriptive, but it cannot alone prove that information was handed from the subject to the readout.

8. **Fixed-L0 P02-to-P10 relay.** `causal_entity_p02_p10_relay.py` fixes the initial intervention: donor subject position P02 replaces recipient P02 at block-0 output. At later relay depths, P10 or the post-subject region is captured from that hybrid trajectory and inserted into a fresh recipient run. Because the early donor intervention is held fixed, changes across relay depth are not explained by increasing the original donor exposure.

9. **Attention sufficiency and necessity.** On the fixed-early-intervention hybrid trajectory, attention-output mediation exchanges the post-output-projection, pre-residual attention contribution at P10. Supplying the hybrid contribution to the recipient tests sufficiency; removing it from the hybrid with the recipient contribution tests necessity. Agreement of these interventions supports causal writing at L23, L24, and L26, including the L25 interruption.

10. **Head and source-edge localization.** The component scan decomposes selected L23-L26 attention outputs by query head and, under fixed softmax weights, by value-weighted source edge. The strongest validated individual heads occur at L23, L24, and L26. For those heads, the subject position P02 is the strongest individual source position. Edge effects are nonlinear at the output metric and must not be summed as an exact percentage decomposition; evidence for a distributed intermediate relay remains interpretive.

11. **MLP mediation.** Matched interventions hold the post-attention residual fixed and compare one current-layer MLP update with a bypass and reconstructed full-post-MLP control. The major P10 attention writes precede the strongest later MLP effects. The validated temporal pattern supports later reshaping primarily at L29-L34, not an MLP-only handoff mechanism.

12. **Absolute position versus relative distance.** Prefix filler shifts subject and readout together while approximately preserving their separation; gap filler increases subject-to-readout distance at a matched final index. Across three validated natural filler families and five lengths, long prefix shifts have little endpoint effect, while long gaps delay the stable subject/readout crossover and dominant attention write. This is evidence about the tested intervention and prompt design, not a universal law of positional encoding.

13. **Template-position causal atlas.** `causal_entity_template_position_map.py` extends the same-prompt map to five sentence templates, no-filler and prefix/gap position conditions, three filler families, 36 blocks, 16 ordered directions, and eight reciprocal clusters. Preflight rejects token-misaligned or baseline-invalid combinations before scientific analysis. The completed run passed smoke, both phases, row-count, finite-value, alignment, control, cluster, analysis, and plot gates; accepted and rejected combinations remain explicit. The strongest validated atlas result is a distance-associated delay across templates. Role-aligned profiles are highly similar for most templates, while the T3 profile difference is an interpretation with design and cluster-count limitations. The committed atlas differs from the saved executed snapshot only by making private reference paths repository-relative. The distance analyzer also replaces an environment-specific Pandas fallback and generated machine-specific commands with portable equivalents.

## Evidence status and scope

Validated conclusions are intervention outcomes that passed the scripts' prespecified alignment, completeness, numerical-control, and reciprocal-cluster checks. Terms such as “write,” “read,” “handoff,” and “reshape” are mechanistic interpretations of those causal outcomes, not proof of a unique internal representation. Confidence intervals are descriptive cluster bootstraps over eight reciprocal entity clusters, recovery is not clipped to `[0, 1]`, and effects near the documented BF16 control floor are not interpreted.

The cautious overall conclusion is:

- Raw subject/readout crossing curves alone do not prove handoff because donor exposure differs with patch depth.
- Stronger controlled evidence shows that, after a fixed early subject intervention, selected attention heads at L23/L24/L26 causally write entity-dependent evidence into the final readout position.
- The strongest individual source position for those heads is the subject position.
- Later MLPs, primarily L29-L34, reshape the readout representation.
- Absolute token position has little endpoint effect in the tested setup, while increased subject-to-readout distance delays the measured causal transition.
- These conclusions currently apply to Qwen3-8B-Base, the tested paired-entity task, and the validated templates only.

## Source inventory

- Baselines and plotting: `patchscope_few_shot.py`, `patchscope_raw_completion.py`, `plot_patchscope_full_matrix_heatmaps.py`.
- Source/target controls: `patchscope_few_shot_context.py`, `patchscope_few_shot_target_suffix.py`, `patchscope_few_shot_target_suffix_entity_specific.py`, `patchscope_no_patch_target_prompt.py`, `patchscope_no_patch_entity_specific_target.py`.
- Window and representation mappings: `patchscope_few_shot_window.py`, `patchscope_few_shot_target_suffix_window.py`, `patchscope_layer_translator.py`, `patchscope_adapter_search.py`.
- Same-prompt causal experiments: `causal_entity_position_handoff.py`, `causal_entity_p02_p10_relay.py`, `causal_entity_component_mediation.py`, `causal_entity_component_analysis.py`, `causal_entity_distance_generality.py`, `analyze_causal_entity_distance_generality.py`, `causal_entity_template_position_map.py`.

All source files above live in `scripts/inspect/`. Their default generated outputs are under the ignored `results/` tree or an explicit output directory.
