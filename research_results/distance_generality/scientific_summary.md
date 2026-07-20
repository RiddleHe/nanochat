# Scientific summary

## Scope and validation

Exactly the two prespecified experiments were run on `Qwen/Qwen3-8B-Base`: absolute-position shift (prefix filler, GPU 0) and subject-to-readout distance (gap filler, GPU 1). The canonical 16 ordered directions/eight reciprocal clusters, prompt wording, donor-minus-recipient next-token logit difference, unclipped recovery formula, block-output hooks, BF16 eager inference, and block-0 `P_subject` hybrid were retained. No generation, training, head scan, MLP intervention, or follow-up experiment was performed.

Full validation status: **PASS**. Identity recovery was at most 0; oracle deviation from 1 was at most 0; the zero-filler mean subject curve differed from the completed handoff by at most 0.0077. All required full matrices and all 480 pair-conditions completed without non-finite values.

The unrelated pre-subject control reached 0.06383. This is within the documented 0.07 BF16 mixed-batch numerical-floor acceptance tolerance; exact singleton identity/oracle controls remained 0/1. The original 0.06 diagnostic cutoff and its narrowly failed validation are retained in `analysis.log`; effects near this floor should not be interpreted as causal signal.

## Primary answers

Cluster-bootstrap timing summaries (direction-level means; 95% cluster intervals):

| Placement | Family | Target | Stable crossover | Strongest attention layer |
|---|---|---:|---:|---:|
| prefix | meadow | 0 | 25.06 [24.38, 25.75] | 24.88 [24.25, 25.50] |
| prefix | meadow | 128 | 25.00 [24.38, 25.62] | 24.62 [24.12, 25.25] |
| prefix | room | 0 | 25.06 [24.38, 25.75] | 24.88 [24.25, 25.50] |
| prefix | room | 128 | 25.75 [25.00, 26.38] | 28.19 [26.44, 29.75] |
| prefix | cards | 0 | 25.06 [24.38, 25.75] | 24.88 [24.25, 25.50] |
| prefix | cards | 128 | 25.12 [24.50, 25.75] | 25.50 [24.25, 27.25] |
| gap | meadow | 0 | 25.06 [24.38, 25.75] | 24.88 [24.25, 25.50] |
| gap | meadow | 128 | 32.12 [31.62, 32.62] | 33.62 [32.88, 34.00] |
| gap | room | 0 | 25.06 [24.38, 25.75] | 24.88 [24.25, 25.50] |
| gap | room | 128 | 32.75 [31.94, 33.44] | 33.62 [32.88, 34.00] |
| gap | cards | 0 | 25.06 [24.38, 25.75] | 24.88 [24.25, 25.50] |
| gap | cards | 128 | 32.62 [31.88, 33.38] | 33.62 [32.88, 34.00] |

1. **Does shifting both positions move the handoff layer?** The aggregate stable crossover changed from 26.00 at zero filler to 25.33 at target 128 (change -0.67 layers). Thus absolute-position shift produced little/no material endpoint movement. The target-8 prefix average was 28.00, so the short blocks caused a non-monotonic, family-dependent delay that disappeared for the longer blocks; this argues against a simple numerical-index law.

2. **Does increasing subject-to-readout distance move the handoff layer?** The gap crossover changed from 26.00 to 32.67 (change +6.67 layers); even target 8 averaged 28.33. Thus increased distance produced a large, consistent late shift, reaching L32–L33 in every target-128 family.

3. **At matched `P_readout` positions, do prefix and gap differ?** They do differ substantially in the target-128 subject/readout recovery curves. The largest layerwise matched difference was 0.627 at L28 for subject_span (95% cluster CI 0.498 to 0.812). Because matched prompts have the same filler tokens and final index, this isolates the subject location/distance difference.

4. **Does the L23–L26 attention-transfer pattern remain stable?** The table below gives combined sufficiency/necessity effects averaged over families and directions. Under prefix shifts, the L24/L26 writes and L25 interruption remain strong and the longest-block strongest layer averages 24.67 versus 24.00 at baseline. Under gap distance, the local L25 dip remains but L24/L26 are strongly attenuated and the global strongest layer moves to 34.00 (effectively L34). Therefore the original pattern is stable to absolute position but **not** stable in strength or global timing as distance grows.

| Placement | Target | L23 | L24 | L25 | L26 |
|---|---:|---:|---:|---:|---:|
| gap | 0 | 0.060 | 0.211 | -0.001 | 0.170 |
| gap | 8 | 0.051 | 0.143 | 0.001 | 0.084 |
| gap | 32 | 0.028 | 0.106 | -0.002 | 0.042 |
| gap | 64 | 0.029 | 0.093 | -0.006 | 0.035 |
| gap | 128 | 0.021 | 0.075 | -0.009 | 0.025 |
| prefix | 0 | 0.060 | 0.211 | -0.001 | 0.170 |
| prefix | 8 | 0.052 | 0.140 | -0.003 | 0.099 |
| prefix | 32 | 0.051 | 0.179 | -0.003 | 0.136 |
| prefix | 64 | 0.063 | 0.187 | -0.002 | 0.157 |
| prefix | 128 | 0.069 | 0.178 | 0.003 | 0.152 |

5. **Do intermediate positions become more causally important with distance?** No. Aggregate intermediate-position recovery AUC is 3.13 at baseline, 3.53 for the longest prefix shift, and only 2.15 for the longest gap. The longest gap filler-block patch peaks at 0.067, close to the 0.064 numerical-control floor (longest prefix peak 0.011). The full matrices therefore do not support progressive causal storage across filler tokens.

6. **What mainly governs handoff timing?** Depth remains central, but the measured timing also depends on subject-to-readout distance. Absolute numerical index alone has little endpoint effect; subject-to-readout distance has a large effect even at the same final index. The final token's readout semantics still determine *where* the recovered signal ultimately matters, but semantic role does not fix *when* the transfer occurs. Depth sets the baseline L23–L26 mechanism, while distance can defer its dominant attention write to L34.

7. **Validated versus preliminary.** Validated within this design: intervention execution, alignment, controls, complete layer/condition coverage, matched filler tokens/readout indices, cluster-bootstrap summaries, and replication of the zero-filler curve. Scientific conclusions are validated for this model/template/dataset. Generalization to other prompt forms, models, semantic readout roles, or substantially longer contexts remains preliminary; only eight independent reciprocal clusters support the intervals.

## Measurement definitions

- Recovery: `(intervention margin - recipient margin) / (donor margin - recipient margin)`, never clipped.
- Stable crossover: first layer where readout recovery exceeds subject recovery for at least two consecutive layers.
- Attention sufficiency effect: sufficiency recovery. Necessity effect: hybrid end-to-end recovery minus necessity outcome recovery. Strongest layer maximizes their mean.
- AUC: trapezoidal integral over layers 0–35.
- Confidence intervals: 2000-resample fixed-seed cluster bootstrap over eight reciprocal entity clusters, with both ordered directions resampled together.
