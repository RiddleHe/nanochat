# Scientific summary

## Scope and metric

Exactly three component-level experiments were completed on `Qwen/Qwen3-8B-Base`: (1) attention-output mediation at P10 over layers 0–35, (2) matched MLP active/bypass/full-post-MLP interventions at P02 and P10 over layers 0–35, and (3) the predefined 32-query-head and 11-source-position scan at layers 23–26. No other experiment was run.

The data are exactly the completed relay's 16 ordered directions in eight reciprocal clusters. Donor P02 block-0 output was patched into recipient P02 to form the hybrid trajectory. Recovery was `(intervention margin - recipient margin) / (donor margin - recipient margin)` without clipping. Confidence intervals are fixed-seed (20260715), 2,000-resample cluster-bootstrap intervals over the eight unordered reciprocal clusters. Necessity effect is hybrid end-to-end recovery minus the necessity intervention's recovery; sufficiency effect is its recovery directly.

## Main results

### 1. When attention writes entity information into P10

The first strong and pair-consistent attention write is layer 23, followed by a much larger peak at layer 24, an abrupt null at layer 25, and a second large write at layer 26. Sufficiency and necessity agree closely:

| Layer | Sufficiency mean / median (95% CI) | Necessity-loss mean / median (95% CI) |
|---:|---:|---:|
| 23 | 0.061 / 0.050 (0.045, 0.077) | 0.059 / 0.060 (0.044, 0.076) |
| 24 | 0.212 / 0.191 (0.173, 0.251) | 0.209 / 0.200 (0.174, 0.250) |
| 25 | 0.002 / 0.004 (-0.006, 0.010) | -0.004 / -0.004 (-0.009, 0.001) |
| 26 | 0.170 / 0.183 (0.130, 0.209) | 0.169 / 0.177 (0.132, 0.206) |

Attention continues to write at layers 27, 29–31, and 34–35. Later sufficiency means are 0.066 at L27, 0.035 at L29, 0.061 at L30, 0.087 at L31, 0.102 at L34, and 0.108 at L35; all corresponding cluster intervals exclude zero. Thus the transition is punctuated, not a monotonic handoff.

### 2. When an MLP changes a fixed post-attention entity state

At P10 the matched MLP effect is essentially absent at L23–24, weakly suppressive at L25, and weakly positive at L26:

| Layer | P10 MLP effect mean / median (95% CI) |
|---:|---:|
| 23 | -0.002 / 0.000 (-0.017, 0.011) |
| 24 | -0.008 / -0.005 (-0.028, 0.011) |
| 25 | -0.010 / 0.000 (-0.019, -0.001) |
| 26 | 0.009 / 0.000 (0.003, 0.018) |

The strongest P10 MLP strengthening is later: L29 0.081 (0.066, 0.101), L30 0.052 (0.022, 0.080), L31 0.031 (0.010, 0.051), L33 0.035 (0.028, 0.043), and L34 0.115 (0.103, 0.125). L35 reverses strongly, -0.144 (-0.198, -0.099). Earlier layers also contain smaller alternating positive and negative transformations. P02 MLP effects around L23–26 are near zero; its largest effect is a suppressive L0 effect of -0.198 (-0.274, -0.123).

MLP-active and full-post-MLP states and outcomes match exactly for every layer and position. Bypass equals the supplied hybrid post-attention residual exactly, confirming that it removes one and only one current-layer MLP update.

### 3. Pair consistency around layers 23–26

The attention timing is highly consistent. Sufficiency is positive for all 16 directions at L23 (range 0.009–0.145), L24 (0.143–0.361), and L26 (0.060–0.290). At L25 it is mixed and near zero: eight positive, four negative, and four exactly zero (range -0.036–0.026). Necessity is positive for all directions at L24 and L26 and for 13/16 at L23. Thus entity pairs agree on the 23→24 peak, 25 gap, and 26 resurgence, although effect magnitude varies.

The MLP transition is not comparably consistent at L23–25: direction-level signs are mixed. At L26, seven directions are positive, three negative, and six zero; the cluster mean is small but positive.

Every direction and layer is reported in `pair_direction_summary.csv`; raw direction-level outcomes are retained in the experiment CSVs and checkpoints.

### 4. Heads performing the transfer

The clearest heads, using the mean of sufficiency and necessity effects, are:

| Layer | Head | Effect (95% cluster CI) | Dominant source positions |
|---:|---:|---:|---|
| 24 | 26 | 0.149 (0.088, 0.217) | P02, then P09/P06 |
| 26 | 20 | 0.096 (0.072, 0.117) | P02, then P03/P04 |
| 23 | 10 | 0.069 (0.056, 0.082) | P02, then P04/P08 |
| 24 | 16 | 0.034 (0.019, 0.049) | P02, then P04/P09 |
| 26 | 22 | 0.022 (0.014, 0.030) | P02, then P03/P09 |

Additional positive heads appear in `top_heads.csv`. The effect is concentrated: L24H26 alone contributes the largest head-level effect at L24, and L26H20 is the largest at L26.

### 5. Direct P02 reading versus relay through P03–P09

The strongest transfer heads read directly from P02: P02 is the dominant source edge for each of the top five heads. Summed over heads, P02 is also the largest *individual* position at L23, L24, and L26. Its summed edge effects are 0.096, 0.208, 0.053, and 0.161 at L23–26; the P02 cluster interval excludes zero at L24 (0.081, 0.320) and L26 (0.019, 0.280).

The seven P03–P09 positions collectively have larger mean sums (0.339–0.388), but their wide cluster intervals cross zero at every scanned layer. Individual-edge intervention effects are not additive, so these sums are not percentages of the full attention effect. The directly supported conclusion is that the dominant heads use a direct P02 path. A distributed relayed contribution through intermediate positions is suggestive in the mean maps but is not stable enough here to claim as the dominant causal route.

### 6. Overall mechanism

The best-supported sequence is **attention transfer followed by later MLP transformation**. Attention creates large P10 causal effects at L24 and L26 while the matched P10 MLP effect is near zero there. MLP strengthening becomes substantial at L29–34, peaking at L34, after the major attention-transfer events. The final L35 MLP update suppresses donor recovery even as L35 attention remains positive. This is neither attention-only nor MLP-only; the temporal ordering supports attention transfer first and later MLP reshaping.

## Direct support versus interpretation

Directly supported:

- The intervention outcomes, their unclipped normalized recoveries, pair directions, and cluster-bootstrap intervals.
- The L23/L24/L26 attention pattern and its all-direction sign consistency.
- Exact matched-state equality for MLP-active/full-post-MLP and exact one-branch bypass.
- The identified head effects and fixed-softmax, value-weighted source-edge effects.
- P02 dominance within the strongest heads and as an individual source position.

Interpretations and limitations:

- “Writing,” “reading,” “entity information,” and the proposed sequence describe causal intervention behavior; they do not establish a unique representational code.
- Edge interventions are nonlinear at the output metric and cannot be summed into an exact decomposition. The collective P03–P09 means therefore remain suggestive.
- Results cover one prompt template, 16 directions, and eight reciprocal clusters; confidence intervals are descriptive with only eight independent clusters.
- Eager attention was used so the actual softmax weights could be captured without rerunning normalization. Previous residual-reference curves were reproduced closely (maximum aggregate-curve delta 0.00784; worst direction/layer delta 0.0469), but BF16/kernel differences remain.
- Recovery is pair-normalized and can exceed conventional bounds; it was never clipped.

## Validation and execution

Full validation passed all checks: exact row counts (1,152 attention; 3,456 MLP; 4,224 head; 45,056 edge; 1,152 reference; 16 identity-control rows), no NaNs or skipped directions, exact layer/head/source coverage, zero identity and wrong-entity recovery, edge-to-head reconstruction maximum error 0.03125 with minimum cosine 0.999996, all-head-to-attention maximum error 0.0625 with minimum cosine 0.999993, and exact agreement between all-head patches and Experiment 1.

The original full workers used 316.38 summed pair-seconds; the slowest shard used 118.22 pair-seconds and the three shards ran concurrently. Two failed smoke implementations are preserved for audit: the first exposed BF16 batch-shape effects and the second showed why fused SDPA weights could not support exact edge decomposition. No full run was launched until the final smoke passed. A layer-0 P02 full-post-MLP formula error found by the full validator was corrected, with pre-fix checkpoints retained; only those 16 layer-0 MLP triplets were recomputed.
