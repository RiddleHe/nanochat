# Late-layer entity information: experiment log (branch `yuchen/patchscope-bov`)

Research question: an entity's identity is decodable at middle layers but seems to
"disappear" at deep layers when read by Patchscope. Is the info lost, moved, or
just unreadable — and does the model use it?

## Common setup

- Repo / branch: `nanochat-patchscope`, `yuchen/patchscope-bov`.
- Python: `/hdd/mh3897/cc/nanochat/.venv/bin/python`
- Env: `NANOCHAT_BASE_DIR=/local-ssd/mh3897` (nanochat models only), `CUDA_VISIBLE_DEVICES=6`
- Models:
  - Attention d24: `arch_d24_gpt_base_100B` (step 95363), 24 layers, d=1536
  - BoV d24: `arch_d24_gpt_base_v_from_value_emb_learn_100B` (step 95368), 24 layers
  - `Qwen/Qwen3-8B-Base`, 36 layers, d=4096
- Shared data (Steps 1, 2, 2b, B, C): 15 entities x 12 templates = 180 sentences.
  Each sentence names the entity early and ENDS in a generic role noun shared
  within its category (all 3 scientists end in "scientist", etc.), so the last
  token's identity must be bound from context. Defined in `probe_step1_robust.py`
  (`ENTITIES`, `TEMPLATES`). We capture the LAST token's hidden state at every layer.
- Linear probe/classifier (Steps 1, B, C): `nn.Linear(d, 15)`, Adam lr 1e-2,
  weight_decay 1e-2, 300 steps, full-batch, cross-entropy, features unit-normalized.

---

## Step 1 — rigorous per-layer entity probe   (`probe_step1_robust.py`)

Is the specific entity linearly decodable at each layer? Five checks:
- held-out-template (PRIMARY, clean): 3 folds, train on 8 templates / test on 4
  unseen ones (shared across entities -> template decorrelated from entity).
- within-category: same split, per category (Einstein vs Newton, ...).
- random split: `per_class_split` 70/30, 10 seeds, mean+/-std (template-confounded, lower).
- random-label control: shuffle labels, 5 seeds -> must sit at chance.
- raw vs normalized hidden states.

Run:
```
NANOCHAT_BASE_DIR=/local-ssd/mh3897 CUDA_VISIBLE_DEVICES=6 <py> -m scripts.inspect.probe_step1_robust --model-tag arch_d24_gpt_base_100B --label Attention --out results/step1
#  BoV : --model-tag arch_d24_gpt_base_v_from_value_emb_learn_100B --label BoV
#  Qwen: --hf-model Qwen/Qwen3-8B-Base --label Qwen3-8B
```
Figures: `results/step1/{Attention,BoV,Qwen3-8B}__step1.png` (+ `.json`).
Result: clean held-out-template ~0.95-0.98 at ALL layers incl. deep (within-cat
~0.98-1.00); random-split lower (confound); random-label at chance ~0.06.
=> entity retained deep; earlier "decline" was a random-split template confound.

## Step 2 — probe vs Patchscope, same hidden states   (`step2_probe_vs_patchscope.py`, Qwen3-8B)

Same last-token hidden state read two ways. Patchscope = candidate-logit over the
15 entity first-tokens; target prompt `Madrid: Madrid\nLincoln: Lincoln\nFerrari:
Ferrari\nx:`; patch the `x` token at a target layer, read the token after `:`.
Controls: norm-matched patch, target-layer sweep [4,10,18,25], random-vector floor,
generation at best target.
```
CUDA_VISIBLE_DEVICES=6 <py> -m scripts.inspect.step2_probe_vs_patchscope --hf-model Qwen/Qwen3-8B-Base --out results/step2
```
Figure: `results/step2/step2_qwen.png`. Result: probe ~0.94-0.99 all layers;
patchscope logit peak 0.68 (target L10) mid, drop 0.22 deep; generation ~0; floor 0.07.

## Step 2b — ORIGINAL free-generation patchscope + source x target matrix   (`step2b_patchscope_matrix.py`, Qwen3-8B)

Prints entity tokenizations (only Beethoven multi-token) and patch/read positions;
free-generation patchscope (greedy, name-substring grade) at target [4,10,18]; full
36x36 source x target matrix (candidate first-token); full-name-logprob cross-check
at best target (matches first-token).
```
CUDA_VISIBLE_DEVICES=6 <py> -m scripts.inspect.step2b_patchscope_matrix --hf-model Qwen/Qwen3-8B-Base --out results/step2
```
Figures: `results/step2/step2b_matrix.png`, `step2b_generation.png`.
Result: free-generation ~0; matrix sweet-spot target band ~L6-17, best cell 0.68
(source L10/target L10), late target dark; late source recoverable only to ~0.4.

## Authentic Patchscope (friend's original)   (`patchscope_few_shot.py`, Qwen3-8B)

The REAL original: few-shot DESCRIPTION target prompt + free generation + semantic
(keyword) grading, canonical 5-entity set (Diana, Alexander, Ali, Jurassic, NYC).
```
CUDA_VISIBLE_DEVICES=6 <py> -m scripts.inspect.patchscope_few_shot --hf-model Qwen/Qwen3-8B-Base --source-set canonical --target-layer 6 --out-dir results/step2/authentic --no-plot
#  also target-layer 4 and 12
```
Figure: `results/step2/authentic/authentic_vs_probe.png` (graded with CRITERIA, avg
over 5 entities, vs the Step-1 probe curve). Result: mid ~0.85, late drops to ~0.42->0
(matches the friend's finding). NOTE: 5 famous entities + description grading, so the
vs-probe (15-entity identity) comparison is QUALITATIVE (shape), not same-data.

## Step B + C — root cause   (`step_root_cause_bc.py`, Qwen3-8B)

- Step B (linear translator): train linear `A: h_Ls -> h_Lt` (MSE, Adam lr 1e-3,
  wd 1e-3, 500 steps) on held-out-template split; patchscope-read raw h_Ls vs A(h_Ls)
  vs true h_Lt (norm-matched candidate readout). Source layers [10,20,24,28,32], ref
  target L10.
- Step C (direction alignment): per layer, |cos| between probe direction (w_i - w_j)
  and unembed/output direction (U_i - U_j), within-category pairs.
```
CUDA_VISIBLE_DEVICES=6 <py> -m scripts.inspect.step_root_cause_bc --hf-model Qwen/Qwen3-8B-Base --out results/step_bc --ref-layer 10
```
Figures: `results/step_bc/step_b_translator.png`, `step_c_direction.png`.
Result B: raw late patchscope 0.24-0.37 -> translated 0.59-0.68 (recovers to upper
bound for L20-24). Result C: |cos| low everywhere (0.02-0.27), INCREASES with depth
=> rules out "off the output direction" as the cause.

## Step D — causal restoration (activation patching)   (`step_d_causal.py`, Qwen3-8B)

Clean vs corrupted differ only in the entity; template
`Everyone knows {name} was a celebrated {role}. The {role} was`. Patch the SUBJECT
token's hidden state per layer clean->corrupted; metric = logit-diff recovery
`(patched - corrupt)/(clean - corrupt)` at the final position, averaged over 16
within-category single-token entity pairs. (`--patch role` was ~0, confounded by the
subject name being directly readable; `--patch subject` is the clean test.)
```
CUDA_VISIBLE_DEVICES=6 <py> -m scripts.inspect.step_d_causal --hf-model Qwen/Qwen3-8B-Base --out results/step_d --patch subject
```
Figure: `results/step_d/step_d_causal.png`. Result: recovery ~0.9-1.0 through L0-22,
drops to 0.41 (L24), 0.16 (L27), 0.07 (L33) => entity causally USED in mid layers,
causally SPENT in late layers.

---

## E1 — inference-time deep-layer window masking   (`deep_window_mask.py`, Qwen3-8B)

Tests "deep layers need no attention beyond a local window": pre-hook replaces
attention_mask with causal window W in layers >= start (first `sink`=4 tokens
stay visible everywhere — deep-layer attention sink, blinding it adds ~0.6
nats of unrelated damage). Metric: mean NLL on 40x2048 wikitext-103 val tokens.
```
CUDA_VISIBLE_DEVICES=0 <py> -m scripts.inspect.deep_window_mask --out results/deep_window
# band variant (mask only [s,e)): --bands 4:24 8:24 24:36 --start-layers --windows
```
Result: NO cliff at the hand-off boundary. Start-layer sweep W=256: L0 +0.141,
L4-20 flat ~+0.12, L24 +0.082, L32 +0.040. Bands: mid L8-24 costs 0.0030
nats/layer, deep L24-36 costs 0.0068/layer (2.3x). W=1024 from L24 nearly free
(+0.007). => windowing is CHEAPEST mid, most expensive late.

## E2 — hand-off vs entity-readout distance   (`handoff_distance.py`, Qwen3-8B)

Same interchange patching as Step D but with N filler tokens between the
entity clause and " The {role} was"; patches BOTH the entity position and the
final position per layer; also logs the clean-corrupt logit-diff (denominator,
stays ~10 up to N=1400 => binding intact).
```
CUDA_VISIBLE_DEVICES=1 <py> -m scripts.inspect.handoff_distance --out results/handoff_distance
```
Result: hand-off ONSET distance-invariant (entity-pos influence starts falling
at L23-24 for all N). COMPLETION distance-dependent: N=0 readout has the info
by L26; N>=100 only by L31-34, with the entity position keeping ~0.4 causal
influence through L30+. Saturates by N~100 (local vs far, two regimes).
=> deep layers DO long-range reads for distant entities; the "deep layers can
all be local" hypothesis is refuted; evidence favors windows early/mid + full
attention late (consistent with E1, Meta FAIR 2510.04800, MixAttention, YOCO).

---

## Combined conclusion

Late-layer entity information is (1) linearly decodable [Step 1, ~0.94],
(2) inaccessible to Patchscope [Step 2, 0.22], (3) linearly translatable back to a
readable form [Step B, ->0.68], (4) NOT a direction-misalignment artifact [Step C],
(5) causally SPENT — used by mid layers, vestigial after [Step D]. The layer where
Patchscope stops reading it (~L22-25) coincides with the layer where the model stops
using it. So the info does not disappear; the model extracts/uses it by mid layers,
after which the late representation is a decodable-but-functionally-vestigial residue.
NOTE: `results/` is gitignored — figures/json are local; scripts are committed.
