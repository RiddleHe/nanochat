# Paper relay: two readout policies, one three-pass intervention

## Scope / 这份代码是什么

This package contains **only the single-token, width-6/8/10/12 paper relay**:

- `open`: final input position can read the entity value at all layers.
- `window`: final input position can read the entity value only at `S < L <= T`.
- Both remove entity-value contributions to intermediate prompt queries and
  generated-token queries at every layer.

**不是**早期 4 donors × 5 recipients 的 Newton/Alice/apple/x/y 排查，
也不包含 K/V、字母、竞争答案或其他机制探针。不要用早期输出表解释本程序。
本 PR 不发布结果评分为论文最终标签，也不重新运行任何 GPU 实验。

## Read these three places first / 从哪里看

1. [Wide-window driver](qwen_relay_wide_single.py): `run()` runs same-window
   no-replacement controls **before** the two relay modes; unknown control
   answers remain pending, rather than being counted as failures.
2. [Intervention functions](qwen_relay_supplied_common.py): `policy()`, `capture()`,
   `run_relay()` and `generate()` specify the affected query positions.
3. Existing [three-pass hooks](qwen_entity_relay_fixed_window.py):
   `capture_donor_entity_states()`, `build_relay_state()`, `greedy_completion()`;
   existing [attention implementation](qwen_entity_attention_ablation.py):
   `disable_entity_attention()` and `entity_zero_attention_forward()`.

The last two files and the existing entity list are **unchanged** in this PR.
Their bytes match the frozen materials used in the completed experiment.

## Exact setup / 准确设置

There are two axes: **token position in the sentence** and **layer depth in the
model**. A layer window never means words before/after the entity in the sentence.

- **Entity position:** the one token representing the donor/recipient name.
- **Intermediate positions:** strictly after the entity and before the final
  prompt position. Entity-prefix positions cannot causally attend to the entity.
- **Final input position:** the last prompt token, before the first answer token
  has been generated. This is not the final transformer layer.
- **Generated queries:** query positions processed while generating subsequent
  answer tokens. Their direct entity-value contributions are removed as well.

For example, donor=`Einstein`, recipient=`Newton`, `S=22`, `T=34`:

1. Run the donor prompt; capture the **entity-position state after block 22**.
2. Run the recipient prompt; replace its **entity-position state once after
   block 22**, continue, and capture its **final-input-position state after block 34**.
3. Run a **fresh recipient prompt**; replace only its **final-input-position state
   after block 34**, then continue generation. The entity in pass 3 is not patched.

The window processes blocks **23 through 34**, inclusive: width `T-S=12`.
The transferred vector in pass 3 can retain donor information after the window.
We do not erase the final position's residual state when direct access is blocked.

| Final prompt query on the stitched relay path | Layers 0–22 | Layers 23–34 | Layer 35 |
|---|---|---|---|
| `open` / 全层可读组 | Recipient entity access allowed | Access to donor-injected entity state allowed | Fresh pass-3 recipient entity access allowed |
| `window` / 仅窗口可读组 | Entity-value contribution removed | Access to donor-injected entity state allowed | Entity-value contribution removed |

In **both** rows, intermediate/generated queries lose entity-value contributions
at **all** layers. This is not a switch that disables their entire attention row.
Other source positions are still available. The operation subtracts the selected
post-softmax weighted value contribution, does not renormalize the remaining
coefficients, and retains the entity key in the softmax denominator. Therefore
we do not claim that every possible entity-dependent information path is cut.

## Cohort, controls, and metric

- Qwen3-8B-Base, 36 blocks, greedy generation, maximum 12 answer tokens.
- BF16-loaded weights upcast to FP32, TF32 disabled; fresh cache in each pass.
- Use the **first 50 single-token names** in the existing
  `entity_attention_data/entities100_balanced12.json`; not the 50 two-token names.
- Templates: IDs `0/1/2/3/7` = direct fact / person in list / friend / visitor
  register / name badge. Validate that donor and recipient are one aligned token
  each, with identical token IDs outside that position.
- Fixed directed pairing: name IDs `i -> i XOR 1`. Both members must recover
  their own full identity in ordinary and middle-blocked/all-layer-readout
  baselines. The frozen single-token cohorts contain `50/34/2/50/46` directed
  pairs respectively (182 template–pair cases, not 182 distinct people).
  The friend template's two directions cannot support a general claim.
- Widths `6/8/10/12` have `30/28/26/24` possible windows. Starts move one block
  at a time, with run order `10/8/12/6`. There is no input-embedding boundary.
- `open` uses the fixed baseline-qualified cohort at every window.
- `window` additionally requires the **recipient's same-window, no-replacement
  control** to recover its full identity. An unknown answer waits for review;
  a failed control means relay is **not run**, not a 0% relay outcome.
- Control eligibility is determined without using the corresponding relay result.
  Whole-response identity labels/aliases are distinct from a name substring hit;
  assistant-reviewed labels require author sign-off. Record raw continuations.
- Baseline-token parity, four historical width-4 outputs and self-replacement
  checks must pass. A same-mask self-replacement must equal the no-replacement
  control token for token, including at wide-window boundaries.

## Packaging provenance (not a new scientific run)

Executed sources remain at `/hdd/mh3897/cc/nanochat-relay-x` on the experiment host.
This PR extracts seven small runtime functions and four GPU-wait functions from
older scripts so **those unrelated experiment entry points are not included**.
The intervention helper now loads the byte-identical attention implementation
and entity list already in `research/skip-ahead`, rather than duplicating them.

[Source manifest](qwen_relay_paper_sources.json) records original source hashes,
package hashes and 26 function AST fingerprints. The fingerprints were compared
against freshly downloaded executed sources; only import statements are excluded
from AST comparison. The preflight source check is deliberately adapted to the
explicit archived-source/package-source map, not silently disabled.

The package uses protocol version `wide-single-paper-package-v1`. It **must use
a new output directory** and cannot resume or append to historical results.
Do not regenerate the manifest casually to bypass a failing provenance check.
Model/output parity on the real model is rechecked by the driver before a new
GPU sweep. CPU tests do not substitute for that check.

## Tests (no GPU/model download)

The verified experiment environment used Python 3.10, torch 2.10.0+cu128,
transformers 4.57.3 and huggingface_hub 0.34.4. From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_relay_paper_package.py' -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_relay_supplied.py' -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_relay_wide_single.py' -v
```

The small randomly initialized models in these tests check intervention mechanics,
not the paper's empirical finding. Span-boundary tests are not extra two-token
experiments in the paper cohort.

Packaging validation on 2026-09-25: all **19 CPU tests passed** in the original
Python 3.10 environment, including the 26-function source fingerprint audit.
Offline, GPU-disabled preflight validated **250 tokenized prompts and 182 fixed
directed template–pairs** against the archived inputs. No real-model forward pass
or new GPU experiment was run while preparing this PR.

## Reproduction inputs and safe invocation

This is an audited continuation of a **frozen experiment**, not a fresh dataset
generator. The following historical inputs are intentionally not committed here:

| Argument | Required artifact relative to the original experiment checkout |
|---|---|
| `--baseline-dir` | `results/relay_supplied_baseline_20260923` (protocol and generated baselines) |
| `--selection` | `results/controlled_review_20260923/selection.json` |
| `--labels` | `wide_single_frozen_labels_20260924.json` |
| `--old-run` | `results/relay_controlled_windows_20260923` (four width-4 integration references) |

The checkpoint must already exist locally. The baseline's exact model snapshot,
selection hashes, source hashes and token alignment are checked. On a different
machine, coordinate a new verified baseline rather than editing paths/hashes to
make the checks pass. Raw outputs, model weights, author decisions, passwords,
and unrelated result directories are not bundled in this code PR.

On the existing experiment host, **from this PR checkout**, first use preflight:

```bash
python -m scripts.inspect.qwen_relay_wide_single \
  --out-dir results/relay_paper_preflight_NEW \
  --baseline-dir /hdd/mh3897/cc/nanochat-relay-x/results/relay_supplied_baseline_20260923 \
  --selection /hdd/mh3897/cc/nanochat-relay-x/results/controlled_review_20260923/selection.json \
  --labels /hdd/mh3897/cc/nanochat-relay-x/wide_single_frozen_labels_20260924.json \
  --old-run /hdd/mh3897/cc/nanochat-relay-x/results/relay_controlled_windows_20260923 \
  --preflight-only
```

`--preflight-only` does not claim a GPU or generate model answers. For an explicitly
authorized new run, use a **different new** output directory and replace that flag
with `--launch`. The worker acquires at most one GPU after five idle checks spaced
30 seconds apart. Locks prevent duplicate workers; it never kills another job.

Additional control review must use the new run's protocol SHA, not the old run's
review-file SHA. Unknown controls remain pending until reviewed. `--resume` is
only for this package's own unchanged run, after checking its PID/locks.

## Excluded historical diagnostic, for locating it only

The earlier 4-donor × 5-recipient table came from the server file
`scripts/inspect/qwen_relay_x_quick_probe.py`, with results under the original
experiment workspace. Its intermediate queries were **not** blocked; its final
prompt query was window-only. It is deliberately **not part of this PR** and must
not be cited as an output of the two policies implemented here.
