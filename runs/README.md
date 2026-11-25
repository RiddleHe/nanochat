# Training preparation and execution

## Branch scope

`master` contains `gpt_base`, the minimal `gpt_base_no_embed_norm` variant, and
both canonical d24 baseline recipes. `research/bank_of_values` additionally
contains the experimental models, diagnostics, and the shared preparation runner
and checks described here. The shared runner has not yet been integrated into
master. Promote shared functionality
through focused changes; keep experiment variants and generated runs on their
research branch or external storage.

## Recipe and run directory

Scientific settings live in [bov/d24_f3p91e19/common.json](bov/d24_f3p91e19/common.json):
D24, width 1,536, 12 heads, context 2,048, SSSL, seed 42, BF16,
**1,048,576 tokens per optimizer step**, and **3.91e19 estimated training FLOPs**.
Each shell entry point selects a model and any ablation flags. The model's FLOP
estimate determines the step count. GPU count and device microbatch change
accumulation while preserving the global token batch. FLOPs exclude evaluation,
optimizer work, normalization, and checkpoint recomputation; they are not runtime.

[lib/pretrain.py](lib/pretrain.py) resolves these settings on CPU/meta tensors.
Calling a recipe directly with `--dry-run` previews them; `--manifest` records
them; `--launch` starts training directly with explicit `CUDA_VISIBLE_DEVICES`.
None of those modes copies code or runs the shared validation suite.

For a frozen research run, use [scripts/run.py](../scripts/run.py). It calls the
same resolver, copies the current source, and generates one launcher per run.
There is no `work/` dependency and no per-model preparation helper to write.

## Prepare a new run

Run from the BoV checkout with an existing environment, tokenizer, dataset, and
required kernel cache. Choose a fresh run ID and a directory outside the checkout:

```bash
python_bin=/hdd/mh3897/nanochat/.venv/bin/python
run_dir=/local-ssd/mh3897/run_queues/my_new_baseline_run
"$python_bin" -m scripts.run prepare \
  --recipe runs/bov/d24_f3p91e19/gpt_base_no_embed_norm.sh \
  --run-dir "$run_dir" --base-dir /local-ssd/mh3897 --gpu-count 2
```

For another model, select its recipe. A JSON recipe requires `--model-type`.
Optional overrides include `--device-batch-size`, `--norm-value-input`,
`--sum-head-attention`, `--attention-sink`, and `--target-layers 1 3 5`; models must support the
selected options. Target indices are zero-based and literal at the configured
depth. Omission preserves the original ScalarRes depth mapping; an explicit list
must be nonempty, unique, and in range. The resolved list is saved in
`pooled_qk_target_layers` and passed unchanged to training. The trainer's D12
hyperparameter reference retains its default mapping.

The normalized single-head placement controls use
`single_qk_norm_input_interleaved_11.sh` (indices 1,3,...,21) and
`single_qk_norm_input_second_half_11.sh` (indices 12,...,22) under
`runs/bov/d24_f3p91e19/`. Both select the existing single-head model with
`--norm-value-input`; only placement changes. Use each resolved manifest's
step count: SSSL gives these selections different FLOPs despite equal parameter
and target-layer counts. These recipes prepare by default and require an explicit
launch command to train.

The SSSL controls split the interleaved-11 targets by window type:
`single_qk_norm_input_s_only_6.sh` selects its six S layers (indices
1,5,9,13,17,21), and `single_qk_norm_input_l_only_5.sh` selects its five L
layers (indices 3,7,11,15,19).
Both reuse the normalized single-head model with no W_v, retained W_o and output
scalar, no attention sink, and `common.json`. Their disjoint target sets recover
interleaved-11 when combined; both exclude the first and final layers. Target
counts, parameter counts, FLOPs, and resolved training horizons differ. Give
each frozen run a descriptive `--run-dir`:
its basename becomes the model tag used for checkpoint directories and log names.

The all-L window comparison uses `gpt_base_no_embed_norm_all_l.sh` and
`single_qk_norm_input_interleaved_11_all_l.sh`. Their shared
`common_all_l.json` differs from `common.json` only in `window-pattern: "L"`;
all 24 layers use full causal attention. Keep the remaining scientific settings
matched when maintaining either recipe. Both entry points use microbatch 2 and
the same global batch and FLOP budget. They reuse existing model classes, with
raw initial embeddings; the interleaved model retains normalized embedding values,
the same 11 targets, and no attention sink. Prepare with the shared runner above;
the resolved horizons differ because full attention increases FLOPs per token.

`single_qk_norm_input_attention_sink.sh` enables `--norm-value-input` and
`--attention-sink` on the original single-head target selection. Each target
has one zero-initialized, learned softmax sink logit with zero value contribution;
there is no token-0 measurement or calibration dependency. Sink logits have
their own AdamW group (base LR 0.001, betas 0.9/0.999, epsilon 1e-10, no decay
or width scaling); the trainer's common batch/LR schedule still applies.
The current budget excludes softmax work, including the added sink entry and
its gradient, and excludes sink scalars from matrix FLOPs. The extra parameters
therefore do not change the calculated training horizon. This is not a claim of
identical runtime. The sink flag defaults off, preserving legacy weights/configs.

Repeated
`--dependency /absolute/path` arguments record additional file/directory hashes.
The ScalarRes shell recipe requires two ranks; the shared runner still takes
its actual GPU selection from `--gpus` at test/launch time. Direct invocation of
that older shell recipe instead fixes devices 0,1.

Preparation does not run tests, wait for GPUs, or start training. It saves:

- `code/`: tracked and untracked nonignored source, including local edits;
  Markdown and Python bytecode are excluded.
- `manifest.json` and its checksum: resolved configuration, command, outputs,
  source/tokenizer/kernel/dependency hashes, and dataset inventory.
- `environment.json`: Python identity and installed package versions.
- `launch.sh`, `prepare.log`, `status.json`, and `events.jsonl`.

The environment and data remain external. Dataset verification checks shard
paths, sizes, and modification times, not complete content hashes. Preparation
does not install packages or download missing inputs. Changed source, settings,
or dependencies require a new run; do not edit or relocate a saved snapshot.

## Check, test, launch, or wait

These commands use the saved source and settings. GPU indices below are examples;
select available devices explicitly. Full GPU UUIDs are also accepted, and the
number of distinct devices must match the prepared GPU count.

| Command | Effect |
|---|---|
| `"$run_dir/launch.sh" check` | Verify saved inputs and environment; no GPU execution. This is also the default with no arguments. |
| `"$run_dir/launch.sh" test` | Run CPU model checks on an unused run. |
| `"$run_dir/launch.sh" test --gpus 6,7` | Run CPU, single-GPU, and distributed checks using the selected devices. |
| `"$run_dir/launch.sh" launch --gpus 6,7` | Recheck inputs and idle GPUs, rerun required model checks, then start training. |
| `"$run_dir/launch.sh" wait --gpus 6,7 --after /absolute/prerequisite/run` | Wait for successful prerequisite completion and idle GPUs, then validate and launch. |

`wait` runs in the foreground; it does not install a background service. Omit
`--after` to wait only for GPUs, or repeat it for several prerequisites. Defaults
are a 30-second poll, two consecutive idle samples, and a 24-hour timeout.
A dependency must report `status: completed`, `exit_code: 0`, and have the expected
final model, metadata, and every optimizer shard. Metadata must match the model
configuration and contain finite final validation BPB. Older runs reporting
`status: exited` / `returncode` do not satisfy this interface; their existing
launchers remain independent. Do not rewrite their recorded status to bypass it.

A GPU is eligible only with no reported compute process, at most 100 MiB used,
and at most 5% utilization. Availability is checked again before each GPU phase
and training. Run locks prevent duplicate controllers; GPU locks coordinate
this user's shared-runner processes. These are not cluster reservations, so
other users and tools can still race an availability check.

## What the checks establish

Shared implementation: [tests/model_checks/check.py](../tests/model_checks/check.py).
Checks use the registry model/config; new models still need correct initialization,
optimizer grouping, cache/checkpoint interfaces, and architecture-specific tests.

| Check | Scope |
|---|---|
| CPU initialization and update | Poison uninitialized storage, initialize weights, require finite outputs/loss/optimized gradients and a working optimizer update. |
| Cache and checkpoint | Compare cached/full predictions; save/load a real checkpoint and compare predictions and the next optimizer step. |
| GPU numerics | Compare small-model FP32/BF16 outputs, loss, and gradients within reported tolerances; compare compiled/eager execution. |
| Full configuration | Run a compiled synthetic optimizer update with the prepared model dimensions and device microbatch. |
| Distributed | Compare a small distributed update with a single-process global-batch reference, require rank agreement, then run a full-configuration distributed update. |

Small checks retain architecture choices but reduce width, vocabulary, and context.
Before the independent full-configuration check, compiler state is reset so
reduced-shape optimizer specializations do not consume the training shape's
compilation-cache limit. Model and optimizer compilation remain enabled.
Synthetic tokens test execution and numerical consistency; they do not establish
training quality, prove a variant's equations, or replace its analytical FLOP and
architecture tests. Reports and logs live in `checks/<timestamp>/`;
`validation.json` binds the result to the manifest and GPU selection. Launch
reruns checks even when an earlier report passed.

The checker has passed CPU, single-GPU BF16/compiled, and two-GPU distributed
checks for the normalized single-head S-only configuration, including full-D24
optimizer updates and agreement between ranks. Validation applies to the tested
configuration and devices; each new launch must pass its own checks. Consult the
frozen run's reports for its actual validation evidence.

Known model blocker: `gpt`, `gpt_nolambda`, `attn_res`, and `attn_res_sink` omit
smear-parameter initialization after meta allocation. The poisoned-storage check
rejects them. The BoV baseline and embedding-value variants are unaffected.
This is an existing model defect; the corresponding
[upstream initialization fix](https://github.com/karpathy/nanochat/commit/9822cc7424aabffd0601f4ddfb465dba269f9765)
is not applied here. Passing checks for another model does not clear this blocker.

## Outputs and existing runs

Training logs are `<base-dir>/arch_d24/<run-id>_train.log`; checkpoints are
`<base-dir>/base_checkpoints/<run-id>/`. The current recipe saves every 1,000 steps
and at the end. The runner refuses existing training outputs and duplicate start
attempts. It does not implement resume; trainer-level resume requires compatible
model/config, metadata, and all optimizer shards, and cannot replay data/RNG
state exactly. Non-d24 studies need a corresponding log-directory change.

Previously frozen external runs keep their own copied code, validation scripts,
and launcher interfaces. Follow each run's saved manifest and instructions;
`--launch --gpus ...` from an older launcher is not the new `launch --gpus ...`
subcommand. Creating new runs does not modify those saved runs.

Diagnostic collection has separate input and measurement contracts. The shared
training runner does not replace the preparation/collection/derivation scripts
under `scripts/inspect/` or their READMEs.
