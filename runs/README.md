# Training recipes

Organize new experiments by study and comparison budget:

```text
runs/
  lib/pretrain.py                    # CPU/meta preflight, manifest, explicit launch
  bov/d24_f3p91e19/
    common.json                     # shared scientific settings
    gpt_base.sh                     # thin model-specific entry point
    gpt_base_attn_res_qkv.sh         # independent Q/K/V/MLP depth routing
    gpt_base_scalar_res_qkv_no_embed_norm.sh       # learned scalar source sums; two fixed GPUs
```

Keep hardware choices and machine paths out of scientific recipes. Add a small
entry point for each future variant, using the same common recipe. Compute its
step count from that model's FLOPs estimate. Keep global tokens per optimizer
step fixed across GPU counts; adjust only gradient accumulation.

From this checkout on Drexel, prepare without launching:

```bash
PYTHON=/hdd/mh3897/nanochat/.venv/bin/python \
NANOCHAT_BASE_DIR=/local-ssd/mh3897 \
bash runs/bov/d24_f3p91e19/gpt_base.sh --gpu-count 2 --dry-run
```

`--manifest /absolute/path/manifest.json` additionally writes a new manifest.
The parent directory must exist; existing files are never overwritten. Preparing
uses CPU/meta tensors and never installs packages, downloads data, starts a queue,
or launches training. `--launch` is a separate, explicit immediate-training mode
requiring `CUDA_VISIBLE_DEVICES`. It redirects output to the printed training log.
Check GPU availability before using that mode.

The baseline uses all-false BoV flags and the smear-free GPTBase. The shared
recipe follows `arch_d24_gpt_base/meta_007809.json`: 3.91e19 FLOPs, 1,048,576
tokens/step, SSSL attention, context 2048, and the historical optimizer/evaluation
settings. Save every 1,000 steps plus final (the old run saved only at the end).
The trainer currently fixes seed 42. With two GPUs and four sequences per GPU,
accumulation is 64; the baseline resolves to 7,809 steps / 8,188,329,984 tokens.

For QKV AttnRes on two GPUs, pass `--device-batch-size 2` to reduce memory use.
The global batch stays fixed, so accumulation becomes 128. Its routing compute
is included in the FLOPs estimate: the same 3.91e19 target resolves to 7,738
steps / 8,113,881,088 tokens. Whole-step rounding makes the actual estimates
slightly different. These are estimated training FLOPs, excluding evaluation
and elementwise normalization/softmax work, not equal wall-clock budgets.

ScalarResQKVNoEmbedNorm uses raw token embeddings as the initial bank source
and separate Q/K/V, MLP, and final mixtures with positive scalar coefficients.
It omits the initial embedding RMSNorm, matching `gpt_base_no_embed_norm`;
receiver and final RMSNorm remain.
Coefficients are `g(z) = z + sqrt(1 + z^2)`, computed in FP32 once per forward,
using the rationalized expression for negative logits. Stored logits start at
zero, so every active coefficient starts at one. The training objective adds
`0.01 * mean(relu(-z - 2)^2)` over active logits only; positive logits are not
penalized. Config fields `scalar_res_gate`, `scalar_res_recovery_weight`, and
`scalar_res_recovery_margin` record these choices. Recovery follows the same
valid-target reduction as cross-entropy and is excluded during evaluation.
Its scalar AdamW group matches BoV `gamma_v`:
`lr=scalar_lr`, betas `(0.96, 0.95)`, epsilon `1e-10`, and no weight decay.
The shared recipe sets scalar LR to 0.5 before the trainer's batch scaling.
At D24 there are 2,377 active gate logits stored in one FP32 parameter
with 2,380 entries (three unused zero-padding entries). Receiver slices preserve
the Q/K/V, MLP, and final-readout ordering. `scalar_res_pad_to=4` is saved in the
model config, so CPU preparation and checkpoint loading use the same layout.
The two-rank optimizer shards the flat parameter's AdamW moments into 1,190
entries per rank; the compiled scalar update sees one fixed shape. A different
world size must divide the stored length when the optimizer shards it.
Parameter counts report active `scalars` and `scalar_padding` separately; `total`
includes storage padding. Mixing is estimated at `6 * width`
training FLOPs per source/receiver pair, excluding coefficients from the matrix
parameter FLOP term. Padding adds no model FLOPs; elementwise gating and recovery
are omitted from the estimate, like normalization. This resolves to 7,773 steps /
8,150,581,248 tokens. Checkpoints identify the model as
`gpt_base_scalar_res_qkv_no_embed_norm` and use `scalar_res_logits_flat`;
direct-coefficient checkpoints are not silently interpreted as gate logits.
Validate compiled GPU optimizer updates and collectives before a new launch.

The ScalarResQKVNoEmbedNorm launcher fixes Drexel GPU indices
`0,1` and two processes. It defaults to two sequences per GPU, giving
128 accumulation steps and the same global token batch. Recheck availability
before launching; the script does not reserve GPUs. Preparation remains the
default and requires no GPU execution:

```bash
PYTHON=/hdd/mh3897/nanochat/.venv/bin/python \
NANOCHAT_BASE_DIR=/local-ssd/mh3897 \
bash runs/bov/d24_f3p91e19/gpt_base_scalar_res_qkv_no_embed_norm.sh --dry-run
```

One-off GPU queues live OUTSIDE the repository under
`/local-ssd/mh3897/run_queues/<run-id>/`. Keep a frozen source snapshot, queue helper,
manifest, status, and queue log there. A manifest records the parent commit plus
source hashes, model configuration, tokenizer hashes, dataset file inventory,
resolved command, and outputs. Dataset inventory records paths, sizes, and mtimes,
not hashes of entire large shards. It describes the current data, not a claim
that the historical run used the same shard set.

### Pooled QK over raw token embeddings

`gpt_base_pooled_qk_embed_no_embed_norm.sh` prepares
`GPTBasePooledQKEmbedNoEmbedNorm` with the same budget and global token batch.
At D24 its target layers are `[9,10,11,12,13,14,15,18,21,22,23]` (zero based).
Other depths use `floor(i*(depth-1)/23 + 0.5)`, sorted and deduplicated; the
resolved indices are serialized as `pooled_qk_target_layers` in model config.
All BoV flags are false. The raw embedding enters the residual stream, while
block and final RMSNorm remain, as in `GPTBaseNoEmbedNorm`.

At target layers, ordinary residual-derived Q/K, RoPE, QK RMSNorm and 1.2
sharpening produce each head's causal/windowed softmax probabilities. Average
those probabilities across heads, mix the full raw token embedding vectors,
then apply that layer's D-by-D output projection. These layers have no `c_v`;
other attention layers and all MLPs are unchanged. Embedding gradients remain
live. No keys are excluded beyond the standard causal/window mask.

Queries are processed in chunks of 128 (`attention_chunk_size`), checkpointed
during gradient-enabled training. This implementation supports equal Q/K head
counts; it rejects GQA and other BoV options. Cached inference stores raw
embeddings in the target layers' existing V-cache storage without projecting
them. KVCache copy/reset and batched prompt expansion retain their semantics;
this does not reduce allocated cache memory.

`estimate_flops()` is the trainer's forward-plus-backward FLOPs **per token**.
It removes the target V matrices from the matrix count, keeps full-width
attention mixing, and adds the head-average forward/backward work.
`estimate_forward_flops()` provides the forward-only estimate per token.
Both retain the baseline's effective-context convention and omit softmax,
normalization, checkpoint recomputation and masked chunk padding. The budget
therefore compares estimated architecture FLOPs, not actual kernel work or time.
Target chunks use explicit softmax; ordinary layers retain FA3/SDPA.

The launcher defaults to two GPUs and two sequences per GPU (128 accumulation
steps). Preparation does not select or reserve GPUs and does not start training:

```bash
PYTHON=/hdd/mh3897/nanochat/.venv/bin/python \
NANOCHAT_BASE_DIR=/local-ssd/mh3897 \
bash runs/bov/d24_f3p91e19/gpt_base_pooled_qk_embed_no_embed_norm.sh --dry-run
```

Prepare a fresh frozen run under `run_queues/<run-id>/` before a real launch.
Its `launch.sh` should verify source/data/tokenizer/kernel dependencies, refuse
occupied GPUs and existing outputs, and require explicit `--launch`.
The implementation tests are `tests/test_gpt_base_pooled_qk_embed_no_embed_norm.py`.

### Head-scaled embedding values

`gpt_base_head_scaled_embed_no_embed_norm.sh` prepares
`GPTBaseHeadScaledEmbedNoEmbedNorm`. It retains the pooled model's target layers,
Q/K computation, masks, raw Emb, and chunked/checkpointed execution. Each target
head has an unconstrained FP32 vector `head_scale[head, embedding_channel]`,
initialized to one. Each head mixes the **full-width** raw embeddings separately,
scales that result elementwise, then the layer averages heads before its existing
D-by-D output projection. Multiplication and head reduction use FP32; matmuls
use the forward dtype. Non-target layers are unchanged. Cached generation keeps
raw embeddings in the inherited cache; scales are applied after mixing.

Scales use the existing coefficient AdamW group (`scalar_lr`, betas 0.96/0.95,
epsilon 1e-10, no decay or width LR scaling), not Muon. Large scale tensors shard
on the head axis, so `n_head` must divide evenly across ranks; the D24 recipe
defaults to two ranks, batch two per rank, and 128 accumulation steps. At D24,
202,752 scale parameters bring the total to 754,391,040.

For each target layer with H heads, width D, and effective context T, relative
to ordinary attention, forward FLOPs/token change by
`-2*D^2 + 2*(H-1)*D*T + 2*H*D`. Training FLOPs/token change by
`-6*D^2 + 6*(H-1)*D*T + (7*H-1)*D`. These remove W_V, count full-width mixing
per head, elementwise scales, the head mean, and shared embedding-gradient
accumulation. Scale parameters are excluded from the dense-matrix FLOP term.
Coefficient gradient reductions use the leading-order N rather than N-1 count.
The same omissions as pooled QK apply: softmax/norms, checkpoint recomputation,
masked chunk padding, and optimizer/evaluation work. This is an architecture
budget, not measured runtime. The 3.91e19 recipe resolves to **6,589 steps**;
global tokens per step stay fixed. It is more expensive than pooling first.

Preparation remains CPU/meta only:

```bash
PYTHON=/hdd/mh3897/nanochat/.venv/bin/python \
NANOCHAT_BASE_DIR=/local-ssd/mh3897 \
bash runs/bov/d24_f3p91e19/gpt_base_head_scaled_embed_no_embed_norm.sh --dry-run
```

Tests live in `tests/test_gpt_base_head_scaled_embed_no_embed_norm.py` and cover
the scale-before-mixing reference, signed scale gradients, identity initialization,
cache/checkpoint round trips, optimizer membership, and analytical FLOP changes.

### Single QK head over raw embeddings

`gpt_base_single_qk_embed_no_embed_norm.sh` prepares
`GPTBaseSingleQKEmbedNoEmbedNorm` with the same fixed targets as pooled QK.
Each target has one Q/K head of width `n_embd / n_head` (128 at D24), while
ordinary layers retain the global head count (12). Q/K project 1536 to 128;
their single softmax distribution mixes the full 1536-channel raw embeddings,
then the existing D-by-D output projection and one learned `a_scale` apply.
Each target owns one unconstrained FP32 scalar initialized to one. It multiplies
the projected output in FP32 before casting back to the forward dtype, including
cached inference. It uses coefficient AdamW at `scalar_lr`, betas (0.96, 0.95),
epsilon 1e-10, and no decay or width LR scaling. There is no W_V or head pooling. Masks, RoPE, Q/K normalization and chunked activation
checkpointing follow pooled QK. The existing uniform inference cache uses only
the first target key slot and retains full-width raw embeddings in value storage;
allocated cache memory is unchanged. Training uses live embeddings with gradients.

Per target/token, let `M = 4*D*d + 2*D^2 + 2*T*(d+D)`. Forward FLOPs are
`M + D`; training FLOPs are `3*M + 4*D`, including the scalar multiplication
and its input/scalar gradients. The scalar is excluded from matrix FLOPs.
The estimator counts the actual smaller Q/K matrices and corrects the attention
score width without shrinking embedding mixing. The same overhead exclusions as
pooled QK apply. D24 has **706,609,163 parameters (11 output scalars)**, **1,415,856,640 forward** and
**4,247,586,816 training FLOPs/token**, resolving to **8,779 steps** at 3.91e19.
The launcher defaults to two ranks and two sequences per rank; preparation never
launches training. The scalar is a required checkpoint key; older unscaled
checkpoints require an explicit migration setting each `a_scale` to one. Prepare
a fresh frozen run for this version; old snapshots retain their original code. Tests: `tests/test_gpt_base_single_qk_embed_no_embed_norm.py`.

### Split embedding chunks with a shared output map

`gpt_base_split_embed_shared_out_no_embed_norm.sh` prepares
`GPTBaseSplitEmbedSharedOutNoEmbedNorm` on the same 11 fixed targets. Each target
head attends to its contiguous D/H-channel chunk of raw Emb. Its result is
multiplied by an unconstrained FP32 `head_scale[H,D/H]`, initialized to ones.
Scaled head results are **summed**, not averaged, in FP32; one bias-free
D-by-(D/H) output matrix maps the sum back into the residual stream. This matrix
is shared among heads within a layer; different layers have independent maps.
Targets have no W_V. Q/K, masks, RoPE, QK normalization and other layers match
NoEmbedNorm. Embedding gradients stay live. FA3/SDPA handles both target and
ordinary attention; inherited `attention_chunk_size` is unused by this variant.
The inference cache stores raw chunks. Each target attention output lies in a
subspace of dimension at most D/H; its residual path and MLP remain full width.

Scales use the existing coefficient AdamW group without weight decay. At D24
their 12-by-128 shape shards on the head axis across two ranks. Matrix optimizer
groups include the smaller 1536-by-128 shared output matrices. Do not treat
these rectangular maps as orthogonal rotations.

The base FLOP estimator counts actual matrices and D total token-mixing
channels, excluding scales from its matrix term. Per target the extra work is
`D + (H-1)*(D/H)` forward, or `4*D + (H-1)*(D/H)` for forward plus backward.
The scale-gradient reduction uses the existing leading-order token convention;
backpropagation through the head sum broadcasts rather than adds gradients.
Relative to baseline, also remove W_V and shrink W_O from D-by-D to D-by-(D/H).
The shared convention excludes softmax/norms, kernel overhead, optimizer,
evaluation, and checkpoint recomputation.

D24: **730,415,616 parameters** including **16,896 scales**;
**1,492,287,104 forward FLOPs/token**, **4,476,847,232 training FLOPs/token**;
**8,329 steps** at 3.91e19 with the unchanged 1,048,576 tokens per step.
Two ranks and batch two per rank give 128 accumulation steps.
Preparation: `PYTHON=/hdd/mh3897/nanochat/.venv/bin/python
NANOCHAT_BASE_DIR=/local-ssd/mh3897 bash
runs/bov/d24_f3p91e19/gpt_base_split_embed_shared_out_no_embed_norm.sh --dry-run`
(as one shell command). Tests are in
`tests/test_gpt_base_split_embed_shared_out_no_embed_norm.py`.
`work/prepare_split_embed.py` freezes a new run, synthetic validation scripts,
environment and dependencies; its `launch.sh` requires passed validation and
explicit idle GPUs. Preparation and validation do not start dataset training.

### Split embedding chunks with per-head output maps

`gpt_base_split_embed_head_out_no_embed_norm.sh` uses the same fixed targets,
raw contiguous embedding chunks, Q/K computation, and FA3/SDPA path as the shared
output variant. It concatenates head results and applies one square D-by-D W_O.
Each head owns a D-by-d column block: `sum_h W_O[:, h*d:(h+1)*d] @ (P_h @ Emb[h])`.
There is no W_V, head scale, or head average. Dedicated output columns can learn
channel scaling directly. Other layers remain ordinary NoEmbedNorm.

The base estimator counts the actual matrices and ordinary D-channel attention
work without any correction. Per target, removing W_V saves `2*D*D` forward or
`6*D*D` training FLOPs/token. D24 has **754,188,288 parameters**,
**1,539,833,856 forward FLOPs/token**, **4,619,501,568 training FLOPs/token**, and
**8,072 steps** at the standard budget and global batch. Matrix parameters use
the existing Muon optimizer; there are no new scalar parameters.

Model tests: `tests/test_gpt_base_split_embed_head_out_no_embed_norm.py`.
`work/prepare_split_embed_head_out.py` freezes code, recipe, dependencies and
synthetic GPU checks. `work/finalize_split_embed_head_out_validation.py` seals
CPU/GPU results; the frozen launcher requires a passed two-rank preflight before
training. `work/queue_head_out_after_single.py` queues it after a successful
Single QK run and complete final checkpoint on idle GPUs 4/7.

To add a sequential run while a queue is live, create a separate frozen queue
with an explicit dependency on the first run's successful completion and complete
final checkpoint. Do not edit/restart the live queue. The dependent queue should
wait without selecting GPUs, stop if its dependency fails, and only begin its
normal GPU-idle checks after the dependency succeeds.

Logs retain the existing `<NANOCHAT_BASE_DIR>/arch_d24/<run-id>_train.log` layout;
checkpoints go to `<NANOCHAT_BASE_DIR>/base_checkpoints/<run-id>/`. Use a fresh run
ID for the no-smear baseline. Do not reuse old smear-enabled checkpoints.
Future non-d24 studies should extend the shared runner's log-directory selection.

An external polling queue is not a cluster reservation system. Its own lock can
prevent duplicate starts, but other users can race its final GPU check. Resume
requires model, metadata, and every rank's optimizer shard with the same training
horizon and GPU count. This trainer does not implement exact data/RNG replay.

### Normalized direct embedding values

The existing pooled, single-QK, and split-head-output model keys accept
`norm_value_input=False` and `sum_head_attention=False` by default, preserving
old checkpoint configurations. The trainer exposes `--norm-value-input` and
`--sum-head-attention`; both are stored in model configuration. RMSNorm has no
learned gain and runs once over full-width Emb for the direct value branch; the
initial residual remains raw. The cache stores these prepared values. Only the
pooled model accepts summed head probabilities; its estimator adds `(H-1)*T`
per target for both forward and training (the sum backward broadcasts gradients).
Normalization remains excluded under the shared FLOP accounting convention.

Separate prepare-only recipes: `pooled_qk_norm_input_sum.sh`,
`single_qk_norm_input.sh`, and `multi_head_norm_input.sh`. Their D24 horizons are
8,072, 8,779, and 8,072 steps. `work/prepare_norm_input.py` freezes distinct run
directories; `work/norm-input-prepared-runs.json` indexes them. Each frozen
`launch.sh --launch --gpus A,B` requires CPU/GPU validation and automatically
runs the two-rank preflight before training. Default invocation only verifies.
The single-head availability queue is `work/queue_norm_input.py`; it starts only
its configured target after two consecutive idle observations of both GPUs.

### Projected single-head normalized values

`gpt_base_single_qk_value_proj_no_embed_norm` implements **+W_v +single head
+norm input** when `--norm-value-input` is enabled. At the same target layers,
one 128-wide head attends to `W_V @ RMSNorm(Emb)` and applies rectangular W_O,
then the existing learned FP32 layer scalar. W_V is 128-by-1536; W_O is
1536-by-128. The initial residual remains raw. The normalization flag defaults
to false, and this separate registry key owns the new checkpoint shapes.

The base estimator counts all four rectangular projections; this model also
corrects both QK score work and value mixing to 128 channels and adds scalar
forward/backward work. D24: 684,982,283 parameters, 4,031,318,016 training
FLOPs/token, 9,250 steps. Targets use FA3/SDPA; the uniform KV cache retains its
allocation but uses only its first key/value head slot at targets.

Recipe: `runs/bov/d24_f3p91e19/single_qk_value_proj_norm_input.sh`.
Prepare and validate a separate frozen standby run with the research Python:

```bash
python work/prepare_norm_input.py --variant single_qk_value_proj_norm_input --index work/value-proj-norm-input-prepared-runs.json
python work/validate_norm_input_cpu.py --index work/value-proj-norm-input-prepared-runs.json
```

The index records the frozen run directory. Its `launch.sh` verifies dependencies
without starting anything; `launch.sh --launch --gpus A,B` requires two idle
GPUs and completes pending synthetic GPU checks before dataset training. No
queue is created by preparation. Model tests independently check dense attention
and gradients, narrow projections, scalar optimizer membership, normalized input,
cache behavior, checkpoint loading, and the analytical budget.
