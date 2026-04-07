"""Visualize cross-layer angular distance heatmaps on CORE benchmarks.

For each model, produces one heatmap per benchmark in BENCHMARK_LABELS.

Each heatmap cell (i, j) shows the angular distance between the hidden state at
layer i and layer j, averaged over the golden-answer tokens in each prompt and
across benchmark samples.

The model is run as a single prefill pass on (few-shot context + prompt + golden
answer), and per-layer attention-input activations are sliced to the answer
span [start_idx:end_idx) — matching how core_eval scores MC/schema tasks by
the per-token loss on the answer span.

Reference: "The Curse of Depth in Large Language Models" (arxiv 2502.05795)

Usage:
    python -m scripts.visualize_angular_distance \
        --model-tags arch_d12_gpt_nolambda arch_d12_attn_res \
        --labels "GPT baseline" "AttnRes" \
        --num-samples 50 \
        --output results/angular_distance_d12.png
"""
import os
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from nanochat.common import get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.core_eval import (
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    batch_sequences_lm,
)

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BENCHMARK_LABELS = ["ARC Challenge", "Winogrande", "SQuAD"]

def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        if os.path.exists(eval_bundle_dir):
            shutil.rmtree(eval_bundle_dir)
        shutil.move(extracted_bundle_dir, eval_bundle_dir)

def _normalize_label(label):
    return "".join(ch.lower() for ch in label if ch.isalnum())

def load_benchmark_data():
    """Load the two hardcoded benchmarks from the CORE eval bundle."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        file_path = download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
        if not os.path.exists(eval_bundle_dir):
            place_eval_bundle(file_path)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task_lookup = {}
    for task in config["icl_tasks"]:
        task_meta = {
            "label": task["label"],
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        task_lookup[_normalize_label(task_meta["label"])] = (task_meta, data)

    benchmark_data = []
    for benchmark_label in BENCHMARK_LABELS:
        key = _normalize_label(benchmark_label)
        if key not in task_lookup:
            available = sorted(task_meta["label"] for task_meta, _ in task_lookup.values())
            raise ValueError(f"Could not find CORE task '{benchmark_label}'. Available labels: {available}")
        benchmark_data.append(task_lookup[key])
    return benchmark_data

def render_gold_with_indices(tokenizer, task_meta, item, fewshot_examples):
    """Render the golden-answer prompt and return (tokens, start_idx, end_idx).

    The returned span [start_idx, end_idx) covers the golden answer tokens
    within the full prompt — matching the convention used by core_eval's
    batch_sequences_* helpers.
    """
    cd = task_meta["continuation_delimiter"]
    task_type = task_meta["task_type"]
    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, cd, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
        gold = item["gold"]
        return tokens[gold], start_idxs[gold], end_idxs[gold]
    elif task_type == "schema":
        prompts = render_prompts_schema(item, cd, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
        gold = item["gold"]
        return tokens[gold], start_idxs[gold], end_idxs[gold]
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, cd, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        return tokens[0], start_idxs[0], end_idxs[0]
    else:
        raise ValueError(f"Unsupported CORE task type: {task_type}")

def build_benchmark_inputs(tokenizer, seq_len, task_meta, data, num_samples):
    """Build few-shot tokenized golden-answer prompts, dropping those too long.

    Few-shot count is taken from task_meta["num_fewshot"] (per core.yaml).
    Returns a list of (tokens, start_idx, end_idx) tuples.
    """
    shuffled_indices = list(range(len(data)))
    random.Random(1337).shuffle(shuffled_indices)
    num_fewshot = task_meta["num_fewshot"]
    inputs = []
    num_total = len(shuffled_indices)
    num_dropped = 0

    for orig_idx in shuffled_indices:
        item = data[orig_idx]
        # Sample few-shot deterministically (mirroring core_eval.evaluate_example)
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + orig_idx)
            available = [i for i in range(len(data)) if i != orig_idx]
            fewshot_indices = rng.sample(available, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]

        tokens, start_idx, end_idx = render_gold_with_indices(
            tokenizer, task_meta, item, fewshot_examples
        )
        if len(tokens) > seq_len or end_idx <= start_idx:
            num_dropped += 1
            continue
        inputs.append((tokens, start_idx, end_idx))
        if len(inputs) == num_samples:
            break

    stats = {
        "num_total": num_total,
        "num_dropped": num_dropped,
        "num_selected": len(inputs),
    }
    return inputs, stats

parser = argparse.ArgumentParser()
parser.add_argument("--model-tags", type=str, nargs="+", required=True)
parser.add_argument("--labels", type=str, nargs="+", default=None)
parser.add_argument("--num-samples", type=int, default=20)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default="results/angular_distance.png")
args = parser.parse_args()

if args.labels is None:
    args.labels = args.model_tags
assert len(args.labels) == len(args.model_tags)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
tokenizer = get_tokenizer()
benchmark_data = load_benchmark_data()

n_models = len(args.model_tags)
fig, axes = plt.subplots(len(BENCHMARK_LABELS), n_models, figsize=(6 * n_models, 5 * len(BENCHMARK_LABELS)))
if len(BENCHMARK_LABELS) == 1 and n_models == 1:
    axes = np.array([[axes]])
elif len(BENCHMARK_LABELS) == 1:
    axes = np.expand_dims(axes, axis=0)
elif n_models == 1:
    axes = np.expand_dims(axes, axis=1)

# Shared colorbar range
all_matrices = []
all_n_layers = [[None for _ in BENCHMARK_LABELS] for _ in range(n_models)]

for model_idx, (model_tag, model_label) in enumerate(zip(args.model_tags, args.labels)):
    print(f"\nProcessing: {model_label} ({model_tag})")
    model, _, meta = load_model("base", device, phase="eval", model_tag=model_tag)
    model.eval()
    config = meta["model_config"]
    n_layer = config["n_layer"]
    seq_len = config["sequence_len"]
    benchmark_inputs = []
    for task_meta, data in benchmark_data:
        sample_inputs, stats = build_benchmark_inputs(
            tokenizer,
            seq_len,
            task_meta,
            data,
            args.num_samples,
        )
        benchmark_inputs.append(sample_inputs)
        print(
            f"  Loaded {task_meta['label']} from {task_meta['dataset_uri']} "
            f"({task_meta['num_fewshot']}-shot, max_len={seq_len}, "
            f"dropped {stats['num_dropped']}/{stats['num_total']}, "
            f"selected {stats['num_selected']})"
        )

    for benchmark_idx, (benchmark_label, sample_inputs) in enumerate(zip(BENCHMARK_LABELS, benchmark_inputs)):
        print(f"  Benchmark: {benchmark_label}")
        n_points = n_layer
        ang_dist_sum = torch.zeros(n_points, n_points)
        count = 0

        # Register attention-input hooks once for this benchmark (cleared per sample)
        layer_inputs = []
        def hook_fn(module, inp, out):
            layer_inputs.append(inp[0].detach())
        hooks = [block.attn.register_forward_hook(hook_fn) for block in model.transformer.h]

        try:
            with torch.no_grad():
                for sample_idx, (prompt_tokens, start_idx, end_idx) in enumerate(sample_inputs):
                    layer_inputs.clear()
                    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                    model(input_ids)

                    # Slice each layer's activations to the golden-answer span: (n_ans, D)
                    reps = [li[0, start_idx:end_idx, :].float() for li in layer_inputs]
                    count += 1

                    for i in range(len(reps)):
                        for j in range(i + 1, len(reps)):
                            cos_sim = torch.nn.functional.cosine_similarity(reps[i], reps[j], dim=-1)
                            cos_sim = cos_sim.clamp(-1, 1)
                            ang_dist = (torch.acos(cos_sim) / torch.pi).mean().cpu()
                            ang_dist_sum[i, j] += ang_dist
                            ang_dist_sum[j, i] += ang_dist

                    if (sample_idx + 1) % 10 == 0:
                        print(f"    {sample_idx + 1}/{len(sample_inputs)}")
        finally:
            for h in hooks:
                h.remove()

        ang_dist_avg = (ang_dist_sum / count).numpy()
        n_pts = ang_dist_avg.shape[0]

        # Reshape into offset matrix: row n = n-th subsequent layer, col = source layer
        max_offset = n_pts - 1
        offset_matrix = np.full((max_offset, n_pts - 1), np.nan)
        for l in range(n_pts - 1):
            for n in range(1, n_pts - l):
                offset_matrix[n - 1, l] = ang_dist_avg[l, l + n]

        all_matrices.append(offset_matrix)
        all_n_layers[model_idx][benchmark_idx] = n_pts

    del model
    if device_type == "cuda":
        torch.cuda.empty_cache()

cmap = plt.cm.viridis_r.copy()
cmap.set_bad(color='white', alpha=0)

im = None
matrix_idx = 0
for model_idx, model_label in enumerate(args.labels):
    for benchmark_idx, benchmark_label in enumerate(BENCHMARK_LABELS):
        matrix = all_matrices[matrix_idx]
        matrix_idx += 1
        ax = axes[benchmark_idx, model_idx]
        n_pts = all_n_layers[model_idx][benchmark_idx]
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', origin='lower')
        ax.set_title(f"{model_label} | {benchmark_label}", fontsize=13)
        ax.set_xlabel("Layer Index $\\ell$", fontsize=12)
        ax.set_ylabel("Subsequent $n^{th}$ Layer", fontsize=12)
        ax.set_xticks(range(0, n_pts - 1, max(1, (n_pts - 1) // 8)))
        ax.set_xticklabels(range(0, n_pts - 1, max(1, (n_pts - 1) // 8)), fontsize=8)
        ax.set_yticks(range(0, n_pts - 1, max(1, (n_pts - 1) // 4)))
        ax.set_yticklabels([i + 1 for i in range(0, n_pts - 1, max(1, (n_pts - 1) // 4))], fontsize=8)

fig.suptitle("Cross-Layer Angular Distance by Benchmark (Answer-Span Prefill)", fontsize=15)
fig.subplots_adjust(bottom=0.18, hspace=0.45)
cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="Angular Distance (0=identical, 1=orthogonal)")
output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved to {args.output}")
