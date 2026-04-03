"""Visualize cross-layer angular distance heatmaps on two CORE benchmarks.

For each model, produces two heatmaps:
- Jeopardy (world knowledge / factual recall)
- ARC Challenge (science QA / reasoning)

Each heatmap cell (i, j) shows the angular distance between the hidden state at
layer i and layer j, averaged over benchmark prompts.

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

from nanochat.common import (
    get_base_dir,
    autodetect_device_type,
    download_file_with_lock,
)
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.core_eval import render_prompts_mc, render_prompts_schema, render_prompts_lm

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BENCHMARK_LABELS = ["Jeopardy", "ARC Challenge"]

def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        if os.path.exists(eval_bundle_dir):
            shutil.rmtree(eval_bundle_dir)
        shutil.move(extracted_bundle_dir, eval_bundle_dir)

def _normalize_label(label):
    return "".join(ch.lower() for ch in label if ch.isalnum())

def resolve_core_task(task_lookup, label):
    """Resolve a CORE task label with exact match first, then fuzzy fallback."""
    key = _normalize_label(label)
    if key in task_lookup:
        return task_lookup[key]

    available = sorted(task_meta["label"] for task_meta, _ in task_lookup.values())
    raise ValueError(f"Could not find CORE task '{label}'. Available labels: {available}")

def load_core_tasks():
    """Load CORE task config and data files from eval_bundle."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        file_path = download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
        if not os.path.exists(eval_bundle_dir):
            place_eval_bundle(file_path)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    with open(config_path, 'r', encoding='utf-8') as f:
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
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        task_lookup[_normalize_label(task_meta["label"])] = (task_meta, data)
    return task_lookup

def build_benchmark_inputs(tokenizer, seq_len, task_meta, data, num_samples):
    """Build tokenized input tensors for a CORE task."""
    shuffled = list(data)
    random.Random(1337).shuffle(shuffled)
    selected = shuffled[:num_samples]
    bos_token = tokenizer.get_bos_token_id()
    inputs = []

    for sample_idx, item in enumerate(selected):
        fewshot_examples = []
        if task_meta["num_fewshot"] > 0:
            rng = random.Random(1234 + sample_idx)
            available_indices = [i for i in range(len(data)) if data[i] is not item]
            fewshot_indices = rng.sample(available_indices, task_meta["num_fewshot"])
            fewshot_examples = [data[i] for i in fewshot_indices]

        if task_meta["task_type"] == "multiple_choice":
            prompts = render_prompts_mc(item, task_meta["continuation_delimiter"], fewshot_examples)
            prompt = prompts[item["gold"]]
        elif task_meta["task_type"] == "schema":
            prompts = render_prompts_schema(item, task_meta["continuation_delimiter"], fewshot_examples)
            prompt = prompts[item["gold"]]
        elif task_meta["task_type"] == "language_modeling":
            _, prompt = render_prompts_lm(item, task_meta["continuation_delimiter"], fewshot_examples)
        else:
            raise ValueError(f"Unsupported CORE task type: {task_meta['task_type']}")

        tokens = tokenizer(prompt, prepend=bos_token)
        if len(tokens) > seq_len:
            tokens = tokens[-seq_len:]
        x_input = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        inputs.append(x_input)

    return inputs

parser = argparse.ArgumentParser()
parser.add_argument("--model-tags", type=str, nargs="+", required=True)
parser.add_argument("--labels", type=str, nargs="+", default=None)
parser.add_argument("--num-samples", type=int, default=50)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--output", type=str, default="results/angular_distance.png")
args = parser.parse_args()

if args.labels is None:
    args.labels = args.model_tags
assert len(args.labels) == len(args.model_tags)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
tokenizer = get_tokenizer()
core_tasks = load_core_tasks()

benchmark_data = []
for benchmark_label in BENCHMARK_LABELS:
    benchmark_data.append(resolve_core_task(core_tasks, benchmark_label))

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
    benchmark_inputs = [
        build_benchmark_inputs(tokenizer, seq_len, task_meta, data, args.num_samples)
        for task_meta, data in benchmark_data
    ]

    for benchmark_idx, (benchmark_label, sample_inputs) in enumerate(zip(BENCHMARK_LABELS, benchmark_inputs)):
        print(f"  Benchmark: {benchmark_label}")
        # Accumulate pairwise angular distances: (n_layer, n_layer)
        # Index i = input to attention at layer i (0-indexed)
        n_points = n_layer
        ang_dist_sum = torch.zeros(n_points, n_points)
        count = 0

        with torch.no_grad():
            for sample_idx, x_input in enumerate(sample_inputs):
                x_input = x_input.to(device)

                # Capture input to each attention sublayer via forward hooks
                layer_inputs = []

                def make_hook(layer_inputs_list):
                    def hook_fn(module, input, output):
                        layer_inputs_list.append(input[0].detach())
                    return hook_fn

                hooks = []
                for block in model.transformer.h:
                    hooks.append(block.attn.register_forward_hook(make_hook(layer_inputs)))

                model(x_input)

                for h in hooks:
                    h.remove()

                # layer_inputs[i] is the normed input to attention at layer i
                # Compute all pairwise angular distances
                reps = [li.float() for li in layer_inputs]  # n_layer tensors of (B, T, D)

                for i in range(len(reps)):
                    for j in range(i + 1, len(reps)):
                        cos_sim = torch.nn.functional.cosine_similarity(reps[i], reps[j], dim=-1)
                        cos_sim = cos_sim.clamp(-1, 1)
                        ang_dist = torch.acos(cos_sim) / torch.pi
                        avg = ang_dist.mean().cpu()
                        ang_dist_sum[i, j] += avg
                        ang_dist_sum[j, i] += avg

                count += 1
                layer_inputs.clear()

                if (sample_idx + 1) % 10 == 0:
                    print(f"    {sample_idx + 1}/{len(sample_inputs)}")

        ang_dist_avg = (ang_dist_sum / count).numpy()
        n_pts = ang_dist_avg.shape[0]

        # Reshape into offset matrix: row n = n-th subsequent layer, col = source layer
        # offset_matrix[n, l] = angular distance between layer l and layer l+n+1
        max_offset = n_pts - 1
        offset_matrix = np.full((max_offset, n_pts - 1), np.nan)
        for l in range(n_pts - 1):          # source layer
            for n in range(1, n_pts - l):   # offset (1, 2, ...)
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

fig.suptitle("Cross-Layer Angular Distance by Benchmark", fontsize=15)
fig.subplots_adjust(bottom=0.18)
cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="Angular Distance (0=identical, 1=orthogonal)")
output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved to {args.output}")
