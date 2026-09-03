"""
Offline probing for entropy-conditioned token gradients across checkpoints.

This script is meant for the first experiment in the entropy-RL research plan:

1. Load one or more HF checkpoints.
2. Sample trajectories from each checkpoint on a fixed prompt subset.
3. Compute response-token entropies and bucket token positions by entropy
   percentile (default: 5 bins = 20-percentile buckets).
4. For selected token positions, compute the gradient of that token's log-prob
   with respect to targeted parameter rows (Qwen k_proj / down_proj rows).
5. Summarize how gradient statistics differ across
   checkpoint x entropy bucket x correctness.

Important caveat:
  This probes per-token *log-prob* gradients under teacher forcing. It does not
  reconstruct the exact token-level contribution to the current RL objective,
  because the current trainer aggregates response tokens into a sample-level
  masked-mean log-prob before applying DAPO/GRPO.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanorl.data import build_rl_dataset, verify_math


@dataclass
class Trajectory:
    checkpoint: str
    checkpoint_label: str
    trajectory_id: str
    example_id: str
    prompt: str
    response: str
    prompt_ids: list[int]
    response_ids: list[int]
    correct: bool
    token_entropies: list[float]
    token_logprobs: list[float]


@dataclass
class TokenProbeRecord:
    checkpoint_label: str
    trajectory_id: str
    example_id: str
    correct: bool
    entropy: float
    entropy_bin: int
    response_pos: int
    token_id: int
    token_text: str


def parse_args():
    parser = argparse.ArgumentParser(description="Probe token gradients across entropy buckets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoints",
        nargs="+",
        help="Checkpoint directories to probe.",
    )
    group.add_argument(
        "--checkpoint-root",
        type=str,
        help="Root directory containing step_XXXXXX checkpoints and/or a final HF checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated step numbers to probe when using --checkpoint-root, e.g. 200,400,800,1000.",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also include the non-step final checkpoint under --checkpoint-root.",
    )
    parser.add_argument("--num-prompts", type=int, default=32, help="Number of prompts sampled from the RL dataset.")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of trajectories sampled per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max decode length for probing samples.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="0,14,27",
        help="Comma-separated transformer layer indices to probe.",
    )
    parser.add_argument(
        "--row-spec",
        type=str,
        default="start,mid,last",
        help="Comma-separated row selectors. Each item can be start, mid, last, or an explicit integer index.",
    )
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=5,
        help="Number of global entropy percentile bins per checkpoint.",
    )
    parser.add_argument(
        "--max-positions-per-group",
        type=int,
        default=12,
        help="Max token positions to probe per (entropy_bin x correctness) group and checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def parse_int_list(spec: str) -> list[int]:
    vals = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals


def discover_checkpoints(args) -> list[str]:
    if args.checkpoints:
        return [str(Path(p).resolve()) for p in args.checkpoints]

    root = Path(args.checkpoint_root).resolve()
    paths = []
    if args.steps:
        for step in parse_int_list(args.steps):
            step_path = root / f"step_{step:06d}"
            if not step_path.is_dir():
                raise FileNotFoundError(f"checkpoint step not found: {step_path}")
            paths.append(str(step_path))
    else:
        for step_path in sorted(root.glob("step_*")):
            if step_path.is_dir():
                paths.append(str(step_path.resolve()))

    if args.include_final:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("Qwen_"):
                paths.append(str(child.resolve()))

    if not paths:
        raise RuntimeError("No checkpoints discovered.")
    return paths


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_examples(num_prompts: int, seed: int):
    dataset = build_rl_dataset()
    n = min(num_prompts, len(dataset))
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), n)
    return [dataset[i] for i in idxs]


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32}
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def compute_response_token_stats(model, full_ids: list[int], prompt_len: int, device: torch.device):
    ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(ids, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :].float()
        labels = ids[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        entropies = -(log_probs.exp() * log_probs).sum(dim=-1)[0]
        token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]

    start = prompt_len - 1
    return entropies[start:].tolist(), token_logprobs[start:].tolist()


def sample_trajectories(
    model,
    tokenizer,
    examples,
    checkpoint_path: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
):
    trajectories: list[Trajectory] = []
    checkpoint_label = Path(checkpoint_path).name
    top_k_arg = None if top_k < 0 else top_k
    do_sample = temperature > 0

    for example_idx, example in enumerate(examples):
        enc = tokenizer(example.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_k=top_k_arg,
                num_return_sequences=num_samples,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        prompt_ids = input_ids[0].tolist()
        prompt_len = len(prompt_ids)
        eos_id = tokenizer.eos_token_id
        for sample_idx, seq in enumerate(outputs):
            seq_ids = seq.tolist()
            response_ids = seq_ids[prompt_len:]
            if not response_ids:
                continue
            # HF generate pads shorter sequences with pad_token after EOS (pad==eos here),
            # which would otherwise pollute entropy/gradient stats. Keep the first EOS
            # itself (its own entropy is a real policy decision) and drop the rest.
            if eos_id is not None and eos_id in response_ids:
                first_eos = response_ids.index(eos_id)
                response_ids = response_ids[:first_eos + 1]
                seq_ids = prompt_ids + response_ids
            response = tokenizer.decode(response_ids, skip_special_tokens=False)
            reward, _ = verify_math(example, response, step=0)
            token_entropies, token_logprobs = compute_response_token_stats(
                model=model,
                full_ids=seq_ids,
                prompt_len=prompt_len,
                device=device,
            )
            trajectories.append(
                Trajectory(
                    checkpoint=checkpoint_path,
                    checkpoint_label=checkpoint_label,
                    trajectory_id=f"{checkpoint_label}:{example_idx}:{sample_idx}",
                    example_id=example.id,
                    prompt=example.prompt,
                    response=response,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    correct=bool(reward == 1.0),
                    token_entropies=token_entropies,
                    token_logprobs=token_logprobs,
                )
            )
    return trajectories


def compute_entropy_edges(trajectories: list[Trajectory], num_bins: int):
    all_entropies = []
    for traj in trajectories:
        all_entropies.extend(traj.token_entropies)
    if not all_entropies:
        raise RuntimeError("No response tokens collected for entropy bucketing.")
    ent = torch.tensor(all_entropies, dtype=torch.float32)
    q = torch.linspace(0.0, 1.0, num_bins + 1)
    edges = torch.quantile(ent, q)
    # Guard against degenerate repeated quantile edges.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges.tolist()


def entropy_bucket(value: float, edges: list[float]) -> int:
    # edges has length bins + 1; return bucket in [0, bins-1]
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if i == len(edges) - 2:
            if left <= value <= right:
                return i
        elif left <= value < right:
            return i
    return len(edges) - 2


def select_probe_positions(
    trajectories: list[Trajectory],
    tokenizer,
    edges: list[float],
    max_positions_per_group: int,
    seed: int,
):
    grouped = defaultdict(list)
    for traj in trajectories:
        for pos, entropy in enumerate(traj.token_entropies):
            bucket = entropy_bucket(entropy, edges)
            token_id = traj.response_ids[pos]
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            record = TokenProbeRecord(
                checkpoint_label=traj.checkpoint_label,
                trajectory_id=traj.trajectory_id,
                example_id=traj.example_id,
                correct=traj.correct,
                entropy=float(entropy),
                entropy_bin=bucket,
                response_pos=pos,
                token_id=token_id,
                token_text=token_text,
            )
            grouped[(bucket, traj.correct)].append(record)

    rng = random.Random(seed)
    selected = []
    for records in grouped.values():
        records = list(records)
        rng.shuffle(records)
        selected.extend(records[:max_positions_per_group])
    return selected


def resolve_row_indices(n_rows: int, row_spec: str) -> list[int]:
    out = []
    for item in row_spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "start":
            idx = 0
        elif item == "mid":
            idx = n_rows // 2
        elif item == "last":
            idx = n_rows - 1
        else:
            idx = int(item)
        if not (0 <= idx < n_rows):
            raise ValueError(f"row index {idx} out of range for n_rows={n_rows}")
        if idx not in out:
            out.append(idx)
    return out


def build_probe_targets(model, layer_indices: list[int], row_spec: str):
    named = dict(model.named_parameters())
    targets = []
    for layer in layer_indices:
        for suffix in ("self_attn.k_proj.weight", "mlp.down_proj.weight"):
            name = f"model.layers.{layer}.{suffix}"
            if name not in named:
                raise KeyError(f"missing parameter in checkpoint: {name}")
            param = named[name]
            rows = resolve_row_indices(param.shape[0], row_spec)
            targets.append({"name": name, "param": param, "rows": rows})
    return targets


def row_stats(grad_row: torch.Tensor, weight_row: torch.Tensor):
    grad_norm = torch.linalg.vector_norm(grad_row).item()
    weight_norm = torch.linalg.vector_norm(weight_row).item()
    if grad_norm == 0.0 or weight_norm == 0.0:
        cosine = 0.0
        proj_energy_frac = 0.0
    else:
        dot = torch.dot(grad_row, weight_row).item()
        cosine = dot / (grad_norm * weight_norm)
        proj_coeff = dot / weight_norm
        proj_energy_frac = (proj_coeff * proj_coeff) / max(grad_norm * grad_norm, 1e-12)
    return {
        "grad_norm": grad_norm,
        "row_cosine": cosine,
        "proj_energy_frac": proj_energy_frac,
    }


def effective_rank(vectors: list[torch.Tensor]) -> float | None:
    if not vectors:
        return None
    mat = torch.stack(vectors, dim=0)
    if mat.shape[0] == 1:
        return 1.0
    s = torch.linalg.svdvals(mat)
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def summarize_scalars(values: list[float]):
    if not values:
        return {}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "median": float(t.median().item()),
    }


def probe_checkpoint(
    checkpoint_path: str,
    examples,
    args,
    device: torch.device,
):
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device=device)
    layer_indices = parse_int_list(args.layer_indices)
    targets = build_probe_targets(model, layer_indices=layer_indices, row_spec=args.row_spec)

    trajectories = sample_trajectories(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        checkpoint_path=checkpoint_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    edges = compute_entropy_edges(trajectories, num_bins=args.entropy_bins)
    selected_positions = select_probe_positions(
        trajectories=trajectories,
        tokenizer=tokenizer,
        edges=edges,
        max_positions_per_group=args.max_positions_per_group,
        seed=args.seed,
    )

    traj_by_id = {traj.trajectory_id: traj for traj in trajectories}
    selected_by_traj = defaultdict(list)
    for record in selected_positions:
        selected_by_traj[record.trajectory_id].append(record)

    summary_acc = defaultdict(lambda: {"grad_norms": [], "cosines": [], "proj_energy_fracs": [], "grad_rows": []})
    token_records = []

    for trajectory_id, records in selected_by_traj.items():
        traj = traj_by_id[trajectory_id]
        full_ids = traj.prompt_ids + traj.response_ids
        ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        attn = torch.ones_like(ids, device=device)
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :].float()
        labels = ids[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
        start = len(traj.prompt_ids) - 1

        ordered_records = sorted(records, key=lambda r: r.response_pos)
        for idx, record in enumerate(ordered_records):
            token_lp = token_logprobs[start + record.response_pos]
            grads = torch.autograd.grad(
                token_lp,
                [t["param"] for t in targets],
                retain_graph=(idx + 1) < len(ordered_records),
                allow_unused=False,
            )

            for target, grad in zip(targets, grads):
                param_name = target["name"]
                weight = target["param"].detach()
                for row_idx in target["rows"]:
                    grad_row = grad[row_idx].detach().float().cpu()
                    weight_row = weight[row_idx].detach().float().cpu()
                    stats = row_stats(grad_row, weight_row)
                    group_key = (
                        traj.checkpoint_label,
                        record.entropy_bin,
                        "correct" if traj.correct else "incorrect",
                        param_name,
                        row_idx,
                    )
                    summary_acc[group_key]["grad_norms"].append(stats["grad_norm"])
                    summary_acc[group_key]["cosines"].append(stats["row_cosine"])
                    summary_acc[group_key]["proj_energy_fracs"].append(stats["proj_energy_frac"])
                    summary_acc[group_key]["grad_rows"].append(grad_row)
                    token_records.append(
                        {
                            "checkpoint_label": traj.checkpoint_label,
                            "trajectory_id": traj.trajectory_id,
                            "example_id": traj.example_id,
                            "correct": traj.correct,
                            "entropy_bin": record.entropy_bin,
                            "entropy": record.entropy,
                            "response_pos": record.response_pos,
                            "token_id": record.token_id,
                            "token_text": record.token_text,
                            "param_name": param_name,
                            "row_idx": row_idx,
                            **stats,
                        }
                    )

        model.zero_grad(set_to_none=True)

    grouped_summary = {}
    for key, acc in summary_acc.items():
        checkpoint_label, entropy_bin_idx, correctness, param_name, row_idx = key
        grouped_summary.setdefault(checkpoint_label, {})
        grouped_summary[checkpoint_label].setdefault(f"entropy_bin_{entropy_bin_idx}", {})
        grouped_summary[checkpoint_label][f"entropy_bin_{entropy_bin_idx}"].setdefault(correctness, {})
        grouped_summary[checkpoint_label][f"entropy_bin_{entropy_bin_idx}"][correctness][f"{param_name}:row_{row_idx}"] = {
            "grad_norm": summarize_scalars(acc["grad_norms"]),
            "row_cosine": summarize_scalars(acc["cosines"]),
            "proj_energy_frac": summarize_scalars(acc["proj_energy_fracs"]),
            "effective_rank": effective_rank(acc["grad_rows"]),
        }

    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_label": Path(checkpoint_path).name,
        "entropy_edges": edges,
        "num_trajectories": len(trajectories),
        "num_selected_positions": len(selected_positions),
        "summary": grouped_summary.get(Path(checkpoint_path).name, {}),
        "selected_positions": [record.__dict__ for record in selected_positions],
        "token_probe_records": token_records,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    checkpoint_paths = discover_checkpoints(args)
    examples = sample_examples(num_prompts=args.num_prompts, seed=args.seed)

    run_summary = {
        "args": vars(args),
        "checkpoint_results": [],
    }

    positions_path = Path(args.output_dir) / "token_probe_records.jsonl"
    with open(positions_path, "w", encoding="utf-8") as positions_f:
        for checkpoint_path in checkpoint_paths:
            result = probe_checkpoint(
                checkpoint_path=checkpoint_path,
                examples=examples,
                args=args,
                device=device,
            )
            run_summary["checkpoint_results"].append(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("selected_positions", "token_probe_records")
                }
            )
            for row in result["token_probe_records"]:
                positions_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_json": str(summary_path),
                "token_probe_records_jsonl": str(positions_path),
                "checkpoints": checkpoint_paths,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote token probe records to {positions_path}")


if __name__ == "__main__":
    main()
