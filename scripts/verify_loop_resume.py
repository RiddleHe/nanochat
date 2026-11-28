"""Compare continuous/resumed loop-training checkpoints, entirely on the CPU.

Example:
    python -m scripts.verify_loop_resume \
        --continuous run_a/checkpoints/step_000004 \
        --resumed run_b/checkpoints/step_000004

Run roots or checkpoint roots are also accepted with ``--step`` (default: 4).
All tensor comparisons are byte-for-byte, including signed zeros and NaN bits.
Only completion reason and elapsed-time metadata are excluded. A mismatch or an
unreadable/incomplete checkpoint yields nonzero exit status and compact JSON.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import re
import struct

import torch


def _finite_json(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


class Comparison:
    """Recursively compare values without including parameter/data values in logs."""

    def __init__(self, name, max_details=20):
        self.result = {
            "name": name, "equal": True, "tensors_checked": 0,
            "tensor_elements_checked": 0, "scalars_checked": 0,
            "mismatches": 0, "details": [],
        }
        self.max_details = max_details

    def fail(self, path, reason, **fields):
        self.result["equal"] = False
        self.result["mismatches"] += 1
        if len(self.result["details"]) < self.max_details:
            self.result["details"].append({"path": path, "reason": reason, **fields})

    def compare(self, expected, actual, path="$"):
        if torch.is_tensor(expected) or torch.is_tensor(actual):
            self._compare_tensor(expected, actual, path)
            return
        if type(expected) is not type(actual):
            self.fail(path, "type", continuous_type=type(expected).__name__,
                      resumed_type=type(actual).__name__)
            return
        if isinstance(expected, dict):
            missing, extra = expected.keys() - actual.keys(), actual.keys() - expected.keys()
            if missing or extra:
                self.fail(path, "dictionary_keys", missing_count=len(missing), extra_count=len(extra),
                          missing_keys=[str(k)[:120] for k in sorted(missing, key=str)[:10]],
                          extra_keys=[str(k)[:120] for k in sorted(extra, key=str)[:10]])
            for key in expected.keys() & actual.keys():
                self.compare(expected[key], actual[key], f"{path}[{str(key)[:120]}]")
            return
        if isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                self.fail(path, "sequence_length", continuous_length=len(expected), resumed_length=len(actual))
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.compare(left, right, f"{path}[{index}]")
            return
        self.result["scalars_checked"] += 1
        if isinstance(expected, float):
            equal = struct.pack("!d", expected) == struct.pack("!d", actual)
        else:
            equal = expected == actual
        if not equal:
            fields = {}
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                fields["absolute_error"] = _finite_json(abs(expected - actual))
            self.fail(path, "scalar_value", **fields)

    def _compare_tensor(self, expected, actual, path):
        if not (torch.is_tensor(expected) and torch.is_tensor(actual)):
            self.fail(path, "tensor_type")
            return
        self.result["tensors_checked"] += 1
        self.result["tensor_elements_checked"] += expected.numel()
        if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.layout != actual.layout:
            self.fail(path, "tensor_metadata", continuous_shape=list(expected.shape),
                      resumed_shape=list(actual.shape), continuous_dtype=str(expected.dtype),
                      resumed_dtype=str(actual.dtype), continuous_layout=str(expected.layout),
                      resumed_layout=str(actual.layout))
            return
        if expected.layout != torch.strided or expected.is_quantized or actual.is_quantized:
            self.fail(path, "unsupported_tensor_layout")
            return
        left = expected.detach().cpu().contiguous().reshape(-1)
        right = actual.detach().cpu().contiguous().reshape(-1)
        left_bytes, right_bytes = left.view(torch.uint8), right.view(torch.uint8)
        if torch.equal(left_bytes, right_bytes):
            return
        different_elements = (left_bytes != right_bytes).view(-1, expected.element_size()).any(dim=1)
        fields = {"differing_elements": int(different_elements.sum().item()),
                  "elements": expected.numel(), "dtype": str(expected.dtype)}
        # Float64 diagnostics are descriptive only; acceptance always uses bytes.
        if not expected.is_complex():
            a, b = left.to(torch.float64), right.to(torch.float64)
            finite = torch.isfinite(a) & torch.isfinite(b)
            fields["nonfinite_pairs"] = int((~finite).sum().item())
            if finite.any():
                absolute = (a[finite] - b[finite]).abs()
                relative = absolute / a[finite].abs().clamp_min(1e-30)
                fields["max_absolute_error"] = _finite_json(absolute.max().item())
                fields["max_relative_error"] = _finite_json(relative.max().item())
        self.fail(path, "tensor_bytes", **fields)


def _checkpoint_directory(path, step):
    root = Path(path).expanduser().resolve()
    if (root / "complete.json").is_file():
        return root
    candidates = (root / "checkpoints" / f"step_{step:06d}", root / f"step_{step:06d}")
    found = [candidate for candidate in candidates if (candidate / "complete.json").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one complete step {step} checkpoint under {root}; found {len(found)}")
    return found[0]


def _rank_inventory(directory):
    ranks = []
    for path in directory.glob("rank*.pt"):
        match = re.fullmatch(r"rank(\d+)\.pt", path.name)
        if match:
            ranks.append(int(match.group(1)))
    return sorted(ranks)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def compare_checkpoints(continuous, resumed, step=4, max_details=20):
    continuous = _checkpoint_directory(continuous, step)
    resumed = _checkpoint_directory(resumed, step)
    result = {
        "equal": True, "continuous": str(continuous), "resumed": str(resumed),
        "requested_step": step, "tensor_policy": "bitwise", "comparisons": [],
    }

    def compare(name, expected, actual):
        comparison = Comparison(name, max_details)
        comparison.compare(expected, actual)
        result["comparisons"].append(comparison.result)
        result["equal"] &= comparison.result["equal"]
        return comparison

    metadata = [json.loads((directory / "complete.json").read_text())
                for directory in (continuous, resumed)]
    ignored = {"reason", "training_seconds", "evaluation_seconds", "optimizer_seconds"}
    for label, value in zip(("continuous", "resumed"), metadata):
        if type(value.get("step")) is not int or value["step"] != step:
            raise ValueError(f"{label} checkpoint step differs from requested step {step}")
        if type(value.get("world_size")) is not int or value["world_size"] < 1:
            raise ValueError(f"{label} checkpoint has invalid world_size")
        if not value.get("config_sha256"):
            raise ValueError(f"{label} checkpoint is missing config_sha256")
    compare("completion_metadata", *({k: v for k, v in value.items() if k not in ignored}
                                     for value in metadata))
    world_size = metadata[0]["world_size"]
    result["world_size"] = world_size
    inventories = [_rank_inventory(directory) for directory in (continuous, resumed)]
    for label, inventory, meta in zip(("continuous", "resumed"), inventories, metadata):
        compare(f"{label}_rank_inventory", list(range(meta["world_size"])), inventory)
    compare("matching_rank_inventory", *inventories)

    expected_model, actual_model = _load(continuous / "model.pt"), _load(resumed / "model.pt")
    if not isinstance(expected_model, dict) or not isinstance(actual_model, dict):
        raise ValueError("model.pt must contain a model state dictionary")
    model_comparison = compare("model", expected_model, actual_model)
    if model_comparison.result["tensors_checked"] == 0:
        raise ValueError("model.pt contains no comparable tensors")
    del expected_model, actual_model
    gc.collect()

    for rank in sorted(set(inventories[0]) & set(inventories[1])):
        expected = _load(continuous / f"rank{rank}.pt")
        actual = _load(resumed / f"rank{rank}.pt")
        for label, record in (("continuous", expected), ("resumed", actual)):
            if not isinstance(record, dict) or not {"optimizer", "training"} <= record.keys():
                raise ValueError(f"{label} rank {rank} is missing optimizer/training state")
            training = record["training"]
            if not isinstance(training, dict) or not {"loader", "rng", "rank", "world_size"} <= training.keys():
                raise ValueError(f"{label} rank {rank} is missing iterator/RNG/rank state")
            compare(f"{label}_rank{rank}_identity", {"rank": rank, "world_size": world_size},
                    {key: training[key] for key in ("rank", "world_size")})
        compare(f"rank{rank}_optimizer", expected["optimizer"], actual["optimizer"])
        compare(f"rank{rank}_loader", expected["training"]["loader"], actual["training"]["loader"])
        compare(f"rank{rank}_rng", expected["training"]["rng"], actual["training"]["rng"])
        compare(f"rank{rank}_training_metadata",
                {key: value for key, value in expected["training"].items() if key not in ("loader", "rng")},
                {key: value for key, value in actual["training"].items() if key not in ("loader", "rng")})
        compare(f"rank{rank}_extra_state",
                {key: value for key, value in expected.items() if key not in ("optimizer", "training")},
                {key: value for key, value in actual.items() if key not in ("optimizer", "training")})
        del expected, actual
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--max-details", type=int, default=20)
    args = parser.parse_args()
    if args.step < 0 or args.max_details < 0:
        parser.error("--step and --max-details must be nonnegative")
    try:
        result = compare_checkpoints(args.continuous, args.resumed, args.step, args.max_details)
        exit_code = 0 if result["equal"] else 1
    except Exception as error:
        result = {"equal": False, "error_type": type(error).__name__, "error": str(error)[:1000]}
        exit_code = 2
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
