"""Collect separate embedding/attention/MLP sources into Q/K/V only; no plots."""
import argparse
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.model.gpt_base import norm
from .collect import capture_forward, now, relative_rms
from .collect_input_norms import run_analysis
from .prepare import write_json
from .summarize_separate_sources import summarize


def check_limit(value, limit, label):
    if not np.isfinite(value) or value > limit:
        raise ValueError(f"{label}: {value} exceeds {limit}")


@torch.inference_mode()
def measure(model, ids, reference, config):
    captured = capture_forward(model, ids)
    sources = [captured["input"][0].float()]
    for layer in range(len(model.transformer.h)):
        sources.extend([captured["attn"][layer].float(), captured["mlp"][layer].float()])
    arrays = {"source_norm": torch.stack([s.norm(dim=-1).flatten() for s in sources], -1).cpu().numpy()}
    checks, denominators = [], []
    limits = config["validation"]
    batch_size = config["source_batch_size"]
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Source batches must contain whole attention/MLP pairs")
    for layer, block in enumerate(model.transformer.h):
        actual = captured["input"][layer]
        denominator = (actual.float().square().mean(-1, keepdim=True) + torch.finfo(actual.dtype).eps).sqrt()
        denominators.append(denominator.flatten())
        available = sources[:2 * layer + 1]
        source_sum = torch.stack(available).sum(0)
        residual_error = relative_rms(source_sum, actual)
        normalization_error = relative_rms(actual.float() / denominator, norm(actual))
        check_limit(residual_error, limits["residual_relative_rms_max"], f"Residual layer {layer}")
        check_limit(normalization_error, limits["normalization_relative_rms_max"], f"Normalization layer {layer}")
        for kind in "QKV":
            weight = getattr(block.attn, f"c_{kind.lower()}").weight.to(actual.dtype).float()
            heads = block.attn.n_head if kind == "Q" else block.attn.n_kv_head
            head_dim = block.attn.head_dim
            projected_sum = torch.zeros_like(captured["projection"][(layer, kind)], dtype=torch.float32)
            parts, combined_parts = [], []
            full_norm_error = 0.0
            # Embedding separately, then batches of complete attention/MLP pairs.
            groups = [available[:1]] + [available[start:start + batch_size]
                                       for start in range(1, len(available), batch_size)]
            for group_index, group in enumerate(groups):
                projected = F.linear(torch.stack(group) / denominator, weight)
                projected_sum += projected.sum(0)
                shaped = projected.reshape(len(group), -1, heads, head_dim)
                head_norms = shaped.norm(dim=-1)
                parts.append(head_norms.permute(1, 2, 0).cpu())
                full_norm_error = max(full_norm_error, relative_rms(
                    head_norms.double().square().sum(-1).sqrt(),
                    projected.norm(dim=-1).reshape(len(group), -1)))
                if group_index == 0:
                    combined_parts.append(parts[-1])
                else:
                    # Add vectors, not their norms, to recover the old measurement.
                    combined = projected.reshape(len(group)//2, 2, *projected.shape[1:]).sum(1)
                    combined_parts.append(combined.reshape(len(group)//2, -1, heads, head_dim)
                                          .norm(dim=-1).permute(1, 2, 0).cpu())
            absolute = torch.cat(parts, -1).numpy()
            combined = torch.cat(combined_parts, -1)
            saved = torch.from_numpy(reference[f"abs_{kind}_{layer:02d}"].copy())
            parent_error = relative_rms(combined, saved)
            linear_error = relative_rms(projected_sum, F.linear(source_sum / denominator, weight))
            forward_error = relative_rms(projected_sum, captured["projection"][(layer, kind)])
            forward_rounding_error = relative_rms(F.linear(actual.float() / denominator, weight),
                                                  captured["projection"][(layer, kind)])
            check_limit(parent_error, limits["recombined_parent_relative_rms_max"], f"Parent {layer} {kind}")
            check_limit(linear_error, limits["linear_reconstruction_relative_rms_max"], f"Linearity {layer} {kind}")
            check_limit(forward_error, limits["projection_relative_rms_max"], f"Projection {layer} {kind}")
            check_limit(full_norm_error, limits["full_norm_relative_rms_max"], f"Full norm {layer} {kind}")
            total = absolute.sum(-1, keepdims=True)
            if not np.isfinite(absolute).all() or np.any(total <= 0):
                raise ValueError(f"Nonfinite norm or undefined share at {layer} {kind}")
            share = absolute / total
            share_error = float(np.abs(share.sum(-1) - 1).max())
            check_limit(share_error, limits["share_sum_max_error"], f"Shares {layer} {kind}")
            arrays[f"abs_{kind}_{layer:02d}"] = absolute
            arrays[f"share_{kind}_{layer:02d}"] = share
            checks.append(dict(layer=layer, kind=kind, residual_relative_rms=residual_error,
                               normalization_relative_rms=normalization_error,
                               linear_reconstruction_relative_rms=linear_error,
                               projection_relative_rms=forward_error,
                               forward_rounding_relative_rms=forward_rounding_error,
                               recombined_parent_relative_rms=parent_error,
                               full_norm_relative_rms=full_norm_error, share_sum_error=share_error))
    arrays["input_denominator"] = torch.stack(denominators, -1).cpu().numpy()
    return arrays, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_analysis(args.run_dir, measure_fn=measure, summarize_fn=summarize)
    except BaseException as error:
        write_json(args.run_dir / "status.json", dict(state="failed", error=repr(error),
                   traceback=traceback.format_exc(), updated_utc=now()))
        raise


if __name__ == "__main__":
    main()
