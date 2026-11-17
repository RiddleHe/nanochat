"""Synthetic validation only: one small optimizer check, no dataset reads or trainer."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
UUID = os.environ["CUDA_VISIBLE_DEVICES"]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == UUID
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
row = next(line for line in rows.splitlines() if line.startswith(UUID))
_, memory, utilization = row.split(",")
assert int(memory) < 100 and int(utilization) <= 5, row
sys.path.insert(0, str(HERE / "code"))
import torch
from nanochat.model.gpt_base_split_embed_head_out_no_embed_norm import GPTBaseSplitEmbedHeadOutNoEmbedNormConfig as Config, GPTBaseSplitEmbedHeadOutNoEmbedNorm as Model
from nanochat.flash_attention import USE_FA3

assert torch.cuda.device_count() == 1
assert USE_FA3, "Expected cached FA3 for ordinary layers on this Hopper GPU"
torch.manual_seed(42)
torch.set_num_threads(1)

def build(config):
    with torch.device("meta"):
        model = Model(config)
    model.to_empty(device="cuda")
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.01)
            block.mlp.c_proj.weight.normal_(std=0.01)
    return model

small = build(Config(sequence_len=128, vocab_size=128, n_layer=4, n_head=2, n_kv_head=2, n_embd=256, attention_chunk_size=32))
tokens = torch.randint(128, (2, 128), device="cuda")
reference = small(tokens, tokens)
reference.backward()
eager_gradients = {n: p.grad.clone() for n, p in small.named_parameters()}
small.zero_grad(set_to_none=True)
compiled = torch.compile(small, dynamic=False, fullgraph=True)
actual = compiled(tokens, tokens)
actual.backward()
torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.002)
relative_errors = {}
for name, p in small.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), name
    ref = eager_gradients[name].float()
    error = ((p.grad.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    relative_errors[name] = error
    assert error < 0.06, (name, error)
print(json.dumps({"small_compile_loss": actual.item(), "max_gradient_relative_l2_error": max(relative_errors.values())}), flush=True)
optimizer = small.setup_optimizer()
output_before = small.transformer.h[-1].attn.c_proj.weight.detach().clone()
optimizer.step()
assert all(torch.isfinite(p).all() for p in small.parameters())
assert not torch.equal(small.transformer.h[-1].attn.c_proj.weight, output_before)
from nanochat.engine import KVCache
small.eval()
cache = KVCache(2, 2, 32, 128, 4, 'cuda', torch.bfloat16)
with torch.no_grad():
    short = tokens[:, :9]
    expected = small(short)
    first = small(short[:, :3], kv_cache=cache)
    rest = small(short[:, 3:], kv_cache=cache)
    torch.testing.assert_close(torch.cat([first, rest], 1), expected, atol=0.004, rtol=0.02)
del cache, short, expected, first, rest
small_loss = actual.item()
max_error = max(relative_errors.values())
del small, compiled, actual, reference, eager_gradients, tokens, p, ref, optimizer, output_before
gc.collect()
torch.cuda.empty_cache()
torch._dynamo.reset()

manifest = json.loads((HERE / "manifest.json").read_text())
config = Config(**manifest["model_config"])
model = build(config)
tokens = torch.randint(config.vocab_size, (2, config.sequence_len), device="cuda")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.monotonic()
# Actual D24 / width1536 / T2048 / batch2, including FA3 target backward.
compiled = torch.compile(model, dynamic=False, fullgraph=True)
loss = compiled(tokens, tokens.roll(-1, 1))
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer = model.setup_optimizer()
output_before = model.transformer.h[-1].attn.c_proj.weight.detach().clone()
assert not any(n.endswith(('a_scale', 'head_scale')) for n, _ in model.named_parameters())
optimizer.step()
torch.cuda.synchronize()
assert all(torch.isfinite(p).all() for p in model.parameters())
assert not torch.equal(model.transformer.h[-1].attn.c_proj.weight, output_before)
result = dict(status="passed", gpu_uuid=UUID, torch=torch.__version__, fa3=USE_FA3,
              small_compiled_loss=small_loss, max_small_gradient_relative_l2_error=max_error,
              full_compiled_loss=loss.item(), full_forward_backward_seconds_including_compile=time.monotonic()-start,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              full_shape=dict(batch=2, sequence_len=config.sequence_len, depth=config.n_layer, width=config.n_embd),
              parameter_count=sum(p.numel() for p in model.parameters()),
              architecture_training_flops_per_token=model.estimate_flops(),
              architecture_forward_flops_per_token=model.estimate_forward_flops(),
              synthetic_small_optimizer_steps=1, full_model_optimizer_steps=1, output_updates_verified=True, cached_generation_verified=True, training_started=False)
(HERE / "gpu_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
