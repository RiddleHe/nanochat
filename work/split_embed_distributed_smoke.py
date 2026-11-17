"""Two-rank synthetic D24 optimizer validation; no dataset or trainer is used."""
from pathlib import Path
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup
from nanochat.flash_attention import USE_FA3
from nanochat.model.gpt_base_split_embed_shared_out_no_embed_norm import (
    GPTBaseSplitEmbedSharedOutNoEmbedNormConfig as Config,
    GPTBaseSplitEmbedSharedOutNoEmbedNorm as Model,
)

ddp, rank, local_rank, world_size, device = compute_init()
assert ddp and world_size == 2 and USE_FA3
torch.set_num_threads(1)
manifest = json.loads((HERE / "manifest.json").read_text())
config = Config(**manifest["model_config"])
with torch.device("meta"):
    model = Model(config)
model.to_empty(device=device)
model.init_weights()
# Nonzero outputs exercise every gradient on the first synthetic step.
with torch.no_grad():
    for b in model.transformer.h:
        b.attn.c_proj.weight.normal_(std=0.01)
        b.mlp.c_proj.weight.normal_(std=0.01)
optimizer = model.setup_optimizer()
scales_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("head_scale")}
output_before = model.transformer.h[-1].attn.c_proj.weight.detach().clone()
torch.manual_seed(100 + rank)  # Distinct data exercises gradient averaging.
tokens = torch.randint(config.vocab_size, (2, config.sequence_len), device=device)
torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
compiled = torch.compile(model, dynamic=False, fullgraph=True)
loss = compiled(tokens, tokens.roll(-1, 1))
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.step()
torch.cuda.synchronize()
assert all(torch.isfinite(p).all() for p in model.parameters())
for n, p in model.named_parameters():
    if n in scales_before:
        assert not torch.equal(p, scales_before[n]), n
        assert optimizer.state[p]["exp_avg"].shape == (6, 128), n
    # Check full synchronized parameter tensors, including rectangular Muon maps.
    reference = p.detach().clone()
    dist.broadcast(reference, src=0)
    torch.testing.assert_close(p, reference, atol=0, rtol=0)
assert not torch.equal(model.transformer.h[-1].attn.c_proj.weight, output_before)
result = dict(rank=rank, status="passed", loss=loss.item(),
              seconds_including_compile=time.monotonic()-start,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_reserved_bytes=torch.cuda.max_memory_reserved())
results = [None] * world_size
dist.all_gather_object(results, result)
if rank == 0:
    summary = dict(status="passed", world_size=world_size, fa3=USE_FA3,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   architecture_training_flops_per_token=model.estimate_flops(),
                   full_shape=dict(batch_per_rank=2, sequence_len=2048, depth=24, width=1536),
                   scales_sharded_shape=[6,128], parameters_identical_after_update=True,
                   synthetic_optimizer_steps=1, training_started=False, ranks=results)
    (HERE / "distributed_validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
compute_cleanup()
