"""Single-GPU full-size optimizer and FA3 cache validation, synthetic data only."""
from pathlib import Path
import json
import os
import sys
import subprocess

HERE = Path(__file__).resolve().parent
uuid = os.environ['CUDA_VISIBLE_DEVICES']
row = next(x for x in subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],text=True).splitlines() if x.startswith(uuid))
_, memory, utilization = row.split(',')
assert int(memory) < 100 and int(utilization) <= 5
sys.path.insert(0, str(HERE/'code'))
import torch
from nanochat.engine import KVCache
from nanochat.flash_attention import USE_FA3
from nanochat.model.gpt_base_split_embed_shared_out_no_embed_norm import GPTBaseSplitEmbedSharedOutNoEmbedNormConfig as Config, GPTBaseSplitEmbedSharedOutNoEmbedNorm as Model

assert USE_FA3 and torch.cuda.device_count() == 1
torch.manual_seed(42)
torch.set_num_threads(1)
def build(cfg):
    with torch.device('meta'):
        model = Model(cfg)
    model.to_empty(device='cuda')
    model.init_weights()
    with torch.no_grad():
        for b in model.transformer.h:
            b.attn.c_proj.weight.normal_(std=.01)
            b.mlp.c_proj.weight.normal_(std=.01)
    return model

small = build(Config(n_layer=3,n_head=2,n_kv_head=2,n_embd=256,vocab_size=128,sequence_len=32))
tokens = torch.randint(128,(1,17),device='cuda')
cache = KVCache(1,2,32,128,3,'cuda',torch.bfloat16)
small.eval()
with torch.no_grad():
    full = small(tokens)
    prefix = small(tokens[:,:6],kv_cache=cache)
    rest = small(tokens[:,6:],kv_cache=cache)
    actual = torch.cat([prefix,rest],1)
    torch.testing.assert_close(actual,full,atol=.01,rtol=.03)
    cache_error = (actual-full).abs().max().item()
del small, tokens, cache, full, actual, prefix, rest
torch.cuda.empty_cache()
cfg = Config(**json.loads((HERE/'manifest.json').read_text())['model_config'])
model = build(cfg)
optimizer = model.setup_optimizer()
before = {n:p.detach().clone() for n,p in model.named_parameters() if n.endswith('head_scale') or (n.endswith('c_proj.weight') and p.shape==(1536,128))}
tokens = torch.randint(cfg.vocab_size,(2,2048),device='cuda')
torch.cuda.reset_peak_memory_stats()
loss = model(tokens,tokens.roll(-1,1))
loss.backward()
assert torch.isfinite(loss)
optimizer.step()
torch.cuda.synchronize()
assert all(torch.isfinite(p).all() for p in model.parameters())
for n,p in model.named_parameters():
    if n in before:
        assert not torch.equal(p,before[n]),n
result=dict(status='passed',gpu_uuid=uuid,fa3_cache_max_absolute_error=cache_error,
            full_shape=dict(batch=2,sequence_len=2048,depth=24,width=1536),
            full_model_optimizer_steps=1,loss=loss.item(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            rectangular_output_and_scale_updates_verified=True,training_started=False)
(HERE/'gpu_extra_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
