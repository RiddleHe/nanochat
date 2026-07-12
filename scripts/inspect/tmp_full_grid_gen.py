"""Full source-layer x target-layer patchscope grid for ONE entity, BARE source
(no few-shot prefix), few-shot description target. Qwen3-8B (36 layers) -> 1296
cells. Batched per target layer (inject all 36 source-layer residuals as one
batch), so the whole grid is ~36 batched generations.

Output: results/patchscopes/<entity>_fullgrid.jsonl, one line per cell:
  {"sl": source_layer, "tl": target_layer, "text": completion}
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.inspect.patchscope_few_shot import TARGET_DEFAULT, SOURCE_SETS

HF = "Qwen/Qwen3-8B-Base"
ENT = "diana"
MAXTOK = 18
DEV = "cuda"
OUT = f"./results/patchscopes/{ENT}_fullgrid.jsonl"

tok = AutoTokenizer.from_pretrained(HF)
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16).to(DEV).eval()
blocks = model.model.layers
N = len(blocks)

src = SOURCE_SETS["canonical"][ENT]   # bare "Diana, princess of Wales"
tgt = TARGET_DEFAULT
src_ids = tok.encode(src, add_special_tokens=False)
tgt_ids = tok.encode(tgt, add_special_tokens=False)
src_pos = len(src_ids) - 1
tgt_pos = len(tgt_ids) - 1
print(f"entity={ENT} src={src!r} N={N} src_pos={src_pos} tgt_pos={tgt_pos}")

# ---- capture every source-layer residual at the last source token (1 forward) ----
res = {}
handles = []
def _mk(i):
    def h(_m, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        res[i] = x[0, src_pos, :].detach().clone()
    return h
for i, b in enumerate(blocks):
    handles.append(b.register_forward_hook(_mk(i)))
with torch.inference_mode():
    model(torch.tensor([src_ids], device=DEV))
for h in handles:
    h.remove()

# ---- batched patched generate: for a fixed target layer, inject all N source vecs ----
@torch.inference_mode()
def gen_for_target(tl):
    stacked = torch.stack([res[sl] for sl in range(N)])  # (N, d)
    x = torch.tensor([tgt_ids] * N, device=DEV)
    def hook(_m, _inp, out):
        is_t = isinstance(out, tuple)
        h = out[0] if is_t else out
        if h.shape[1] <= tgt_pos:        # only patch the prefill step
            return out
        h = h.clone()
        h[:, tgt_pos, :] = stacked.to(h.dtype)
        return (h,) + out[1:] if is_t else h
    handle = blocks[tl].register_forward_hook(hook)
    o = model.generate(x, max_new_tokens=MAXTOK, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    handle.remove()
    return [tok.decode(o[b, x.shape[1]:], skip_special_tokens=True).split("\n")[0]
            for b in range(N)]

with open(OUT, "w") as f:
    for tl in range(N):
        outs = gen_for_target(tl)
        for sl in range(N):
            f.write(json.dumps({"sl": sl, "tl": tl, "text": outs[sl]}) + "\n")
        if tl % 6 == 0 or tl == N - 1:
            print(f"target layer {tl}/{N-1} done")
print(f"wrote {OUT}  ({N*N} cells)")
