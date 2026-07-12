"""Third mode: standard few-shot examples (NO whitespace), but after the patched
[x] we append the tokens of ' refer to ' and then sample. So the target reads
'...corporation, x refer to <completion>'. Source layer swept 0..N-1 at target
L6, all 5 phrases, dumped for hand-reading."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.inspect.patchscope_few_shot import SOURCE_SETS, ENTITIES

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
TGT = 6
MAXTOK = 28
CANON = SOURCE_SETS["canonical"]
EX = [("Syria", "Country in the Middle East"),
      ("Leonardo DiCaprio", "American actor"),
      ("Samsung", "South Korean multinational major appliance and consumer electronics corporation")]
REFER = ": a historical figure who"
OUT = "results/patchscopes/referto.json"

tok = AutoTokenizer.from_pretrained(HF)
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16).to(DEV).eval()
blocks = model.model.layers
N = len(blocks)


def enc(s):
    return tok.encode(s, add_special_tokens=False)


target = ", ".join(f"{n}: {d}" for n, d in EX) + ", x"
tgt_ids = enc(target)
xpos = len(tgt_ids) - 1
refer_ids = enc(REFER)
prefill = tgt_ids + refer_ids
print("REFER tokenizes to:", [tok.decode([t]) for t in refer_ids])
print("prefill tail:", [tok.decode([t]) for t in prefill[-5:]])


@torch.inference_mode()
def capture(src_ids):
    res = {}
    handles = []
    def mk(i):
        def h(_m, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            res[i] = x[0, -1, :].detach().clone().float()
        return h
    for i, b in enumerate(blocks):
        handles.append(b.register_forward_hook(mk(i)))
    model(torch.tensor([src_ids], device=DEV))
    for h in handles:
        h.remove()
    return res


@torch.inference_mode()
def gen(src_vec):
    x = torch.tensor([prefill], device=DEV)
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= xpos:
            return out
        h = h.clone()
        h[:, xpos, :] = src_vec.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    handle = blocks[TGT].register_forward_hook(hook)
    o = model.generate(x, max_new_tokens=MAXTOK, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    handle.remove()
    return tok.decode(o[0, x.shape[1]:], skip_special_tokens=True).split("\n")[0]


data = {}
for ent in ENTITIES:
    H = capture(enc(CANON[ent]))
    data[ent] = [{"L": L, "referto": gen(H[L])} for L in range(N)]
    print(f"[{ent}] done")
json.dump(data, open(OUT, "w"), indent=1)
print("wrote", OUT)
