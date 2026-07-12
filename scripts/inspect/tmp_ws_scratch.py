"""Whitespace-scratch patchscope.

Idea: give the patched [x] token extra forward-pass positions to 'think' before
the definition is sampled, and keep the few-shot format consistent by inserting
the same whitespace gap after each example's colon.

Build:
  examples : "Name:<WS>definition, ..."   (WS = 6 spaces inserted after each ':')
  target   : "...corporation, x"  ->  after x we APPEND <WS> tokens, run the
             forward pass over them (with x patched), THEN sample the completion.
Control: the plain standard patchscope (no WS anywhere), same source vectors.

Sweep source layer 0..N-1 at target L6, both variants, dump to JSON for reading.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.inspect.patchscope_few_shot import SOURCE_SETS, ENTITIES

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
TGT = 6
MAXTOK = 28
WS = "      "                         # 6 spaces
CANON = SOURCE_SETS["canonical"]
EX = [("Syria", "Country in the Middle East"),
      ("Leonardo DiCaprio", "American actor"),
      ("Samsung", "South Korean multinational major appliance and consumer electronics corporation")]
OUT = "results/patchscopes/ws_scratch.json"

tok = AutoTokenizer.from_pretrained(HF)
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16).to(DEV).eval()
blocks = model.model.layers
N = len(blocks)


def enc(s):
    return tok.encode(s, add_special_tokens=False)


# ---- build the two targets ----
target_ws = ", ".join(f"{n}:{WS}{d}" for n, d in EX) + ", x"
target_plain = ", ".join(f"{n}: {d}" for n, d in EX) + ", x"
tgt_ws_ids = enc(target_ws)
tgt_plain_ids = enc(target_plain)
# CONSISTENCY: use the exact whitespace token that appears after the colons in
# the examples (not a standalone-tokenized WS, which differs). Grab the token
# right after the first ':' (id 25).
colon_i = tgt_ws_ids.index(25)
ws_ids = [tgt_ws_ids[colon_i + 1]]
xpos_ws = len(tgt_ws_ids) - 1        # the 'x' token
xpos_plain = len(tgt_plain_ids) - 1
prefill_ws = tgt_ws_ids + ws_ids     # append the scratch WS after x
prefill_plain = tgt_plain_ids

print("WS tokenizes to:", ws_ids, "=", [tok.decode([t]) for t in ws_ids])
print("x token (ws target):", tok.decode([tgt_ws_ids[xpos_ws]]),
      " | x token (plain):", tok.decode([tgt_plain_ids[xpos_plain]]))
print("prefill_ws tail:", [tok.decode([t]) for t in prefill_ws[-4:]])
print(f"target_ws = {target_ws!r}")


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
def gen(prefill_ids, x_pos, src_vec):
    x = torch.tensor([prefill_ids], device=DEV)
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= x_pos:
            return out
        h = h.clone()
        h[:, x_pos, :] = src_vec.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    handle = blocks[TGT].register_forward_hook(hook)
    o = model.generate(x, max_new_tokens=MAXTOK, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    handle.remove()
    return tok.decode(o[0, x.shape[1]:], skip_special_tokens=True).split("\n")[0]


data = {}
for ent in ENTITIES:
    sids = enc(CANON[ent])
    H = capture(sids)
    rows = []
    for L in range(N):
        t_plain = gen(prefill_plain, xpos_plain, H[L])
        t_ws = gen(prefill_ws, xpos_ws, H[L])
        rows.append({"L": L, "plain": t_plain, "ws": t_ws})
    data[ent] = rows
    print(f"[{ent}] done")

with open(OUT, "w") as f:
    json.dump(data, f, indent=1)
print("wrote", OUT)
