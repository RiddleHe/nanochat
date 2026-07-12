"""TEMP: test the claim — patch source LAST-layer residual of the entity's last
token into the target x-slot at the LAST layer (S35->T35 on Qwen3-8B, 36 layers).
Everything identical except the SOURCE PHRASE:
  bare : "Diana, princess of Wales"
  fs   : PREFIX + "Diana, princess of Wales"
If the claim holds: bare -> garbage, fs -> a real description, purely because the
fs source's last-layer residual is already 'about to emit the description'."""
import torch
from scripts.inspect.patchscope_few_shot import (
    _load_hf, TARGET_DEFAULT, ENTITIES, SOURCE_SETS, run_one_source,
)

PREFIX = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer "
    "electronics corporation, "
)
CANON = SOURCE_SETS["canonical"]
HF_MODEL = "Qwen/Qwen3-8B-Base"
MAX_TOKENS = 24


def main():
    adapter = _load_hf(HF_MODEL, torch.device("cuda"))
    N = adapter["n_layer"]
    TGT = N - 1  # last layer
    print(f"Model {adapter['name']}  n_layer={N}  target_layer={TGT} (LAST)")
    conds = [("bare    ", lambda e: e), ("few-shot", lambda e: PREFIX + e)]
    rows = {}
    for label, fn in conds:
        rows[label] = {}
        for ent in ENTITIES:
            res = run_one_source(adapter, fn(CANON[ent]), TARGET_DEFAULT,
                                 TGT, MAX_TOKENS, inject_mode="residual")
            d = dict(res)
            rows[label][ent] = {L: d[L] for L in (TGT, TGT-1, TGT-2)}

    for ent in ENTITIES:
        print(f"\n================ {ent}  (last tok {CANON[ent].split()[-1]!r}) ================")
        for L in (TGT, TGT-1, TGT-2):
            print(f"  -- source layer {L} -> target layer {TGT} --")
            for label, _ in conds:
                print(f"     {label}: {rows[label][ent][L]}")


if __name__ == "__main__":
    main()
