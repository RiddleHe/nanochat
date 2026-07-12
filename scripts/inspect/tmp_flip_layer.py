"""Dead-simple version: for each phrase, take the source layer(s) that flip
no->yes in the vanilla patchscope (target L6), record that layer's RAW output
h_L (no magnitude change at all), patch it at L6, and read the output. If there
are several flip layers, also patch their raw average. No normalization anywhere.
"""
import torch
from scripts.inspect.patchscope_few_shot import (
    _load_hf, TARGET_DEFAULT, SOURCE_SETS, ENTITIES, capture_source_hiddens,
    patched_generate,
)

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
TGT = 6
MAXTOK = 28
CANON = SOURCE_SETS["canonical"]

# no->yes source layers at target L6, from my hand-graded raw decode sets
FLIP = {
    "diana":     [8, 10, 13, 17, 24],
    "alexander": [6],
    "ali":       [5],
    "jurassic":  [6],
    "nyc":       [5, 19],
}


def clean(t):
    return t.lstrip().split("\n")[0]


def main():
    adapter = _load_hf(HF, torch.device(DEV))
    N = adapter["n_layer"]
    tgt_ids = adapter["encode"](TARGET_DEFAULT)
    tgt_pos = len(tgt_ids) - 1
    for ent in ENTITIES:
        ids = adapter["encode"](CANON[ent])
        if len(ids) < 2:
            ids = [ids[0]] + ids
        H = capture_source_hiddens(adapter, ids, len(ids) - 1)
        Hf = {L: H[L].float() for L in range(N)}
        fl = FLIP[ent]
        print(f"\n#### {ent}  flip layers={fl} ####")
        for L in fl:
            out = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos, Hf[L],
                                         MAXTOK, op="replace"))
            print(f"  h{L:02d} (|h|={Hf[L].norm():5.0f}) -> {out}")
        if len(fl) > 1:
            avg = torch.stack([Hf[L] for L in fl]).mean(0)
            out = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos, avg,
                                         MAXTOK, op="replace"))
            print(f"  AVG{fl} (|avg|={avg.norm():5.0f}) -> {out}")


if __name__ == "__main__":
    main()
