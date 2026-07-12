"""The missing clean experiment: the flip layer's OWN contribution
Delta_L = h_L - h_{L-1}, at its NATURAL magnitude (no unit-normalizing, no
rescaling). Patch the raw diff (individually, and the raw per-phrase average)
at L6, read by hand. Nothing else touched.
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
        diffs = []
        for L in fl:
            d = Hf[L] - Hf[L - 1]          # raw contribution, natural magnitude
            diffs.append(d)
            out = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos, d,
                                         MAXTOK, op="replace"))
            print(f"  d{L:02d}=h{L}-h{L-1}  (|d|={d.norm():4.0f}) -> {out}")
        if len(fl) > 1:
            avg = torch.stack(diffs).mean(0)   # raw average, natural magnitude
            out = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos, avg,
                                         MAXTOK, op="replace"))
            print(f"  AVGraw    (|avg|={avg.norm():4.0f}) -> {out}")


if __name__ == "__main__":
    main()
