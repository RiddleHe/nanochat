"""Disentangle direction vs magnitude for the pure (transition-diff) vector.

Patch the pure direction and the L19-oracle direction at L6 across a range of
FIXED magnitudes, and read every output. Also report cosine(pure_dir, L19_dir):
if the pure (backbone-free) direction is nearly orthogonal to the decodable L19
direction, then decoding relies on the backbone-aligned component that the diff
removed -- i.e. the 'pure' vector, though entity-specific, is not the readable
one.
"""
import numpy as np
import torch

from scripts.inspect.patchscope_few_shot import (
    _load_hf, TARGET_DEFAULT, SOURCE_SETS, ENTITIES, capture_source_hiddens,
    patched_generate,
)

HF = "Qwen/Qwen3-8B-Base"
DEV = "cuda"
TGT = 6
MAXTOK = 24
MAGS = [50, 100, 150, 250]
CANON = SOURCE_SETS["canonical"]
RAW = {
    "diana":     [8,10,13,17,18,19,20,21,22,24],
    "alexander": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "ali":       [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    "jurassic":  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "nyc":       [5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29],
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
        dec = set(RAW[ent])
        trans = [L for L in range(1, N) if L in dec and (L - 1) not in dec]
        unit = [(Hf[L] - Hf[L - 1]) / ((Hf[L] - Hf[L - 1]).norm() + 1e-8) for L in trans]
        pure = torch.stack(unit).mean(0)
        pure = pure / (pure.norm() + 1e-8)
        o19 = Hf[19] / Hf[19].norm()
        cos_po = (pure @ o19).item()
        print(f"\n############ {ent}  (|L19|={Hf[19].norm():.0f}, "
              f"cos(pure,L19)={cos_po:+.2f}, transitions={trans}) ############")
        for mag in MAGS:
            op = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos,
                                        pure * mag, MAXTOK, op="replace"))
            oo = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos,
                                        o19 * mag, MAXTOK, op="replace"))
            print(f"  mag={mag:3d} | PURE: {op[:70]}")
            print(f"          | L19 : {oo[:70]}")


if __name__ == "__main__":
    main()
