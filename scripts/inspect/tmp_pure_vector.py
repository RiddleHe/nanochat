"""'Pure entity vector' from no->yes transition diffs.

Idea: in the vanilla patchscope (raw h_L patched at target L6), the decode flips
no->yes at certain source layers. The block contribution Delta_L = h_L - h_{L-1}
at that transition is what pushed the residual across the decode boundary; the
shared backbone (in both h_{L-1} and h_L) cancels in the diff, so Delta_L is a
purer entity direction than any single residual. For each phrase:
  - transitions = layers that are decode-yes while the previous layer is no
    (from my hand-graded raw decode sets)
  - pure_dir = mean of the unit Delta_L over those transitions
  - patch pure_dir at L6, rescaled to the L6 norm (normalize=True), read output
Compared against patching the L19 oracle the same way. Also reports how
entity-specific each vector is (mean cross-entity cosine; lower = purer).
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
MAXTOK = 28
CANON = SOURCE_SETS["canonical"]

# hand-graded RAW decode layers (I read every completion)
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

    pure = {}
    oracle19 = {}
    for ent in ENTITIES:
        ids = adapter["encode"](CANON[ent])
        if len(ids) < 2:
            ids = [ids[0]] + ids
        H = capture_source_hiddens(adapter, ids, len(ids) - 1)
        Hf = {L: H[L].float() for L in range(N)}
        oracle19[ent] = Hf[19]

        dec = set(RAW[ent])
        trans = [L for L in range(1, N) if L in dec and (L - 1) not in dec]
        diffs = [Hf[L] - Hf[L - 1] for L in trans]
        unit = [d / (d.norm() + 1e-8) for d in diffs]
        pure_dir = torch.stack(unit).mean(0)
        pure_dir = pure_dir / (pure_dir.norm() + 1e-8)
        pure[ent] = pure_dir

        out_pure = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos,
                                          pure_dir, MAXTOK, op="replace",
                                          normalize=True))
        out_o19 = clean(patched_generate(adapter, tgt_ids, TGT, tgt_pos,
                                         oracle19[ent], MAXTOK, op="replace",
                                         normalize=True))
        print(f"\n[{ent}]  transitions={trans}  ({len(trans)} diffs averaged)")
        print(f"   PURE   @L6: {out_pure}")
        print(f"   L19    @L6: {out_o19}")

    # purity: mean cross-entity cosine (lower = more entity-specific)
    def mean_cross(vecs):
        vs = [vecs[e] for e in ENTITIES]
        cs = []
        for i in range(len(vs)):
            for j in range(len(vs)):
                if i < j:
                    a, b = vs[i], vs[j]
                    cs.append((a @ b / (a.norm() * b.norm())).item())
        return np.mean(cs)

    print("\n=== purity (mean cross-entity cosine; lower = purer) ===")
    print(f"   L19 oracles : {mean_cross(oracle19):.3f}")
    print(f"   pure vectors: {mean_cross(pure):.3f}")


if __name__ == "__main__":
    main()
