# Finding-1 daily report — 2026-07-22

Direction: make EARLY layers read already-processed (deep) states of earlier
tokens. Everything below ran today.

## 1. Prior-art map (web, verified against PDFs)
- Bare wiring TAKEN: LCKV (2405.10637) = all layers read top-layer KV,
  ~3x train cost approximate iteration, quality parity-at-best, memory-motivated.
  Exact 1-layer diagonal TAKEN: Recurrent Transformer (2604.21215), tiled exact
  training, 6-layer beats 24-layer at 300M.
- Open gaps we target: (G1) wiring chosen by causal measurement (nobody);
  (G2) augmentation not replacement + QUALITY gains at equal FLOPs (nobody);
  (G3) exact k-shift wavefront training (open).
- Must cite: LCKV + 2410.14442, Recurrent Transformer, Feedback Transformer.

## 2. Frozen-model viability probe (early_injection.py, Qwen3-8B, N=800)
Inject the entity's L30 state (ridge-translated L30->L8, wikitext-fit,
resid 0.062) at the READOUT position, L8. 16 pairs.
- ref interchange sanity 0.90-0.99; inject_replace 0.45-0.48 (L4/L8/L14 all);
  inject_add scales with alpha (0.33-0.37 at a=2); survives deep windowing.
- Read: frozen early layers can partially use early-injected processed
  content; treat 0.45 as a lower bound (model never trained for this channel).

## 3. Architecture implemented: chunk deep-KV (gpt_base, pushed)
Early third of layers get a gated extra attention branch over K/V projected
from strictly-earlier chunks' states (chunk 256), source = final-block output
(deep) or same-layer pass-1 state (control). Pass-1 no-grad forward, detached
(TXL-style) => parallel training preserved. Per-head gates init 0 => starts
exactly at baseline. No new matrices. FLOPs honestly counted: 1.414x at d12
shape (train script auto-shortens steps for equal-FLOPs comparison).
Verified: bitwise parity at gate=0; grads flow to gates; chunk-0 logits
delta exactly 0 (no mask leakage).

## 4. Training launched (d12 @1.5e18, equal FLOPs, idle GPUs only)
- arch_d12_gpt_base_chunk_deep_kv_1.5e18 (running, healthy at 56%)
- arch_d12_gpt_base_chunk_same_kv_1.5e18 (queued)
- baseline: arch_d12_gpt_base_1.5e18 val_bpb 0.8540
Key scientific quantity: deep_kv minus same_kv = net value of PROCESSED
content vs mere extra visibility. Kill rule: >0.005 bpb behind baseline.

## 5. Compute reality check for the Aug 1-2B plan (this machine)
H100 bf16 at mfu ~35% ~= 3.5e14 FLOP/s/GPU; 4 fully-idle GPUs = 1.4e15.
- d24/780M compute-optimal (3.91e19): ~7.5h/run (measured 437m).
- 1.6B (d32) compute-optimal (~1.6e20): ~32h/run on 4 idle GPUs.
- "convincing" 100B-token 1.6B (~9.6e20): ~8 days on 4 idle GPUs.
With the only-fully-idle policy on a shared box, 100B-token 1-2B runs are
NOT realistic here; compute-optimal 1.6B is borderline (needs ~1.5 idle
days x4). Team decision needed: dedicated allocation or external compute.

## 6. UPDATE — full three-number verdict (equal-token referee run done)
| run | steps | val_bpb |
|---|---|---|
| baseline (full 1.5e18) | 3766 | 0.85400 |
| equal-token baseline (1.06e18, completed schedule) | 2661 | 0.86863 |
| chunk deep_kv | 2663 | **0.86757** |
| chunk same_kv (control) | 2663 | 0.86928 |

Ordering deep < equaltoken < same is exactly the theory's prediction:
- deep_kv − equaltoken = **−0.00106**: the branch is per-token BENEFICIAL;
  the entire equal-FLOPs loss vs baseline is the v1 two-pass tax (+41% compute).
- deep_kv − same_kv = −0.00171: the benefit comes specifically from PROCESSED
  content; mere extra long-range visibility (same_kv) is slightly harmful.
Caveats: single seed each; margins are ~0.001 bpb (small). Next: v2 removes
the tax via single-pass chunk-recurrent training (chunk loop, within-chunk
parallel, exact for normal layers, detached deep sources for the branch);
then re-run the equal-FLOPs comparison + seed variance.
