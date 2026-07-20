# Step 1: Main Claim

## Working Claim

Entity and category information can remain present in deep source-layer hidden states, but it is not naturally decoded when inserted into deep target layers. The same deep source vectors can produce entity/category evidence when inserted into earlier target layers, where the model still has enough downstream computation to integrate and verbalize the signal.

In short:

> Deep-layer information is recoverable through an earlier decoding path, but late target-layer patching exposes a generative readout mismatch.

## Evidence We Want To Emphasize

1. Target-frame wording strongly affects readout. `x refers to` improves strict entity-name recovery, while `x was` mostly improves category/description recovery.
2. No-patch controls for the new neutral-ish target frames do not directly leak the five target entities under the current strict/category criteria.
3. Deep source layers 24-35 still produce hits when injected into early target layers, especially target layers 5-12.
4. Late target-layer injection rarely recovers entity/category information, even with better target frames.

## Interpretation

The result is not simply that late layers have lost entity information. A linear probe or an earlier target-layer patch can still reveal information. The stronger interpretation is that late-layer residual states are not in a format that the model's remaining layers can easily use for generative entity reconstruction.

## Caveats

- Strict entity-name hits and category/description hits should be reported separately.
- Category-only hits are weaker evidence than strict name hits.
- Entity-specific suffix prompts are confounded by prompt leakage and should be presented as exploratory or controlled separately.
- The current entity set is small; the claim should be framed as a mechanistic observation from these controlled examples, not a universal law.
