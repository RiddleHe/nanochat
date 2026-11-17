"""GPTBase with raw token embeddings entering the residual stream.

Initialization, blocks, optimizer setup, and FLOP accounting are inherited from
GPTBase. Keep forward aligned with GPTBase.forward except for the omitted
embedding RMSNorm; block pre-norms and the final RMSNorm are unchanged.
"""

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.model.gpt_base import GPTBase, norm


class GPTBaseNoEmbedNorm(GPTBase):
    def _prepare_embedding_values(self, embeddings):
        """Direct-value variants may transform Emb without changing the residual."""
        return embeddings

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens without normalizing the initial residual stream.
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)

        # Forward the trunk of the Transformer.
        # x0 = prepared embedding value input; raw by default.
        # detach_init_value: detach once here (idempotent — same as detaching per-block).
        # x is passed un-detached, so add_init_v's layer-0 v capture is unaffected.
        # ve = layer-0 v capture, only consumed by add_init_v (shared-mode value residual).
        x0 = self._prepare_embedding_values(x)
        if self.config.detach_init_value:
            x0 = x0.detach()
        ve = None
        capture_v1 = self.config.add_init_v or self.config.v_from_v1
        for i, block in enumerate(self.transformer.h):
            # ve fed to block: per-late-layer VE table lookup under
            # v_from_value_emb / add_init_value_emb / add_init_value_emb_nanogpt,
            # otherwise the captured layer-0 v (for add_init_v / v_from_v1) or None.
            if (self.config.v_from_value_emb or self.config.add_init_value_emb or self.config.add_init_value_emb_nanogpt) and str(i) in self.value_embeds:
                ve_in = self.value_embeds[str(i)](idx).to(x.dtype)
            else:
                ve_in = ve
            x, v_init = block(x, x0, ve_in, cos_sin, kv_cache)
            if capture_v1 and i == 0:
                ve = v_init
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits
