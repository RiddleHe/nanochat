"""GPTBase with raw token embeddings entering the residual stream.

Only the initial embedding RMSNorm is omitted. Block/final norms, initialization,
optimizer, FLOP accounting, loss, and cached inference are inherited unchanged.
"""

from nanochat.model.gpt_base import GPTBase


class GPTBaseNoEmbedNorm(GPTBase):
    def _prepare_embeddings(self, embeddings):
        return embeddings
