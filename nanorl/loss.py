"""
RL loss functions for policy-gradient training (per-token formulation).

Each loss takes a uniform signature:

    loss_fn(logprobs, old_logprobs, advantages, response_mask, **kwargs) -> scalar

where

    logprobs, old_logprobs : FloatTensor [B, T]   per-token log-probs (shifted)
    advantages             : FloatTensor [B]      per-sample advantages
    response_mask          : FloatTensor [B, T]   1 where position is a response
                                                  token, 0 for prompt/pad/eos-pad

The loss is formed per token (ratio, clip, advantage × logprob), then
aggregated. Two aggregation strategies are provided:

  - ``_masked_token_mean`` ("token-mean"): sum over all response tokens in
    the batch / total response-token count. Longer responses contribute
    proportionally more. Used by DAPO (its signature aggregation, argued
    for in the paper for long-CoT).
  - ``_masked_sequence_mean`` ("sequence-mean"): mean over response tokens
    *within each sample* first, then mean over samples. Every sample
    contributes equally regardless of length. Used by GRPO and REINFORCE
    — their advantages are per-sample scalars, so length-normalizing
    avoids implicitly weighting a sample by however long its response
    happened to be.

Why *per-token* rather than per-sequence log-probs:
    ratio = exp(logπ_new - logπ_old). If these logπ values are the
    mean-pooled log-probs of a whole sequence, then per-sequence
    ratio = geometric-mean of the individual token ratios, which is
    almost always ~1 — the PPO/DAPO clip bounds (e.g. 0.8/1.28) never
    bind. Per-token clipping is what makes clip-higher / clip-low
    actually do anything.

Add new algorithms by registering them in ALGORITHMS below.
"""

import torch


def compute_advantages(
    algorithm: str,
    rewards: torch.Tensor,
    num_samples_per_prompt: int = 1,
) -> torch.Tensor:
    """Per-sample advantages. grpo/dapo use *group-relative* normalization.

    For grpo/dapo, rewards are reshaped into ``[num_prompts, num_samples_per_prompt]``
    and normalized within each prompt's group. This matches the original
    GRPO/DAPO formulation where the baseline for an action is the *other
    samples from the same prompt*, not a global batch mean. Normalizing
    across the whole batch mixes prompt-level variance into each sample's
    advantage, which is strictly worse signal-to-noise.

    For reinforce, we fall back to plain mean-subtraction over the batch.

    Callers that have only a single sample per prompt (or don't know
    their group size) may pass ``num_samples_per_prompt=1`` and receive
    the batch-mean-subtracted version.
    """
    if algorithm in ("grpo", "dapo", "gspo"):
        if num_samples_per_prompt <= 1:
            return rewards - rewards.mean()
        n = rewards.numel()
        assert n % num_samples_per_prompt == 0, (
            f"rewards size {n} not divisible by num_samples_per_prompt "
            f"{num_samples_per_prompt}"
        )
        grouped = rewards.view(-1, num_samples_per_prompt)
        group_mean = grouped.mean(dim=-1, keepdim=True)
        group_std = grouped.std(dim=-1, keepdim=True)
        return ((grouped - group_mean) / (group_std + 1e-8)).view(-1)
    if algorithm == "reinforce":
        return rewards - rewards.mean()
    raise ValueError(f"unknown RL algorithm: {algorithm!r}")


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DAPO token-mean: sum over all response tokens / count of response tokens."""
    return (values * mask).sum() / mask.sum().clamp(min=1)


def _masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GRPO/REINFORCE sequence-mean: per-sample token-mean, then mean over samples."""
    seq_mean = (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)   # [B]
    return seq_mean.mean()


def _apply_entropy_top_frac(mask: torch.Tensor, entropy: torch.Tensor, top_frac: float) -> torch.Tensor:
    """Narrow `mask` to keep only tokens whose entropy is in the top `top_frac` of
    the masked positions in the batch. Implements the paper's "forking tokens"
    selective-gradient scheme: only top-k high-entropy response tokens contribute.

    If ``top_frac`` is None/<=0/>=1, returns mask unchanged.
    """
    if top_frac is None or top_frac <= 0.0 or top_frac >= 1.0:
        return mask
    mask_bool = mask.bool()
    ent_masked = entropy.float()[mask_bool]
    if ent_masked.numel() == 0:
        return mask
    thresh = torch.quantile(ent_masked, 1.0 - top_frac)
    keep = (entropy >= thresh).to(mask.dtype) * mask
    return keep


def grpo_loss(
    logprobs,
    old_logprobs,
    advantages,
    response_mask,
    entropy=None,
    entropy_top_frac=None,
    clip=0.2,
    kl_coeff=0.0,
    **kwargs,
):
    """Per-token PPO-style surrogate with symmetric clip, sequence-mean aggregated."""
    # Cast to fp32 for numerically stable exp / min.
    lp = logprobs.float()
    old_lp = old_logprobs.float()
    adv = advantages.float().unsqueeze(-1)                          # [B, 1]
    mask = response_mask.float()
    if entropy is not None:
        mask = _apply_entropy_top_frac(mask, entropy, entropy_top_frac)

    ratio = (lp - old_lp).exp()                                      # [B, T]
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    per_token = -torch.min(ratio * adv, clipped * adv)               # [B, T]
    loss = _masked_sequence_mean(per_token, mask)

    if kl_coeff > 0:
        # Simple per-token KL estimator: E[old_logp - new_logp] (low-variance
        # on-policy approximation; sign convention matches original GRPO code).
        kl_per_token = old_lp - lp                                   # [B, T]
        kl = _masked_sequence_mean(kl_per_token, mask)
        loss = loss + kl_coeff * kl
    return loss


def dapo_loss(
    logprobs,
    old_logprobs,
    advantages,
    response_mask,
    entropy=None,
    entropy_top_frac=None,
    clip_low=0.8,
    clip_high=1.28,
    **kwargs,
):
    """DAPO per-token clipped surrogate with asymmetric (clip-higher) bounds.

    When ``entropy_top_frac`` is given (e.g. 0.2) and ``entropy`` is provided,
    gradient is restricted to the top-fraction of highest-entropy response
    tokens per batch — this is the 80/20 paper's "forking tokens" selective
    training scheme.
    """
    lp = logprobs.float()
    old_lp = old_logprobs.float()
    adv = advantages.float().unsqueeze(-1)
    mask = response_mask.float()
    if entropy is not None:
        mask = _apply_entropy_top_frac(mask, entropy, entropy_top_frac)

    ratio = (lp - old_lp).exp()
    clipped = torch.clamp(ratio, clip_low, clip_high)
    per_token = -torch.min(ratio * adv, clipped * adv)
    return _masked_token_mean(per_token, mask)


def reinforce_loss(
    logprobs,
    old_logprobs,
    advantages,
    response_mask,
    entropy=None,
    entropy_top_frac=None,
    **kwargs,
):
    """Per-token REINFORCE with mean-subtracted advantage, sequence-mean aggregated."""
    lp = logprobs.float()
    adv = advantages.float().unsqueeze(-1)
    mask = response_mask.float()
    if entropy is not None:
        mask = _apply_entropy_top_frac(mask, entropy, entropy_top_frac)
    per_token = -(lp * adv)
    return _masked_sequence_mean(per_token, mask)


def gspo_loss(
    logprobs,
    old_logprobs,
    advantages,
    response_mask,
    clip=0.2,
    kl_coeff=0.0,
    **kwargs,
):
    """GSPO sequence-level clipped surrogate with length-normalized log-ratio."""
    lp = logprobs.float()
    old_lp = old_logprobs.float()
    adv = advantages.float().unsqueeze(-1)                          # [B, 1]
    mask = response_mask.float()

    seq_lengths = mask.sum(dim=-1).clamp(min=1)                     # [B]
    seq_log_ratio = ((lp - old_lp) * mask).sum(dim=-1) / seq_lengths  # [B]
    ratio = seq_log_ratio.exp().unsqueeze(-1)                       # [B, 1]
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)            # [B, 1]
    per_seq = -torch.min(ratio * adv, clipped * adv)                # [B, 1]
    loss = _masked_sequence_mean(per_seq, mask)                     # broadcasts vs [B, T] mask

    if kl_coeff > 0:
        kl_per_token = old_lp - lp                                  # [B, T]
        kl = _masked_sequence_mean(kl_per_token, mask)
        loss = loss + kl_coeff * kl
    return loss


def cispo_loss(
    logprobs,
    old_logprobs,
    advantages,
    response_mask,
    clip_high=0.2,
    **kwargs,
):
    """CISPO loss (MiniMax-M1). One-sided IS-weight clip, stop-gradient on weight,
    gradient flows through log π. Token-mean aggregation across the batch."""
    lp = logprobs.float()
    old_lp = old_logprobs.float()
    adv = advantages.float().unsqueeze(-1)                    # [B, 1]
    mask = response_mask.float()

    ratio = (lp - old_lp).exp()                               # [B, T]
    clipped = torch.clamp(ratio, max=1.0 + clip_high).detach()
    per_token = -clipped * adv * lp                           # [B, T]
    return _masked_token_mean(per_token, mask)


ALGORITHMS = {
    "grpo": grpo_loss,
    "dapo": dapo_loss,
    "reinforce": reinforce_loss,
    "gspo": gspo_loss,
    "cispo": cispo_loss,
}
