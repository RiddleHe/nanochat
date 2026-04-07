"""
RL loss functions for policy-gradient training.

Each loss takes a uniform signature:
    loss_fn(logprobs, old_logprobs, rewards, **kwargs) -> scalar tensor

where `logprobs` and `old_logprobs` are per-sample masked-mean log-probs over
response tokens (shape [B]) and `rewards` is a scalar reward per sample
(shape [B]). Add new algorithms by registering them in ALGORITHMS below.
"""

import torch


def grpo_loss(logprobs, old_logprobs, rewards, clip=0.2, kl_coeff=0.0, **kwargs):
    """GRPO: group-relative policy optimization."""
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    ratio = (logprobs - old_logprobs).exp()
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    if kl_coeff > 0:
        kl = (old_logprobs - logprobs).mean()
        pg_loss = pg_loss + kl_coeff * kl
    return pg_loss


def dapo_loss(logprobs, old_logprobs, rewards, clip_low=0.8, clip_high=1.28, **kwargs):
    """DAPO: decoupled asymmetric policy optimization."""
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    ratio = (logprobs - old_logprobs).exp()
    clipped = torch.clamp(ratio, clip_low, clip_high)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def reinforce_loss(logprobs, old_logprobs, rewards, **kwargs):
    """Simple REINFORCE with mean-subtracted advantages."""
    advantages = rewards - rewards.mean()
    return -(logprobs * advantages).mean()


ALGORITHMS = {
    "grpo": grpo_loss,
    "dapo": dapo_loss,
    "reinforce": reinforce_loss,
}
