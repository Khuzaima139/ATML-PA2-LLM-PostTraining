"""Task 3 helpers shared by the continuation loop and the cached group-size study.

A prompt group is informative when the population standard deviation of its rewards is
greater than INFORMATIVE_TOL. The tolerance equals the default eps of
task3_grpo.grpo.group_relative_advantages, so a group flagged uninformative is one whose advantage
denominator is clamped to eps.
"""
from __future__ import annotations

import torch

INFORMATIVE_TOL = 1.0e-6
INFORMATIVE_RULE = f"informative iff population std of the group's rewards > {INFORMATIVE_TOL:g}"


def group_population_std(rewards: torch.Tensor) -> torch.Tensor:
    """Population std (divide by K), the same convention as group_relative_advantages."""
    return rewards.std(unbiased=False)


def is_informative(rewards: torch.Tensor) -> bool:
    return bool(group_population_std(rewards) > INFORMATIVE_TOL)


def zero_gradient_token_shares(lengths, truncated, informative: bool) -> dict:
    """Share of the update's generated tokens whose surrogate (advantage) gradient is zero, by cause.

    Causes are assigned in this order, so every token is counted once:
      masked_truncation    tokens of completions masked by mask_truncated_sequences (no EOS at the cap);
                           these get no gradient from the surrogate or the KL term.
      uninformative_group  tokens of the remaining completions when the group is uninformative
                           (advantages are 0; the KL term still has a gradient on them).
    """
    total = int(sum(lengths))
    masked = int(sum(n for n, t in zip(lengths, truncated) if t))
    unmasked = total - masked
    uninf = 0 if informative else unmasked
    share = (lambda x: x / total) if total else (lambda x: float("nan"))
    return {
        "generated_tokens": total,
        "masked_truncation_tokens": masked,
        "uninformative_group_tokens": uninf,
        "masked_truncation_share": share(masked),
        "uninformative_group_share": share(uninf),
        "total_share": share(masked + uninf),
    }
