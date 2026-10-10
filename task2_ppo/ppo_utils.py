"""Task 2 helpers shared by the continuation loop, the cached clipping study and the evaluation.

Tensor conventions: per-token tensors are [batch, response_steps]; response_mask is 1.0 on valid
response tokens (up to and including the first EOS) and 0.0 on padding.

Run-level rules recorded in every Task 2 JSON (see common.rollouts.PROMPT_RULE, DROPOUT_RULE):
  P. A prompt is used only if its chat-template encoding (add_generation_prompt=True) has at most
     max_prompt_length tokens, so batch_generate never truncates it. Excluded IDs are logged.
  D. Dropout is disabled (p=0 on every nn.Dropout) in the trainable policy and critic, so old and
     new log-probs come from the same deterministic function and the epoch-1 ratio is 1.
"""
from __future__ import annotations

import torch
from torch import nn

from common.metrics import masked_mean
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards

DROPOUT_RULE = "p set to 0 on every torch.nn.Dropout of the trainable policy and critic (LoRA dropout 0.05 inactive)"


# ---------------------------------------------------------------- rewards, advantages, positions

def effective_terminal_rewards(raw: torch.Tensor, terminated_with_eos, penalty: float) -> torch.Tensor:
    """raw minus missing_eos_penalty for responses that did not emit EOS."""
    no_eos = (~torch.as_tensor(terminated_with_eos, dtype=torch.bool, device=raw.device)).to(raw.dtype)
    return raw - float(penalty) * no_eos


def response_slice(per_position: torch.Tensor, prompt_width: int, steps: int) -> torch.Tensor:
    """Select the positions that predict response tokens 0..steps-1.

    Response token t sits at sequence index prompt_width + t and is predicted from the hidden state
    at index prompt_width + t - 1. This is the shift used by common.generation.response_token_logprobs,
    so V(s_t) taken here lines up with log pi(a_t | s_t).
    """
    return per_position[:, prompt_width - 1 : prompt_width - 1 + steps]


def build_targets(effective_reward, old_logp, ref_logp, values, mask, beta_kl, gamma, lam) -> dict:
    """KL-shaped token rewards (terminal reward at the last valid token), GAE advantages, returns = A + V.

    Values are zeroed on padding so the bootstrap after the last valid token is 0. No whitening.
    """
    values = values * mask
    rewards = shaped_rewards(effective_reward, old_logp, ref_logp, mask, beta_kl)
    advantages, returns = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
    return {"rewards": rewards, "advantages": advantages, "returns": returns, "values": values}


# ---------------------------------------------------------------- diagnostics

def affected_fraction(ratio, advantage, mask, eps) -> torch.Tensor:
    """Tokens where the clipped branch is the one min() keeps and it is flat in rho:
    (rho > 1+eps and A > 0) or (rho < 1-eps and A < 0)."""
    hit = ((ratio > 1.0 + eps) & (advantage > 0)) | ((ratio < 1.0 - eps) & (advantage < 0))
    return masked_mean(hit.to(ratio.dtype), mask)


def unclipped_surrogate(ratio, advantage, mask) -> torch.Tensor:
    return masked_mean(ratio * advantage, mask)


def clip_study(new_logp, old_logp, advantage, mask, eps_values) -> list[dict]:
    """Per eps: clip fraction (before clipping), affected fraction, mean clipped and unclipped surrogate."""
    out = []
    for eps in eps_values:
        loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=float(eps))
        out.append({
            "eps": float(eps),
            "clip_fraction": float(clip_frac),
            "affected_fraction": float(affected_fraction(ratio, advantage, mask, float(eps))),
            "clipped_surrogate": float(-loss.detach()),
            "unclipped_surrogate": float(unclipped_surrogate(ratio, advantage, mask)),
        })
    return out


def check_clip_identities(results: list[dict], tol: float = 1.0e-12) -> dict:
    """Clip fraction non-increasing in eps; clipped surrogate non-decreasing in eps; affected <= clip."""
    rs = sorted(results, key=lambda r: r["eps"])
    checks = {
        "clip_fraction_non_increasing_in_eps": all(b["clip_fraction"] <= a["clip_fraction"] + tol for a, b in zip(rs, rs[1:])),
        "clipped_surrogate_non_decreasing_in_eps": all(b["clipped_surrogate"] >= a["clipped_surrogate"] - tol for a, b in zip(rs, rs[1:])),
        "affected_not_above_clip_fraction": all(r["affected_fraction"] <= r["clip_fraction"] + tol for r in rs),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"clip-study identity checks failed: {failed}")
    return checks


def stability_statistic(new_logp, old_logp, mask) -> float:
    """Masked token mean of (rho - 1) - log rho, rho = pi_new / pi_old. Non-negative, 0 iff rho = 1.

    Computed as expm1(d) - d in float64 (d = log rho) to avoid cancellation when rho is close to 1.
    """
    d = (new_logp.double() - old_logp.double())
    return float(masked_mean(torch.expm1(d) - d, mask.double()))


def explained_variance(returns, values, mask) -> float:
    """1 - Var(returns - values) / Var(returns) over valid tokens (population variances). NaN if Var(returns) = 0."""
    sel = mask.bool()
    r = returns[sel].double()
    v = values[sel].double()
    var_r = r.var(unbiased=False)
    if r.numel() == 0 or float(var_r) == 0.0:
        return float("nan")
    return float(1.0 - (r - v).var(unbiased=False) / var_r)


# ---------------------------------------------------------------- model preparation

def disable_dropout(model) -> int:
    """Rule D. Set p=0 on every nn.Dropout; returns how many modules were changed."""
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            n += 1
    return n


def critic_head_to_fp32(value_model) -> dict:
    """Cast the trainable critic head to float32 and cast its input to float32.

    PEFT upcasts LoRA weights to float32 but keeps the modules_to_save copy of `score` in the base
    dtype. The pre-hook makes head(hidden) run in float32 whether it is called by
    common.models.token_values or by the classifier's own forward.
    """
    head = value_model.score
    trainable = [m for m in head.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad]
    if not trainable:
        raise RuntimeError("critic head has no trainable Linear layer")
    before = sorted({str(m.weight.dtype) for m in trainable})
    for m in trainable:
        m.float()
    handle = head.register_forward_pre_hook(lambda mod, args: tuple(a.float() if torch.is_tensor(a) else a for a in args))
    return {"head_dtype_before": before, "head_dtype_after": sorted({str(m.weight.dtype) for m in trainable}), "hook": handle}


