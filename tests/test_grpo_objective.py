"""Validate task3_grpo.grpo against independent references of the manual's GRPO equations.

Manual, Section 3 (GRPO), "Introduction and Terminology":

    A_k = (r_k - mu_r) / (sigma_r + eps),   mu_r = (1/K) sum_j r_j        (within one prompt group)
    L_GRPO = -E_k[ (1/T_k) sum_t min(rho_kt A_k, clip(rho_kt, 1 - eps, 1 + eps) A_k) ] + beta D_KL(pi_theta || pi_ref)

Manual, Section 3, "Dataset, Models, and Experimental Setup": the baseline uses K = 4 completions per
prompt "with max-length completions masked from the training loss". Step 3 compares canonical
sequence normalization with "the supplied Dr. GRPO-style normalization"; the release implements it as
loss_type="dr_grpo", which divides each completion's token sum by the constant max_completion_length
instead of its own length T_k.

Conventions used by the references (the manual does not fix them; they match the release):
  - sigma_r is the population standard deviation (divide by K) of the group's rewards.
  - E_k is the mean over all completions in the batch; a masked completion has no valid tokens, so its
    term is 0.
  - The KL term uses the release estimator per token, k3 = exp(l_ref - l_pol) - (l_ref - l_pol) - 1,
    averaged over all valid tokens of the batch. Its direction is checked against an exact
    KL(pi_theta || pi_ref) on a tiny distribution.

Layout: per-token tensors are [completions, response_steps]; token_mask is 1.0 on valid response
tokens and 0.0 on right padding. Padded positions hold large finite garbage to show they never leak.
The references are plain Python floats written directly from the equations above.
"""
from __future__ import annotations

import math

import pytest
import torch

from task3_grpo.grpo import grpo_policy_loss, group_relative_advantages, mask_truncated_sequences

PAD_GARBAGE = 37.0
STD_EPS = 1.0e-6  # default eps of group_relative_advantages


def t64(x):
    return torch.tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------- references from the manual


def ref_group_advantages(rewards, group_ids, eps=STD_EPS):
    """A_k = (r_k - mu) / (sigma + eps), with mu and population sigma taken inside each group."""
    out = [None] * len(rewards)
    for g in set(group_ids):
        idx = [i for i, gid in enumerate(group_ids) if gid == g]
        r = [rewards[i] for i in idx]
        mu = sum(r) / len(r)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in r) / len(r))
        for i in idx:
            out[i] = (rewards[i] - mu) / (sigma + eps)
    return out


def ref_clip(x, lo, hi):
    return min(max(x, lo), hi)


def ref_surrogate_token(rho, adv, eps):
    return min(rho * adv, ref_clip(rho, 1.0 - eps, 1.0 + eps) * adv)


def ref_k3(l_pol, l_ref):
    """Per-token release KL estimator; its expectation under pi_theta is KL(pi_theta || pi_ref)."""
    d = l_ref - l_pol
    return math.exp(d) - d - 1.0


def ref_policy_term(new_logp, old_logp, adv, mask, eps, normalizer=None):
    """-(1/N) sum_k (1/T_k) sum_t surrogate. normalizer=None uses T_k; a number is the Dr. GRPO constant."""
    per_seq = []
    for nrow, orow, a, mrow in zip(new_logp, old_logp, adv, mask):
        terms = [ref_surrogate_token(math.exp(n - o), a, eps) for n, o, m in zip(nrow, orow, mrow) if m]
        if normalizer is None:
            per_seq.append(sum(terms) / len(terms) if terms else 0.0)
        else:
            per_seq.append(sum(terms) / normalizer)
    return -sum(per_seq) / len(per_seq)


def ref_kl_term(new_logp, ref_logp, mask):
    vals = [ref_k3(n, r) for nrow, rrow, mrow in zip(new_logp, ref_logp, mask) for n, r, m in zip(nrow, rrow, mrow) if m]
    return sum(vals) / len(vals) if vals else 0.0


def ref_loss(new_logp, old_logp, adv, mask, ref_logp, eps, beta, normalizer=None):
    return ref_policy_term(new_logp, old_logp, adv, mask, eps, normalizer) + beta * ref_kl_term(new_logp, ref_logp, mask)


# ---------------------------------------------------------------- B1: group-relative advantages


def test_single_group_matches_reference():
    rewards = [0.5, 2.0, -1.0, 1.5]
    got = group_relative_advantages(t64(rewards), torch.tensor([0, 0, 0, 0]))
    assert got.tolist() == pytest.approx(ref_group_advantages(rewards, [0, 0, 0, 0]), rel=1e-5, abs=1e-9)


def test_two_groups_use_their_own_mean_and_std():
    # Group 0 is high-reward, group 1 low-reward; each must be normalized inside itself.
    rewards = [3.0, 4.0, 5.0, 6.0, -2.0, -1.5, -1.0, 0.5]
    gids = [0, 0, 0, 0, 1, 1, 1, 1]
    got = group_relative_advantages(t64(rewards), torch.tensor(gids))
    assert got.tolist() == pytest.approx(ref_group_advantages(rewards, gids), rel=1e-5, abs=1e-9)
    # Each group's advantages are centred on 0.
    assert float(got[:4].sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(got[4:].sum()) == pytest.approx(0.0, abs=1e-9)


def test_interleaved_and_noncontiguous_group_ids():
    rewards = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
    gids = [7, 3, 7, 3, 7, 3]
    got = group_relative_advantages(t64(rewards), torch.tensor(gids))
    assert got.tolist() == pytest.approx(ref_group_advantages(rewards, gids), rel=1e-5, abs=1e-9)
    # Same ordering of rewards inside each group, so both groups get identical advantages.
    assert got[0::2].tolist() == pytest.approx(got[1::2].tolist(), rel=1e-9)


def test_zero_std_group_gets_zero_advantage_next_to_informative_group():
    rewards = [1.25, 1.25, 1.25, 1.25, 0.0, 1.0, 0.5, 2.5]
    gids = [0, 0, 0, 0, 1, 1, 1, 1]
    got = group_relative_advantages(t64(rewards), torch.tensor(gids))
    assert torch.isfinite(got).all()
    assert got[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert got.tolist() == pytest.approx(ref_group_advantages(rewards, gids), rel=1e-5, abs=1e-9)


def test_zero_std_float32_group_is_finite_and_zero():
    got = group_relative_advantages(torch.full((4,), 0.1, dtype=torch.float32), torch.zeros(4, dtype=torch.long))
    assert torch.isfinite(got).all()
    assert got.abs().max().item() == 0.0


def test_shifting_one_group_does_not_change_any_advantage():
    rewards = [0.0, 1.0, 3.0, 2.0, 0.5, 0.0, 1.5, 4.0]
    gids = [0, 0, 0, 0, 1, 1, 1, 1]
    shifted = [r + (100.0 if g == 1 else 0.0) for r, g in zip(rewards, gids)]
    a = group_relative_advantages(t64(rewards), torch.tensor(gids))
    b = group_relative_advantages(t64(shifted), torch.tensor(gids))
    assert b.tolist() == pytest.approx(a.tolist(), rel=1e-6, abs=1e-9)


def test_population_std_convention():
    # K = 2: population sigma = |r1 - r2| / 2, so the advantages are -1 and +1 (up to eps).
    got = group_relative_advantages(t64([0.0, 2.0]), torch.tensor([0, 0]))
    assert got.tolist() == pytest.approx([-1.0, 1.0], rel=1e-5)


# ---------------------------------------------------------------- B1: clipped surrogate, per token


@pytest.mark.parametrize(
    "rho,adv,expected",
    [
        (1.5, 1.0, 1.2),    # A > 0, rho above 1 + eps: clipped value is smaller
        (0.5, 1.0, 0.5),    # A > 0, rho below 1 - eps: unclipped value is smaller
        (1.5, -1.0, -1.5),  # A < 0, rho above 1 + eps: unclipped value is smaller
        (0.5, -1.0, -0.8),  # A < 0, rho below 1 - eps: clipped value is smaller
        (1.1, -2.0, -2.2),  # A < 0, inside the interval
        (0.9, 2.0, 1.8),    # A > 0, inside the interval
    ],
)
def test_single_token_surrogate(rho, adv, expected):
    eps = 0.2
    assert ref_surrogate_token(rho, adv, eps) == pytest.approx(expected)
    new = t64([[math.log(rho)]])
    old = t64([[0.0]])
    loss, stats = grpo_policy_loss(new, old, t64([adv]), t64([[1.0]]), new.clone(), eps=eps, beta=0.0)
    assert float(loss) == pytest.approx(-expected, rel=1e-9)
    assert float(stats["clip_fraction"]) == (1.0 if abs(rho - 1.0) > eps else 0.0)


# ---------------------------------------------------------------- B1: 1/T_k normalization, padding, masking


def _ragged_batch():
    """Three completions with T = 3, 1, 5 (right padding filled with garbage)."""
    lengths = [3, 1, 5]
    width = 5
    new = [[0.10, -0.30, 0.25, 0, 0], [0.40, 0, 0, 0, 0], [-0.05, 0.15, -0.40, 0.30, 0.02]]
    old = [[0.00, -0.10, 0.05, 0, 0], [0.00, 0, 0, 0, 0], [0.10, 0.00, -0.10, 0.00, 0.00]]
    ref = [[-0.20, 0.10, 0.00, 0, 0], [0.10, 0, 0, 0, 0], [0.00, 0.05, -0.20, 0.10, -0.10]]
    mask = [[1.0 if t < n else 0.0 for t in range(width)] for n in lengths]
    for rows in (new, old, ref):
        for r, n in zip(rows, lengths):
            for t in range(n, width):
                r[t] = PAD_GARBAGE
    adv = [1.3, -0.7, 0.4]
    return new, old, ref, mask, adv


@pytest.mark.parametrize("eps", [0.05, 0.20, 0.50])
@pytest.mark.parametrize("beta", [0.0, 0.10])
def test_grpo_loss_matches_reference_on_ragged_batch(eps, beta):
    new, old, ref, mask, adv = _ragged_batch()
    loss, stats = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=eps, beta=beta)
    assert float(loss) == pytest.approx(ref_loss(new, old, adv, mask, ref, eps, beta), rel=1e-9, abs=1e-12)
    assert float(stats["policy_term"]) == pytest.approx(ref_policy_term(new, old, adv, mask, eps), rel=1e-9, abs=1e-12)


def test_sequence_mean_differs_from_token_pooled_mean():
    # Guard: the 1/T_k per-completion mean is not the same as one mean over all batch tokens.
    new, old, ref, mask, adv = _ragged_batch()
    eps = 0.2
    pooled = [ref_surrogate_token(math.exp(n - o), a, eps) for nr, orow, a, mr in zip(new, old, adv, mask) for n, o, m in zip(nr, orow, mr) if m]
    token_pooled = -sum(pooled) / len(pooled)
    loss, _ = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=eps, beta=0.0)
    assert abs(float(loss) - token_pooled) > 1e-3
    assert float(loss) == pytest.approx(ref_policy_term(new, old, adv, mask, eps), rel=1e-9)


def test_gradient_at_ratio_one_is_minus_adv_over_T_over_N():
    # rho = 1 everywhere: d(-surrogate)/d new_logp_kt = -A_k / (T_k N) on valid tokens, 0 on padding.
    _, _, ref, mask, adv = _ragged_batch()
    old = t64([[0.0] * 5] * 3)
    new = old.clone().requires_grad_(True)
    loss, _ = grpo_policy_loss(new, old, t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.0)
    loss.backward()
    lengths = [3, 1, 5]
    expected = [[(-a / (n * 3)) if t < n else 0.0 for t in range(5)] for a, n in zip(adv, lengths)]
    assert new.grad.tolist() == [pytest.approx(row, rel=1e-9, abs=1e-12) for row in expected]


def test_padding_garbage_never_changes_loss():
    new, old, ref, mask, adv = _ragged_batch()
    a, _ = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.1)
    for rows in (new, old, ref):
        for r, m in zip(rows, mask):
            for t, mt in enumerate(m):
                if not mt:
                    r[t] = -PAD_GARBAGE * 3
    b, _ = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.1)
    assert float(a) == pytest.approx(float(b), rel=1e-12)


def test_mask_truncated_sequences_zeroes_only_flagged_rows():
    mask = t64([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    out = mask_truncated_sequences(mask, [False, True, False])
    assert out.tolist() == [[1, 1, 0], [0, 0, 0], [1, 0, 0]]
    out2 = mask_truncated_sequences(mask, torch.tensor([True, False, True]))
    assert out2.tolist() == [[0, 0, 0], [1, 1, 1], [0, 0, 0]]


@pytest.mark.parametrize("loss_type", ["grpo", "dr_grpo"])
def test_masked_completion_contributes_nothing(loss_type):
    new, old, ref, mask, adv = _ragged_batch()
    masked = mask_truncated_sequences(t64(mask), [False, False, True])  # completion 2 hit the cap
    kw = {"max_completion_length": 5} if loss_type == "dr_grpo" else {}
    new_t = t64(new).requires_grad_(True)
    loss, _ = grpo_policy_loss(new_t, t64(old), t64(adv), masked, t64(ref), eps=0.2, beta=0.1, loss_type=loss_type, **kw)
    loss.backward()
    assert new_t.grad[2].abs().max().item() == 0.0
    assert new_t.grad[:2].abs().sum().item() > 0.0
    # Any values in the masked completion leave the loss unchanged.
    new2, old2, ref2 = [list(r) for r in new], [list(r) for r in old], [list(r) for r in ref]
    new2[2], old2[2], ref2[2] = [5.0] * 5, [-5.0] * 5, [9.0] * 5
    loss2, _ = grpo_policy_loss(t64(new2), t64(old2), t64(adv), masked, t64(ref2), eps=0.2, beta=0.1, loss_type=loss_type, **kw)
    assert float(loss2) == pytest.approx(float(loss.detach()), rel=1e-12)
    # And it matches the reference with the completion's mask row set to 0.
    m_ref = [list(mask[0]), list(mask[1]), [0.0] * 5]
    norm = 5 if loss_type == "dr_grpo" else None
    assert float(loss.detach()) == pytest.approx(ref_loss(new, old, adv, m_ref, ref, 0.2, 0.1, norm), rel=1e-9)


def test_all_completions_masked_gives_zero_loss_and_no_nan():
    new, old, ref, mask, adv = _ragged_batch()
    masked = mask_truncated_sequences(t64(mask), [True, True, True])
    loss, stats = grpo_policy_loss(t64(new), t64(old), t64(adv), masked, t64(ref), eps=0.2, beta=0.1)
    assert float(loss) == 0.0
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in stats.values())


# ---------------------------------------------------------------- B1: KL term, sign and direction


def test_kl_term_is_zero_when_policy_equals_reference():
    new, old, _, mask, adv = _ragged_batch()
    a, s = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(new), eps=0.2, beta=0.5)
    b, _ = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(new), eps=0.2, beta=0.0)
    assert float(s["sampled_kl"]) == pytest.approx(0.0, abs=1e-15)
    assert float(a) == pytest.approx(float(b), rel=1e-12)


def test_kl_enters_loss_with_plus_beta():
    new, old, ref, mask, adv = _ragged_batch()
    base, _ = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.0)
    with_kl, stats = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.3)
    kl = ref_kl_term(new, ref, mask)
    assert kl > 0.0
    assert float(stats["sampled_kl"]) == pytest.approx(kl, rel=1e-9)
    assert float(with_kl) - float(base) == pytest.approx(0.3 * kl, rel=1e-9)


def test_kl_gradient_pulls_policy_toward_reference():
    # beta-only loss: where l_pol > l_ref the gradient is positive, so a descent step lowers l_pol.
    old = t64([[0.0, 0.0]])
    ref = t64([[-1.0, 0.5]])
    new = t64([[0.0, 0.0]]).requires_grad_(True)
    loss, _ = grpo_policy_loss(new, old, t64([0.0]), t64([[1.0, 1.0]]), ref, eps=0.2, beta=1.0)
    loss.backward()
    assert new.grad[0, 0].item() > 0.0  # l_pol above l_ref: pushed down
    assert new.grad[0, 1].item() < 0.0  # l_pol below l_ref: pushed up


def test_kl_estimator_direction_is_policy_to_reference():
    # pi_theta = [1/2, 1/4, 1/4], pi_ref = [1/8, 1/2, 3/8]. Samples drawn from pi_theta in exact
    # proportion (tokens a, a, b, c), so the token mean of k3 equals E_pi[k3] = KL(pi_theta || pi_ref).
    pi = [0.5, 0.25, 0.25]
    pr = [0.125, 0.5, 0.375]
    kl_forward = sum(p * math.log(p / q) for p, q in zip(pi, pr))   # KL(pi_theta || pi_ref)
    kl_reverse = sum(q * math.log(q / p) for p, q in zip(pi, pr))   # KL(pi_ref || pi_theta)
    assert abs(kl_forward - kl_reverse) > 0.05
    toks = [0, 0, 1, 2]
    l_pol = t64([[math.log(pi[i]) for i in toks]])
    l_ref = t64([[math.log(pr[i]) for i in toks]])
    _, stats = grpo_policy_loss(l_pol, l_pol.clone(), t64([0.0]), t64([[1.0] * 4]), l_ref, eps=0.2, beta=1.0)
    assert float(stats["sampled_kl"]) == pytest.approx(kl_forward, rel=1e-9)


# ---------------------------------------------------------------- B1: Dr. GRPO normalization (release definition)


@pytest.mark.parametrize("beta", [0.0, 0.10])
def test_dr_grpo_matches_reference_with_constant_normalizer(beta):
    new, old, ref, mask, adv = _ragged_batch()
    L = 512
    loss, stats = grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=0.2, beta=beta, loss_type="dr_grpo", max_completion_length=L)
    assert float(loss) == pytest.approx(ref_loss(new, old, adv, mask, ref, 0.2, beta, normalizer=L), rel=1e-9, abs=1e-12)


def test_dr_grpo_per_token_gradient_does_not_depend_on_length():
    _, _, ref, mask, _ = _ragged_batch()
    adv = [1.0, 1.0, 1.0]
    old = t64([[0.0] * 5] * 3)
    lengths = [3, 1, 5]
    L = 8
    for loss_type, kw in (("grpo", {}), ("dr_grpo", {"max_completion_length": L})):
        new = old.clone().requires_grad_(True)
        loss, _ = grpo_policy_loss(new, old, t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.0, loss_type=loss_type, **kw)
        loss.backward()
        first = [new.grad[k, 0].item() for k in range(3)]
        if loss_type == "grpo":
            assert first == pytest.approx([-1.0 / (n * 3) for n in lengths], rel=1e-9)
        else:
            assert first == pytest.approx([-1.0 / (L * 3)] * 3, rel=1e-9)


def test_dr_grpo_requires_max_completion_length():
    new, old, ref, mask, adv = _ragged_batch()
    with pytest.raises(ValueError):
        grpo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), t64(ref), eps=0.2, beta=0.1, loss_type="dr_grpo")


# ---------------------------------------------------------------- end to end: rewards -> advantages -> loss


def test_end_to_end_two_prompt_groups():
    # Two prompts, K = 2 each; the loss must use within-group advantages.
    rewards = [5.0, 7.0, -3.0, -2.0]
    gids = [0, 0, 1, 1]
    adv = group_relative_advantages(t64(rewards), torch.tensor(gids))
    ref_adv = ref_group_advantages(rewards, gids)
    new = [[0.1, 0.0], [0.0, -0.2], [0.3, 0.0], [-0.1, 0.05]]
    old = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    mask = [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    loss, _ = grpo_policy_loss(t64(new), t64(old), adv, t64(mask), t64(old), eps=0.2, beta=0.0)
    assert float(loss) == pytest.approx(ref_policy_term(new, old, ref_adv, mask, 0.2), rel=1e-5)
