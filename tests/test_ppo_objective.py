"""Validate task2_ppo.ppo against independent references of the manual's PPO equations.

Manual, Section 2 (PPO), "Introduction and Terminology":

    rho_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
    L_clip(theta) = E_t[ min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t) ]
    delta_t = r_t + gamma V(s_{t+1}) - V(s_t),   A_t^GAE = sum_{k>=0} (gamma lambda)^k delta_{t+k}
    r_t = r_task 1[t = T] - beta_KL (log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t))

Manual, "Common Metric Definitions", PPO clip fraction:

    fraction of valid response tokens for which rho_t lies outside [1 - eps, 1 + eps] before clipping.

Layout used by the code: tensors are [batch, response_steps]. common.generation.batch_generate
builds response_mask with ones from the first response token up to and including the first EOS,
and zeros after it, so padding is on the RIGHT of each response row. The tests use that layout and
put large finite garbage on padded positions to show that padding never leaks into a result.

The references below are written in plain Python floats, directly from the equations above.
"""
from __future__ import annotations

import math

import pytest
import torch

from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards, value_mse_loss

EPS_VALUES = [0.05, 0.20, 0.50]  # configs/ppo.yaml clip_values
PAD_GARBAGE = 37.0


def t64(x):
    return torch.tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------- references from the manual


def ref_clip(x, lo, hi):
    return min(max(x, lo), hi)


def ref_surrogate_token(rho, adv, eps):
    """min(rho A, clip(rho, 1-eps, 1+eps) A) for one token."""
    return min(rho * adv, ref_clip(rho, 1.0 - eps, 1.0 + eps) * adv)


def ref_policy_loss(new_logp, old_logp, adv, mask, eps):
    """Negative of the clipped surrogate, averaged over valid tokens of the whole batch."""
    total, count = 0.0, 0
    for nrow, orow, arow, mrow in zip(new_logp, old_logp, adv, mask):
        for n, o, a, m in zip(nrow, orow, arow, mrow):
            if m:
                total += ref_surrogate_token(math.exp(n - o), a, eps)
                count += 1
    return -total / count


def ref_clip_fraction(new_logp, old_logp, mask, eps):
    out, count = 0, 0
    for nrow, orow, mrow in zip(new_logp, old_logp, mask):
        for n, o, m in zip(nrow, orow, mrow):
            if m:
                rho = math.exp(n - o)
                out += int(rho < 1.0 - eps or rho > 1.0 + eps)
                count += 1
    return out / count


def ref_gae_row(rewards, values, n_valid, gamma, lam):
    """A_t = sum_k (gamma lam)^k delta_{t+k} over the first n_valid tokens; V after the last valid = 0."""
    v = list(values[:n_valid]) + [0.0]
    deltas = [rewards[t] + gamma * v[t + 1] - v[t] for t in range(n_valid)]
    adv = []
    for t in range(n_valid):
        adv.append(sum((gamma * lam) ** k * deltas[t + k] for k in range(n_valid - t)))
    return adv


# ---------------------------------------------------------------- importance ratio


def test_ratio_is_exp_of_logp_difference():
    new = [[-1.0, -2.0, -0.5]]
    old = [[-1.2, -1.5, -0.5]]
    adv = [[1.0, -1.0, 0.3]]
    mask = [[1.0, 1.0, 1.0]]
    _, ratio, _ = ppo_policy_loss(t64(new), t64(old), t64(adv), t64(mask), eps=0.2)
    expected = [[math.exp(n - o) for n, o in zip(new[0], old[0])]]
    # exp(0.2) = 1.221403, exp(-0.5) = 0.606531, exp(0) = 1
    assert torch.allclose(ratio, t64(expected), atol=1e-12)
    assert ratio[0, 0].item() == pytest.approx(1.2214027581601699, abs=1e-12)
    assert not ratio.requires_grad


# ---------------------------------------------------------------- clipped surrogate (min)

# (rho, A, expected surrogate for eps = 0.2, which branch binds)
#   A>0, rho=1.5 outside high: min(1.5, 1.2) = 1.2   clipped term binds
#   A>0, rho=0.5 outside low : min(0.5, 0.8) = 0.5   unclipped term
#   A<0, rho=1.5 outside high: min(-1.5, -1.2) = -1.5 unclipped term
#   A<0, rho=0.5 outside low : min(-0.5, -0.8) = -0.8 clipped term binds
#   A>0, rho=1.1 inside      : 1.1
#   A<0, rho=0.9 inside      : -0.9
CASES = [
    (1.5, 1.0, 1.2, "clipped"),
    (0.5, 1.0, 0.5, "unclipped"),
    (1.5, -1.0, -1.5, "unclipped"),
    (0.5, -1.0, -0.8, "clipped"),
    (1.1, 1.0, 1.1, "unclipped"),
    (0.9, -1.0, -0.9, "unclipped"),
]


@pytest.mark.parametrize("rho,adv,expected,branch", CASES)
def test_single_token_surrogate_uses_min(rho, adv, expected, branch):
    old = torch.zeros(1, 1, dtype=torch.float64)
    new = t64([[math.log(rho)]])
    loss, _, _ = ppo_policy_loss(new, old, t64([[adv]]), t64([[1.0]]), eps=0.2)
    assert ref_surrogate_token(rho, adv, 0.2) == pytest.approx(expected, abs=1e-12)
    assert loss.item() == pytest.approx(-expected, abs=1e-12)


def test_all_cases_in_one_batch_hand_computed():
    # Surrogates from CASES: 1.2 + 0.5 - 1.5 - 0.8 + 1.1 - 0.9 = -0.4 ; mean = -0.4 / 6 = -0.0666667
    # Loss = -mean = +0.0666667
    rhos = [c[0] for c in CASES]
    advs = [c[1] for c in CASES]
    new = t64([[math.log(r) for r in rhos]])
    old = torch.zeros_like(new)
    loss, _, _ = ppo_policy_loss(new, old, t64([advs]), torch.ones_like(new), eps=0.2)
    assert loss.item() == pytest.approx(0.4 / 6.0, abs=1e-12)


@pytest.mark.parametrize("eps", EPS_VALUES)
def test_matches_reference_on_random_batch(eps):
    g = torch.Generator().manual_seed(6304)
    new = torch.randn(4, 9, generator=g, dtype=torch.float64) * 0.5 - 2.0
    old = new + torch.randn(4, 9, generator=g, dtype=torch.float64) * 0.4
    adv = torch.randn(4, 9, generator=g, dtype=torch.float64)
    lengths = [9, 6, 3, 1]
    mask = torch.zeros(4, 9, dtype=torch.float64)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1.0
    loss, _, clip_frac = ppo_policy_loss(new, old, adv, mask, eps=eps)
    ref = ref_policy_loss(new.tolist(), old.tolist(), adv.tolist(), mask.tolist(), eps)
    assert loss.item() == pytest.approx(ref, abs=1e-12)
    # The code casts the outside-interval indicator with .float(), so clip_fraction is float32.
    assert clip_frac.item() == pytest.approx(
        ref_clip_fraction(new.tolist(), old.tolist(), mask.tolist(), eps), abs=1e-6
    )


# ---------------------------------------------------------------- gradients


def test_gradient_zero_where_clip_binds_nonzero_where_unclipped():
    # d loss / d new_logp_t = -(1/N) * d surrogate_t / d new_logp_t.
    # Unclipped term selected: surrogate = rho A, d rho / d logp = rho, so grad = -rho A / N.
    # Clipped term selected (outside the interval): surrogate is constant, grad = 0.
    # N = 6 tokens. Expected grads, in CASES order:
    #   rho=1.5,A=+1 clipped   -> 0
    #   rho=0.5,A=+1 unclipped -> -0.5/6 = -0.0833333
    #   rho=1.5,A=-1 unclipped -> +1.5/6 = +0.25
    #   rho=0.5,A=-1 clipped   -> 0
    #   rho=1.1,A=+1 inside    -> -1.1/6 = -0.1833333
    #   rho=0.9,A=-1 inside    -> +0.9/6 = +0.15
    rhos = [c[0] for c in CASES]
    advs = [c[1] for c in CASES]
    new = t64([[math.log(r) for r in rhos]]).requires_grad_(True)
    old = torch.zeros(1, 6, dtype=torch.float64)
    loss, _, _ = ppo_policy_loss(new, old, t64([advs]), torch.ones(1, 6, dtype=torch.float64), eps=0.2)
    loss.backward()
    expected = []
    for rho, adv, _, branch in CASES:
        expected.append(0.0 if branch == "clipped" else -rho * adv / 6.0)
    assert torch.allclose(new.grad, t64([expected]), atol=1e-12)
    for (_, _, _, branch), gval in zip(CASES, new.grad[0].tolist()):
        if branch == "clipped":
            assert gval == 0.0
        else:
            assert gval != 0.0


# ---------------------------------------------------------------- loss sign


def test_minimizing_loss_maximizes_surrogate():
    # One SGD step on the loss must increase the manual's surrogate L_clip (computed by the
    # reference, not by the code). Tokens start at rho = 1 (inside the interval), so every token
    # is in the unclipped branch and has a non-zero gradient.
    old = t64([[-1.0, -2.0, -0.7, -1.5]])
    adv = t64([[1.0, -0.5, 2.0, -1.0]])
    mask = torch.ones_like(old)
    new = old.clone().requires_grad_(True)
    eps = 0.2

    def surrogate(n):
        return -ref_policy_loss(n.tolist(), old.tolist(), adv.tolist(), mask.tolist(), eps)

    before = surrogate(new.detach())
    loss, _, _ = ppo_policy_loss(new, old, adv, mask, eps=eps)
    assert loss.item() == pytest.approx(-before, abs=1e-12)
    loss.backward()
    with torch.no_grad():
        stepped = new - 0.05 * new.grad
    after = surrogate(stepped)
    assert after > before
    # Positive-advantage tokens became more likely, negative-advantage tokens less likely.
    diff = (stepped - old)[0].tolist()
    for d, a in zip(diff, adv[0].tolist()):
        assert (d > 0) == (a > 0)


# ---------------------------------------------------------------- masking


def test_padding_is_ignored_and_mean_is_over_valid_tokens():
    # Row 0: 3 valid tokens. Row 1: 1 valid token then 2 right-padded positions with garbage.
    # eps = 0.2. Valid tokens:
    #   (rho=1.0, A=+2)  -> 2.0
    #   (rho=1.5, A=+1)  -> min(1.5, 1.2) = 1.2
    #   (rho=0.5, A=-1)  -> min(-0.5, -0.8) = -0.8
    #   (rho=1.1, A=+1)  -> 1.1
    # Sum = 3.5 over N = 4 valid tokens -> mean 0.875 -> loss -0.875.
    # A mean over all 6 slots would give -3.5/6 = -0.583333 instead.
    rhos = [[1.0, 1.5, 0.5], [1.1, 9.0, 0.01]]
    advs = [[2.0, 1.0, -1.0], [1.0, PAD_GARBAGE, -PAD_GARBAGE]]
    mask = [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    new = t64([[math.log(r) for r in row] for row in rhos])
    old = torch.zeros_like(new)
    loss, _, clip_frac = ppo_policy_loss(new, old, t64(advs), t64(mask), eps=0.2)
    assert loss.item() == pytest.approx(-0.875, abs=1e-12)
    # Valid ratios outside [0.8, 1.2]: 1.5 and 0.5 -> 2 of 4. Padded ratios 9.0 and 0.01 ignored.
    assert clip_frac.item() == pytest.approx(0.5, abs=1e-12)


def test_padded_positions_receive_no_gradient():
    new = t64([[0.0, 0.1, 0.3, -0.4]]).requires_grad_(True)
    old = torch.zeros(1, 4, dtype=torch.float64)
    adv = t64([[1.0, -1.0, PAD_GARBAGE, PAD_GARBAGE]])
    mask = t64([[1.0, 1.0, 0.0, 0.0]])
    loss, _, _ = ppo_policy_loss(new, old, adv, mask, eps=0.2)
    loss.backward()
    assert new.grad[0, 2].item() == 0.0
    assert new.grad[0, 3].item() == 0.0


# ---------------------------------------------------------------- clip fraction


@pytest.mark.parametrize("eps", EPS_VALUES)
def test_clip_fraction_counts_ratio_outside_interval_before_clipping(eps):
    # Ratios chosen relative to eps: two clearly inside, one below 1-eps, one above 1+eps,
    # plus one padded position far outside. Advantage signs are mixed on purpose: the metric
    # depends on rho only, not on which branch of min() is selected.
    rhos = [[1.0, 1.0 + eps / 2, 1.0 - 1.5 * eps, 1.0 + 1.5 * eps, 50.0]]
    advs = [[1.0, -1.0, 1.0, 1.0, 1.0]]
    mask = [[1.0, 1.0, 1.0, 1.0, 0.0]]
    new = t64([[math.log(r) for r in rhos[0]]])
    old = torch.zeros_like(new)
    _, _, clip_frac = ppo_policy_loss(new, old, t64(advs), t64(mask), eps=eps)
    # 2 of 4 valid tokens are outside [1 - eps, 1 + eps].
    assert clip_frac.item() == pytest.approx(0.5, abs=1e-12)
    assert not clip_frac.requires_grad


# ---------------------------------------------------------------- GAE


def test_gae_hand_computed_with_padded_row():
    # gamma = 0.9, lambda = 0.8, gamma*lambda = 0.72.
    # Row 0 (3 valid): r = [0.1, -0.2, 1.0], V = [0.5, 0.4, 0.3], V after last valid = 0.
    #   delta_2 = 1.0 + 0.9*0   - 0.3 =  0.7
    #   delta_1 = -0.2 + 0.9*0.3 - 0.4 = -0.33
    #   delta_0 = 0.1 + 0.9*0.4 - 0.5 = -0.04
    #   A_2 = 0.7
    #   A_1 = -0.33 + 0.72*0.7   = 0.174
    #   A_0 = -0.04 + 0.72*0.174 = 0.08528
    #   returns = A + V = [0.58528, 0.574, 1.0]
    # Row 1 (2 valid, 1 right pad): r = [0.2, 0.5, 0], V = [0.6, 0.1, garbage 37.0].
    #   delta_1 = 0.5 + 0.9*0   - 0.1 =  0.4     (V after last valid = 0, not the padded 37.0)
    #   delta_0 = 0.2 + 0.9*0.1 - 0.6 = -0.31
    #   A_1 = 0.4 ; A_0 = -0.31 + 0.72*0.4 = -0.022 ; A_pad = 0
    #   returns on valid tokens = [0.578, 0.5]
    rewards = t64([[0.1, -0.2, 1.0], [0.2, 0.5, 0.0]])
    values = t64([[0.5, 0.4, 0.3], [0.6, 0.1, PAD_GARBAGE]])
    mask = t64([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=0.9, lam=0.8)
    assert torch.allclose(adv[0], t64([0.08528, 0.174, 0.7]), atol=1e-12)
    assert torch.allclose(adv[1], t64([-0.022, 0.4, 0.0]), atol=1e-12)
    assert torch.allclose(ret[0], t64([0.58528, 0.574, 1.0]), atol=1e-12)
    assert torch.allclose(ret[1, :2], t64([0.578, 0.5]), atol=1e-12)
    # Independent loop reference agrees.
    for b, n in enumerate([3, 2]):
        ref = ref_gae_row(rewards[b].tolist(), values[b].tolist(), n, 0.9, 0.8)
        assert adv[b, :n].tolist() == pytest.approx(ref, abs=1e-12)


def test_gae_four_tokens_gamma_lambda_one_is_monte_carlo():
    # gamma = lambda = 1: A_t = sum_{k>=t} r_k - V_t (telescoping), returns = reward-to-go.
    # r = [-0.1, -0.1, -0.1, 2.0], V = [1.0, 0.5, 0.2, -0.3]
    # reward-to-go = [1.7, 1.8, 1.9, 2.0] ; A = [0.7, 1.3, 1.7, 2.3]
    rewards = t64([[-0.1, -0.1, -0.1, 2.0]])
    values = t64([[1.0, 0.5, 0.2, -0.3]])
    mask = torch.ones_like(rewards)
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, t64([[0.7, 1.3, 1.7, 2.3]]), atol=1e-12)
    assert torch.allclose(ret, t64([[1.7, 1.8, 1.9, 2.0]]), atol=1e-12)


@pytest.mark.parametrize("gamma,lam", [(1.0, 0.95), (0.9, 0.8), (0.99, 0.0)])
def test_gae_matches_sum_formula_on_random_right_padded_batch(gamma, lam):
    g = torch.Generator().manual_seed(6304)
    lengths = [7, 4, 1, 0]
    rewards = torch.randn(4, 7, generator=g, dtype=torch.float64)
    values = torch.randn(4, 7, generator=g, dtype=torch.float64)
    mask = torch.zeros(4, 7, dtype=torch.float64)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1.0
        values[b, n:] = PAD_GARBAGE
        rewards[b, n:] = 0.0
    adv, ret = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
    for b, n in enumerate(lengths):
        ref = ref_gae_row(rewards[b].tolist(), values[b].tolist(), n, gamma, lam)
        assert adv[b, :n].tolist() == pytest.approx(ref, abs=1e-10)
        assert torch.all(adv[b, n:] == 0.0)
        # Critic target used by the code: returns = advantages + values.
        assert ret[b, :n].tolist() == pytest.approx(
            [a + v for a, v in zip(ref, values[b, :n].tolist())], abs=1e-10
        )


# ---------------------------------------------------------------- KL-shaped reward


def test_shaped_rewards_hand_computed():
    # beta_KL = 0.1. r_t = -0.1 (logp_t - ref_t) on every valid token, plus r_task on t = T.
    # Row 0 (3 valid): logp - ref = [0.5, -0.2, 0.3]; r_task = 2.0
    #   r = [-0.05, 0.02, -0.03 + 2.0] = [-0.05, 0.02, 1.97]
    # Row 1 (1 valid, 2 right pads with garbage log-probs): logp - ref = [1.0]; r_task = -1.0
    #   r = [-0.1 - 1.0, 0, 0] = [-1.1, 0, 0]
    policy = t64([[-1.0, -2.2, -0.7], [-0.5, PAD_GARBAGE, -PAD_GARBAGE]])
    ref = t64([[-1.5, -2.0, -1.0], [-1.5, 0.0, 0.0]])
    mask = t64([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    task = t64([2.0, -1.0])
    r = shaped_rewards(task, policy, ref, mask, beta_kl=0.1)
    assert torch.allclose(r, t64([[-0.05, 0.02, 1.97], [-1.1, 0.0, 0.0]]), atol=1e-12)


def test_shaped_rewards_beta_zero_is_terminal_only_and_empty_row_is_zero():
    policy = t64([[-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0]])
    ref = t64([[-2.0, -2.5, -0.1], [-1.0, -1.0, -1.0]])
    mask = t64([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    task = t64([0.7, 5.0])
    r = shaped_rewards(task, policy, ref, mask, beta_kl=0.0)
    assert torch.allclose(r, t64([[0.0, 0.7, 0.0], [0.0, 0.0, 0.0]]), atol=1e-12)


def test_shaped_rewards_kl_sum_matches_beta_times_sampled_kl_sum():
    # Sum over valid tokens of the KL part = -beta * sum (logp - ref); terminal reward appears once.
    g = torch.Generator().manual_seed(6304)
    policy = torch.randn(3, 6, generator=g, dtype=torch.float64) - 2.0
    ref = torch.randn(3, 6, generator=g, dtype=torch.float64) - 2.0
    mask = torch.zeros(3, 6, dtype=torch.float64)
    for b, n in enumerate([6, 2, 4]):
        mask[b, :n] = 1.0
    task = t64([1.0, -2.0, 0.5])
    beta = 0.2
    r = shaped_rewards(task, policy, ref, mask, beta_kl=beta)
    for b, n in enumerate([6, 2, 4]):
        kl_sum = sum(p - q for p, q in zip(policy[b, :n].tolist(), ref[b, :n].tolist()))
        assert r[b].sum().item() == pytest.approx(task[b].item() - beta * kl_sum, abs=1e-12)
        assert torch.all(r[b, n:] == 0.0)


# ---------------------------------------------------------------- value loss


def test_value_loss_is_masked_mean_squared_error():
    # Form implemented in the code: mean over valid tokens of (V - R)^2. No 0.5 factor and no
    # value clipping inside this function; value_coef from the config is applied by the caller.
    # Valid: (V-R) = [0.5, -1.0, 2.0] -> squares [0.25, 1.0, 4.0] -> mean 5.25/3 = 1.75
    pred = t64([[1.0, 0.0], [3.0, PAD_GARBAGE]]).requires_grad_(True)
    ret = t64([[0.5, 1.0], [1.0, 0.0]])
    mask = t64([[1.0, 1.0], [1.0, 0.0]])
    loss = value_mse_loss(pred, ret, mask)
    assert loss.item() == pytest.approx(1.75, abs=1e-12)
    loss.backward()
    # d/dV = 2 (V - R) / N on valid tokens, 0 on padding: [1/3, -2/3, 4/3, 0]
    assert torch.allclose(pred.grad, t64([[1.0 / 3, -2.0 / 3], [4.0 / 3, 0.0]]), atol=1e-12)
