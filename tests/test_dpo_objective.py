"""Validate task1_dpo.dpo.dpo_loss against an independent reference of the manual's objective.

Manual, Section 1 (DPO), "Introduction and Terminology":

    L_DPO(theta) = -E[ log sigma( beta * [ log pi_theta(y+|x)/pi_ref(y+|x)
                                          - log pi_theta(y-|x)/pi_ref(y-|x) ] ) ]

Manual, "Common Metric Definitions", DPO preference accuracy:

    m_theta = [log pi_theta(y+|x) - log pi_ref(y+|x)] - [log pi_theta(y-|x) - log pi_ref(y-|x)]
    accuracy = fraction of pairs with m_theta > 0

So the loss per pair is -log sigma(beta * m_theta) = log(1 + exp(-beta * m_theta)),
averaged over pairs. Inputs are summed response-token log-probabilities, shape (B,).
"""
from __future__ import annotations

import math

import pytest
import torch

from task1_dpo.dpo import dpo_loss

BETAS = [0.03, 0.10, 0.30]  # configs/dpo.yaml betas


def reference_margin(pc, pr, rc, rr):
    """m_theta from the manual, written per pair in plain Python floats."""
    return [(a - c) - (b - d) for a, b, c, d in zip(pc, pr, rc, rr)]


def reference_loss(pc, pr, rc, rr, beta):
    """Mean over pairs of -log sigma(beta * m_theta), via log1p(exp(-z)) in float math."""
    ms = reference_margin(pc, pr, rc, rr)
    per_pair = []
    for m in ms:
        z = beta * m
        # stable softplus(-z) = -log sigma(z)
        per_pair.append(math.log1p(math.exp(-z)) if z > -30 else -z + math.log1p(math.exp(z)))
    return sum(per_pair) / len(per_pair)


def as_t(x):
    return torch.tensor(x, dtype=torch.float64)


def test_hand_computed_two_pairs():
    # Pair 1: pc=-10, pr=-12, rc=-11, rr=-11
    #   policy margin = -10 - (-12) = 2 ; ref margin = -11 - (-11) = 0 ; m = 2 - 0 = 2
    #   z = 0.1 * 2 = 0.2 ; loss = log(1 + e^-0.2) = log(1.818731) = 0.598139
    # Pair 2: pc=-10, pr=-12, rc=-9, rr=-14
    #   policy margin = 2 ; ref margin = -9 - (-14) = 5 ; m = 2 - 5 = -3
    #   z = 0.1 * -3 = -0.3 ; loss = log(1 + e^0.3) = log(2.349859) = 0.854355
    # Mean loss = (0.598139 + 0.854355) / 2 = 0.726247
    # Accuracy = mean([2 > 0, -3 > 0]) = 0.5 ; logit mean = (0.2 + -0.3) / 2 = -0.05
    pc, pr, rc, rr = [-10.0, -10.0], [-12.0, -12.0], [-11.0, -9.0], [-11.0, -14.0]
    loss, diag = dpo_loss(as_t(pc), as_t(pr), as_t(rc), as_t(rr), beta=0.1)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.7262470569250594, abs=1e-9)
    assert loss.item() == pytest.approx(reference_loss(pc, pr, rc, rr, 0.1), abs=1e-12)
    assert diag["preference_accuracy"].item() == pytest.approx(0.5)
    assert diag["logit_mean"].item() == pytest.approx(-0.05, abs=1e-12)
    assert diag["policy_margin_mean"].item() == pytest.approx(2.0)


@pytest.mark.parametrize("beta", BETAS)
def test_policy_equals_reference_gives_log2(beta):
    # If pi_theta == pi_ref then each log-ratio is 0, so m_theta = 0 and
    # -log sigma(0) = log 2 = 0.693147 per pair, for any beta. The reference
    # margins are deliberately non-zero (5 and -3) so a sign error shows up.
    # Pair 1: c = -20, r = -25 -> policy margin 5, ref margin 5, m = 0
    # Pair 2: c = -30, r = -27 -> policy margin -3, ref margin -3, m = 0
    c, r = [-20.0, -30.0], [-25.0, -27.0]
    loss, diag = dpo_loss(as_t(c), as_t(r), as_t(c), as_t(r), beta=beta)
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-12)
    assert diag["logit_mean"].item() == pytest.approx(0.0, abs=1e-12)
    # m_theta = 0 is not > 0, so no pair counts as correct
    assert diag["preference_accuracy"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("seed", [0, 1, 6304])
@pytest.mark.parametrize("beta", BETAS)
def test_seeded_random_matches_reference(seed, beta):
    g = torch.Generator().manual_seed(seed)
    B = 64
    # magnitudes typical of summed sequence log-probs (hundreds of nats)
    pc, pr, rc, rr = (-(torch.rand(B, generator=g, dtype=torch.float64) * 400.0) for _ in range(4))
    loss, diag = dpo_loss(pc, pr, rc, rr, beta=beta)

    lists = [t.tolist() for t in (pc, pr, rc, rr)]
    ms = reference_margin(*lists)
    assert loss.item() == pytest.approx(reference_loss(*lists, beta), rel=1e-10, abs=1e-10)
    assert diag["preference_accuracy"].item() == pytest.approx(sum(m > 0 for m in ms) / B)
    assert diag["logit_mean"].item() == pytest.approx(beta * sum(ms) / B, rel=1e-10, abs=1e-10)
    assert diag["policy_margin_mean"].item() == pytest.approx(
        sum(a - b for a, b in zip(lists[0], lists[1])) / B, rel=1e-10
    )


@pytest.mark.parametrize("beta", BETAS)
def test_gradient_matches_reference(beta):
    # d/d pc of -log sigma(beta*m) = -beta * sigma(-beta*m) / B ; pr gets the opposite sign.
    g = torch.Generator().manual_seed(6304)
    B = 16
    pc = (-(torch.rand(B, generator=g, dtype=torch.float64) * 200.0)).requires_grad_(True)
    pr = (-(torch.rand(B, generator=g, dtype=torch.float64) * 200.0)).requires_grad_(True)
    rc = -(torch.rand(B, generator=g, dtype=torch.float64) * 200.0)
    rr = -(torch.rand(B, generator=g, dtype=torch.float64) * 200.0)
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=beta)
    loss.backward()

    ms = reference_margin(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())
    expected = [-beta / (1.0 + math.exp(beta * m)) / B for m in ms]
    assert pc.grad.tolist() == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert pr.grad.tolist() == pytest.approx([-e for e in expected], rel=1e-9, abs=1e-12)


def test_diagnostics_are_detached():
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    _, diag = dpo_loss(x, x - 1.0, torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), beta=0.1)
    assert all(not v.requires_grad for v in diag.values())
