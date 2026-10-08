"""Mac-only checks of task1_dpo.summarize (no model is loaded).

1. Paired bootstrap differences on synthetic data with a known answer.
2. Stratified and cluster resamplers keep strata sizes and prompt clusters intact.
3. Qualitative selection rules on hand-made rows, including the tie rules.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from task1_dpo.evaluate import pooled_kl
from task1_dpo.summarize import (
    accuracy_stat,
    align,
    bootstrap,
    cluster_resampler,
    diff_stat,
    gap_stat,
    iid_resampler,
    largest_compliance_gap,
    mean_stat,
    pooled_ratio_stat,
    prompt_key,
    stratified_resampler,
    top_rm_gain,
    wordlimit_violations,
)

SEED = 6304


# ---------------------------------------------------------------- 1. paired bootstrap

def test_paired_difference_constant_shift_is_exact():
    # b = a - 0.5 item by item: every paired resample gives exactly 0.5.
    a = np.random.default_rng(0).normal(size=300)
    b = a - 0.5
    out = bootstrap(diff_stat(mean_stat(a), mean_stat(b)), 300, iid_resampler(300), SEED, n_resamples=2000)
    assert out["point"] == pytest.approx(0.5)
    assert out["ci_low"] == pytest.approx(0.5) and out["ci_high"] == pytest.approx(0.5)


def test_paired_difference_matches_normal_theory():
    # a = b + d with d ~ N(1, 1) and large shared noise in b. The paired interval must have
    # half-width close to 1.96 / sqrt(n), unaffected by the noise in b.
    rng = np.random.default_rng(1)
    n = 2000
    b = rng.normal(scale=10.0, size=n)
    d = rng.normal(loc=1.0, scale=1.0, size=n)
    a = b + d
    out = bootstrap(diff_stat(mean_stat(a), mean_stat(b)), n, iid_resampler(n), SEED)
    half = (out["ci_high"] - out["ci_low"]) / 2
    expected = 1.96 * d.std() / np.sqrt(n)
    assert out["point"] == pytest.approx(d.mean())
    assert half == pytest.approx(expected, rel=0.1)
    assert out["ci_low"] < out["point"] < out["ci_high"]


def test_paired_accuracy_difference_known():
    m_a = np.array([1.0, 2.0, 0.5, 3.0])     # accuracy 1
    m_b = np.array([-1.0, -2.0, -0.5, -3.0])  # accuracy 0
    out = bootstrap(diff_stat(accuracy_stat(m_a), accuracy_stat(m_b)), 4, iid_resampler(4), SEED, n_resamples=500)
    assert (out["point"], out["ci_low"], out["ci_high"]) == (1.0, 1.0, 1.0)


def test_bootstrap_is_reproducible():
    x = np.random.default_rng(2).normal(size=50)
    r1 = bootstrap(mean_stat(x), 50, iid_resampler(50), SEED, n_resamples=500)
    r2 = bootstrap(mean_stat(x), 50, iid_resampler(50), SEED, n_resamples=500)
    assert r1 == r2


def test_pooled_ratio_equals_course_pooled_kl():
    pol = [[-1.0, -2.0, -0.5], [-0.1], [-3.0, -1.0]]
    ref = [[-1.5, -2.0, -1.0], [-0.4], [-2.0, -2.5]]
    summed = [sum(p) - sum(r) for p, r in zip(pol, ref)]
    lengths = [len(p) for p in pol]
    assert pooled_ratio_stat(summed, lengths)(np.arange(3)) == pytest.approx(pooled_kl(pol, ref))


def test_gap_stat_known():
    m = np.array([1, 1, -1, 1, -1, -1, 1, -1], dtype=float)
    s = np.array(["preferred_longer"] * 4 + ["rejected_longer"] * 4)
    # preferred_longer accuracy 3/4, rejected_longer accuracy 1/4.
    assert gap_stat(m, s)(np.arange(8)) == pytest.approx(0.5)


# ---------------------------------------------------------------- 2. resamplers

def test_cluster_resampler_keeps_clusters_intact():
    clusters = np.array(["wl01"] * 5 + ["wl02"] * 5 + ["wl03"] * 3 + ["wl04"] * 1)
    members = {c: set(np.flatnonzero(clusters == c).tolist()) for c in set(clusters.tolist())}
    draw = cluster_resampler(clusters)
    rng = np.random.default_rng(SEED)
    for _ in range(300):
        idx = draw(rng)
        counts = Counter(idx.tolist())
        n_drawn, n_items = 0, 0
        for c, items in members.items():
            k = {counts.get(i, 0) for i in items}
            assert len(k) == 1, f"cluster {c} split: item counts {k}"  # all items of a cluster drawn equally often
            times = k.pop()
            n_drawn += times
            n_items += times * len(items)
        assert n_drawn == len(members)  # as many clusters drawn as exist
        assert len(idx) == n_items      # nothing outside the drawn clusters


def test_stratified_resampler_keeps_stratum_sizes():
    strata = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2)
    draw = stratified_resampler(strata)
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        idx = draw(rng)
        assert Counter(strata[idx].tolist()) == {"a": 3, "b": 5, "c": 2}


def test_align_rejects_different_ids():
    a = [{"prompt_id": 1}, {"prompt_id": 2}]
    b = [{"prompt_id": 2}, {"prompt_id": 3}]
    with pytest.raises(SystemExit):
        align(a, b, prompt_key, "test")
    _, b2 = align(a, [{"prompt_id": 2, "x": 1}, {"prompt_id": 1, "x": 0}], prompt_key, "test")
    assert [r["x"] for r in b2] == [0, 1]  # reordered to match a


# ---------------------------------------------------------------- 3. qualitative rules

def gen_row(pid, rm, text="t"):
    return {"prompt_id": pid, "rm_score": rm, "text": text, "token_length": len(text), "truncated": False}


def test_top_rm_gain_order_and_ties():
    std = [gen_row("p1", 1.0), gen_row("p2", 3.0), gen_row("p3", 2.0), gen_row("p4", 0.0)]
    sft = [gen_row("p3", 0.0), gen_row("p1", 0.5), gen_row("p2", 1.0), gen_row("p4", 1.0)]
    # gains: p1 0.5, p2 2.0, p3 2.0, p4 -1.0 -> p2 and p3 tie, lower ID first.
    out = top_rm_gain(std, sft, k=3)
    assert [r["prompt_id"] for r in out] == ["p2", "p3", "p1"]
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert out[0]["rm_standard_minus_sft"] == pytest.approx(2.0)
    assert out[0]["standard"]["rm_score"] == 3.0 and out[0]["sft"]["rm_score"] == 1.0


def wl_row(pid, s, compliant, rm=0.0, words=10):
    return {"prompt_id": pid, "sample_index": s, "compliant": compliant, "rm_score": rm, "limit": 30,
            "word_count": words, "token_length": words, "truncated": False, "text": f"{pid}-{s}"}


def test_wordlimit_violations_rule():
    rows = [wl_row("wl01", 0, 0.0, rm=1.0), wl_row("wl01", 1, 1.0, rm=9.0), wl_row("wl02", 0, 0.0, rm=2.0),
            wl_row("wl02", 1, 0.0, rm=1.0), wl_row("wl03", 0, 0.0, rm=0.5), wl_row("wl03", 1, 0.0, rm=3.0),
            wl_row("wl04", 0, 0.0, rm=-1.0)]
    out = wordlimit_violations(rows, k=5)
    # compliant rows excluded; descending RM; tie at rm 1.0 between (wl01,0) and (wl02,1) -> wl01 first.
    assert [(r["prompt_id"], r["sample_index"]) for r in out] == [("wl03", 1), ("wl02", 0), ("wl01", 0), ("wl02", 1), ("wl03", 0)]
    assert len(wordlimit_violations(rows[:2], k=5)) == 1


def test_largest_compliance_gap_with_tie():
    std, bal = [], []
    # wl01: 1.0 vs 1.0 (gap 0); wl02: 0.2 vs 0.8 (|gap| 0.6); wl03: 0.8 vs 0.2 (|gap| 0.6); wl04: 0.4 vs 0.6.
    plan = {"wl01": (5, 5), "wl02": (1, 4), "wl03": (4, 1), "wl04": (2, 3)}
    for pid, (ns, nb) in plan.items():
        for s in range(5):
            std.append(wl_row(pid, s, float(s < ns)))
            bal.append(wl_row(pid, s, float(s < nb)))
    out = largest_compliance_gap(std, list(reversed(bal)))
    assert out["prompt_id"] == "wl02"  # tie with wl03 on |gap| -> lowest prompt ID
    assert out["compliance_standard"] == pytest.approx(0.2) and out["compliance_length_balanced"] == pytest.approx(0.8)
    assert out["compliance_standard_minus_balanced"] == pytest.approx(-0.6)
    assert out["standard"]["sample_index"] == 0 and out["length_balanced"]["sample_index"] == 0
    assert out["standard"]["text"] == "wl02-0" and out["length_balanced"]["text"] == "wl02-0"
