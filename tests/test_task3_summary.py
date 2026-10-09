"""Mac-only checks of task3_grpo.summarize on small synthetic result dicts (no files, no models).

1. OLS slope equals numpy.polyfit; constant x gives NaN; the bootstrap point is the full-sample slope.
2. Standard-run section: informative count, token totals and zero-gradient shares.
3. Protocol checks pass on a consistent fixture and each targeted corruption fails its own check.
4. Qualitative selection: rule maxima, ties broken by the lowest prompt_id, uninformative listing.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from task3_grpo.summarize import (CANONICAL, DR, RUNS, STANDARD, ols_slope, pick, protocol, qualitative,
                                  slope_bootstrap, standard_section)

SEED = 6304
GEN = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_new_tokens": 512}


def hist_row(u, informative=True, n_masked=0, lengths=(10, 20, 30, 40), reward=0.0):
    gen = int(sum(lengths))
    masked = 512 * n_masked
    uninf = 0 if informative else gen - masked
    return {
        "update": u, "prompt_id": f"p{u:02d}", "rewards": [reward] * 4, "group_reward_mean": reward,
        "group_reward_std": 0.5 if informative else 0.0, "informative": informative, "n_masked": n_masked,
        "policy_loss": 0.1, "surrogate_term": 0.0, "beta_kl_term": 0.1, "kl_k3": 1.0, "kl_sampled": 0.01 * u,
        "entropy_sampled": 0.5, "grad_norm_before_clip": 1.0, "clip_fraction": 0.0, "inputs_finite": True,
        "step_taken": True, "completion_length": {"mean": float(np.mean(lengths)), "sd": 1.0},
        "generated_tokens": gen,
        "zero_gradient_tokens": {"masked_truncation_tokens": masked, "uninformative_group_tokens": uninf,
                                 "masked_truncation_share": masked / gen, "uninformative_group_share": uninf / gen},
        "seconds": {"total": 1.0},
        "length_diagnostic": [{"grad_norm": 1.0, "T_k": 10, "abs_advantage": 1.0}] * (4 - n_masked),
        "term_gradients": {"grad_norm_surrogate": 0.5, "grad_norm_beta_kl": 0.01},
    }


def fixture():
    def train(n):
        h = [hist_row(u) for u in range(1, n + 1)]
        return {"status": "completed", "updates_completed": n, "history": h, "git": {"commit": "7ced6a30", "dirty": False},
                "seed": SEED, "n_nonfinite_steps": 0, "n_rm_input_truncated": 0,
                "config": {"lora": {"dropout": 0.05}},
                "decisions": {"dropout": "LoRA dropout 0.05 active in the training forward (train mode)"},
                "effective": {"max_completion_length": 512, "num_generations": 4, "kl_beta": 0.1, "clip_epsilon": 0.2,
                              "learning_rate": 5.0e-6, "policy_epochs": 1, "mask_truncated_completions": True,
                              "reward_max_length": 1024, "generation": dict(GEN)},
                "peak_vram_bytes": 2 ** 30, "wall_clock_seconds": 10.0, "setup_seconds": 1.0,
                "hardware": {"device": "Tesla T4"}, "dtype": "float16", "generated_tokens_total": 100 * n}

    def ev():
        return {"status": "completed", "smoke": False, "git": {"commit": "bf1fdc7f", "dirty": False}, "seed": SEED,
                "prompts": {"n_evaluated": 165}, "metrics": {"n_responses": 165, "n_rm_input_truncated": 0},
                "settings": {"max_new_tokens": 768, "reward_max_length": 1280, "samples_per_prompt": 1, "temperature": 0.7,
                             "top_p": 0.9, "seed_reset_before_generation": SEED,
                             "effective_generation": {**GEN, "max_new_tokens": 768}}}
    trains = {STANDARD: train(20), CANONICAL: train(8), DR: train(8)}
    evals = {r: ev() for r in RUNS}
    roll = [{"update": 1, "sample_index": k, "response_ids_sha256": f"h{k}", "token_length": 10 + k} for k in range(4)]
    rollouts = {r: copy.deepcopy(roll) for r in RUNS}
    gens = {r: [{"prompt_id": f"e{i:03d}"} for i in range(165)] for r in RUNS}
    return trains, evals, rollouts, gens


def failed(checks):
    return [c["check"] for c in checks if not c["passed"]]


# ---------------------------------------------------------------- 1. slope

def test_ols_slope_matches_polyfit():
    rng = np.random.default_rng(0)
    x, y = np.arange(1, 21, dtype=float), rng.normal(size=20)
    assert ols_slope(y, x) == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-12)
    assert math.isnan(ols_slope([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]))


def test_slope_bootstrap_point_and_interval():
    y = 2.0 * np.arange(1, 21) + np.random.default_rng(1).normal(scale=0.1, size=20)
    out = slope_bootstrap(y, SEED, n_resamples=500)
    assert out["point"] == pytest.approx(np.polyfit(np.arange(1, 21), y, 1)[0], rel=1e-12)
    assert out["ci_low"] <= out["point"] <= out["ci_high"] and 1.9 < out["ci_low"] and out["ci_high"] < 2.1
    assert slope_bootstrap(y, SEED, n_resamples=500) == out  # seeded


# ---------------------------------------------------------------- 2. standard section

def test_standard_section_counts():
    t = fixture()[0][STANDARD]
    t["history"][2] = hist_row(3, n_masked=1, lengths=(512, 20, 30, 40))
    t["history"][5] = hist_row(6, informative=False, lengths=(2, 2, 2, 2))
    s = standard_section(t, SEED)
    assert s["informative_updates"] == {"n_informative": 19, "n_updates": 20, "uninformative_updates": [6]}
    gen = 18 * 100 + 602 + 8
    assert s["zero_gradient_tokens"]["generated_tokens"] == gen
    assert s["zero_gradient_tokens"]["zero_grad_masked_truncation_tokens"] == 512 and s["zero_gradient_tokens"]["zero_grad_uninformative_tokens"] == 8
    assert s["zero_gradient_tokens"]["zero_grad_masked_truncation_share"] == pytest.approx(512 / gen)
    assert s["zero_gradient_tokens"]["zero_grad_total_share"] == pytest.approx(520 / gen)
    assert s["update_trends"]["sampled_kl"]["point"] == pytest.approx(0.01)
    assert s["run"]["peak_vram_gib"] == 1.0 and len(s["per_update"]) == 20


# ---------------------------------------------------------------- 3. protocol

def test_protocol_passes_on_consistent_fixture():
    assert failed(protocol(*fixture(), SEED)) == []


@pytest.mark.parametrize("corrupt, expected", [
    (lambda t, e, r, g: t[STANDARD]["history"][4].update(clip_fraction=0.01), "standard: clip fraction 0 on every update"),
    (lambda t, e, r, g: t[DR]["git"].update(commit="1234567x"), "fork_dr_grpo train: commit"),
    (lambda t, e, r, g: e[CANONICAL]["git"].update(dirty=True), "fork_grpo eval: commit"),
    (lambda t, e, r, g: t[CANONICAL]["history"][3].update(prompt_id="other"), "fork_grpo: same prompt_ids"),
    (lambda t, e, r, g: r[DR][2].update(response_ids_sha256="x"), "update-1 completions identical"),
    (lambda t, e, r, g: g[DR].reverse(), "eval prompt_ids identical"),
    (lambda t, e, r, g: t[DR]["effective"]["generation"].update(top_k=50), "effective training generation"),
    (lambda t, e, r, g: t[STANDARD]["effective"].update(kl_beta=0.2), "standard: training settings"),
    (lambda t, e, r, g: e[STANDARD]["settings"].update(reward_max_length=1024), "standard: eval settings"),
    (lambda t, e, r, g: t[DR]["history"][1].update(length_diagnostic=[]), "fork_dr_grpo: length-diagnostic rows"),
    (lambda t, e, r, g: t[CANONICAL]["history"][1]["term_gradients"].update(grad_norm_beta_kl=float("nan")),
     "fork_grpo: length-diagnostic rows"),
    (lambda t, e, r, g: t[STANDARD].update(updates_completed=19), "standard: train completed"),
    (lambda t, e, r, g: e[DR]["metrics"].update(n_rm_input_truncated=2), "RM-truncated inputs"),
    (lambda t, e, r, g: t[CANONICAL]["history"][0].update(step_taken=False), "fork_grpo: no non-finite steps"),
])
def test_protocol_detects_each_corruption(corrupt, expected):
    data = fixture()
    corrupt(*data)
    bad = failed(protocol(*data, SEED))
    assert bad and all(b.startswith(expected) for b in bad), bad


# ---------------------------------------------------------------- 4. qualitative

def gen_row(pid, length, rm):
    return {"prompt_id": pid, "text": f"{pid}-" + "x" * 1000, "token_length": length, "rm_score": rm, "truncated": False}


def test_qualitative_rules_and_ties():
    gens = {CANONICAL: [gen_row("b", 100, 1.0), gen_row("a", 50, 2.0), gen_row("c", 10, 0.0)],
            DR: [gen_row("b", 10, 3.0), gen_row("a", 140, 0.0), gen_row("c", 20, 0.5)]}
    prompts = {p: p * 500 for p in "abc"}
    train = {"history": [hist_row(1), hist_row(2, informative=False)]}
    roll = [{"update": 2, "sample_index": k, "text": f"r{k}" * 200, "token_length": 2, "truncated": False} for k in (3, 1, 0, 2)]
    q = qualitative(gens, prompts, train, roll)
    h = q["heldout"]
    # |length diff| is 90 for both a and b: tie goes to the lowest prompt_id.
    assert h["largest_abs_length_difference"]["prompt_id"] == "a" and h["largest_abs_length_difference"]["rule_value"] == 90
    assert h["largest_rm_dr_minus_grpo"]["prompt_id"] == "b" and h["largest_rm_grpo_minus_dr"]["prompt_id"] == "a"
    assert len(h["largest_rm_dr_minus_grpo"]["prompt_text"]) == 400
    assert len(h["largest_rm_dr_minus_grpo"]["dr"]["response"]) == 600
    u = q["standard_uninformative_updates"]
    assert [x["update"] for x in u] == [2] and [c["sample_index"] for c in u[0]["completions"]] == [0, 1, 2, 3]
    assert all(len(c["text"]) == 200 for c in u[0]["completions"])


def test_pick_tie_lowest_prompt_id():
    rows = [{"prompt_id": "z", "v": 1}, {"prompt_id": "m", "v": 1}, {"prompt_id": "a", "v": 0}]
    assert pick(rows, lambda r: r["v"])["prompt_id"] == "m"
