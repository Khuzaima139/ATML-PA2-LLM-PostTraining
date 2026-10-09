"""Mac-only checks of the Task 3 GRPO pipeline pieces (tiny random Qwen2 models, float32, CPU).

Continuation loop:
1. One step with --length-diagnostic gives bit-identical parameters and RNG state to one step without
   it (LoRA dropout on, non-reentrant gradient checkpointing on, as on Kaggle).
2. The diagnostic gradient norm equals an independent per-completion gradient written from the
   manual's surrogate at ratio 1: grad of (A_k / N_k) * sum_t log pi(y_t), N_k = T_k or 512.
   The term gradient norms equal independent gradients of the manual surrogate and of beta * k3 KL.
3. Clip fraction 0 and ratio 1 (old = new.detach()); masked completions get no diagnostic row.
4. Zero-gradient token shares; eval-mode log-probs restore train mode and match the reference.
5. Effective generation settings take unset keys from the model's generation_config.
Group-size study:
6. Consecutive generation_index partition, subset enumeration, tertile binning with ties.
7. Pooled advantage variance equals the informative rate; regrouping noise and sign flips on hand cases.
8. Std-bias table (c4 values, Monte Carlo) and the uninformative-at-K8 qualitative selection.
Normalization comparison: Spearman, bootstraps, protocol checks, term-gradient section.
"""
from __future__ import annotations

import copy
import math
from itertools import combinations

import numpy as np
import pytest
import torch
from peft import get_peft_model
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from common.generation import response_token_logprobs
from common.models import make_lora_config, reference_mode, trainable_parameters
from task3_grpo.analyze_group_size import (
    difficulty_bins,
    partition_advantages,
    prompt_stats,
    quantity_fns,
    regroup_equal_generation_budget,
    run_study,
    stack,
    subset_advantages,
    subsets_containing,
)
from task3_grpo.continue_train import check_release_settings, eval_logprobs, grpo_update
from task3_grpo.grpo import group_relative_advantages
from task3_grpo.grpo_utils import effective_generation_settings, is_informative, zero_gradient_token_shares

LORA = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
PAD, EOS = 0, 2
MAXLEN = 512


def tiny_policy(seed=0, dropout=0.05, checkpointing=True):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, pad_token_id=PAD, use_cache=False)
    model = get_peft_model(Qwen2ForCausalLM(cfg), make_lora_config({"lora": {**LORA, "dropout": dropout}}))
    with torch.no_grad():  # non-zero LoRA B so the policy differs from the reference
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.05)
    model.train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def tiny_batch(seed=2):
    """4 completions of one 5-token prompt: EOS at t=3, truncated (6 tokens, masked), EOS at t=5, EOS at t=1."""
    g = torch.Generator().manual_seed(seed)
    P, R = 5, 6
    prompt = torch.randint(3, 1000, (1, P), generator=g).repeat(4, 1)
    resp = torch.randint(3, 1000, (4, R), generator=g)
    resp[0, 3], resp[0, 4:] = EOS, PAD
    resp[2, 5] = EOS
    resp[3, 1], resp[3, 2:] = EOS, PAD
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6, [1] * 6, [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    seq = torch.cat([prompt, resp], dim=1)
    return {
        "sequences": seq,
        "attention_mask": torch.ones_like(seq),
        "prompt_width": P,
        "response_ids": resp,
        "response_mask": mask,
        "truncated": [False, True, False, False],
        "advantages": group_relative_advantages(torch.tensor([0.3, -1.0, 1.2, 0.1]), torch.zeros(4, dtype=torch.long)),
    }


def with_ref(model, batch):
    b = dict(batch)
    b["ref_logp"] = eval_logprobs(model, b, reference=True)
    return b


def one_step(diag: bool, loss_type: str, seed=0):
    model = tiny_policy(seed)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1.0e-2)
    batch = with_ref(model, tiny_batch())
    torch.manual_seed(123)  # dropout stream for the training forward
    out = grpo_update(model, opt, batch, 0.2, 0.1, loss_type, MAXLEN, 1.0, length_diag=diag)
    return model, out, torch.get_rng_state()


# ---------------------------------------------------------------- continuation loop

@pytest.mark.parametrize("loss_type", ["grpo", "dr_grpo"])
def test_length_diagnostic_leaves_the_step_unchanged(loss_type):
    m0, out0, rng0 = one_step(False, loss_type)
    m1, out1, rng1 = one_step(True, loss_type)
    assert torch.equal(rng0, rng1), "diagnostic consumed RNG"
    for (n0, p0), (n1, p1) in zip(m0.named_parameters(), m1.named_parameters()):
        assert n0 == n1 and torch.equal(p0, p1), n0
    for key in ("policy_loss", "surrogate_term", "beta_kl_term", "kl_k3", "grad_norm_before_clip"):
        assert out0[key] == out1[key], key
    assert out0["length_diagnostic"] is None and len(out1["length_diagnostic"]) == 3
    assert out0["term_gradients"] is None
    tg = out1["term_gradients"]
    assert tg["grad_norm_surrogate"] > 0 and tg["grad_norm_beta_kl"] > 0
    assert tg["surrogate_value"] == pytest.approx(out1["surrogate_term"], abs=1e-6)
    assert tg["beta_kl_value"] == pytest.approx(out1["beta_kl_term"], abs=1e-7)
    # the step actually moved the LoRA parameters
    ref = tiny_policy(0)
    moved = any(not torch.equal(p, q) for (n, p), (_, q) in zip(m0.named_parameters(), ref.named_parameters()) if "lora_" in n)
    assert moved


@pytest.mark.parametrize("loss_type", ["grpo", "dr_grpo"])
def test_diagnostic_grad_norm_matches_manual_surrogate_gradient(loss_type):
    # dropout 0 so an independent forward reproduces the same function
    model = tiny_policy(0, dropout=0.0, checkpointing=False)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=0.0)
    batch = with_ref(model, tiny_batch())
    out = grpo_update(model, opt, batch, 0.2, 0.1, loss_type, MAXLEN, 1.0, length_diag=True)
    params = trainable_parameters(model)
    adv = batch["advantages"]
    for row in out["length_diagnostic"]:
        k = row["sample_index"]
        sl = slice(k, k + 1)
        logp = response_token_logprobs(model, batch["sequences"][sl], batch["attention_mask"][sl], batch["prompt_width"], batch["response_ids"][sl])[0]
        t_k = int(batch["response_mask"][k].sum())
        n_k = t_k if loss_type == "grpo" else MAXLEN
        # manual: at rho = 1 the gradient of (1/N_k) sum_t min(rho A, clip(rho) A) is (A_k / N_k) sum_t grad log pi
        surrogate = float(adv[k]) / n_k * (logp[0, :t_k]).sum()
        grads = torch.autograd.grad(surrogate, params, allow_unused=True)
        expected = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads if g is not None))
        assert row["grad_norm"] == pytest.approx(expected, rel=1e-5)
        assert row["T_k"] == t_k
        assert row["abs_advantage"] == pytest.approx(abs(float(adv[k])))
        assert row["analytic_token_weight"] == pytest.approx(abs(float(adv[k])) / n_k)


@pytest.mark.parametrize("loss_type", ["grpo", "dr_grpo"])
def test_term_gradient_norms_match_manual_terms(loss_type):
    model = tiny_policy(0, dropout=0.0, checkpointing=False)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=0.0)
    batch = with_ref(model, tiny_batch())
    beta = 0.1
    out = grpo_update(model, opt, batch, 0.2, beta, loss_type, MAXLEN, 1.0, length_diag=True)
    params = trainable_parameters(model)
    logp = response_token_logprobs(model, batch["sequences"], batch["attention_mask"], batch["prompt_width"], batch["response_ids"])[0]
    keep = torch.tensor([0.0 if t else 1.0 for t in batch["truncated"]])[:, None]
    mask = batch["response_mask"] * keep
    adv = batch["advantages"]
    K = mask.shape[0]
    # manual surrogate at rho = 1: -(1/K) sum_k (A_k / N_k) sum_t log pi, masked rows contribute 0
    n_k = mask.sum(-1).clamp_min(1.0) if loss_type == "grpo" else torch.full((K,), float(MAXLEN))
    surrogate = -((adv / n_k) * (logp * mask).sum(-1)).sum() / K
    # manual k3: exp(ref - logp) - (ref - logp) - 1, token mean over unmasked tokens
    d = batch["ref_logp"] - logp
    kl_term = beta * ((torch.exp(d) - d - 1.0) * mask).sum() / mask.sum()

    def norm(t):
        grads = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        return math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads if g is not None))

    tg = out["term_gradients"]
    assert tg["grad_norm_surrogate"] == pytest.approx(norm(surrogate), rel=1e-5)
    assert tg["grad_norm_beta_kl"] == pytest.approx(norm(kl_term), rel=1e-5)
    assert all(p.grad is None for p in params)


def test_clip_fraction_zero_ratio_one_and_masked_rows_skipped():
    _, out, _ = one_step(True, "grpo")
    assert out["clip_fraction"] == 0.0 and out["ratio_mean"] == 1.0
    assert out["n_masked"] == 1
    assert [r["sample_index"] for r in out["length_diagnostic"]] == [0, 2, 3]
    assert out["policy_loss"] == pytest.approx(out["surrogate_term"] + out["beta_kl_term"])
    assert out["step_taken"]


def test_grpo_update_refuses_eval_mode():
    model = tiny_policy(0)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1.0e-3)
    batch = with_ref(model, tiny_batch())
    model.eval()
    with pytest.raises(RuntimeError):
        grpo_update(model, opt, batch, 0.2, 0.1, "grpo", MAXLEN, 1.0)


def test_eval_logprobs_restores_train_mode_and_reference_is_adapter_off():
    model = tiny_policy(0)
    batch = tiny_batch()
    pol = eval_logprobs(model, batch, reference=False)
    ref = eval_logprobs(model, batch, reference=True)
    assert model.training
    with torch.no_grad(), reference_mode(model):
        direct = response_token_logprobs(model, batch["sequences"], batch["attention_mask"], 5, batch["response_ids"])[0]
    assert torch.allclose(ref, direct * batch["response_mask"], atol=1e-5)
    assert not torch.allclose(pol, ref)
    assert float((pol * (1 - batch["response_mask"])).abs().sum()) == 0.0


def test_zero_gradient_token_shares():
    s = zero_gradient_token_shares([10, 512, 30, 48], [False, True, False, False], informative=True)
    assert s["generated_tokens"] == 600 and s["masked_truncation_tokens"] == 512 and s["uninformative_group_tokens"] == 0
    s = zero_gradient_token_shares([10, 512, 30, 48], [False, True, False, False], informative=False)
    assert s["uninformative_group_tokens"] == 88
    assert s["total_share"] == pytest.approx(1.0)
    assert s["masked_truncation_share"] == pytest.approx(512 / 600)


def test_informative_flag_uses_population_std_tolerance():
    assert not is_informative(torch.tensor([1.25, 1.25, 1.25, 1.25]))
    assert is_informative(torch.tensor([1.0, 1.0, 1.0, 1.00001]))


def test_effective_generation_settings_records_model_keys():
    class M:
        generation_config = GenerationConfig(top_k=20, repetition_penalty=1.05, temperature=0.1, top_p=0.8, do_sample=True)

    class T:
        pad_token_id, eos_token_id = 0, 2

    s = effective_generation_settings(M(), T(), {"temperature": 0.7, "top_p": 0.9, "do_sample": True}, 512)
    assert s["temperature"] == 0.7 and s["top_p"] == 0.9 and s["max_new_tokens"] == 512
    assert s["top_k"] == 20 and s["repetition_penalty"] == 1.05
    assert "top_k" in s["from_model_generation_config"] and "temperature" not in s["from_model_generation_config"]


def test_release_settings_guard():
    ok = {"policy_epochs": 1, "prompts_per_update": 1, "mask_truncated_completions": True}
    check_release_settings(ok)
    for bad in ({"policy_epochs": 2}, {"prompts_per_update": 2}, {"mask_truncated_completions": False}):
        with pytest.raises(SystemExit):
            check_release_settings({**ok, **bad})


# ---------------------------------------------------------------- group-size study

def fake_cache(rewards_by_prompt: dict[str, list[float]]):
    by_prompt = {}
    for i, (pid, rs) in enumerate(rewards_by_prompt.items()):
        by_prompt[str(i)] = [{"prompt_id": pid, "source_index": i, "generation_index": j, "reward": r,
                              "clipped_at_max": False, "terminated_with_eos": True} for j, r in enumerate(rs)]
    return by_prompt


def test_partition_is_consecutive_generation_index():
    by_prompt = fake_cache({"a": list(range(8)), "b": list(range(10, 18))})
    for k in (2, 4, 8):
        groups = regroup_equal_generation_budget(by_prompt, k)
        assert len(groups) == 2 * 8 // k
        assert sum(len(g["rewards"]) for g in groups) == 16  # equal generation budget
        for g in groups:
            j = g["group_index"]
            assert g["generation_indices"] == list(range(j * k, (j + 1) * k))
    a4 = partition_advantages(np.arange(8.0), 4)
    expected = torch.cat([group_relative_advantages(torch.arange(4.0, dtype=torch.float64), torch.zeros(4, dtype=torch.long))] * 2)
    assert np.allclose(a4, expected.numpy())
    with pytest.raises(ValueError):
        regroup_equal_generation_budget(by_prompt, 3)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_subset_enumeration(k):
    subs = subsets_containing(8, k)
    for i, ss in subs.items():
        assert len(ss) == math.comb(7, k - 1)
        assert all(i in s and len(s) == k for s in ss)
        assert len(set(ss)) == len(ss)
    r = np.array([0.1, 0.5, -0.2, 0.9, 0.3, 0.3, -1.0, 2.0])
    sub = subset_advantages(r, k)
    for i in range(8):
        assert len(sub[i]) == math.comb(7, k - 1)
        # recompute each subset advantage directly from the manual formula
        direct = []
        for s in combinations(range(8), k):
            if i in s:
                rs = r[list(s)]
                direct.append((r[i] - rs.mean()) / max(rs.std(), 1e-6))
        assert np.allclose(sorted(sub[i]), sorted(direct))


def test_difficulty_bins_tertiles_with_ties_by_prompt_id():
    ids = ["f", "e", "d", "c", "b", "a"]
    means = [0.0, 0.0, 1.0, 1.0, 2.0, 0.0]
    bins = difficulty_bins(ids, means)
    # sorted by (mean, id): a(0), e(0), f(0), c(1), d(1), b(2) -> bins 0,0,1,1,2,2
    assert dict(zip(ids, bins)) == {"a": 0, "e": 0, "f": 1, "c": 1, "d": 2, "b": 2}
    with pytest.raises(ValueError):
        difficulty_bins(ids[:5], means[:5])


def test_pooled_variance_equals_informative_rate_and_k8_has_no_regroup_noise():
    rng = np.random.default_rng(0)
    rewards = [rng.normal(size=8) for _ in range(5)] + [np.full(8, 0.7), np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])]
    for k in (2, 4, 8):
        q = quantity_fns(stack([prompt_stats(r, k) for r in rewards]))
        idx = np.arange(len(rewards))
        assert q["pooled_advantage_var"](idx) == pytest.approx(q["informative_rate"](idx), abs=1e-9)
        assert q["informative_rate"](idx) + q["uninformative_rate"](idx) == pytest.approx(1.0)
        assert q["sign_flip_rate"](idx) + q["no_signal_rate"](idx) + q["agree_rate"](idx) == pytest.approx(1.0)
        if k == 8:
            assert q["regrouping_noise"](idx) == 0.0 and q["sign_flip_rate"](idx) == 0.0
    # last prompt, K=2 groups {1,1},{2,2},{3,3},{3,3}: all uninformative
    assert prompt_stats(rewards[-1], 2)["n_informative"] == 0
    assert prompt_stats(rewards[-1], 4)["n_informative"] == 1


def test_sign_flip_hand_case():
    # rewards 0,0,0,0,0,0,0,8: mean8 = 1. Completion 0 (r=0 < mean) at K=2 pairs with six 0s (A=0, no signal)
    # and with 8 (A=-1, agrees). Completion 7 (r=8) is always above its partner (A=+1, agrees).
    r = np.array([0.0] * 7 + [8.0])
    s = prompt_stats(r, 2)
    assert s["n_eligible"] == 8
    assert s["sum_flip"] == 0.0
    assert s["sum_no_signal"] == pytest.approx(7 * (6 / 7))
    # a flip: rewards 0,2,2,...: mean8 = 1.75; completion 1 (r=2 > mean) paired with 2 gives A=0, with 0 gives +1;
    # completion with r=2 never flips. Use r = [0, 1.9, 2, 2, 2, 2, 2, 2]: mean = 1.7375, r=1.9 > mean but
    # paired with any 2 its A = -1 (flip) in 6 of 7 pairs.
    r = np.array([0.0, 1.9, 2, 2, 2, 2, 2, 2])
    sub = subset_advantages(r, 2)
    assert np.sum(sub[1] < 0) == 6
    s = prompt_stats(r, 2)
    assert s["sum_flip"] == pytest.approx(6 / 7)


def test_run_study_shapes_and_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    rewards = {f"p{i:02d}": list(rng.normal(loc=i / 6, size=8)) for i in range(6)}
    study = run_study(fake_cache(rewards), [2, 4, 8], seed=6304, n_resamples=200)
    assert set(study["results"]) == {"all", "bin1_low_reward", "bin2_mid_reward", "bin3_high_reward"}
    for scope, block in study["results"].items():
        assert set(block) == {"K2", "K4", "K8", "K2_minus_K8"}
        for q, v in block["K2"].items():
            if isinstance(v, dict):
                assert v["ci_low"] - 1e-12 <= v["point"] <= v["ci_high"] + 1e-12 or v["n_nonfinite_resamples"] > 0
    assert study["results"]["all"]["K2"]["n_groups"] == 24
    assert len(study["scopes"]["bin1_low_reward"]) == 2


# ---------------------------------------------------------------- normalization comparison

from task3_grpo.compare_normalization import (  # noqa: E402
    CANONICAL,
    DR,
    diagnostic_rows,
    length_section,
    protocol,
    spearman,
    term_gradient_section,
    terms_section,
    two_sample_bootstrap,
    update1_identity,
)


def test_spearman_average_ranks_and_monotone_invariance():
    x = [1, 2, 3, 4, 5]
    assert spearman(x, [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman(x, np.exp(-np.array(x, dtype=float))) == pytest.approx(-1.0)
    # ties: ranks of y = [1.5, 1.5, 3, 4]; Pearson of [1,2,3,4] with those ranks
    expected = np.corrcoef([1, 2, 3, 4], [1.5, 1.5, 3, 4])[0, 1]
    assert spearman([1, 2, 3, 4], [5, 5, 6, 7]) == pytest.approx(expected)
    assert math.isnan(spearman([1, 2, 3], [1, 1, 1]))


def test_two_sample_bootstrap_point_is_b_minus_a():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([5.0, 7.0])
    r = two_sample_bootstrap(lambda i: a[i].mean(), 3, lambda i: b[i].mean(), 2, seed=0, n_resamples=500)
    assert r["point"] == pytest.approx(4.0)
    assert r["ci_low"] <= r["point"] <= r["ci_high"]


def fake_train(loss_type, diag_rows, terms=((-0.1, 0.02),), prompt_ids=("p1", "p2"), grads=((0.4, 0.1),)):
    hist = []
    for u, rows in enumerate(diag_rows, start=1):
        s, k = terms[min(u - 1, len(terms) - 1)]
        gs, gk = grads[min(u - 1, len(grads) - 1)]
        hist.append({"update": u, "length_diagnostic": rows, "surrogate_term": s, "beta_kl_term": k, "n_masked": 0, "informative": gs > 0,
                     "term_gradients": {"grad_norm_surrogate": gs, "grad_norm_beta_kl": gk}})
    return {"status": "completed", "seed": 6304, "updates_completed": len(diag_rows), "generated_tokens_total": 100,
            "prompts": {"prompt_ids": list(prompt_ids)},
            "effective": {"loss_type": loss_type, "length_diagnostic": True, "updates": len(diag_rows), "kl_beta": 0.1},
            "history": hist}


def diag(T, A, g):
    return {"sample_index": 0, "T_k": T, "abs_advantage": A, "grad_norm": g, "analytic_token_weight": A / T}


def test_diagnostic_rows_exclude_zero_advantage_and_length_section():
    canon = fake_train("grpo", [[diag(10, 1.0, 0.5), diag(300, 0.0, 0.0), diag(400, 2.0, 0.2)], [diag(50, 0.5, 0.1), diag(500, 1.0, 0.01)]])
    dr = fake_train("dr_grpo", [[diag(10, 1.0, 0.01), diag(300, 1.0, 0.3), diag(400, 2.0, 0.9)], [diag(50, 0.5, 0.02)]])
    rows, excluded = diagnostic_rows(canon)
    assert excluded == 1 and len(rows) == 4 and rows[2]["update"] == 2
    assert rows[1]["y"] == pytest.approx(0.1)
    out = length_section({CANONICAL: canon, DR: dr}, seed=0, n_resamples=200)
    pc = out["per_fork"][CANONICAL]
    assert pc["n_completions"] == 4 and pc["n_T_le_256"] == 2 and pc["n_T_gt_256"] == 2
    assert pc["mean_grad_per_abs_adv_T_le_256"]["point"] == pytest.approx((0.5 + 0.2) / 2)
    assert pc["spearman_T_vs_grad_per_abs_adv"]["point"] == pytest.approx(-1.0)
    assert out["per_fork"][DR]["spearman_T_vs_grad_per_abs_adv"]["point"] == pytest.approx(1.0)
    assert out["dr_grpo_minus_grpo"]["spearman_T_vs_grad_per_abs_adv"]["point"] == pytest.approx(2.0)


def test_terms_section_keeps_values_without_ratio():
    t = terms_section({CANONICAL: fake_train("grpo", [[]], terms=((-0.1, 0.02),))})
    assert t[CANONICAL][0]["surrogate_term"] == -0.1 and t[CANONICAL][0]["beta_kl_term"] == 0.02
    assert not any("over" in key for key in t[CANONICAL][0])


def test_term_gradient_section_ratio_of_means():
    grads = ((0.4, 0.1), (0.0, 0.3), (0.2, 0.2))  # update 2 uninformative: surrogate grad exactly 0
    out = term_gradient_section({CANONICAL: fake_train("grpo", [[]] * 3, grads=grads)}, seed=0, n_resamples=300)[CANONICAL]
    assert out["n_updates"] == 3 and out["n_zero_surrogate_grad"] == 1
    assert out["mean_grad_norm_surrogate"]["point"] == pytest.approx(0.2)
    assert out["mean_grad_norm_beta_kl"]["point"] == pytest.approx(0.2)
    assert out["ratio_beta_kl_over_surrogate"]["point"] == pytest.approx(1.0)
    assert out["ratio_beta_kl_over_surrogate"]["n_nonfinite_resamples"] > 0  # resamples drawing only update 2
    r = out["ratio_beta_kl_over_surrogate"]
    assert r["ci_low"] <= r["point"] <= r["ci_high"]
    assert [u["grad_norm_beta_kl"] for u in out["per_update"]] == [0.1, 0.3, 0.2]


def roll(update, idx, sha, n, r):
    return {"update": update, "sample_index": idx, "response_ids_sha256": sha, "token_length": n, "reward": r}


def test_update1_identity_and_protocol_checks():
    ra = [roll(1, 0, "a", 5, 0.1), roll(1, 1, "b", 7, 0.2), roll(2, 0, "c", 3, 0.0)]
    rb = [roll(1, 1, "b", 7, 0.2), roll(1, 0, "a", 5, 0.1), roll(2, 0, "z", 9, 1.0)]
    ident = update1_identity({CANONICAL: ra, DR: rb})
    assert ident["token_ids_identical"] and ident["n_completions"] == 2 and ident["max_abs_reward_difference"] == 0.0
    rb_bad = [roll(1, 0, "x", 5, 0.1), roll(1, 1, "b", 7, 0.2)]
    assert not update1_identity({CANONICAL: ra, DR: rb_bad})["token_ids_identical"]

    ev = {"status": "completed", "smoke": False, "settings": {"max_new_tokens": 768}, "prompts": {"n_evaluated": 165}}
    trains = {CANONICAL: fake_train("grpo", [[]]), DR: fake_train("dr_grpo", [[]])}
    checks = protocol(trains, {CANONICAL: ev, DR: dict(ev)}, {CANONICAL: ra, DR: rb})
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    trains[DR]["effective"]["kl_beta"] = 0.2
    trains[DR]["prompts"]["prompt_ids"] = ["p2", "p1"]
    failed = {c["check"] for c in protocol(trains, {CANONICAL: ev, DR: dict(ev)}, {CANONICAL: ra, DR: rb}) if not c["passed"]}
    assert failed == {"same effective settings apart from loss_type", "same prompt IDs and order"}
    del trains[DR]["history"][0]["term_gradients"]
    failed = {c["check"] for c in protocol(trains, {CANONICAL: ev, DR: dict(ev)}, {CANONICAL: ra, DR: rb}) if not c["passed"]}
    assert f"{DR} term gradients logged at every update" in failed


def test_length_section_survives_an_empty_bucket():
    canon = fake_train("grpo", [[diag(10, 1.0, 0.5), diag(20, 2.0, 0.2), diag(30, 1.0, 0.1)]])
    dr = fake_train("dr_grpo", [[diag(10, 1.0, 0.01), diag(20, 1.0, 0.3), diag(40, 2.0, 0.9)]])
    out = length_section({CANONICAL: canon, DR: dr}, seed=0, n_resamples=50)
    gt = out["per_fork"][CANONICAL]["mean_grad_per_abs_adv_T_gt_256"]
    assert math.isnan(gt["point"]) and gt["n_resamples"] == 0
    assert math.isnan(out["dr_grpo_minus_grpo"]["mean_grad_per_abs_adv_T_gt_256"]["ci_low"])
    assert np.isfinite(out["per_fork"][CANONICAL]["mean_grad_per_abs_adv_T_le_256"]["point"])


# ---------------------------------------------------------------- group-size follow-ups

from task3_grpo.group_size_followup import c4, expected_population_std_ratio, qualitative, std_bias  # noqa: E402


def test_c4_known_values_and_monte_carlo():
    assert c4(2) == pytest.approx(math.sqrt(2 / math.pi))
    assert c4(4) == pytest.approx(0.921318, abs=1e-6)
    assert c4(8) == pytest.approx(0.965030, abs=1e-6)
    rng = np.random.default_rng(0)
    for k in (2, 4, 8):
        x = rng.normal(scale=2.0, size=(200_000, k))  # 200k x 8 float64 = 13 MB
        assert x.std(axis=1).mean() / 2.0 == pytest.approx(expected_population_std_ratio(k), rel=5e-3)


def test_std_bias_table_ratios():
    def entry(v):
        return {"mean_within_group_std": {"point": v}}
    gs = {"group_sizes": [2, 4, 8], "results": {"all": {"K2": entry(0.3), "K4": entry(0.45), "K8": entry(0.6)}}}
    out = std_bias(gs)
    assert out["reference_K"] == 8
    assert out["per_K"]["K8"]["expected_ratio_to_K8"] == pytest.approx(1.0)
    assert out["per_K"]["K2"]["observed_ratio_to_K8"] == pytest.approx(0.5)
    assert out["per_K"]["K2"]["expected_ratio_to_K8"] == pytest.approx(
        math.sqrt(2 / math.pi) * math.sqrt(0.5) / (c4(8) * math.sqrt(7 / 8)))


def test_qualitative_selects_uninformative_at_k8_and_dedups_texts():
    by_prompt = fake_cache({"flat": [0.5] * 8, "varied": [0.1 * j for j in range(8)]})
    for j, r in enumerate(by_prompt["0"]):
        r["completion"] = "same text " * 50 if j < 6 else f"other {j % 2}"
        r["completion_tokens"] = 100 + j
    for r in by_prompt["1"]:
        r["completion"], r["completion_tokens"] = "x", 1
    per_prompt = [{"prompt_id": "flat", "bin": "bin1_low_reward"}, {"prompt_id": "varied", "bin": "bin3_high_reward"}]
    out = qualitative(by_prompt, per_prompt)
    assert [p["prompt_id"] for p in out] == ["flat"]
    p = out[0]
    assert p["tertile"] == "bin1_low_reward" and p["rewards"] == [0.5] * 8
    assert not p["all_texts_identical"] and p["n_distinct_texts"] == 3
    assert [d["generation_indices"] for d in p["distinct_completions"]] == [[0, 1, 2, 3, 4, 5], [6], [7]]
    assert len(p["distinct_completions"][0]["first_chars"]) == 300
    assert p["completion_tokens"] == list(range(100, 108)) and all(p["terminated_with_eos"])
