"""Mac-only checks of the Task 2 PPO pipeline pieces (tiny random Qwen2 models, float32, CPU).

1. One rollout-to-loss step: score_rollout -> rewards/GAE -> ppo_update on a tiny policy and critic.
2. Prompt-sequence identity across runs with different eps/beta (and a shorter run is a prefix).
3. Epoch-1 clip fraction is 0 (dropout disabled, same per-row shapes as the rollout scoring).
4. Affected-fraction formula on a hand example.
5. Stability statistic is non-negative and 0 at rho = 1.
6. Explained variance: 1 for perfect values, 0 for a constant equal to the mean return.
7. Value positions aligned with log-prob positions (V(s_t) and log pi(a_t|s_t) see tokens < t only).
8. EOS-penalty reward assembly.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification

from common.generation import response_token_logprobs
from common.metrics import masked_mean
from common.models import make_lora_config, make_value_lora_config, token_values, trainable_parameters, value_parameter_groups
from common.rollouts import plan_prompts, pooled_token_metrics
from task2_ppo.continue_train import critic_values, policy_logprobs, ppo_update, score_rollout
from task2_ppo.ppo_utils import (
    affected_fraction,
    build_targets,
    check_clip_identities,
    clip_study,
    critic_head_to_fp32,
    disable_dropout,
    effective_terminal_rewards,
    explained_variance,
    response_slice,
    stability_statistic,
)

LORA = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
CFG = {"lora": LORA, "value_lora": LORA}
PAD, EOS = 0, 2


def tiny_config(**kw):
    return Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, pad_token_id=PAD, **kw)


def tiny_policy(seed=0):
    torch.manual_seed(seed)
    model = get_peft_model(Qwen2ForCausalLM(tiny_config()), make_lora_config(CFG))
    with torch.no_grad():  # non-zero LoRA B so the policy differs from the reference
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.05)
    model.train()
    return model


def tiny_critic(seed=1):
    torch.manual_seed(seed)
    model = get_peft_model(Qwen2ForSequenceClassification(tiny_config(num_labels=1)), make_value_lora_config(CFG))
    critic_head_to_fp32(model)
    model.train()
    return model


def tiny_batch(seed=2):
    """3 responses to one 5-token prompt: EOS at t=3, no EOS (6 tokens), EOS at t=5."""
    g = torch.Generator().manual_seed(seed)
    P, R = 5, 6
    prompt = torch.randint(3, 1000, (1, P), generator=g).repeat(3, 1)
    resp = torch.randint(3, 1000, (3, R), generator=g)
    resp[0, 3], resp[0, 4:] = EOS, PAD
    resp[2, 5] = EOS
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6, [1] * 6], dtype=torch.float32)
    seq = torch.cat([prompt, resp], dim=1)
    return {
        "sequences": seq,
        "attention_mask": torch.ones_like(seq),
        "prompt_width": P,
        "response_ids": resp,
        "response_mask": mask,
        "terminated_with_eos": [True, False, True],
    }


# ---------------------------------------------------------------- 1 and 3. rollout-to-loss step

def test_rollout_to_loss_step_and_epoch1_clip_fraction_zero():
    policy, critic = tiny_policy(), tiny_critic()
    assert disable_dropout(policy) > 0 and disable_dropout(critic) > 0
    batch = tiny_batch()
    mask = batch["response_mask"]
    batch.update(score_rollout(policy, critic, batch))
    assert batch["old_logp"].shape == mask.shape == batch["values"].shape
    assert float((batch["old_logp"] - batch["ref_logp"]).abs().max()) > 0  # policy != reference

    raw = torch.tensor([0.5, 2.0, -1.0])
    eff = effective_terminal_rewards(raw, batch["terminated_with_eos"], 1.0)
    t = build_targets(eff, batch["old_logp"], batch["ref_logp"], batch["values"], mask, 0.1, 1.0, 0.95)
    batch["advantages"], batch["returns"] = t["advantages"], t["returns"]

    p_opt = torch.optim.AdamW(trainable_parameters(policy), lr=1e-2)
    v_opt = torch.optim.AdamW(value_parameter_groups(critic, 1e-2, 1e-2), weight_decay=0.0)
    p_before = [p.detach().clone() for p in trainable_parameters(policy)]
    v_before = [p.detach().clone() for p in trainable_parameters(critic)]
    out = ppo_update(policy, critic, p_opt, v_opt, batch, eps=0.2, value_coef=0.5, ppo_epochs=2, max_grad_norm=1.0)

    e1, e2 = out["epochs"]
    assert e1["clip_fraction"] == 0.0
    assert e1["max_abs_log_ratio_before_step"] < 1e-5
    # At rho = 1 both surrogate branches equal A, so the loss is -mean(A) over valid tokens.
    assert e1["policy_loss"] == pytest.approx(-float(masked_mean(t["advantages"], mask)), abs=1e-5)
    # Before any critic step, returns - V = A on valid tokens.
    assert e1["value_loss"] == pytest.approx(float(masked_mean(t["advantages"] ** 2, mask)), rel=1e-5)
    for e in (e1, e2):
        assert math.isfinite(e["policy_loss"]) and math.isfinite(e["value_loss"])
        assert e["policy_grad_norm_before_clip"] > 0 and e["value_grad_norm_before_clip"] > 0
        assert e["policy_step_taken"] and e["value_step_taken"]
    assert out["n_nonfinite_steps"] == 0
    assert any(not torch.equal(a, b) for a, b in zip(p_before, trainable_parameters(policy)))
    assert any(not torch.equal(a, b) for a, b in zip(v_before, trainable_parameters(critic)))

    with torch.no_grad():
        new = torch.cat([policy_logprobs(policy, batch, i) for i in range(3)]) * mask
    assert out["stability_statistic"] == pytest.approx(stability_statistic(new, batch["old_logp"], mask), rel=1e-9)
    assert out["stability_statistic"] > 0


def test_epoch1_clip_fraction_assertion_fires_when_old_logp_is_stale():
    policy, critic = tiny_policy(), tiny_critic()
    disable_dropout(policy), disable_dropout(critic)
    batch = tiny_batch()
    batch.update(score_rollout(policy, critic, batch))
    batch["old_logp"] = batch["old_logp"] - 1.0 * batch["response_mask"]  # rho = e on every token
    t = build_targets(torch.zeros(3), batch["old_logp"], batch["ref_logp"], batch["values"], batch["response_mask"], 0.1, 1.0, 0.95)
    batch["advantages"], batch["returns"] = t["advantages"], t["returns"]
    p_opt = torch.optim.AdamW(trainable_parameters(policy), lr=1e-3)
    v_opt = torch.optim.AdamW(value_parameter_groups(critic, 1e-3, 1e-3), weight_decay=0.0)
    with pytest.raises(AssertionError):
        ppo_update(policy, critic, p_opt, v_opt, batch, eps=0.2, value_coef=0.5, ppo_epochs=1, max_grad_norm=1.0)


def test_critic_head_is_float32_and_dropout_disabled():
    torch.manual_seed(0)
    m = get_peft_model(Qwen2ForSequenceClassification(tiny_config(num_labels=1)).to(torch.float16), make_value_lora_config(CFG))
    info = critic_head_to_fp32(m)
    assert info["head_dtype_before"] == ["torch.float16"] and info["head_dtype_after"] == ["torch.float32"]
    assert {str(p.dtype) for p in trainable_parameters(m)} == {"torch.float32"}
    assert {str(p.dtype) for p in m.parameters() if not p.requires_grad} == {"torch.float16"}
    assert disable_dropout(m) > 0 and disable_dropout(m) == 0
    assert all(mod.p == 0.0 for mod in m.modules() if isinstance(mod, torch.nn.Dropout))


# ---------------------------------------------------------------- 2. prompt sequence identity

class WordTokenizer:
    """One token per whitespace word plus 3 template tokens."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(3 + sum(len(m["content"].split()) for m in messages)))


def prompt_rows():
    g = torch.Generator().manual_seed(5)
    lengths = torch.randint(1, 15, (60,), generator=g).tolist()
    return [{"prompt_id": f"p{i}", "messages": [{"role": "user", "content": " ".join(["w"] * n)}]} for i, n in enumerate(lengths)]


def test_prompt_sequence_identical_across_eps_and_beta():
    rows, tok = prompt_rows(), WordTokenizer()
    base = {"seed": 6304, "max_prompt_length": 12, "updates": 8}
    a = plan_prompts({**base, "clip_epsilon": 0.05, "kl_beta": 0.10}, tok, rows)
    torch.manual_seed(123)
    _ = torch.rand(1000)  # the global RNG must not matter
    b = plan_prompts({**base, "clip_epsilon": 0.50, "kl_beta": 0.0}, tok, rows)
    std = plan_prompts({**base, "updates": 20, "clip_epsilon": 0.20, "kl_beta": 0.10}, tok, rows)
    assert a["prompt_ids"] == b["prompt_ids"]
    assert std["prompt_ids"][:8] == a["prompt_ids"]
    assert len(set(std["prompt_ids"])) == 20
    # rule P: every planned prompt fits; excluded ones are exactly the long ones
    fits = {r["prompt_id"] for r in rows if 3 + len(r["messages"][0]["content"].split()) <= 12}
    assert set(std["prompt_ids"]) <= fits
    assert {e["prompt_id"] for e in std["filter"]["excluded"]} == {r["prompt_id"] for r in rows} - fits
    assert std["filter"]["n_kept"] + std["filter"]["n_excluded"] == len(rows)


# ---------------------------------------------------------------- 4. affected fraction

def test_affected_fraction_hand_example():
    ratio = torch.tensor([[1.3, 0.7, 1.3, 0.7, 1.1, 0.9, 2.0]])
    adv = torch.tensor([[1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)
    # affected: (1.3, +) and (0.7, -); clipped but not affected: (1.3, -) and (0.7, +); last token masked
    assert float(affected_fraction(ratio, adv, mask, 0.2)) == pytest.approx(2 / 6)
    res = clip_study(torch.log(ratio), torch.zeros_like(ratio), adv, mask, [0.2])[0]
    assert res["clip_fraction"] == pytest.approx(4 / 6)
    assert res["affected_fraction"] == pytest.approx(2 / 6)
    # clipped: 1.2, -0.8, -1.3, 0.7, 1.1, -0.9 ; unclipped: 1.3, -0.7, -1.3, 0.7, 1.1, -0.9
    assert res["clipped_surrogate"] == pytest.approx((1.2 - 0.8 - 1.3 + 0.7 + 1.1 - 0.9) / 6, abs=1e-6)
    assert res["unclipped_surrogate"] == pytest.approx((1.3 - 0.7 - 1.3 + 0.7 + 1.1 - 0.9) / 6, abs=1e-6)


def test_clip_identities_hold_on_random_batch_and_violation_raises():
    g = torch.Generator().manual_seed(9)
    new = 0.3 * torch.randn(4, 50, generator=g, dtype=torch.float64)
    adv = torch.randn(4, 50, generator=g, dtype=torch.float64)
    mask = (torch.rand(4, 50, generator=g) > 0.2).double()
    res = clip_study(new, torch.zeros_like(new), adv, mask, [0.05, 0.20, 0.50])
    assert all(check_clip_identities(res).values())
    bad = [dict(r) for r in res]
    bad[2]["clip_fraction"] = bad[0]["clip_fraction"] + 0.1
    with pytest.raises(AssertionError):
        check_clip_identities(bad)


# ---------------------------------------------------------------- 5. stability statistic

def test_stability_statistic_zero_at_rho_one_and_non_negative():
    g = torch.Generator().manual_seed(3)
    old = -3 * torch.rand(5, 20, generator=g)
    mask = (torch.rand(5, 20, generator=g) > 0.3).float()
    assert stability_statistic(old.clone(), old, mask) == 0.0
    for scale in (1e-4, 1e-2, 1.0):
        new = old + scale * torch.randn(5, 20, generator=g)
        s = stability_statistic(new, old, mask)
        d = (new - old).double()
        assert s >= 0.0
        assert s == pytest.approx(float(((torch.exp(d) - 1 - d) * mask).sum() / mask.sum()), rel=1e-6, abs=1e-15)


# ---------------------------------------------------------------- 6. explained variance

def test_explained_variance_perfect_and_constant():
    g = torch.Generator().manual_seed(4)
    ret = torch.randn(3, 10, generator=g)
    mask = torch.ones(3, 10)
    mask[0, 7:] = 0
    ret_pad = ret.clone()
    ret_pad[0, 7:] = 1e6  # padding must be ignored
    assert explained_variance(ret_pad, ret_pad, mask) == pytest.approx(1.0)
    const = torch.full_like(ret, float(ret[mask.bool()].mean()))
    assert explained_variance(ret_pad, const, mask) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(explained_variance(torch.ones(2, 3), torch.zeros(2, 3), torch.ones(2, 3)))


# ---------------------------------------------------------------- 7. value / log-prob alignment

def test_value_and_logprob_positions_aligned():
    policy, critic = tiny_policy(), tiny_critic()
    policy.eval(), critic.eval()
    b = tiny_batch()
    P, R = b["prompt_width"], b["response_ids"].shape[1]
    seq = b["sequences"][1:2]  # the full-length row
    with torch.no_grad():
        v = response_slice(token_values(critic, seq, torch.ones_like(seq)), P, R)
        lp = response_token_logprobs(policy, seq, torch.ones_like(seq), P, b["response_ids"][1:2])[0]
        assert torch.allclose(critic_values(critic, b, 1), v)
        assert torch.allclose(policy_logprobs(policy, b, 1), lp)
        for t in range(R):
            prefix = seq[:, : P + t]  # tokens before response token t
            v_t = token_values(critic, prefix, torch.ones_like(prefix))[0, -1]
            logits = policy(input_ids=prefix, attention_mask=torch.ones_like(prefix)).logits[0, -1]
            lp_t = F.log_softmax(logits.float(), dim=-1)[b["response_ids"][1, t]]
            assert float(v[0, t]) == pytest.approx(float(v_t), abs=1e-5)
            assert float(lp[0, t]) == pytest.approx(float(lp_t), abs=1e-5)


# ---------------------------------------------------------------- 8. EOS penalty and reward assembly

def test_eos_penalty_reward_assembly():
    raw = torch.tensor([1.5, 2.0, -0.5], dtype=torch.float64)
    eff = effective_terminal_rewards(raw, [True, False, True], 1.0)
    assert eff.tolist() == [1.5, 1.0, -0.5]
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float64)
    old = torch.tensor([[-1.0, -2.0, 9.0], [-1.0, -1.5, -0.5], [-0.2, 9.0, 9.0]], dtype=torch.float64)
    ref = torch.tensor([[-1.5, -1.0, 9.0], [-1.0, -2.0, -1.0], [-0.7, 9.0, 9.0]], dtype=torch.float64)
    values = torch.tensor([[0.3, 0.2, 7.0], [0.1, 0.0, -0.1], [0.4, 7.0, 7.0]], dtype=torch.float64)
    t = build_targets(eff, old, ref, values, mask, 0.1, 1.0, 0.95)
    kl = -0.1 * (old - ref) * mask
    expected = kl.clone()
    expected[0, 1] += 1.5
    expected[1, 2] += 1.0
    expected[2, 0] += -0.5
    assert torch.allclose(t["rewards"], expected)
    assert torch.all(t["rewards"][mask == 0] == 0)
    assert torch.all(t["values"][mask == 0] == 0)
    assert torch.allclose(t["returns"] * mask, (t["advantages"] + t["values"]) * mask)
    # row 2 has one valid token: A = r - V (bootstrap after the last valid token is 0)
    assert float(t["advantages"][2, 0]) == pytest.approx(float(expected[2, 0] - 0.4))


def test_pooled_token_metrics_are_token_level():
    pol = [[-1.0, -2.0], [-0.5]]
    ref = [[-1.5, -1.0], [-1.0]]
    m = pooled_token_metrics(pol, ref)
    assert m["n_tokens"] == 3
    assert m["kl"] == pytest.approx((0.5 - 1.0 + 0.5) / 3)
    assert m["entropy"] == pytest.approx((1.0 + 2.0 + 0.5) / 3)
