"""Mac-only checks of the Task 1 pipeline pieces that need no pretrained model.

1. Rule A filter (skip prompts >= max_length, flag truncated responses) on synthetic rows,
   and its agreement with common.data.encode_prompt_response.
2. Two DataLoaders built with the same seed give the same order.
3. Gradient accumulation: 8 micro-batches of loss/8 equal one batch of 16; remainder step.
4. Evaluation aggregation on hand-made inputs: accuracy, per-stratum accuracy, length stats,
   word-limit compliance, pooled sampled KL.
"""
from __future__ import annotations

import math

import pytest
import torch

from common.data import encode_prompt_response, preference_responses, prompt_messages_from_preference
from common.metrics import word_limit_compliance
from task1_dpo.evaluate import compliance_summary, length_summary, pair_summary, pooled_kl, stratified_summary
from task1_dpo.train import accumulate_and_step, build_loader, filter_pairs, filter_report

TEMPLATE_TOKENS = 3  # fake chat template overhead per prompt


class FakeTokenizer:
    """One token per whitespace word; the chat template adds TEMPLATE_TOKENS tokens."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        words = sum(len(m["content"].split()) for m in messages)
        return list(range(1, 1 + TEMPLATE_TOKENS + words))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7] * len(text.split())}


def make_row(prompt_words, chosen_words, rejected_words, pid=None, stratum=None):
    prompt = " ".join(["p"] * prompt_words)
    row = {
        "prompt": prompt,
        "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": " ".join(["c"] * chosen_words)}],
        "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": " ".join(["r"] * rejected_words)}],
    }
    if pid is not None:
        row["prompt_id"] = pid
    if stratum is not None:
        row["length_stratum"] = stratum
    return row


# ---------------------------------------------------------------- 1. rule A filter

MAX_LEN = 10


def synthetic_rows():
    # prompt tokens = 3 + prompt_words; skip when >= 10, i.e. prompt_words >= 7
    return [
        make_row(2, 4, 5, pid="a", stratum="s1"),  # prompt 5, content budget 10-5-1 = 4: rejected truncated
        make_row(6, 1, 1, pid="b", stratum="s1"),  # prompt 9 < 10: kept, budget 0: both truncated
        make_row(7, 1, 1, pid="c", stratum="s2"),  # prompt 10 >= 10: skipped
        make_row(9, 1, 1, pid="d", stratum="s2"),  # prompt 12: skipped
        make_row(1, 2, 2, stratum="s2"),           # no prompt_id: ID is the row index 4; not truncated
    ]


def test_rule_a_skips_and_flags():
    rows = synthetic_rows()
    kept, recs, skipped = filter_pairs(FakeTokenizer(), rows, MAX_LEN)
    assert [r["id"] for r in recs] == ["a", "b", 4]
    assert [r["index"] for r in recs] == [0, 1, 4]
    assert [r["id"] for r in skipped] == ["c", "d"]
    assert [r["prompt_tokens"] for r in skipped] == [10, 12]
    assert kept == [rows[0], rows[1], rows[4]]
    assert [(r["chosen_truncated"], r["rejected_truncated"]) for r in recs] == [(False, True), (True, True), (False, False)]

    rep = filter_report("synthetic", MAX_LEN, recs, skipped)
    assert rep["n_input"] == 5 and rep["n_retained"] == 3 and rep["n_skipped_prompt_too_long"] == 2
    assert rep["n_pairs_any_response_truncated"] == 2 and rep["n_chosen_truncated"] == 1 and rep["n_rejected_truncated"] == 2
    assert rep["truncated_ids"] == ["a", "b"]
    assert rep["per_stratum"]["s1"]["n_retained"] == 2 and rep["per_stratum"]["s1"]["n_skipped_prompt_too_long"] == 0
    assert rep["per_stratum"]["s2"]["n_retained"] == 1 and rep["per_stratum"]["s2"]["n_skipped_prompt_too_long"] == 2


def test_rule_a_agrees_with_encoder():
    """Kept rows encode; skipped rows are exactly those encode_prompt_response rejects; flags match."""
    tok = FakeTokenizer()
    rows = synthetic_rows()
    _, recs, skipped = filter_pairs(tok, rows, MAX_LEN)
    for rec in recs:
        row = rows[rec["index"]]
        msgs = prompt_messages_from_preference(row)
        for resp, flag in zip(preference_responses(row), (rec["chosen_truncated"], rec["rejected_truncated"])):
            ids, mask = encode_prompt_response(tok, msgs, resp, MAX_LEN)
            raw = len(resp.split())
            assert (sum(mask) - 1 < raw) == flag  # encoded content excludes the appended EOS
            assert ids[-1] == tok.eos_token_id  # rule B: EOS kept even when truncated
    for rec in skipped:
        row = rows[rec["index"]]
        with pytest.raises(ValueError):
            encode_prompt_response(tok, prompt_messages_from_preference(row), "x", MAX_LEN)


# ---------------------------------------------------------------- 2. loader order

def loader_order(seed):
    items = [(i, None) for i in range(23)]
    loader = build_loader(items, 2, seed, collate_fn=lambda b: [i for i, _ in b])
    return [i for batch in loader for i in batch]


def test_same_seed_same_order():
    torch.manual_seed(1)
    a = loader_order(6304)
    torch.manual_seed(999)  # the global RNG must not affect the order
    _ = torch.rand(100)
    b = loader_order(6304)
    assert a == b
    assert sorted(a) == list(range(23))  # one epoch visits every row once
    assert a != list(range(23))  # actually shuffled
    assert loader_order(6305) != a


# ---------------------------------------------------------------- 3. gradient accumulation

def tiny_setup():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1).double()
    x = torch.randn(16, 4, dtype=torch.float64)
    y = torch.randn(16, 1, dtype=torch.float64)
    return model, x, y


def mse_step(model):
    def step(batch):
        xb, yb = batch
        loss = ((model(xb) - yb) ** 2).mean()
        return loss, {"n": xb.shape[0]}
    return step


def test_eight_micro_batches_equal_one_batch_of_16():
    model, x, y = tiny_setup()
    w0 = [p.detach().clone() for p in model.parameters()]

    # Reference: one batch of 16, mean loss.
    ref = torch.nn.Linear(4, 1).double()
    ref.load_state_dict(model.state_dict())
    ((ref(x) - y) ** 2).mean().backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]
    ref_norm = torch.sqrt(sum((g ** 2).sum() for g in ref_grads)).item()

    micro = [(x[i : i + 2], y[i : i + 2]) for i in range(0, 16, 2)]
    opt = torch.optim.SGD(model.parameters(), lr=1.0)  # update = -grad
    calls = []
    n_steps = accumulate_and_step(micro, mse_step(model), list(model.parameters()), opt, 8, 1.0e9,
                                  lambda step, stats, gn: calls.append((step, len(stats), gn)))
    assert n_steps == 1
    assert calls[0][0] == 1 and calls[0][1] == 8
    assert calls[0][2] == pytest.approx(ref_norm, rel=1e-12)
    for p, p0, g in zip(model.parameters(), w0, ref_grads):
        assert torch.allclose(p0 - p.detach(), g, atol=1e-12, rtol=1e-10)


def test_remainder_step_and_clipping():
    model, x, y = tiny_setup()
    micro = [(x[i : i + 2], y[i : i + 2]) for i in range(0, 16, 2)] + [(x[:2], y[:2]), (x[2:4], y[2:4])]
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    calls = []
    n_steps = accumulate_and_step(micro, mse_step(model), list(model.parameters()), opt, 8, 1.0e-6,
                                  lambda step, stats, gn: calls.append((step, len(stats))))
    assert n_steps == 2
    assert calls == [(1, 8), (2, 2)]


# ---------------------------------------------------------------- 4. evaluation aggregation

def pair(m, stratum=None):
    # policy_chosen_logp = m, everything else 0, so the DPO margin is m
    return {"policy_chosen_logp": m, "policy_rejected_logp": 0.0, "ref_chosen_logp": 0.0, "ref_rejected_logp": 0.0,
            "chosen_logratio": m, "rejected_logratio": 0.0, "m": m, "stratum": stratum}


def test_pair_summary_accuracy_loss_margin():
    beta = 0.1
    ms = [2.0, -1.0, 0.0, 3.0]
    s = pair_summary([pair(m) for m in ms], beta)
    assert s["n"] == 4
    assert s["preference_accuracy"] == 0.5  # m = 0 is not counted as correct
    assert s["mean_margin"] == pytest.approx(1.0)
    expect = sum(math.log1p(math.exp(-beta * m)) for m in ms) / 4
    assert s["dpo_loss"] == pytest.approx(expect, rel=1e-12)


def test_stratified_summary():
    recs = [pair(1.0, "a"), pair(-1.0, "a"), pair(2.0, "b"), pair(3.0, "b"), pair(-2.0, "c")]
    s = stratified_summary(recs, 0.1)
    assert s["overall"]["n"] == 5 and s["overall"]["preference_accuracy"] == pytest.approx(0.6)
    assert {k: (v["n"], v["preference_accuracy"]) for k, v in s["per_stratum"].items()} == {
        "a": (2, 0.5), "b": (2, 1.0), "c": (1, 0.0)}


def test_length_summary():
    s = length_summary([10, 20, 30, 40], [False, False, True, False])
    assert s["n"] == 4 and s["mean"] == 25.0
    assert s["std"] == pytest.approx(math.sqrt(125.0))  # population std (ddof=0)
    assert s["median"] == 25.0 and s["q25"] == 17.5 and s["q75"] == 32.5 and s["iqr"] == 15.0
    assert s["truncated_fraction"] == 0.25 and s["n_truncated"] == 1


def test_compliance_summary():
    prompt = "Answer in at most 3 words."
    texts = {"p1": ["one two three", "one two three four"], "p2": ["yes", "a b"]}
    recs = []
    for pid, ts in texts.items():
        for t in ts:
            recs.append({"prompt_id": pid, "limit": 3, "word_count": len(t.split()),
                         "compliant": word_limit_compliance(prompt, t)})
    s = compliance_summary(recs)
    assert s["n_responses"] == 4 and s["n_unparsed_limit"] == 0
    assert s["compliance"] == pytest.approx(0.75)
    assert s["per_prompt"]["p1"]["compliance"] == 0.5 and s["per_prompt"]["p2"]["compliance"] == 1.0
    assert s["per_prompt"]["p1"]["mean_word_count"] == 3.5


def test_pooled_kl_is_token_level():
    # log-ratios: response 1 = [4], response 2 = [0, 0, 0]
    kl = pooled_kl([[4.0], [-1.0, -2.0, -3.0]], [[0.0], [-1.0, -2.0, -3.0]])
    assert kl == pytest.approx(1.0)  # 4 / 4 tokens, not the per-response mean (4 + 0) / 2 = 2
