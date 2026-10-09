"""Task 5 evaluation code: mocks and synthetic strings only (no model weights, no Task 5 data texts)."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pytest
import torch

from task5_feedback import protocol as P
from task5_feedback.compare_feedback import (
    agreement_block, build_summary, category_block, drop_section, judge_outcome, swap_block, win_rate_block, win_scores,
)
from task5_feedback.judge_wrapper import judge_call, parity_check, released_swap
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.score_perturbations import verifier_pairs


# ---------------------------------------------------------------- compliance and failure types

@pytest.mark.parametrize("text,gold,truncated,expected", [
    ("so 3 + 4 = 7\n#### 7", "7", False, "correct"),
    ("#### 1,800", "1800", False, "correct"),
    ("we had 7 apples, then 9\n#### 9", "7", False, "wrong_compliant_gold_present"),
    ("total 12\n#### 12", "7", False, "wrong_compliant_gold_absent"),
    ("the answer is 7", "7", False, "noncompliant_ended"),
    ("the answer is 7 and then we keep going", "7", True, "noncompliant_truncated"),
    ("#### 7 ... more text cut at the cap", "7", True, "correct"),
    ("#### 8 ... cut at the cap", "7", True, "wrong_compliant_gold_absent"),
    ("#### $7", "7", False, "noncompliant_ended"),
    ("#### 7 first, then #### 8", "7", False, "wrong_compliant_gold_present"),
    ("x = 5-7 so\n#### 2", "7", False, "wrong_compliant_gold_present"),
    ("17 and 70 and 7.5\n#### 2", "7", False, "wrong_compliant_gold_absent"),
])
def test_classify_response(text, gold, truncated, expected):
    out = P.classify_response(text, gold, truncated)
    assert out["failure_type"] == expected
    assert out["compliant"] == (expected not in ("noncompliant_truncated", "noncompliant_ended"))
    assert out["correct"] == (expected == "correct")


def test_gold_mentioned_reads_commas_and_signs():
    assert P.gold_mentioned("costs 1,200 dollars", "1200")
    assert P.gold_mentioned("drop of -18 degrees", "18")
    assert P.gold_mentioned("drop of -18 degrees", "-18")
    assert not P.gold_mentioned("118 and 181", "18")


# ---------------------------------------------------------------- judge mocks

class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, rule):
        self.rule, self.shown = rule, None

    def apply_chat_template(self, messages, return_tensors=None, add_generation_prompt=True):
        text = messages[0]["content"]
        a = text.split("Candidate A:\n")[1].split("\n\nCandidate B:")[0]
        b = text.split("Candidate B:\n")[1].split("\n\nPreference:")[0]
        self.shown = (a, b)
        return torch.zeros((1, 5 + len(text) % 7), dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        return self.rule(*self.shown)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.calls = []

    def generate(self, ids, **kw):
        self.calls.append(kw)
        return torch.cat([ids, torch.ones((1, 2), dtype=torch.long)], dim=1)


def fake_judge(rule, tmp_path):
    j = object.__new__(PairwiseAIJudge)
    j.cfg = {"ai_judge_model": "fake/judge"}
    j.cache, j.cache_path = {}, tmp_path / "released_cache.json"
    j.tokenizer, j.model = FakeTokenizer(rule), FakeModel()
    return j


def content_rule(a, b):
    """Prefers the candidate containing 'good'; ' tie ' if both or neither."""
    ga, gb = "good" in a, "good" in b
    return " A" if ga and not gb else (" b." if gb and not ga else " Tie")


def pairs_corpus(n=60):
    rng = random.Random(0)
    out = []
    for i in range(n):
        a = f"resp {i} {'good' if rng.random() < 0.5 else 'bad'} {rng.random()}"
        b = f"resp {i}b {'good' if rng.random() < 0.5 else 'bad'} {rng.random()}"
        out.append((f"problem {i}", a, b))
    return out


def test_wrapper_matches_released_compare(tmp_path):
    judge = fake_judge(content_rule, tmp_path)
    for problem, a, b in pairs_corpus():
        rec = judge_call(judge, problem, a, b)
        assert rec["label"] == PairwiseAIJudge.compare(judge, problem, a, b)
        assert rec["released_order"] and rec["parse_matched"]
        assert rec["physical_order"] == ("BA" if released_swap(judge, problem, a, b) else "AB")
    # Same generate arguments as the release.
    assert judge.model.calls[0] == judge.model.calls[1] == {
        "max_new_tokens": 4, "do_sample": False, "pad_token_id": 99, "eos_token_id": 99}


def test_wrapper_parse_failure_is_tie_and_flagged(tmp_path):
    judge = fake_judge(lambda a, b: "Candidate", tmp_path)
    rec = judge_call(judge, "p", "x", "y")
    assert rec["label"] == "TIE" and not rec["parse_matched"] and rec["raw"] == "Candidate"
    assert PairwiseAIJudge.compare(judge, "p", "x", "y") == "TIE"


def test_parity_check_leaves_cache_untouched(tmp_path):
    judge = fake_judge(content_rule, tmp_path)
    judge.cache = {"sentinel": "A"}
    rec = judge_call(judge, "p", "good one", "bad one")
    out = parity_check(judge, "p", "good one", "bad one", rec["label"])
    assert out["match"] and judge.cache == {"sentinel": "A"} and judge.cache_path == tmp_path / "released_cache.json"
    assert not (tmp_path / "released_cache.json").exists()


def test_swap_gives_opposite_physical_order(tmp_path):
    judge = fake_judge(content_rule, tmp_path)
    for problem, a, b in pairs_corpus():
        r, s = judge_call(judge, problem, a, b), judge_call(judge, problem, a, b, swap_order=True)
        assert r["physical_order"] != s["physical_order"] and not s["released_order"]
        shown_first = judge.tokenizer.shown[0]
        assert shown_first == (a if s["physical_order"] == "AB" else b)
        assert r["label"] == s["label"]  # content-based judge: unchanged
    # A judge that always picks the first slot flips its content-level decision under the swap.
    biased = fake_judge(lambda a, b: "A", tmp_path)
    for problem, a, b in pairs_corpus(10):
        r, s = judge_call(biased, problem, a, b), judge_call(biased, problem, a, b, swap_order=True)
        assert {r["label"], s["label"]} == {"A", "B"}


# ---------------------------------------------------------------- diagnostic pairs and rates

def synthetic_groups(n=20):
    groups = {}
    for pid in range(n):
        g = {}
        for v in [P.ANCHOR, *P.CATEGORIES.values()]:
            correct = v in ("clean_correct", "corrupt_reasoning_correct_final", "persuasive_filler_correct")
            g[v] = {"problem_id": pid, "question": f"q{pid}", "gold_final": 10 + pid, "variant_type": v,
                    "response": f"{v} text\n#### {10 + pid if correct else 1}", "expected_exact_reward": float(correct)}
        groups[str(pid)] = g
    return groups


def test_build_pairs():
    pairs = P.build_pairs(synthetic_groups())
    assert len(pairs) == P.N_PAIRS == 80
    assert len({p["pair_id"] for p in pairs}) == 80
    for p in pairs:
        assert p["better_variant"] == "clean_correct"
        assert p["other_variant"] == P.CATEGORIES[p["category"]]
        assert p["better_response"].startswith("clean_correct") and p["other_response"].startswith(p["other_variant"])
    assert [p["category"] for p in pairs[:4]] == list(P.CATEGORIES)


def test_verifier_pairs_and_s_rates():
    vp = verifier_pairs(P.build_pairs(synthetic_groups()))
    by = {c: [r["preference"] for r in vp if r["category"] == c] for c in P.CATEGORIES}
    assert set(by["reasoning_corrupt"]) == {"tie"} and set(by["filler"]) == {"tie"}
    assert set(by["wrong_final"]) == {"better"} and set(by["gold_distractor"]) == {"better"}
    blk = category_block(by["reasoning_corrupt"], seed=1, n_boot=20)
    assert blk["better_rate"]["point"] == 0.0 and blk["tie_rate"]["point"] == 1.0 and blk["resolution_points"] == 5.0


def test_preference_mappings():
    assert P.reward_preference(1, 0) == "better" and P.reward_preference(1, 1) == "tie" and P.reward_preference(0, 1) == "wrong"
    assert P.judge_preference("A", True) == "better" and P.judge_preference("B", True) == "wrong"
    assert P.judge_preference("TIE", True) == "tie" and P.judge_preference("TIE", False) == "parse_failure"


def test_category_block_arithmetic():
    prefs = ["better"] * 12 + ["tie"] * 3 + ["wrong"] * 4 + ["parse_failure"]
    blk = category_block(prefs, seed=1, n_boot=20)
    assert blk["better_rate"]["point"] == pytest.approx(0.60)
    assert blk["tie_rate"]["point"] == pytest.approx(0.15)
    assert blk["wrong_rate"]["point"] == pytest.approx(0.20)
    assert blk["parse_failure_rate"]["point"] == pytest.approx(0.05)


def test_swap_block():
    rel = [{"pair_id": i, "preference": p, "physical_order": "AB"} for i, p in enumerate(["better", "wrong", "tie", "better"])]
    swp = [{"pair_id": i, "preference": p, "physical_order": "BA"} for i, p in enumerate(["better", "better", "tie", "wrong"])]
    b = swap_block(rel, swp)
    assert b["unchanged"] == 2 and b["unchanged_rate"] == 0.5
    assert b["released_counts"]["better"] == 2 and b["swapped_counts"]["better"] == 2 and b["swapped_counts"]["wrong"] == 1


# ---------------------------------------------------------------- win rate and agreement A

def rec(label, matched=True, a_correct=False, b_correct=False):
    return {"label": label, "parse_matched": matched, "a_correct": a_correct, "b_correct": b_correct}


def test_win_rate_counts_parse_failures_as_ties():
    recs = [rec("A"), rec("A"), rec("B"), rec("TIE"), rec("TIE", matched=False)]
    assert [judge_outcome(r) for r in recs] == ["win", "win", "loss", "tie", "parse_failure"]
    assert win_scores(recs).tolist() == [1, 1, 0, 0.5, 0.5]
    blk = win_rate_block(recs, seed=1, n_boot=20)
    assert blk["win_rate"]["point"] == pytest.approx(3 / 5)
    assert blk["counts"] == {"win": 2, "loss": 1, "tie": 1, "parse_failure": 1}
    assert blk["ties_incl_parse_failures"] == 2 and blk["resolution_points"] == 20.0


def test_agreement_A():
    recs = [
        rec("A", a_correct=True), rec("B", a_correct=True), rec("TIE", a_correct=True),   # trained correct only
        rec("B", b_correct=True), rec("TIE", matched=False, b_correct=True),            # sft correct only
        rec("A", a_correct=True, b_correct=True), rec("TIE", a_correct=True, b_correct=True),
        rec("B"),
    ]
    a = agreement_block(recs)
    one = a["exactly_one_correct"]
    assert one["n"] == 5 and one["judge_prefers_correct"] == 2 and one["judge_prefers_wrong"] == 1
    assert one["judge_tie"] == 1 and one["parse_failure"] == 1 and one["agreement_rate"] == pytest.approx(0.4)
    assert one["n_correct_is_trained"] == 3
    assert a["both_correct"]["n"] == 2 and a["both_correct"]["judge_counts_trained_side"] == {"win": 1, "loss": 0, "tie": 1, "parse_failure": 0}
    assert a["both_wrong"]["n"] == 1 and a["both_wrong"]["judge_counts_trained_side"]["loss"] == 1


# ---------------------------------------------------------------- drops

def test_drop_arithmetic():
    def rows(acc, comp, length, n):
        k_acc, k_comp = round(acc * n), round(comp * n)
        return [{"correct": i < k_acc, "compliant": i < k_comp, "token_length": length} for i in range(n)]
    gens = {}
    for pol, (ga, sa) in {"sft": (0.5, 0.4), "rlvr": (0.7, 0.45), "rlaif": (0.6, 0.6)}.items():
        gens[("gsm", pol)] = rows(ga, 0.9, 200, 300)
        gens[("transfer", pol)] = rows(sa, 0.8, 150, 100)
    judges = {
        "gsm": (None, [{"comparison": c, "label": "A" if i < 180 else "B"} for c in ("rlvr_vs_sft", "rlaif_vs_sft") for i in range(300)]),
        "transfer": (None, [{"comparison": c, "label": "A" if i < 50 else "TIE"} for c in ("rlvr_vs_sft", "rlaif_vs_sft") for i in range(100)]),
    }
    d = drop_section(gens, judges, seed=1, n_boot=20)
    assert d["rlvr"]["accuracy"]["point"] == pytest.approx(0.45 - 0.7)
    assert d["rlaif"]["accuracy"]["point"] == pytest.approx(0.0)
    assert d["sft"]["format_compliance"]["point"] == pytest.approx(-0.1)
    assert d["sft"]["mean_length"]["point"] == pytest.approx(-50)
    assert d["rlvr"]["win_rate_vs_sft"]["point"] == pytest.approx(0.75 - 0.6)
    assert "win_rate_vs_sft" not in d["sft"]


# ---------------------------------------------------------------- qualitative rule

def qrow(pid, label, matched=True):
    return {"problem_id": pid, "pair_id": f"{pid}:x", "label": label, "parse_matched": matched}


def test_qualitative_rule_branches():
    assert P.qualitative_choice([qrow(5, "A"), qrow(9, "B"), qrow(7, "B"), qrow(1, "TIE")]) == {
        "problem_id": 7, "pair_id": "7:x", "branch": "perturbed_preferred", "judge_label": "B", "judge_parse_matched": True}
    out = P.qualitative_choice([qrow(5, "A"), qrow(9, "TIE", matched=False), qrow(3, "TIE")])
    assert (out["problem_id"], out["branch"]) == (3, "tie")
    out = P.qualitative_choice([qrow(5, "A"), qrow(2, "A")])
    assert (out["problem_id"], out["branch"]) == (2, "lowest_id")


# ---------------------------------------------------------------- end to end on synthetic result files

def write_synthetic_results(d, rng):
    commit = {"commit": "abc", "dirty": False}
    eff = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20, "repetition_penalty": 1.1, "max_new_tokens": 512}
    gens = {}
    for ds, n in P.DATASET_ROWS.items():
        path = "data/gsm8k_eval.jsonl" if ds == "gsm" else "data/math_transfer_eval.jsonl"
        ids = [f"{ds}:{i}" for i in range(n)]
        for pol in P.POLICIES:
            rows = []
            for i, pid in enumerate(ids):
                ft = P.FAILURE_TYPES[rng.integers(0, 5)]
                rows.append({"index": i, "prompt_id": pid, "token_length": int(rng.integers(50, 513)), "truncated": ft == "noncompliant_truncated",
                             "correct": ft == "correct", "compliant": not ft.startswith("noncompliant"), "failure_type": ft})
            gens[(ds, pol)] = rows
            adapter = None if pol == "sft" else f"checkpoints/{pol}_policy"
            meta = {"status": "completed", "smoke": False, "git": commit, "seed": P.SEED,
                    "adapter": adapter or "none (base model)",
                    "adapter_sha256": P.FILE_SHA256[P.adapter_file(adapter)] if adapter else None,
                    "data": {"path": path, "sha256": P.FILE_SHA256[path], "prompt_ids": ids, "prompt_tokens": {"max": 200},
                             "n_prompts_over_cap": 0, "n_skipped": 0, "n_truncated_prompts": 0},
                    "settings": {"samples_per_prompt": 1, "gen_batch_size": 16, "effective_generation": eff},
                    "metrics": {"n_responses": n}, "timing": {"seconds_per_prompt": 1.0}}
            (d / f"gen_{ds}_{pol}.json").write_text(json.dumps(meta))
            (d / f"gen_{ds}_{pol}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        jr = []
        for c in ("rlvr_vs_sft", "rlaif_vs_sft"):
            pol = c.split("_")[0]
            for i, pid in enumerate(ids):
                lab = ["A", "B", "TIE"][rng.integers(0, 3)]
                jr.append({"comparison": c, "prompt_id": pid, "index": i, "a_policy": pol, "b_policy": "sft",
                           "a_correct": gens[(ds, pol)][i]["correct"], "b_correct": gens[(ds, "sft")][i]["correct"],
                           "label": lab, "physical_label": lab, "parse_matched": rng.random() > 0.05,
                           "physical_order": "AB", "released_order": True})
        jm = {"status": "completed", "smoke": False, "git": commit, "n_calls_done": len(jr),
              "parity": [{"match": True}], "parity_all_match": True, "settings": {}, "timing": {}}
        (d / f"judge_{ds}.json").write_text(json.dumps(jm))
        (d / f"judge_{ds}.jsonl").write_text("\n".join(json.dumps(r) for r in jr))
    groups = synthetic_groups()
    pairs = P.build_pairs(groups)
    from task5_feedback.score_perturbations import verifier_rows
    for order in ("released", "swapped"):
        dr = []
        for p in pairs:
            lab = ["A", "B", "TIE"][rng.integers(0, 3)]
            matched = rng.random() > 0.1
            phys = "AB" if (hashlib.sha256(p["pair_id"].encode()).digest()[0] % 2 == 0) == (order == "released") else "BA"
            dr.append({"pair_id": p["pair_id"], "problem_id": p["problem_id"], "category": p["category"], "label": lab if matched else "TIE",
                       "physical_label": lab, "parse_matched": matched, "physical_order": phys, "released_order": order == "released",
                       "preference": P.judge_preference(lab if matched else "TIE", matched)})
        dm = {"status": "completed", "smoke": False, "git": commit,
              "data": {"path": "data/task5_controlled_reward_diagnostics.jsonl",
                       "sha256": P.FILE_SHA256["data/task5_controlled_reward_diagnostics.jsonl"], "n_pairs_judged": 80},
              "verifier_responses": verifier_rows(groups), "verifier_pairs": verifier_pairs(pairs),
              "parity": [{"match": True}] if order == "released" else [], "parity_all_match": True if order == "released" else None,
              "timing": {}}
        (d / f"diagnostics_{order}.json").write_text(json.dumps(dm))
        (d / f"diagnostics_{order}.jsonl").write_text("\n".join(json.dumps(r) for r in dr))
    return gens


def test_build_summary_end_to_end(tmp_path):
    gens = write_synthetic_results(tmp_path, np.random.default_rng(0))
    summary, rows = build_summary(tmp_path, seed=1, n_boot=30)
    bad = [c for c in summary["protocol_checks"] if not c["ok"]]
    assert not bad, bad
    acc = np.mean([r["correct"] for r in gens[("gsm", "rlvr")]])
    assert summary["policies"]["gsm"]["rlvr"]["accuracy"]["point"] == pytest.approx(acc)
    diff = summary["policies"]["transfer"]["paired_differences"]["rlvr_minus_sft"]["mean_length"]["point"]
    assert diff == pytest.approx(np.mean([r["token_length"] for r in gens[("transfer", "rlvr")]])
                                 - np.mean([r["token_length"] for r in gens[("transfer", "sft")]]))
    dg = summary["diagnostics"]
    assert dg["S_reason"]["verifier"]["point"] == 0.0 and dg["S_outcome"]["verifier"]["point"] == 1.0
    assert set(summary["qualitative_candidates"]) == set(P.QUALITATIVE_CATEGORIES)
    assert rows and all("section" in r for r in rows)


def test_summary_detects_reordered_prompts(tmp_path):
    write_synthetic_results(tmp_path, np.random.default_rng(1))
    p = tmp_path / "gen_gsm_rlaif.jsonl"
    lines = p.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    p.write_text("\n".join(lines))
    summary, _ = build_summary(tmp_path, seed=1, n_boot=5)
    failed = {c["check"] for c in summary["protocol_checks"] if not c["ok"]}
    assert "gsm: identical prompt order across policies" in failed
