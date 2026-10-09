"""Task 5 summary: in-domain, diagnostic and transfer results combined (manual Section 5, Required Evidence).

Runs on the Mac from saved result files only (no model is loaded). Reads results/task5_feedback/
gen_{gsm,transfer}_{sft,rlvr,rlaif}, judge_{gsm,transfer}, diagnostics_{released,swapped} and writes
summary.json and summary.csv there. Never reads files under smoke/.

Uncertainty: 95% percentile bootstrap, protocol.N_BOOT resamples, a fresh
numpy default_rng(protocol.SEED) per quantity. Policy metrics and policy differences: resample prompts
(paired: the same prompt indices for both policies). SVAMP minus GSM8K drops: the two prompt sets are
different, so each is resampled independently (two-sample bootstrap). Diagnostic rates: resample the 20
problems (one pair per problem and category, so pairs of a category are resampled with their problem).
Every rate carries its resolution: one item = 100 / n points.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from task1_dpo.dataset_stats import describe
from task1_dpo.summarize import bootstrap, iid_resampler, write_csv
from task1_dpo.train import display_path, git_state
from task3_grpo.compare_normalization import two_sample_bootstrap
from task5_feedback import protocol as P

DATASETS = ("gsm", "transfer")
COMPARISONS = ("rlvr_vs_sft", "rlaif_vs_sft")
DIFF_PAIRS = (("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif"))
METRICS = ("accuracy", "format_compliance", "mean_length")
JUDGE_OUTCOMES = ("win", "loss", "tie", "parse_failure")
PREFS = ("better", "tie", "wrong", "parse_failure")


# ---------------------------------------------------------------- per-item arrays and statistics (pure)

def metric_array(rows: list[dict], metric: str) -> np.ndarray:
    if metric == "accuracy":
        return np.array([float(r["correct"]) for r in rows])
    if metric == "format_compliance":
        return np.array([float(r["compliant"]) for r in rows])
    if metric == "mean_length":
        return np.array([float(r["token_length"]) for r in rows])
    raise ValueError(metric)


def mean_of(x):
    x = np.asarray(x, dtype=float)
    return lambda idx: float(np.mean(x[idx]))


def judge_outcome(rec: dict) -> str:
    """Outcome from the trained policy's side (argument a). Parse failures are released ties."""
    if not rec["parse_matched"]:
        return "parse_failure"
    return {"A": "win", "B": "loss", "TIE": "tie"}[rec["label"]]


def win_scores(recs: list[dict]) -> np.ndarray:
    """Released win rate per comparison: win 1, tie 0.5, loss 0; parse failures are TIE (0.5)."""
    return np.array([P.JUDGE_SCORE[r["label"]] for r in recs])


def win_rate_block(recs: list[dict], seed: int, n_boot: int) -> dict:
    n = len(recs)
    counts = {o: sum(judge_outcome(r) == o for r in recs) for o in JUDGE_OUTCOMES}
    s = win_scores(recs)
    return {
        "n": n,
        "win_rate": bootstrap(mean_of(s), n, iid_resampler(n), seed, n_boot, "iid over prompts"),
        "counts": counts,
        "rates": {o: c / n for o, c in counts.items()},
        "ties_incl_parse_failures": counts["tie"] + counts["parse_failure"],
        "resolution_points": 100.0 / n,
    }


def agreement_block(recs: list[dict]) -> dict:
    """Definition A. a = trained policy, b = SFT; correctness from the released verifier."""
    one = [r for r in recs if r["a_correct"] != r["b_correct"]]
    agree = sum(1 for r in one if r["parse_matched"] and r["label"] == ("A" if r["a_correct"] else "B"))
    disagree = sum(1 for r in one if r["parse_matched"] and r["label"] == ("B" if r["a_correct"] else "A"))
    tie = sum(1 for r in one if r["parse_matched"] and r["label"] == "TIE")
    pf = sum(1 for r in one if not r["parse_matched"])
    n1 = len(one)
    out = {
        "exactly_one_correct": {
            "n": n1,
            "judge_prefers_correct": agree, "judge_prefers_wrong": disagree, "judge_tie": tie, "parse_failure": pf,
            "agreement_rate": agree / n1 if n1 else None,
            "rates": {k: (v / n1 if n1 else None) for k, v in
                      (("prefers_correct", agree), ("prefers_wrong", disagree), ("tie", tie), ("parse_failure", pf))},
            "n_correct_is_trained": sum(1 for r in one if r["a_correct"]),
            "resolution_points": 100.0 / n1 if n1 else None,
        },
    }
    for name, both in (("both_correct", True), ("both_wrong", False)):
        sub = [r for r in recs if r["a_correct"] == r["b_correct"] == both]
        c = {o: sum(judge_outcome(r) == o for r in sub) for o in JUDGE_OUTCOMES}
        out[name] = {"n": len(sub), "judge_counts_trained_side": c,
                     "rates": {o: (v / len(sub) if sub else None) for o, v in c.items()}}
    return out


def agreement_with_ci(recs: list[dict], seed: int, n_boot: int) -> dict:
    blk = agreement_block(recs)
    one = [r for r in recs if r["a_correct"] != r["b_correct"]]
    if one:
        hit = np.array([float(r["parse_matched"] and r["label"] == ("A" if r["a_correct"] else "B")) for r in one])
        blk["exactly_one_correct"]["agreement_rate_ci"] = bootstrap(mean_of(hit), len(one), iid_resampler(len(one)), seed, n_boot,
                                                                    "iid over disagreeing prompts")
    return blk


def drop_point(svamp_value: float, gsm_value: float) -> float:
    return svamp_value - gsm_value


def pref_counts(prefs: list[str]) -> dict:
    return {k: sum(p == k for p in prefs) for k in PREFS}


def category_block(prefs: list[str], seed: int, n_boot: int) -> dict:
    n = len(prefs)
    out = {"n": n, "counts": pref_counts(prefs), "resolution_points": 100.0 / n}
    for k in PREFS:
        x = np.array([float(p == k) for p in prefs])
        out[f"{k}_rate"] = bootstrap(mean_of(x), n, iid_resampler(n), seed, n_boot, "iid over problems")
    return out


def swap_block(released: list[dict], swapped: list[dict]) -> dict:
    """Per pair: content-level released labels in both orders (parse failure kept distinct)."""
    by_id = {r["pair_id"]: r for r in swapped}
    rows = []
    for r in released:
        s = by_id[r["pair_id"]]
        assert r["physical_order"] != s["physical_order"]
        rows.append((r["preference"], s["preference"]))
    n = len(rows)
    return {
        "n": n,
        "unchanged": sum(a == b for a, b in rows),
        "unchanged_rate": sum(a == b for a, b in rows) / n if n else None,
        "released_counts": pref_counts([a for a, _ in rows]),
        "swapped_counts": pref_counts([b for _, b in rows]),
        "resolution_points": 100.0 / n if n else None,
    }


def physical_position_counts(recs: list[dict]) -> dict:
    """Which slot the judge picked, ignoring content: first shown (A), second (B), TIE, parse failure."""
    c = {"first_shown": 0, "second_shown": 0, "tie": 0, "parse_failure": 0}
    for r in recs:
        if not r["parse_matched"]:
            c["parse_failure"] += 1
        else:
            c[{"A": "first_shown", "B": "second_shown", "TIE": "tie"}[r["physical_label"]]] += 1
    return c


# ---------------------------------------------------------------- inputs

class Inputs:
    def __init__(self, d: Path):
        self.d = d
        self.checks: list[dict] = []

    def json(self, name):
        return load_json(self.d / name)

    def jsonl(self, name):
        return read_jsonl(self.d / name)

    def check(self, name, ok, detail=None):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})


def load_all(inp: Inputs):
    gens, gen_meta = {}, {}
    for ds in DATASETS:
        for pol in P.POLICIES:
            m = inp.json(f"gen_{ds}_{pol}.json")
            gen_meta[(ds, pol)] = m
            gens[(ds, pol)] = inp.jsonl(f"gen_{ds}_{pol}.jsonl")
    judges = {ds: (inp.json(f"judge_{ds}.json"), inp.jsonl(f"judge_{ds}.jsonl")) for ds in DATASETS}
    diags = {o: (inp.json(f"diagnostics_{o}.json"), inp.jsonl(f"diagnostics_{o}.jsonl")) for o in ("released", "swapped")}
    return gens, gen_meta, judges, diags


def protocol_checks(inp: Inputs, gens, gen_meta, judges, diags):
    eff = {}
    for (ds, pol), m in gen_meta.items():
        tag = f"gen {ds} {pol}"
        n = P.DATASET_ROWS[ds]
        rows = gens[(ds, pol)]
        inp.check(f"{tag}: completed, not smoke", m["status"] == "completed" and not m["smoke"], m["status"])
        inp.check(f"{tag}: {n} responses", len(rows) == n == m["metrics"]["n_responses"], len(rows))
        inp.check(f"{tag}: data sha256", m["data"]["sha256"] == P.FILE_SHA256[m["data"]["path"]], m["data"]["sha256"])
        if pol != "sft":
            inp.check(f"{tag}: adapter sha256", m["adapter_sha256"] == P.FILE_SHA256[P.adapter_file(m["adapter"])], m["adapter_sha256"])
        else:
            inp.check(f"{tag}: base model, no adapter", m["adapter_sha256"] is None, m["adapter"])
        inp.check(f"{tag}: no prompt over cap, none skipped or truncated",
                  m["data"]["n_prompts_over_cap"] == m["data"]["n_skipped"] == m["data"]["n_truncated_prompts"] == 0
                  and m["data"]["prompt_tokens"]["max"] <= P.PROMPT_TOKEN_CAP, m["data"]["prompt_tokens"]["max"])
        inp.check(f"{tag}: seed {P.SEED}, 1 sample, batch {P.GEN_BATCH_SIZE}",
                  m["seed"] == P.SEED and m["settings"]["samples_per_prompt"] == 1 and m["settings"]["gen_batch_size"] == P.GEN_BATCH_SIZE)
        inp.check(f"{tag}: per-row ids follow the data file order", [r["prompt_id"] for r in rows] == m["data"]["prompt_ids"])
        eff[(ds, pol)] = {k: v for k, v in m["settings"]["effective_generation"].items()}
    for ds in DATASETS:
        ids = [[r["prompt_id"] for r in gens[(ds, p)]] for p in P.POLICIES]
        inp.check(f"{ds}: identical prompt order across policies", ids[0] == ids[1] == ids[2])
        inp.check(f"{ds}: identical effective generation settings across policies",
                  eff[(ds, "sft")] == eff[(ds, "rlvr")] == eff[(ds, "rlaif")], eff[(ds, "sft")])
        jm, jr = judges[ds]
        n = P.DATASET_ROWS[ds]
        inp.check(f"judge {ds}: completed, not smoke", jm["status"] == "completed" and not jm["smoke"], jm["status"])
        inp.check(f"judge {ds}: {2 * n} calls", len(jr) == 2 * n == jm["n_calls_done"], len(jr))
        inp.check(f"judge {ds}: all calls in released order", all(r["released_order"] for r in jr))
        inp.check(f"judge {ds}: wrapper parity", jm["parity_all_match"] and len(jm["parity"]) > 0, jm["parity"])
        for c in COMPARISONS:
            pid = [r["prompt_id"] for r in jr if r["comparison"] == c]
            inp.check(f"judge {ds} {c}: one call per prompt, generation order", pid == ids[0])
        corr = {p: {r["prompt_id"]: r["correct"] for r in gens[(ds, p)]} for p in P.POLICIES}
        inp.check(f"judge {ds}: verifier labels match generation files",
                  all(r["a_correct"] == corr[r["a_policy"]][r["prompt_id"]] and r["b_correct"] == corr["sft"][r["prompt_id"]] for r in jr))
    for o, (dm, dr) in diags.items():
        inp.check(f"diagnostics {o}: completed, not smoke", dm["status"] == "completed" and not dm["smoke"], dm["status"])
        inp.check(f"diagnostics {o}: {P.N_PAIRS} pairs", len(dr) == P.N_PAIRS == dm["data"]["n_pairs_judged"], len(dr))
        inp.check(f"diagnostics {o}: {P.N_DIAGNOSTIC_ROWS} verifier rows", len(dm["verifier_responses"]) == P.N_DIAGNOSTIC_ROWS)
        inp.check(f"diagnostics {o}: data sha256", dm["data"]["sha256"] == P.FILE_SHA256[dm["data"]["path"]])
        inp.check(f"diagnostics {o}: orientation flag", all(r["released_order"] == (o == "released") for r in dr))
    inp.check("diagnostics released: wrapper parity", diags["released"][0]["parity_all_match"], diags["released"][0]["parity"])
    rel = {r["pair_id"]: r["physical_order"] for r in diags["released"][1]}
    inp.check("diagnostics swapped: every pair in the opposite physical order",
              all(rel[r["pair_id"]] != r["physical_order"] for r in diags["swapped"][1]) and set(rel) == {r["pair_id"] for r in diags["swapped"][1]})
    commits = sorted({m["git"]["commit"] for m in gen_meta.values()} | {judges[d][0]["git"]["commit"] for d in DATASETS}
                     | {diags[o][0]["git"]["commit"] for o in diags})
    inp.check("all runs from one commit", len(commits) == 1, commits)


# ---------------------------------------------------------------- sections

def policy_section(gens, seed, n_boot):
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for pol in P.POLICIES:
            rows = gens[(ds, pol)]
            n = len(rows)
            blk = {"n": n, "resolution_points": 100.0 / n}
            for m in METRICS:
                blk[m] = bootstrap(mean_of(metric_array(rows, m)), n, iid_resampler(n), seed, n_boot, "iid over prompts")
            blk["length"] = describe([r["token_length"] for r in rows])
            blk["n_truncated_at_cap"] = sum(r["truncated"] for r in rows)
            blk["failure_type_counts"] = {t: sum(r["failure_type"] == t for r in rows) for t in P.FAILURE_TYPES}
            out[ds][pol] = blk
        out[ds]["paired_differences"] = {}
        for a, b in DIFF_PAIRS:
            ra, rb = gens[(ds, a)], gens[(ds, b)]
            n = len(ra)
            out[ds]["paired_differences"][f"{a}_minus_{b}"] = {
                m: bootstrap((lambda xa, xb: lambda idx: float(np.mean(xa[idx]) - np.mean(xb[idx])))(metric_array(ra, m), metric_array(rb, m)),
                             n, iid_resampler(n), seed, n_boot, "paired over prompts") for m in METRICS}
    return out


def judge_section(judges, seed, n_boot):
    out = {}
    for ds in DATASETS:
        jm, jr = judges[ds]
        by = {c: [r for r in jr if r["comparison"] == c] for c in COMPARISONS}
        out[ds] = {c: {"win": win_rate_block(by[c], seed, n_boot), "agreement_A": agreement_with_ci(by[c], seed, n_boot)}
                   for c in COMPARISONS}
        out[ds]["agreement_A_pooled"] = agreement_block(jr)
        sa, sb = win_scores(by["rlvr_vs_sft"]), win_scores(by["rlaif_vs_sft"])
        out[ds]["win_rate_rlvr_minus_rlaif"] = bootstrap(lambda idx: float(np.mean(sa[idx]) - np.mean(sb[idx])), len(sa),
                                                         iid_resampler(len(sa)), seed, n_boot, "paired over prompts")
        out[ds]["judge_seconds_per_call"] = jm["timing"].get("judge_seconds_per_call")
        out[ds]["judge_settings"] = jm["settings"].get("judge")
        out[ds]["physical_position_counts"] = physical_position_counts(jr)
    return out


def drop_section(gens, judges, seed, n_boot):
    out = {}
    for pol in P.POLICIES:
        g, s = gens[("gsm", pol)], gens[("transfer", pol)]
        out[pol] = {m: two_sample_bootstrap(mean_of(metric_array(g, m)), len(g), mean_of(metric_array(s, m)), len(s), seed, n_boot,
                                            "SVAMP minus GSM8K; two-sample, each prompt set resampled independently") for m in METRICS}
    for c in COMPARISONS:
        pol = c.split("_")[0]
        g = win_scores([r for r in judges["gsm"][1] if r["comparison"] == c])
        s = win_scores([r for r in judges["transfer"][1] if r["comparison"] == c])
        out[pol]["win_rate_vs_sft"] = two_sample_bootstrap(mean_of(g), len(g), mean_of(s), len(s), seed, n_boot,
                                                           "SVAMP minus GSM8K; two-sample")
    return out


def diagnostic_section(diags, seed, n_boot):
    dm, dr = diags["released"]
    sm, sr = diags["swapped"]
    vpairs = dm["verifier_pairs"]
    out = {"verifier_responses": {}, "verifier": {}, "judge": {}, "position_swap": {}}
    for variant in sorted({r["variant_type"] for r in dm["verifier_responses"]}):
        rs = [r for r in dm["verifier_responses"] if r["variant_type"] == variant]
        out["verifier_responses"][variant] = {
            "n": len(rs), "mean_reward": float(np.mean([r["reward"] for r in rs])),
            "compliance": float(np.mean([r["compliant"] for r in rs])),
            "n_reward_differs_from_expected": sum(r["reward"] != r["expected_exact_reward"] for r in rs)}
    for cat in P.CATEGORIES:
        v = sorted((r for r in vpairs if r["category"] == cat), key=lambda r: r["problem_id"])
        j = sorted((r for r in dr if r["category"] == cat), key=lambda r: r["problem_id"])
        s = [r for r in sr if r["category"] == cat]
        out["verifier"][cat] = category_block([r["preference"] for r in v], seed, n_boot)
        out["judge"][cat] = category_block([r["preference"] for r in j], seed, n_boot)
        out["position_swap"][cat] = swap_block(j, s)
        out["position_swap"][cat]["released_physical"] = physical_position_counts(j)
        out["position_swap"][cat]["swapped_physical"] = physical_position_counts(s)
    out["S_reason"] = {m: out[m]["reasoning_corrupt"]["better_rate"] for m in ("verifier", "judge")}
    out["S_outcome"] = {m: out[m]["wrong_final"]["better_rate"] for m in ("verifier", "judge")}
    out["definitions"] = {
        "S_reason": "Pr[R(clean_correct) > R(corrupt_reasoning_correct_final)] over the 20 problems",
        "S_outcome": "Pr[R(clean_correct) > R(good_reasoning_wrong_final)] over the 20 problems; gold_distractor reported separately",
        "judge_rates": "released order; parse failures are a separate category (the release would read them as TIE)",
    }
    out["judge_seconds_per_call"] = {"released": dm["timing"].get("judge_seconds_per_call"), "swapped": sm["timing"].get("judge_seconds_per_call")}
    return out


def qualitative_section(diags) -> dict:
    dr = diags["released"][1]
    return {cat: P.qualitative_choice([r for r in dr if r["category"] == cat]) for cat in P.QUALITATIVE_CATEGORIES}


# ---------------------------------------------------------------- csv

def ci_row(section, dataset, subject, quantity, b, n=None, resolution=None):
    return {"section": section, "dataset": dataset, "subject": subject, "quantity": quantity,
            "point": b["point"], "ci_low": b["ci_low"], "ci_high": b["ci_high"], "n": n, "resolution_points": resolution}


def csv_rows(summary: dict) -> list[dict]:
    rows = []
    pol = summary["policies"]
    for ds in DATASETS:
        for p in P.POLICIES:
            b = pol[ds][p]
            for m in METRICS:
                rows.append(ci_row("policy", ds, p, m, b[m], b["n"], b["resolution_points"]))
            for t, c in b["failure_type_counts"].items():
                rows.append({"section": "failure_type", "dataset": ds, "subject": p, "quantity": t, "point": c, "n": b["n"]})
            rows.append({"section": "policy", "dataset": ds, "subject": p, "quantity": "length_sd", "point": b["length"]["std"], "n": b["n"]})
            rows.append({"section": "policy", "dataset": ds, "subject": p, "quantity": "length_iqr", "point": b["length"]["iqr"], "n": b["n"]})
            rows.append({"section": "policy", "dataset": ds, "subject": p, "quantity": "n_truncated_at_cap", "point": b["n_truncated_at_cap"], "n": b["n"]})
        for d, blk in pol[ds]["paired_differences"].items():
            for m in METRICS:
                rows.append(ci_row("paired_difference", ds, d, m, blk[m]))
        for c in COMPARISONS:
            w = summary["judge"][ds][c]["win"]
            rows.append(ci_row("win_rate", ds, c, "win_rate", w["win_rate"], w["n"], w["resolution_points"]))
            for o, k in w["counts"].items():
                rows.append({"section": "win_counts", "dataset": ds, "subject": c, "quantity": o, "point": k, "n": w["n"]})
            a = summary["judge"][ds][c]["agreement_A"]["exactly_one_correct"]
            if a["n"]:
                rows.append(ci_row("agreement_A", ds, c, "judge_prefers_verifier_correct", a["agreement_rate_ci"], a["n"], a["resolution_points"]))
        rows.append(ci_row("win_rate", ds, "rlvr_minus_rlaif", "win_rate_difference", summary["judge"][ds]["win_rate_rlvr_minus_rlaif"]))
    for p, blk in summary["drops"].items():
        for q, b in blk.items():
            rows.append(ci_row("drop_svamp_minus_gsm", "both", p, q, b))
    dg = summary["diagnostics"]
    for mech in ("verifier", "judge"):
        for cat in P.CATEGORIES:
            b = dg[mech][cat]
            for k in PREFS:
                rows.append(ci_row(f"diagnostic_{mech}", "diagnostics", cat, f"{k}_rate", b[f"{k}_rate"], b["n"], b["resolution_points"]))
        rows.append(ci_row(f"diagnostic_{mech}", "diagnostics", "S_reason", "better_rate", dg["S_reason"][mech], 20, 5.0))
        rows.append(ci_row(f"diagnostic_{mech}", "diagnostics", "S_outcome", "better_rate", dg["S_outcome"][mech], 20, 5.0))
    for cat, b in dg["position_swap"].items():
        rows.append({"section": "position_swap", "dataset": "diagnostics", "subject": cat, "quantity": "unchanged_rate",
                     "point": b["unchanged_rate"], "n": b["n"], "resolution_points": b["resolution_points"]})
    return rows


# ---------------------------------------------------------------- main

def build_summary(d: Path, seed: int = P.SEED, n_boot: int = P.N_BOOT) -> tuple[dict, list[dict]]:
    inp = Inputs(d)
    gens, gen_meta, judges, diags = load_all(inp)
    protocol_checks(inp, gens, gen_meta, judges, diags)
    summary = {
        "policies": policy_section(gens, seed, n_boot),
        "judge": judge_section(judges, seed, n_boot),
        "drops": drop_section(gens, judges, seed, n_boot),
        "diagnostics": diagnostic_section(diags, seed, n_boot),
        "qualitative_candidates": qualitative_section(diags),
        "generation_seconds_per_prompt": {f"{ds}_{p}": m["timing"].get("seconds_per_prompt") for (ds, p), m in gen_meta.items()},
        "effective_generation": gen_meta[("gsm", "sft")]["settings"]["effective_generation"],
        "protocol_checks": inp.checks,
        "all_checks_ok": all(c["ok"] for c in inp.checks),
    }
    return summary, csv_rows(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(f"{cfg['results_dir']}/task5_feedback")
    out_json, out_csv = d / "summary.json", d / "summary.csv"
    for p in (out_json, out_csv):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    summary, rows = build_summary(d, int(cfg["seed"]))
    summary = {"script": "task5_feedback.compare_feedback", "git": git_state(), "seed": int(cfg["seed"]),
               "n_resamples": P.N_BOOT, **summary, "wall_clock_seconds": elapsed()}
    save_json(out_json, summary)
    write_csv(out_csv, rows)
    n_bad = sum(not c["ok"] for c in summary["protocol_checks"])
    print(f"[summary] checks={len(summary['protocol_checks'])} failed={n_bad} rows={len(rows)} t={elapsed():.0f}s; "
          f"wrote {display_path(out_json)}, {display_path(out_csv)}", flush=True)


if __name__ == "__main__":
    main()
