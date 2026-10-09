"""Task 3 summary of notebook 4 (Mac, from saved result files only; no model is loaded).

Inputs (results dir): train_<run>.json, rollouts_<run>.jsonl, eval_<run>.json and
generations_eval_<run>.jsonl for the standard run and the two normalization forks;
data/rl_prompt_pool_eval.jsonl for prompt text in the qualitative file.

Outputs:
  summary_standard.json / .csv
      protocol_checks       pass/fail list over all three runs (status, update counts, commit, settings A to L
                            values, finite steps, fork diagnostics, clip fraction, prompts, update-1 identity,
                            evaluated prompts, effective generation settings, RM-truncated input counts)
      per_update            one row per update of the standard run
      run                   peak VRAM, wall-clock, hardware, dtype
      update_trends         OLS slope against update number 1..20 of reward mean, sampled_kl, entropy and mean
                            length, with a 95% percentile bootstrap over updates ((update, value) pairs resampled)
      informative_updates   informative updates out of 20
      zero_gradient_tokens  generated tokens and the zero-gradient shares (masked truncation, uninformative group)
      heldout               standard held-out evaluation summary (task2_ppo.summarize.standard_heldout)
  normalization_qualitative.json
      three held-out prompts picked by rules fixed in code (largest |length dr - length grpo|, largest
      RM dr - RM grpo, largest RM grpo - RM dr; ties by lowest prompt_id) and every uninformative
      update of the standard run with its 4 completions.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from task1_dpo.summarize import bootstrap, iid_resampler, write_csv
from task1_dpo.train import git_state
from task2_ppo.summarize import standard_heldout

N_RESAMPLES = 10_000
STANDARD, CANONICAL, DR = "standard", "fork_grpo", "fork_dr_grpo"
RUNS = (STANDARD, CANONICAL, DR)
# Commits the notebook 4 runs were allowed to use (prompt 3c).
ALLOWED_COMMITS = ("bf1fdc7", "7ced6a3")
EXPECTED_UPDATES = {STANDARD: 20, CANONICAL: 8, DR: 8}
# Setting values that every run must carry (effective settings of the training runs).
EXPECTED_TRAIN = {"max_completion_length": 512, "num_generations": 4, "kl_beta": 0.1, "clip_epsilon": 0.2,
                  "learning_rate": 5.0e-6, "policy_epochs": 1, "mask_truncated_completions": True}
EXPECTED_EVAL = {"max_new_tokens": 768, "reward_max_length": 1280, "samples_per_prompt": 1,
                 "temperature": 0.7, "top_p": 0.9}
N_EVAL_PROMPTS = 165
PROMPT_CHARS, RESPONSE_CHARS, UNINFORMATIVE_CHARS = 400, 600, 200


# ---------------------------------------------------------------- statistics

def ols_slope(y, x) -> float:
    """Least-squares slope of y on x; NaN when x is constant (a degenerate bootstrap draw)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    return float((xc * (y - y.mean())).sum() / sxx) if sxx > 0 else float("nan")


def slope_bootstrap(y, seed: int, n_resamples: int = N_RESAMPLES) -> dict:
    """OLS slope of y against update number 1..n; 95% percentile bootstrap over (update, y) pairs."""
    y = np.asarray(y, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    return bootstrap(lambda idx: ols_slope(y[idx], x[idx]), len(y), iid_resampler(len(y)), seed, n_resamples,
                     f"OLS slope vs update number, {n_resamples} resamples of the {len(y)} (update, value) pairs")


# ---------------------------------------------------------------- standard run

def update_row(h: dict) -> dict:
    z = h["zero_gradient_tokens"]
    return {
        "update": h["update"], "prompt_id": h["prompt_id"],
        "reward_mean": h["group_reward_mean"], "group_reward_std": h["group_reward_std"],
        "informative": h["informative"], "n_masked": h["n_masked"],
        "policy_loss": h["policy_loss"], "surrogate_term": h["surrogate_term"], "beta_kl_term": h["beta_kl_term"],
        "kl_k3": h["kl_k3"], "sampled_kl": h["kl_sampled"], "entropy": h["entropy_sampled"],
        "grad_norm_before_clip": h["grad_norm_before_clip"], "clip_fraction": h["clip_fraction"],
        "length_mean": h["completion_length"]["mean"], "length_sd": h["completion_length"]["sd"],
        "generated_tokens": h["generated_tokens"],
        "zero_grad_masked_truncation_tokens": z["masked_truncation_tokens"],
        "zero_grad_uninformative_tokens": z["uninformative_group_tokens"],
        "zero_grad_masked_truncation_share": z["masked_truncation_share"],
        "zero_grad_uninformative_share": z["uninformative_group_share"],
        "seconds": h["seconds"]["total"],
    }


TREND_KEYS = ("reward_mean", "sampled_kl", "entropy", "length_mean")


def standard_section(train: dict, seed: int) -> dict:
    rows = [update_row(h) for h in train["history"]]
    col = lambda k: [r[k] for r in rows]
    gen = int(sum(col("generated_tokens")))
    masked, uninf = int(sum(col("zero_grad_masked_truncation_tokens"))), int(sum(col("zero_grad_uninformative_tokens")))
    return {
        "per_update": rows,
        "run": {"peak_vram_bytes": train["peak_vram_bytes"], "peak_vram_gib": train["peak_vram_bytes"] / 2 ** 30,
                "wall_clock_seconds": train["wall_clock_seconds"], "setup_seconds": train["setup_seconds"],
                "hardware": train["hardware"]["device"], "dtype": train["dtype"]},
        "update_trends": {k: slope_bootstrap(col(k), seed) for k in TREND_KEYS},
        "informative_updates": {"n_informative": int(sum(col("informative"))), "n_updates": len(rows),
               "uninformative_updates": [r["update"] for r in rows if not r["informative"]]},
        "zero_gradient_tokens": {"generated_tokens": gen, "generated_tokens_total_logged": train["generated_tokens_total"],
               "zero_grad_masked_truncation_tokens": masked, "zero_grad_uninformative_tokens": uninf,
               "zero_grad_masked_truncation_share": masked / gen, "zero_grad_uninformative_share": uninf / gen,
               "zero_grad_total_share": (masked + uninf) / gen,
               "n_masked_completions": int(sum(col("n_masked"))), "n_completions": 4 * len(rows)},
    }


# ---------------------------------------------------------------- protocol checks

def protocol(trains: dict, evals: dict, rollouts: dict, gens: dict, seed: int) -> list[dict]:
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for r in RUNS:
        t, e = trains[r], evals[r]
        check(f"{r}: train completed with {EXPECTED_UPDATES[r]} updates",
              t["status"] == "completed" and t["updates_completed"] == EXPECTED_UPDATES[r] == len(t["history"]),
              [t["status"], t["updates_completed"], len(t["history"])])
        check(f"{r}: eval completed, not smoke", e["status"] == "completed" and not e["smoke"], [e["status"], e["smoke"]])
    for kind, runs in (("train", trains), ("eval", evals)):
        for r in RUNS:
            g = runs[r]["git"]
            check(f"{r} {kind}: commit in {ALLOWED_COMMITS} and clean tree",
                  g["commit"][:7] in ALLOWED_COMMITS and g["dirty"] is False, [g["commit"][:7], g["dirty"]])
    for r in RUNS:
        t, e = trains[r], evals[r]
        bad = {k: t["effective"][k] for k, v in EXPECTED_TRAIN.items() if t["effective"][k] != v}
        check(f"{r}: training settings (cap 512, K 4, beta 0.1, eps 0.2, lr 5e-6, 1 epoch, masking)", not bad, bad)
        check(f"{r}: seed {seed}", t["seed"] == seed == e["seed"], [t["seed"], e["seed"]])
        check(f"{r}: LoRA dropout 0.05 active in the training forward",
              t["config"]["lora"]["dropout"] == 0.05 and "dropout 0.05 active" in t["decisions"]["dropout"],
              t["decisions"]["dropout"])
        check(f"{r}: training RM cap 1024 (release default, recorded)", t["effective"]["reward_max_length"] == 1024,
              t["effective"]["reward_max_length"])
        s = e["settings"]
        bad = {k: s[k] for k, v in EXPECTED_EVAL.items() if s[k] != v}
        check(f"{r}: eval settings (cap 768, RM cap 1280, 1 sample, T 0.7, top_p 0.9)", not bad, bad)
        check(f"{r}: {N_EVAL_PROMPTS} eval prompts",
              e["prompts"]["n_evaluated"] == e["metrics"]["n_responses"] == len(gens[r]) == N_EVAL_PROMPTS,
              [e["prompts"]["n_evaluated"], e["metrics"]["n_responses"], len(gens[r])])
        check(f"{r}: eval seed reset to {seed} before generation", s["seed_reset_before_generation"] == seed,
              s["seed_reset_before_generation"])
        check(f"{r}: no non-finite steps",
              t["n_nonfinite_steps"] == 0 and all(h["inputs_finite"] and h["step_taken"] for h in t["history"]),
              t["n_nonfinite_steps"])
        clip = [h["clip_fraction"] for h in t["history"]]
        check(f"{r}: clip fraction 0 on every update", all(c == 0.0 for c in clip), max(clip))
    for r in (CANONICAL, DR):
        bad = []
        for h in trains[r]["history"]:
            ld, tg = h.get("length_diagnostic") or [], h.get("term_gradients") or {}
            ok = (len(ld) == 4 - h["n_masked"] and all(math.isfinite(x["grad_norm"]) for x in ld)
                  and tg and all(math.isfinite(tg[k]) for k in ("grad_norm_surrogate", "grad_norm_beta_kl")))
            if not ok:
                bad.append(h["update"])
        check(f"{r}: length-diagnostic rows for every unmasked completion and finite term gradients", not bad, bad)
        std_ids = [h["prompt_id"] for h in trains[STANDARD]["history"][:EXPECTED_UPDATES[r]]]
        check(f"{r}: same prompt_ids as standard updates 1-{EXPECTED_UPDATES[r]}",
              [h["prompt_id"] for h in trains[r]["history"]] == std_ids)

    def first(rows):
        return sorted((x["sample_index"], x["response_ids_sha256"], x["token_length"]) for x in rows if x["update"] == 1)
    u1 = {r: first(rollouts[r]) for r in RUNS}
    check("update-1 completions identical across the three runs (token-id sha256)",
          len(u1[STANDARD]) == 4 and u1[STANDARD] == u1[CANONICAL] == u1[DR], {r: len(v) for r, v in u1.items()})
    ids = {r: [g["prompt_id"] for g in gens[r]] for r in RUNS}
    check("eval prompt_ids identical across the three evals (same order)", ids[STANDARD] == ids[CANONICAL] == ids[DR],
          {r: len(set(v)) for r, v in ids.items()})
    tg = [trains[r]["effective"]["generation"] for r in RUNS]
    eg = [evals[r]["settings"]["effective_generation"] for r in RUNS]
    check("effective training generation settings recorded and identical", tg[0] and tg[0] == tg[1] == tg[2], tg[0])
    check("effective eval generation settings recorded and identical", eg[0] and eg[0] == eg[1] == eg[2], eg[0])
    drop = lambda d: {k: v for k, v in d.items() if k != "max_new_tokens"}
    check("training and eval generation settings identical apart from the cap", drop(tg[0]) == drop(eg[0]))
    rm = {f"{r} train": trains[r]["n_rm_input_truncated"] for r in RUNS}
    rm.update({f"{r} eval": evals[r]["metrics"]["n_rm_input_truncated"] for r in RUNS})
    check("RM-truncated inputs: 0 in every run", all(v == 0 for v in rm.values()), rm)
    return checks


# ---------------------------------------------------------------- qualitative

def pick(rows: list[dict], key) -> dict:
    """Row with the largest key(row); ties broken by the lowest prompt_id."""
    return min(rows, key=lambda r: (-key(r), r["prompt_id"]))


def qualitative(gens: dict, prompts: dict, train: dict, rollouts: list[dict]) -> dict:
    g = {x["prompt_id"]: x for x in gens[CANONICAL]}
    d = {x["prompt_id"]: x for x in gens[DR]}
    if set(g) != set(d):
        raise SystemExit("fork eval prompt_id sets differ")
    pairs = [{"prompt_id": p, "grpo": g[p], "dr": d[p]} for p in sorted(g)]
    rules = {
        "largest_abs_length_difference": lambda r: abs(r["dr"]["token_length"] - r["grpo"]["token_length"]),
        "largest_rm_dr_minus_grpo": lambda r: r["dr"]["rm_score"] - r["grpo"]["rm_score"],
        "largest_rm_grpo_minus_dr": lambda r: r["grpo"]["rm_score"] - r["dr"]["rm_score"],
    }
    picked = {}
    for name, key in rules.items():
        r = pick(pairs, key)
        picked[name] = {
            "prompt_id": r["prompt_id"], "rule_value": key(r),
            "prompt_text": prompts[r["prompt_id"]][:PROMPT_CHARS],
            **{lab: {"response": r[lab]["text"][:RESPONSE_CHARS], "token_length": r[lab]["token_length"],
                     "rm_score": r[lab]["rm_score"], "truncated": r[lab]["truncated"]} for lab in ("grpo", "dr")},
        }
    uninformative = []
    for h in train["history"]:
        if h["informative"]:
            continue
        comps = sorted((x for x in rollouts if x["update"] == h["update"]), key=lambda x: x["sample_index"])
        uninformative.append({"update": h["update"], "prompt_id": h["prompt_id"], "rewards": h["rewards"],
                              "completions": [{"sample_index": x["sample_index"], "text": x["text"][:UNINFORMATIVE_CHARS],
                                               "token_length": x["token_length"], "truncated": x["truncated"]} for x in comps]})
    return {"rules": {"selection": "largest value of each rule over the held-out prompts, ties by lowest prompt_id",
                      "grpo": CANONICAL, "dr": DR,
                      "truncation": f"prompt {PROMPT_CHARS} chars, responses {RESPONSE_CHARS} chars, "
                                    f"uninformative completions {UNINFORMATIVE_CHARS} chars"},
            "heldout": picked, "standard_uninformative_updates": uninformative}


# ---------------------------------------------------------------- output

def csv_rows(s: dict) -> list[dict]:
    rows = [{"section": "per_update", **r} for r in s["per_update"]]
    for k, v in s["run"].items():
        rows.append({"section": "run", "metric": k, "point": v})
    for k, v in s["update_trends"].items():
        rows.append({"section": "update_trends", "metric": k, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
    rows.append({"section": "informative_updates", "metric": "n_informative", "point": s["informative_updates"]["n_informative"]})
    for k, v in s["zero_gradient_tokens"].items():
        rows.append({"section": "zero_gradient_tokens", "metric": k, "point": v})
    h = s["heldout"]
    for k in ("rm_mean", "rm_sd", "kl", "entropy", "length_mean", "length_sd", "truncation_rate_at_768", "n_truncated_at_768"):
        rows.append({"section": "heldout", "metric": k, "point": h[k]})
    for k, v in h["bootstrap"].items():
        rows.append({"section": "heldout_bootstrap", "metric": k, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = cfg["results_dir"]
    out_json, out_csv = repo_path(f"{rd}/summary_standard.json"), repo_path(f"{rd}/summary_standard.csv")
    out_q = repo_path(f"{rd}/normalization_qualitative.json")
    for p in (out_json, out_csv, out_q):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    seed = int(cfg["seed"])
    trains = {r: load_json(f"{rd}/train_{r}.json") for r in RUNS}
    evals = {r: load_json(f"{rd}/eval_{r}.json") for r in RUNS}
    rollouts = {r: read_jsonl(f"{rd}/rollouts_{r}.jsonl") for r in RUNS}
    gens = {r: read_jsonl(f"{rd}/generations_eval_{r}.jsonl") for r in RUNS}
    prompts = {x["prompt_id"]: x["messages"][-1]["content"] for x in read_jsonl(evals[STANDARD]["prompts"]["path"])}

    checks = protocol(trains, evals, rollouts, gens, seed)
    summary = {"script": "task3_grpo.summarize", "git": git_state(), "seed": seed, "run": STANDARD,
               "bootstrap": f"{N_RESAMPLES} resamples, seed {seed}, 95% percentile interval",
               "protocol_checks": checks, "n_failed_checks": sum(not c["passed"] for c in checks),
               **standard_section(trains[STANDARD], seed),
               "heldout": standard_heldout(evals[STANDARD], gens[STANDARD], seed)}
    summary["wall_clock_seconds"] = elapsed()
    save_json(out_json, summary)
    write_csv(out_csv, csv_rows(summary))
    save_json(out_q, {"script": "task3_grpo.summarize", "git": summary["git"],
                      **qualitative(gens, prompts, trains[STANDARD], rollouts[STANDARD])})
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    print(f"failed checks: {summary['n_failed_checks']}; wrote {out_json}, {out_csv}, {out_q} in {elapsed():.0f}s")


if __name__ == "__main__":
    main()
