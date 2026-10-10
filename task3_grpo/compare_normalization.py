"""Task 3 Step 3: canonical GRPO vs Dr. GRPO normalization (Mac, from saved result files only).

Inputs (results dir): train_<run>.json, rollouts_<run>.jsonl, eval_<run>.json and
generations_eval_<run>.jsonl for the canonical fork (loss_type grpo) and the Dr. GRPO fork.

Outputs normalization.json and normalization.csv:
  protocol     checks that the forks differ only in loss_type (same midpoint settings, seed, prompts,
               generation settings, update count, evaluation settings) and that the update-1 completions
               are identical (same parameters, same seed); generated tokens per fork.
  heldout      paired bootstrap over held-out prompts (common.stats.heldout_block): reward, KL,
               entropy, length per fork and dr_grpo minus grpo.
  length       from the --length-diagnostic rows (unmasked completions, all updates); completions with
               A_k = 0 are excluded (counted). y_k = ||g_k|| / |A_k|.
               Spearman(T_k, y_k) per fork with a bootstrap CI over completions, and the difference
               dr_grpo minus grpo (the two forks' completions resampled independently);
               mean y_k for T_k <= LENGTH_SPLIT and T_k > LENGTH_SPLIT; mean analytic per-token weight.
  terms        per update: surrogate term and beta * KL term (k3) values, per fork (logged only).
  term_gradients  per update: ||grad surrogate term|| and ||grad beta * KL term|| over the trainable
               parameters (from --length-diagnostic). Per fork: mean of each over updates and the ratio of
               means mean(KL grad) / mean(surrogate grad), each with a bootstrap CI over updates. The ratio
               of means is used because the surrogate gradient is exactly 0 on an uninformative update.
Bootstrap: N_RESAMPLES resamples, numpy default_rng(seed), 95% percentile interval.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from common.metrics import safe_corr
from common.run_info import git_state
from common.stats import bootstrap, heldout_block, iid_resampler, two_sample_bootstrap, write_csv

N_RESAMPLES = 10_000
LENGTH_SPLIT = 256
CANONICAL, DR = "fork_grpo", "fork_dr_grpo"
# Keys allowed to differ between the two fork runs.
FORK_SPECIFIC = {"loss_type"}


# ---------------------------------------------------------------- statistics

def spearman(x, y) -> float:
    """Pearson correlation of average ranks (ties share their mean rank); NaN if either is constant."""
    return safe_corr(pd.Series(np.asarray(x, dtype=float)).rank().to_numpy(), pd.Series(np.asarray(y, dtype=float)).rank().to_numpy())


def boot(stat, n: int, seed: int, n_resamples: int, method: str) -> dict:
    """common.stats.bootstrap, or NaN (with the reason) when the statistic is undefined on the full
    sample, e.g. an empty length bucket or a constant input to Spearman."""
    point = stat(np.arange(n)) if n else float("nan")
    if not np.isfinite(point):
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_resamples": 0,
                "n_nonfinite_resamples": 0, "seed": seed, "method": method + "; undefined on the full sample, not bootstrapped"}
    return bootstrap(stat, n, iid_resampler(n), seed, n_resamples, method)


def diagnostic_rows(train: dict) -> tuple[list[dict], int]:
    """Length-diagnostic rows with A_k != 0 (tagged with their update), and how many were excluded."""
    rows, excluded = [], 0
    for h in train["history"]:
        for r in h.get("length_diagnostic") or []:
            if r["abs_advantage"] == 0.0:
                excluded += 1
                continue
            rows.append({**r, "update": h["update"], "y": r["grad_norm"] / r["abs_advantage"]})
    return rows, excluded


def length_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in rows], dtype=float) for k in ("T_k", "y", "analytic_token_weight", "grad_norm")}


def length_stats(a: dict[str, np.ndarray]) -> dict:
    """quantity -> f(completion index array) -> float."""
    def bucket_mean(key, long: bool):
        def f(idx):
            t, v = a["T_k"][idx], a[key][idx]
            sel = t > LENGTH_SPLIT if long else t <= LENGTH_SPLIT
            return float(v[sel].mean()) if sel.any() else float("nan")
        return f
    return {
        "spearman_T_vs_grad_per_abs_adv": lambda idx: spearman(a["T_k"][idx], a["y"][idx]),
        f"mean_grad_per_abs_adv_T_le_{LENGTH_SPLIT}": bucket_mean("y", False),
        f"mean_grad_per_abs_adv_T_gt_{LENGTH_SPLIT}": bucket_mean("y", True),
        "mean_analytic_token_weight": lambda idx: float(a["analytic_token_weight"][idx].mean()),
        f"mean_analytic_token_weight_T_le_{LENGTH_SPLIT}": bucket_mean("analytic_token_weight", False),
        f"mean_analytic_token_weight_T_gt_{LENGTH_SPLIT}": bucket_mean("analytic_token_weight", True),
    }


def length_section(trains: dict[str, dict], seed: int, n_resamples: int = N_RESAMPLES) -> dict:
    arrays, out = {}, {"per_fork": {}, "dr_grpo_minus_grpo": {}}
    for name, train in trains.items():
        rows, excluded = diagnostic_rows(train)
        a = length_arrays(rows)
        arrays[name] = a
        n = len(rows)
        out["per_fork"][name] = {
            "n_completions": n,
            "n_excluded_zero_advantage": excluded,
            f"n_T_le_{LENGTH_SPLIT}": int((a["T_k"] <= LENGTH_SPLIT).sum()),
            f"n_T_gt_{LENGTH_SPLIT}": int((a["T_k"] > LENGTH_SPLIT).sum()),
            **{q: boot(f, n, seed, n_resamples, f"bootstrap over {n} completions")
               for q, f in length_stats(a).items()},
        }
    ca, da = arrays[CANONICAL], arrays[DR]
    sa, sd = length_stats(ca), length_stats(da)
    for q in sa:
        out["dr_grpo_minus_grpo"][q] = two_sample_bootstrap(sa[q], len(ca["T_k"]), sd[q], len(da["T_k"]), seed, n_resamples,
                                                            "independent bootstrap of each fork's completions")
    return out


def terms_section(trains: dict[str, dict]) -> dict:
    out = {}
    for name, train in trains.items():
        out[name] = [{"update": h["update"], "surrogate_term": h["surrogate_term"], "beta_kl_term": h["beta_kl_term"],
                      "n_masked": h["n_masked"], "informative": h["informative"]} for h in train["history"]]
    return out


def term_gradient_section(trains: dict[str, dict], seed: int, n_resamples: int = N_RESAMPLES) -> dict:
    out = {}
    for name, train in trains.items():
        per_update = [{"update": h["update"], "informative": h["informative"], "n_masked": h["n_masked"],
                       "grad_norm_surrogate": h["term_gradients"]["grad_norm_surrogate"],
                       "grad_norm_beta_kl": h["term_gradients"]["grad_norm_beta_kl"]} for h in train["history"]]
        s = np.array([r["grad_norm_surrogate"] for r in per_update], dtype=float)
        k = np.array([r["grad_norm_beta_kl"] for r in per_update], dtype=float)
        n = len(per_update)
        method = f"bootstrap over {n} updates"
        out[name] = {
            "n_updates": n,
            "n_zero_surrogate_grad": int((s == 0.0).sum()),
            "mean_grad_norm_surrogate": boot(lambda idx: float(s[idx].mean()), n, seed, n_resamples, method),
            "mean_grad_norm_beta_kl": boot(lambda idx: float(k[idx].mean()), n, seed, n_resamples, method),
            "ratio_beta_kl_over_surrogate": boot(lambda idx: float(k[idx].mean() / s[idx].mean()) if s[idx].mean() > 0 else float("nan"),
                                                 n, seed, n_resamples, method + "; ratio of means"),
            "per_update": per_update,
        }
    return out


# ---------------------------------------------------------------- protocol checks

def update1_identity(rollouts: dict[str, list[dict]]) -> dict:
    def first(rows):
        return sorted(((r["sample_index"], r["response_ids_sha256"], r["token_length"], r["reward"]) for r in rows if r["update"] == 1))
    a, b = first(rollouts[CANONICAL]), first(rollouts[DR])
    same_tokens = [x[:3] for x in a] == [x[:3] for x in b] and len(a) > 0
    reward_diff = max((abs(x[3] - y[3]) for x, y in zip(a, b)), default=float("nan"))
    return {"n_completions": len(a), "token_ids_identical": same_tokens, "max_abs_reward_difference": reward_diff}


def protocol(trains: dict, evals: dict, rollouts: dict) -> list[dict]:
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    c, d = trains[CANONICAL], trains[DR]
    for name, t in trains.items():
        check(f"{name} train completed", t["status"] == "completed", t["status"])
        check(f"{name} length diagnostic on", t["effective"]["length_diagnostic"] is True)
        check(f"{name} eval completed and not smoke", evals[name]["status"] == "completed" and not evals[name]["smoke"],
              {"status": evals[name]["status"], "smoke": evals[name]["smoke"]})
    check("loss types", c["effective"]["loss_type"] == "grpo" and d["effective"]["loss_type"] == "dr_grpo",
          [c["effective"]["loss_type"], d["effective"]["loss_type"]])
    eff = lambda t: {k: v for k, v in t["effective"].items() if k not in FORK_SPECIFIC}
    diff = sorted(k for k in set(eff(c)) | set(eff(d)) if eff(c).get(k) != eff(d).get(k))
    check("same effective settings apart from loss_type", not diff, diff)
    check("same seed", c["seed"] == d["seed"], [c["seed"], d["seed"]])
    check("same prompt IDs and order", c["prompts"]["prompt_ids"] == d["prompts"]["prompt_ids"])
    check("same updates completed", c["updates_completed"] == d["updates_completed"], [c["updates_completed"], d["updates_completed"]])
    check("same evaluation settings", evals[CANONICAL]["settings"] == evals[DR]["settings"])
    check("same evaluated prompts", evals[CANONICAL]["prompts"] == evals[DR]["prompts"])
    ident = update1_identity(rollouts)
    check("update-1 completions identical (token ids)", ident["token_ids_identical"], ident)
    for name, t in trains.items():
        missing = [h["update"] for h in t["history"] if not h.get("term_gradients")]
        check(f"{name} term gradients logged at every update", not missing, missing)
    return checks


# ---------------------------------------------------------------- output

def csv_rows(result: dict) -> list[dict]:
    rows = []
    h = result["heldout"]
    for cond, ms in h["per_condition"].items():
        for m, v in ms.items():
            rows.append({"section": "heldout", "condition": cond, "metric": m, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
    for cond, ms in h["paired_differences"].items():
        for m, v in ms.items():
            rows.append({"section": "heldout_paired_difference", "condition": cond, "metric": m, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
    for cond, ms in result["length"]["per_fork"].items():
        for m, v in ms.items():
            if isinstance(v, dict):
                rows.append({"section": "length", "condition": cond, "metric": m, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
            else:
                rows.append({"section": "length", "condition": cond, "metric": m, "point": v})
    for m, v in result["length"]["dr_grpo_minus_grpo"].items():
        rows.append({"section": "length_difference", "condition": "dr_grpo-grpo", "metric": m, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
    for cond, us in result["terms"].items():
        for u in us:
            for m in ("surrogate_term", "beta_kl_term"):
                rows.append({"section": "terms", "condition": cond, "update": u["update"], "metric": m, "point": u[m]})
    for cond, block in result["term_gradients"].items():
        for m, v in block.items():
            if isinstance(v, dict):
                rows.append({"section": "term_gradients", "condition": cond, "metric": m, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
        for u in block["per_update"]:
            for m in ("grad_norm_surrogate", "grad_norm_beta_kl"):
                rows.append({"section": "term_gradients", "condition": cond, "update": u["update"], "metric": m, "point": u[m]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = cfg["results_dir"]
    out_json, out_csv = repo_path(f"{rd}/normalization.json"), repo_path(f"{rd}/normalization.csv")
    for p in (out_json, out_csv):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    seed = int(cfg["seed"])
    runs = (CANONICAL, DR)
    trains = {r: load_json(f"{rd}/train_{r}.json") for r in runs}
    evals = {r: load_json(f"{rd}/eval_{r}.json") for r in runs}
    rollouts = {r: read_jsonl(f"{rd}/rollouts_{r}.jsonl") for r in runs}
    gens = {r: read_jsonl(f"{rd}/generations_eval_{r}.jsonl") for r in runs}

    checks = protocol(trains, evals, rollouts)
    result = {
        "script": "task3_grpo.compare_normalization",
        "git": git_state(),
        "seed": seed,
        "runs": {"grpo": CANONICAL, "dr_grpo": DR},
        "bootstrap": f"{N_RESAMPLES} resamples, seed {seed}, 95% percentile interval",
        "protocol_checks": checks,
        "n_failed_checks": sum(not c["passed"] for c in checks),
        "update1_identity": update1_identity(rollouts),
        "generated_tokens": {r: trains[r]["generated_tokens_total"] for r in runs},
        "heldout": heldout_block(gens, {"grpo": CANONICAL, "dr_grpo": DR}, [("dr_grpo", "grpo")], seed),
        "length": length_section(trains, seed),
        "terms": terms_section(trains),
        "term_gradients": term_gradient_section(trains, seed),
    }
    result["wall_clock_seconds"] = elapsed()
    save_json(out_json, result)
    write_csv(out_csv, csv_rows(result))
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    print(f"failed checks: {result['n_failed_checks']}; wrote {out_json} and {out_csv} in {elapsed():.0f}s")


if __name__ == "__main__":
    main()
