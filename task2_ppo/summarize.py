"""Task 2 summary: protocol checks, standard-run table, cached clipping study, clipping and KL forks.

Runs on the Mac from saved result files only (no model is loaded). The fork equality check also reads
the small final LoRA adapter files in outputs/task2_ppo (local copies of the Kaggle adapters).
Writes, into the results dir: summary.json (everything) and summary.csv (long format: one row per
section, condition, metric).

Uncertainty: percentile bootstrap, N_RESAMPLES resamples, a fresh numpy default_rng(seed) for every
quantity (so every quantity uses the same prompt resamples), paired over the held-out prompts.
KL and entropy are recomputed per resample as (sum of per-response token sums) / (sum of token counts).
Trend tests: OLS slope against update number with a 95% t interval (df = n - 2).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json, wall_timer
from task1_dpo.summarize import (Inputs, align, bootstrap, diff_stat, iid_resampler, mean_stat,
                                 pooled_ratio_stat, prompt_key, write_csv)
from task1_dpo.train import git_state

STANDARD = "standard"
CLIP_FORKS = {0.05: "fork_eps0p05_kl0p10", 0.20: "fork_eps0p20_kl0p10", 0.50: "fork_eps0p50_kl0p10"}
KL_FORKS = {0.0: "fork_eps0p20_kl0p00", 0.10: "fork_eps0p20_kl0p10", 0.20: "fork_eps0p20_kl0p20"}
FORKS = sorted(set(CLIP_FORKS.values()) | set(KL_FORKS.values()))
EVALS = [STANDARD, *FORKS]
CLIP_PAIRS = ((0.05, 0.20), (0.05, 0.50), (0.20, 0.50))
KL_PAIRS = ((0.0, 0.20), (0.0, 0.10), (0.10, 0.20))
N_RESAMPLES = 10_000
TOL = 1.0e-8
T975 = {18: 2.10092204024096}  # Student t 97.5% quantile; df 18 = 20 updates - 2 (scipy is not a dependency)
# History keys that measure time or memory, not the optimisation; skipped by the fork equality check.
NON_ALGORITHMIC_KEYS = {"seconds", "elapsed_seconds", "peak_vram_bytes"}


class Checks(Inputs):
    """Inputs that records failed checks instead of stopping, so the summary reports a count."""

    def check(self, name: str, ok: bool, detail=None):
        self.checks.append({"check": name, "passed": bool(ok), "detail": detail})


# ---------------------------------------------------------------- statistics

def ols(y) -> dict:
    """OLS slope of y against update number 1..n with a 95% t interval."""
    y = np.asarray(y, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - (y.mean() + slope * xc)
    df = len(y) - 2
    se = math.sqrt((resid ** 2).sum() / df / (xc ** 2).sum())
    t = T975[df]
    return {"slope": slope, "se": se, "ci_low": slope - t * se, "ci_high": slope + t * se, "df": df, "n": len(y)}


def eval_stats(rows: list[dict]) -> dict:
    """Per-prompt arrays from a generations file, in file order."""
    return {
        "rm": mean_stat([r["rm_score"] for r in rows]),
        "kl": pooled_ratio_stat([r["sum_log_ratio"] for r in rows], [r["token_length"] for r in rows]),
        "entropy": pooled_ratio_stat([r["sum_neg_logp"] for r in rows], [r["token_length"] for r in rows]),
        "length": mean_stat([r["token_length"] for r in rows]),
    }


METRICS = ("rm", "kl", "entropy", "length")


def boot(stat, n, seed, method):
    return bootstrap(stat, n, iid_resampler(n), seed, N_RESAMPLES, method)


def heldout_block(gens: dict[str, list[dict]], labels: dict, pairs, seed: int) -> dict:
    """Per-condition estimates and paired differences (a minus b) for every metric."""
    names = list(labels.values())
    base = gens[names[0]]
    aligned = {n: align(base, gens[n], prompt_key, f"held-out {names[0]} vs {n}")[1] for n in names}
    n = len(base)
    stats = {lab: eval_stats(aligned[name]) for lab, name in labels.items()}
    per = {str(lab): {m: boot(stats[lab][m], n, seed, f"{m}, {N_RESAMPLES} prompt resamples") for m in METRICS}
           for lab in labels}
    diffs = {f"{a}-{b}": {m: boot(diff_stat(stats[a][m], stats[b][m]), n, seed, f"paired {m} difference")
                          for m in METRICS} for a, b in pairs}
    return {"n_prompts": n, "conditions": {str(k): v for k, v in labels.items()}, "per_condition": per,
            "paired_differences": diffs}


def flat_numbers(obj, prefix=""):
    """{path: float} for every numeric leaf, skipping timing and memory keys."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in NON_ALGORITHMIC_KEYS:
                out.update(flat_numbers(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flat_numbers(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def history_max_diff(ha: list[dict], hb: list[dict]) -> dict:
    fa, fb = flat_numbers(ha), flat_numbers(hb)
    keys_same = set(fa) == set(fb)
    common = sorted(set(fa) & set(fb))
    diffs = {k: abs(fa[k] - fb[k]) for k in common}
    worst = max(diffs, key=diffs.get) if diffs else None
    return {"same_keys": keys_same, "n_values": len(common), "max_abs_diff": diffs[worst] if worst else None,
            "argmax": worst, "n_values_differing": sum(d > 0 for d in diffs.values())}


def adapter_max_diff(run_a: str, run_b: str, part: str) -> dict:
    from safetensors.torch import load_file
    a = load_file(str(repo_path(f"outputs/task2_ppo/{run_a}/{part}/adapter_model.safetensors")))
    b = load_file(str(repo_path(f"outputs/task2_ppo/{run_b}/{part}/adapter_model.safetensors")))
    if set(a) != set(b):
        return {"same_tensor_names": False}
    per = {k: float((a[k].double() - b[k].double()).abs().max()) for k in sorted(a)}
    worst = max(per, key=per.get)
    return {"same_tensor_names": True, "n_tensors": len(per), "max_abs_diff": per[worst], "argmax": worst,
            "n_tensors_differing": sum(v > 0 for v in per.values()), "per_tensor": per}


def first_constant_sign(d) -> dict:
    """First update u such that sign(d_u..d_last) is constant and non-zero; 'none' if no such u."""
    s = np.sign(np.asarray(d, dtype=float))
    for u in range(len(s)):
        if s[u] != 0 and np.all(s[u:] == s[u]):
            return {"first_update": u + 1, "sign": int(s[u])}
    return {"first_update": "none", "sign": None}


# ---------------------------------------------------------------- per-update tables

def update_row(h: dict) -> dict:
    e1, e2 = h["epochs"]
    return {
        "update": h["update"], "prompt_id": h["prompt_id"],
        "reward_raw_mean": h["reward_raw"]["mean"], "reward_effective_mean": h["reward_effective"]["mean"],
        "kl": h["kl_sampled"], "entropy": h["entropy_sampled"],
        "policy_loss_ep1": e1["policy_loss"], "policy_loss_ep2": e2["policy_loss"],
        "value_loss_ep1": e1["value_loss"], "value_loss_ep2": e2["value_loss"],
        "policy_grad_norm_ep1": e1["policy_grad_norm_before_clip"], "policy_grad_norm_ep2": e2["policy_grad_norm_before_clip"],
        "value_grad_norm_ep1": e1["value_grad_norm_before_clip"], "value_grad_norm_ep2": e2["value_grad_norm_before_clip"],
        "clip_fraction_ep1": e1["clip_fraction"], "clip_fraction_ep2": e2["clip_fraction"],
        "max_abs_log_ratio_ep2": e2["max_abs_log_ratio_before_step"],
        "length_mean": h["response_length"]["mean"], "length_sd": h["response_length"]["sd"],
        "n_eos": h["n_eos"], "n_truncated": h["n_truncated"], "generated_tokens": h["generated_tokens"],
        "explained_variance": h["explained_variance"], "stability_statistic": h["stability_statistic"],
        "n_rm_input_truncated": h["n_rm_input_truncated"],
    }


def col(rows, key):
    return [r[key] for r in rows]


# ---------------------------------------------------------------- sections

def protocol(inp: Checks, trains: dict, evals: dict, gens: dict, cached: dict) -> dict:
    for run, t in trains.items():
        n_exp = t["effective"]["updates"]
        inp.check(f"{run}: training completed", t["status"] == "completed" and t["updates_completed"] == n_exp
                  and len(t["history"]) == n_exp, {"status": t["status"], "updates_completed": t["updates_completed"]})
        inp.check(f"{run}: no skipped optimizer steps", t["n_nonfinite_steps"] == 0, t["n_nonfinite_steps"])
        inp.check(f"{run}: reference adapter base matches config", t["reference"]["match"], t["reference"])
    for run in FORKS:
        inp.check(f"{run}: update budget equals fork_updates", trains[run]["effective"]["updates"] == trains[run]["config"]["fork_updates"],
                  trains[run]["effective"]["updates"])
    for name, e in evals.items():
        inp.check(f"eval {name}: completed full run", e["status"] == "completed" and not e["smoke"] and e["limit"] is None
                  and e["metrics"]["n_responses"] == 165, {"status": e["status"], "n": e["metrics"]["n_responses"]})
        inp.check(f"eval {name}: adapter is the trained {name} policy",
                  repo_path(e["adapter"]).resolve() == repo_path(trains[name]["policy_adapter_path"]).resolve(), e["adapter"])
    inp.check("cached study: completed", cached["status"] == "completed" and not cached["smoke"], cached["status"])

    std_ids = trains[STANDARD]["prompts"]["prompt_ids"]
    inp.check("standard: logged prompt IDs match history", std_ids == [h["prompt_id"] for h in trains[STANDARD]["history"]], None)
    for run in FORKS:
        ids = [h["prompt_id"] for h in trains[run]["history"]]
        inp.check(f"{run}: prompt-ID sequence equals standard updates 1-8", ids == std_ids[:8] == trains[run]["prompts"]["prompt_ids"],
                  {"n_mismatch": sum(a != b for a, b in zip(ids, std_ids[:8]))})

    id_lists = {n: [r["prompt_id"] for r in g] for n, g in gens.items()}
    ref = id_lists[STANDARD]
    for n in EVALS[1:]:
        inp.check(f"eval {n}: same 165 prompt IDs in the same order as standard", id_lists[n] == ref, None)
    inp.check("eval: 165 unique prompt IDs", len(ref) == 165 == len(set(ref)), len(set(ref)))
    seeds = {n: (e["seed"], e["settings"]["seed_reset_before_generation"]) for n, e in evals.items()}
    inp.check("eval: identical seed and seed reset across evaluations", len(set(seeds.values())) == 1, seeds)
    settings = {n: e["settings"] for n, e in evals.items()}
    inp.check("eval: identical settings across evaluations", all(s == settings[STANDARD] for s in settings.values()), None)
    gen_cfg = {n: (t["effective"]["generation"], t["effective"]["model_generation_config"]) for n, t in trains.items()}
    inp.check("training: identical generation settings across runs", all(v == gen_cfg[STANDARD] for v in gen_cfg.values()), None)

    commits = {**{f"train_{n}": t["git"] for n, t in trains.items()}, **{f"eval_{n}": e["git"] for n, e in evals.items()},
               "cached_clip_study": cached["git"]}
    inp.check("same commit across all runs", len({c["commit"] for c in commits.values()}) == 1,
              sorted({c["commit"] for c in commits.values()}))
    inp.check("no run from a dirty tree", not any(c["dirty"] for c in commits.values()), None)

    for name, e in evals.items():
        inp.check(f"eval {name}: KL recomputed from per-prompt sums",
                  abs(pooled_ratio_stat([r["sum_log_ratio"] for r in gens[name]], [r["token_length"] for r in gens[name]])(np.arange(165))
                      - e["metrics"]["kl"]) <= TOL, None)
        inp.check(f"eval {name}: entropy recomputed from per-prompt sums",
                  abs(pooled_ratio_stat([r["sum_neg_logp"] for r in gens[name]], [r["token_length"] for r in gens[name]])(np.arange(165))
                      - e["metrics"]["entropy"]) <= TOL, None)
        inp.check(f"eval {name}: RM mean recomputed", abs(np.mean([r["rm_score"] for r in gens[name]]) - e["metrics"]["reward"]["mean"]) <= TOL, None)

    passed = [c["check"] for c in inp.checks if c["passed"]]
    failed = [c for c in inp.checks if not c["passed"]]
    # Effective sampling: generate() merges the model's generation_config.json with the passed kwargs
    # (temperature, top_p, do_sample), so keys not passed keep the generation_config value.
    mgc = trains[STANDARD]["effective"]["model_generation_config"]
    passed_kwargs = trains[STANDARD]["effective"]["generation"]
    effective = {"temperature": passed_kwargs["temperature"], "top_p": passed_kwargs["top_p"],
                 "do_sample": passed_kwargs["do_sample"], "top_k": mgc.get("top_k"),
                 "repetition_penalty": mgc.get("repetition_penalty")}
    return {
        "n_checks": len(inp.checks), "n_passed": len(passed), "n_failed": len(failed), "failed": failed,
        "effective_generation_settings": {"effective": effective, "passed_by_code": passed_kwargs,
                                          "model_generation_config": mgc,
                                          "note": "top_k and repetition_penalty are not passed by common.generation.batch_generate, "
                                                  "so generate() keeps the model generation_config values; identical across all runs"},
        "rm_input_truncated": {
            "eval": {n: e["metrics"]["n_rm_input_truncated"] for n, e in evals.items()},
            "train": {n: int(sum(h["n_rm_input_truncated"] for h in t["history"])) for n, t in trains.items()},
        },
        "commit": sorted({c["commit"] for c in commits.values()}),
    }


def standard_section(train: dict) -> tuple[dict, list[dict]]:
    rows = [update_row(h) for h in train["history"]]
    first, last = slice(0, 5), slice(15, 20)
    critic = {}
    for key in ("value_loss_ep1", "value_loss_ep2", "explained_variance"):
        v = col(rows, key)
        critic[key] = {"ols": ols(v), "mean_updates_1_5": float(np.mean(v[first])), "mean_updates_16_20": float(np.mean(v[last]))}
    ep2 = col(rows, "clip_fraction_ep2")
    return {
        "per_update": rows,
        "peak_vram_bytes": train["peak_vram_bytes"], "peak_vram_gib": train["peak_vram_bytes"] / 2 ** 30,
        "wall_clock_seconds": train["wall_clock_seconds"], "setup_seconds": train["setup_seconds"],
        "generated_tokens_total": train["generated_tokens_total"],
        "kl_trend": ols(col(rows, "kl")),
        "critic_trends": critic,
        "max_epoch2_clip_fraction": {"max": max(ep2), "update": int(np.argmax(ep2)) + 1,
                                     "max_epoch1_clip_fraction": max(col(rows, "clip_fraction_ep1")),
                                     "max_abs_log_ratio_ep2": max(col(rows, "max_abs_log_ratio_ep2"))},
    }, rows


def cached_section(c: dict) -> dict:
    return {
        "eps_results": c["eps_results"],
        "rule": c["rule"],
        "branch_fired": c["rule"]["branch"],
        "log_prob_difference": {"mean_abs_token_pooled": c["recompute"]["mean_abs_diff_token_pooled"],
                                "max_abs": c["recompute"]["max_abs_diff"], "n_tokens": c["recompute"]["n_tokens"]},
        "long_prompt_diagnostic": c["recompute"]["long_prompt_diagnostic"]["rows"],
        "token_rebuild": c["token_rebuild"], "identity_checks": c["identity_checks"],
    }


def clipping_section(trains: dict, gens: dict, seed: int) -> dict:
    per_update, stab = {}, {}
    for eps, run in CLIP_FORKS.items():
        rows = [update_row(h) for h in trains[run]["history"]]
        s = col(rows, "stability_statistic")
        per_update[str(eps)] = {"stability_statistic": s, "clip_fraction_ep2": col(rows, "clip_fraction_ep2"),
                                "max_abs_log_ratio_ep2": col(rows, "max_abs_log_ratio_ep2")}
        stab[str(eps)] = {"max": max(s), "mean": float(np.mean(s))}
    eq = {}
    for a, b in CLIP_PAIRS:
        ra, rb = CLIP_FORKS[a], CLIP_FORKS[b]
        eq[f"{a}-{b}"] = {"policy_lora": adapter_max_diff(ra, rb, "policy"), "value_lora": adapter_max_diff(ra, rb, "value"),
                          "history": history_max_diff(trains[ra]["history"], trains[rb]["history"]),
                          "heldout_generations_identical_text": [x["text"] for x in gens[ra]] == [x["text"] for x in gens[rb]]}
    return {"per_update": per_update, "stability_summary": stab, "equality": eq,
            "heldout": heldout_block(gens, CLIP_FORKS, CLIP_PAIRS, seed)}


def kl_section(trains: dict, gens: dict, seed: int) -> dict:
    rows = {beta: [update_row(h) for h in trains[run]["history"]] for beta, run in KL_FORKS.items()}
    per_update = {str(b): {k: col(r, k) for k in ("kl", "reward_raw_mean", "entropy", "length_mean")} for b, r in rows.items()}
    sign = {}
    for key in ("kl", "reward_raw_mean", "entropy", "length_mean"):
        d = [x - y for x, y in zip(col(rows[0.0], key), col(rows[0.20], key))]
        sign[key] = {"diff_beta0_minus_beta0p20": d, **first_constant_sign(d)}
    return {"per_update": per_update, "beta0_vs_beta0p20": sign, "heldout": heldout_block(gens, KL_FORKS, KL_PAIRS, seed)}


def standard_heldout(e: dict, gens: list[dict], seed: int) -> dict:
    m = e["metrics"]
    st = eval_stats(gens)
    return {"rm_mean": m["reward"]["mean"], "rm_sd": m["reward"]["std"], "kl": m["kl"], "entropy": m["entropy"],
            "length_mean": m["length"]["mean"], "length_sd": m["length"]["std"],
            "truncation_rate_at_768": m["truncation_rate_at_cap"], "n_truncated_at_768": m["n_truncated_at_cap"],
            "cap": e["settings"]["max_new_tokens"], "n_prompts": len(gens),
            "bootstrap": {k: boot(st[k], len(gens), seed, k) for k in METRICS},
            "note": "for Task 4; not compared with forks (20 vs 8 updates)"}


def budget_section(trains: dict) -> dict:
    return {run: {"updates_completed": trains[run]["updates_completed"], "generated_tokens_total": trains[run]["generated_tokens_total"],
                  "prompts_per_update": trains[run]["config"]["prompts_per_update"],
                  "responses_per_prompt": trains[run]["effective"]["responses_per_prompt"],
                  "ppo_epochs": trains[run]["effective"]["ppo_epochs"]} for run in [STANDARD, *FORKS]}


# ---------------------------------------------------------------- csv

def csv_rows(s: dict) -> list[dict]:
    rows = []

    def add(section, condition, metric, value, lo=None, hi=None):
        rows.append({"section": section, "condition": condition, "metric": metric, "value": value, "ci_low": lo, "ci_high": hi})

    for r in s["standard"]["per_update"]:
        for k, v in r.items():
            if k not in ("update", "prompt_id"):
                add("standard_per_update", f"update_{r['update']}", k, v)
    kl = s["standard"]["kl_trend"]
    add("kl_trend", STANDARD, "kl_ols_slope_per_update", kl["slope"], kl["ci_low"], kl["ci_high"])
    for k, v in s["standard"]["critic_trends"].items():
        add("critic_trends", STANDARD, f"{k}_ols_slope", v["ols"]["slope"], v["ols"]["ci_low"], v["ols"]["ci_high"])
        add("critic_trends", STANDARD, f"{k}_mean_updates_1_5", v["mean_updates_1_5"])
        add("critic_trends", STANDARD, f"{k}_mean_updates_16_20", v["mean_updates_16_20"])
    add("max_epoch2_clip_fraction", STANDARD, "max_epoch2_clip_fraction", s["standard"]["max_epoch2_clip_fraction"]["max"])
    for r in s["cached_study"]["eps_results"]:
        for k in ("clip_fraction", "affected_fraction", "clipped_surrogate", "unclipped_surrogate"):
            add("cached_clip_study", f"eps_{r['eps']}", k, r[k])
    for eps, v in s["clipping_forks"]["stability_summary"].items():
        add("clip_fork_stability", f"eps_{eps}", "max", v["max"])
        add("clip_fork_stability", f"eps_{eps}", "mean", v["mean"])
    for sec, block in (("clip_fork_heldout", s["clipping_forks"]["heldout"]), ("kl_fork_heldout", s["kl_forks"]["heldout"])):
        for cond, ms in block["per_condition"].items():
            for m, b in ms.items():
                add(sec, cond, m, b["point"], b["ci_low"], b["ci_high"])
        for pair, ms in block["paired_differences"].items():
            for m, b in ms.items():
                add(sec, f"diff_{pair}", m, b["point"], b["ci_low"], b["ci_high"])
    for k, v in s["kl_forks"]["beta0_vs_beta0p20"].items():
        for u, d in enumerate(v["diff_beta0_minus_beta0p20"], 1):
            add("kl_fork_beta0_vs_beta0p20", f"update_{u}", f"{k}_diff_beta0_minus_beta0p20", d)
        add("kl_fork_beta0_vs_beta0p20", "first_constant_sign_update", k, v["first_update"])
    for k, b in s["standard_heldout"]["bootstrap"].items():
        add("standard_heldout", STANDARD, k, b["point"], b["ci_low"], b["ci_high"])
    add("standard_heldout", STANDARD, "truncation_rate_at_768", s["standard_heldout"]["truncation_rate_at_768"])
    for run, v in s["budget"].items():
        add("budget", run, "updates_completed", v["updates_completed"])
        add("budget", run, "generated_tokens_total", v["generated_tokens_total"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--results-dir", help="default: config results_dir")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    elapsed = wall_timer()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    results_dir = args.results_dir or cfg["results_dir"]
    inp = Checks(results_dir)
    outputs = ["summary.json", "summary.csv"]
    clash = [o for o in outputs if inp.path(o).exists()]
    if clash and not args.overwrite:
        raise SystemExit(f"{clash} exist in {results_dir}; pass --overwrite to replace them.")
    inp.require([f"train_{r}.json" for r in [STANDARD, *FORKS]] + [f"eval_{r}.json" for r in EVALS]
                + [f"generations_eval_{r}.jsonl" for r in EVALS] + ["cached_clip_study.json"])

    trains = {r: inp.json(f"train_{r}.json") for r in [STANDARD, *FORKS]}
    evals = {r: inp.json(f"eval_{r}.json") for r in EVALS}
    gens = {r: inp.jsonl(f"{results_dir}/generations_eval_{r}.jsonl") for r in EVALS}
    cached = inp.json("cached_clip_study.json")

    summary = {
        "script": "task2_ppo.summarize", "config_path": args.config, "results_dir": results_dir, "git": git_state(), "seed": seed,
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": seed, "interval": "95% percentile (2.5th, 97.5th)",
                      "paired_over": "165 held-out prompt IDs",
                      "rng": "a fresh numpy.random.default_rng(seed) for every quantity",
                      "kl_entropy": "per resample: sum of per-prompt token sums / sum of token counts"},
        "definitions": {
            "ols": "slope of the per-update value against update number 1..n, 95% t interval with df n-2",
            "value_loss": "value_loss_ep1 is the critic loss on fresh rollouts before any critic step of that update; ep2 after one step",
            "first_constant_sign_update": "first update u such that the sign of (beta 0 minus beta 0.20) is constant and non-zero for u..8",
            "fork_equality": "max absolute difference over all tensors of the final adapters, and over all numeric "
                             "per-update logged values except timing and memory"},
    }
    summary["protocol"] = protocol(inp, trains, evals, gens, cached)
    summary["standard"], _ = standard_section(trains[STANDARD])
    summary["cached_study"] = cached_section(cached)
    summary["clipping_forks"] = clipping_section(trains, gens, seed)
    summary["kl_forks"] = kl_section(trains, gens, seed)
    summary["standard_heldout"] = standard_heldout(evals[STANDARD], gens[STANDARD], seed)
    summary["budget"] = budget_section(trains)
    summary["checks"] = inp.checks
    summary["input_sha256"] = inp.hashes
    summary["wall_clock_seconds"] = elapsed()

    save_json(inp.path("summary.json"), summary)
    write_csv(inp.path("summary.csv"), csv_rows(summary))
    p = summary["protocol"]
    print(f"{p['n_passed']}/{p['n_checks']} protocol checks passed; wrote {', '.join(outputs)} to {results_dir}")
    for f in p["failed"]:
        print("FAILED:", f["check"], f["detail"])


if __name__ == "__main__":
    main()
