"""Held-out reward-model differences between the Task 4 policies, from saved per-prompt scores only (Mac).

GRPO standard minus PPO standard: both evaluated on the same held-out RL prompts (task2_ppo.evaluate and
task3_grpo.evaluate). The script first checks the same prompt ids and identical protocol fields, then
computes a paired percentile bootstrap over prompts (task1_dpo.summarize.bootstrap).

DPO standard minus SFT: copied from the Task 1 summary (its own prompt set and caps), cited by path and hash.
The two differences use different prompt sets and caps and are stored side by side, not combined.

Written before any XSTest result exists. Writes results/task4_safety/reward_pairs.json.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.summarize import N_RESAMPLES, bootstrap, diff_stat, iid_resampler, mean_stat
from task1_dpo.train import git_state
from task4_safety.protocol import sha256_file, task_dir

PPO_EVAL, PPO_GEN = "results/task2_ppo/eval_standard.json", "results/task2_ppo/generations_eval_standard.jsonl"
GRPO_EVAL, GRPO_GEN = "results/task3_grpo/eval_standard.json", "results/task3_grpo/generations_eval_standard.jsonl"
DPO_SUMMARY = "results/task1_dpo/summary_steps12.json"
DPO_EVALS = ("results/task1_dpo/eval_standard.json", "results/task1_dpo/eval_sft.json")
PROTOCOL_KEYS = ("samples_per_prompt", "temperature", "top_p", "do_sample", "max_prompt_length", "max_new_tokens",
                 "seed_reset_before_generation", "gen_batch_size", "reward_model", "reward_max_length")
EXPECTED_ADAPTERS = {"ppo": "outputs/task2_ppo/standard/policy", "grpo": "outputs/task3_grpo/standard/policy"}
TOL = 1.0e-8


def protocol_check(ev: dict) -> dict:
    """Both held-out evaluations must be complete, non-smoke, and share prompts, seed and every protocol field."""
    a, b = ev["ppo"], ev["grpo"]
    fields = {k: {"ppo": a["settings"].get(k), "grpo": b["settings"].get(k)} for k in PROTOCOL_KEYS}
    out = {
        "fields": fields,
        "fields_equal": all(v["ppo"] == v["grpo"] and v["ppo"] is not None for v in fields.values()),
        "seed_equal": a["seed"] == b["seed"],
        "prompt_filter_equal": a["prompts"] == b["prompts"],
        "completed_not_smoke": all(e["status"] == "completed" and not e["smoke"] and e.get("limit") is None for e in (a, b)),
        "adapters": {k: ev[k]["adapter"] for k in ev},
        "adapters_standard": all(ev[k]["adapter"] == EXPECTED_ADAPTERS[k] for k in ev),
    }
    out["passed"] = all(out[k] for k in ("fields_equal", "seed_equal", "prompt_filter_equal", "completed_not_smoke", "adapters_standard"))
    return out


def grpo_minus_ppo(ppo_rows: list[dict], grpo_rows: list[dict], ev: dict, seed: int) -> dict:
    ids_p, ids_g = [r["prompt_id"] for r in ppo_rows], [r["prompt_id"] for r in grpo_rows]
    if len(set(ids_p)) != len(ids_p) or set(ids_p) != set(ids_g):
        raise SystemExit("PPO and GRPO held-out files do not cover the same unique prompt ids")
    g_by = {r["prompt_id"]: r for r in grpo_rows}
    rp = np.array([r["rm_score"] for r in ppo_rows], dtype=float)
    rg = np.array([g_by[i]["rm_score"] for i in ids_p], dtype=float)
    for name, x in (("ppo", rp), ("grpo", rg)):
        if abs(float(x.mean()) - ev[name]["metrics"]["reward"]["mean"]) > TOL:
            raise SystemExit(f"{name}: recomputed RM mean differs from its eval JSON")
    n = len(ids_p)
    return {
        "n_prompts": n,
        "same_prompt_ids": True,
        "same_order": ids_p == ids_g,
        "n_rm_input_truncated": {"ppo": int(sum(r["rm_input_truncated"] for r in ppo_rows)), "grpo": int(sum(r["rm_input_truncated"] for r in grpo_rows))},
        "mean_rm": {"ppo": float(rp.mean()), "grpo": float(rg.mean())},
        "rm_mean": bootstrap(diff_stat(mean_stat(rg), mean_stat(rp)), n, iid_resampler(n), seed,
                             method="paired percentile bootstrap over prompt ids: the same resampled indices for both; difference = GRPO minus PPO"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    out = task_dir(cfg) / "reward_pairs.json"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it.")

    ev = {"ppo": load_json(PPO_EVAL), "grpo": load_json(GRPO_EVAL)}
    proto = protocol_check(ev)
    if not proto["passed"]:
        raise SystemExit(f"protocol mismatch between PPO and GRPO held-out evaluations: {proto}")
    gp = grpo_minus_ppo(read_jsonl(PPO_GEN), read_jsonl(GRPO_GEN), ev, seed)

    t1 = load_json(DPO_SUMMARY)
    dpo_gen = {p: load_json(p)["modes"]["generate"] for p in DPO_EVALS}
    t1_settings = {p: {k: g["settings"].get(k) for k in ("temperature", "top_p", "do_sample", "max_new_tokens", "max_prompt_length",
                                                         "seed_reset_before_pass", "rm_max_length", "reward_model")}
                   for p, g in dpo_gen.items()}
    result = {
        "script": "task4_safety.reward_pairs",
        "git": git_state(),
        "seed": seed,
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": seed, "interval": "95% percentile (2.5th, 97.5th)",
                      "rng": "a fresh numpy.random.default_rng(seed) for every quantity"},
        "grpo_minus_ppo": {
            **gp,
            "prompts": ev["ppo"]["prompts"],
            "protocol_check": proto,
            "sources": {p: sha256_file(repo_path(p)) for p in (PPO_EVAL, PPO_GEN, GRPO_EVAL, GRPO_GEN)},
        },
        "dpo_minus_sft": {
            "rm_mean": t1["uncertainty"]["standard_minus_sft"]["rm_mean"],
            "source": f"{DPO_SUMMARY} uncertainty.standard_minus_sft.rm_mean",
            "source_sha256": sha256_file(repo_path(DPO_SUMMARY)),
            "source_git": t1["git"],
            "n_prompts": {p: g["metrics"]["n_responses"] if "metrics" in g else None for p, g in dpo_gen.items()},
            "prompts_path": {p: g["filter"]["path"] for p, g in dpo_gen.items()},
            "settings": t1_settings,
        },
        "note": "the two differences use different prompt sets, response caps and RM caps; they are not on one scale",
    }
    save_json(out, result)
    r = gp["rm_mean"]
    print(f"GRPO-PPO RM mean difference over {gp['n_prompts']} prompts: {r['point']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]; "
          f"DPO-SFT (Task 1): {result['dpo_minus_sft']['rm_mean']['point']:.4f} "
          f"[{result['dpo_minus_sft']['rm_mean']['ci_low']:.4f}, {result['dpo_minus_sft']['rm_mean']['ci_high']:.4f}]; wrote {out.relative_to(repo_path('.'))}")


if __name__ == "__main__":
    main()
