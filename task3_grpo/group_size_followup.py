"""Task 3 Step 2 follow-ups on the K=8 cache (Mac, no model). Reads group_size.json; never rewrites it.

group_size_std_bias.json
  For K in group_sizes: the normal-theory expectation of the population std of K i.i.d. samples,
  E[s_pop] / sigma = c4(K) * sqrt((K-1)/K), with c4(K) = sqrt(2/(K-1)) * Gamma(K/2) / Gamma((K-1)/2);
  its ratio to the largest K; and the observed ratio of mean_within_group_std (scope "all", point
  estimate from group_size.json) to the largest K.

group_size_qualitative.json
  Rule fixed before looking at any text: every cached prompt whose 8 completions form an uninformative
  group at K=8 (task3_grpo.grpo_utils.INFORMATIVE_RULE). Per prompt: tertile, the 8 rewards, completion_tokens, terminated_with_eos,
  whether the 8 texts are identical, the number of distinct texts, and the first QUAL_CHARS characters
  of each distinct text (in first-appearance order by generation_index, with the indices that share it).
"""
from __future__ import annotations

import argparse
import math

import torch

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from common.run_info import git_state
from task3_grpo.analyze_group_size import N_CACHED, check_cache, load_k8_cache
from task3_grpo.grpo_utils import INFORMATIVE_RULE, is_informative

QUAL_CHARS = 300
STD_BIAS_NOTE = ("regroup noise and sign-flip at K=8 are 0 by construction (reference = the same 8 completions), "
                 "so that measure applies to K=2 and K=4 only")


def c4(k: int) -> float:
    """E[sample std with ddof 1] / sigma for k i.i.d. normal samples."""
    return math.sqrt(2.0 / (k - 1)) * math.exp(math.lgamma(k / 2) - math.lgamma((k - 1) / 2))


def expected_population_std_ratio(k: int) -> float:
    """E[population std (ddof 0) of k i.i.d. normal samples] / sigma = c4(k) * sqrt((k-1)/k)."""
    return c4(k) * math.sqrt((k - 1) / k)


def std_bias(group_size: dict) -> dict:
    sizes = [int(k) for k in group_size["group_sizes"]]
    top = max(sizes)
    observed = {k: float(group_size["results"]["all"][f"K{k}"]["mean_within_group_std"]["point"]) for k in sizes}
    rows = {}
    for k in sizes:
        rows[f"K{k}"] = {
            "c4": c4(k),
            "sqrt_(K-1)/K": math.sqrt((k - 1) / k),
            "expected_pop_std_over_sigma": expected_population_std_ratio(k),
            f"expected_ratio_to_K{top}": expected_population_std_ratio(k) / expected_population_std_ratio(top),
            "observed_mean_within_group_std": observed[k],
            f"observed_ratio_to_K{top}": observed[k] / observed[top],
        }
    return {"reference_K": top, "per_K": rows, "observed_source": "group_size.json results.all.K*.mean_within_group_std.point"}


def qualitative(by_prompt, per_prompt: list[dict]) -> list[dict]:
    bins = {p["prompt_id"]: p["bin"] for p in per_prompt}
    out = []
    for rows in by_prompt.values():
        rewards = [float(r["reward"]) for r in rows]
        if is_informative(torch.tensor(rewards, dtype=torch.float64)):
            continue
        texts = [r["completion"] for r in rows]
        distinct: dict[str, list[int]] = {}
        for r in rows:
            distinct.setdefault(r["completion"], []).append(int(r["generation_index"]))
        out.append({
            "prompt_id": rows[0]["prompt_id"],
            "source_index": rows[0]["source_index"],
            "tertile": bins[rows[0]["prompt_id"]],
            "rewards": rewards,
            "completion_tokens": [int(r["completion_tokens"]) for r in rows],
            "terminated_with_eos": [bool(r["terminated_with_eos"]) for r in rows],
            "all_texts_identical": len(set(texts)) == 1,
            "n_distinct_texts": len(distinct),
            "distinct_completions": [{"generation_indices": idx, "first_chars": t[:QUAL_CHARS]} for t, idx in distinct.items()],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = cfg["results_dir"]
    out_bias, out_qual = repo_path(f"{rd}/group_size_std_bias.json"), repo_path(f"{rd}/group_size_qualitative.json")
    for p in (out_bias, out_qual):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    group_size = load_json(f"{rd}/group_size.json")
    by_prompt = load_k8_cache(cfg["group_cache"])
    cache = check_cache(by_prompt)
    meta = {"git": git_state(), "seed": int(cfg["seed"]), "source": f"{rd}/group_size.json",
            "cache": {"path": cfg["group_cache"], **cache}}

    bias = {"script": "task3_grpo.group_size_followup", **meta,
            "formula": "E[population std of K i.i.d. normal samples] / sigma = c4(K) * sqrt((K-1)/K)",
            **std_bias(group_size), "note": STD_BIAS_NOTE}
    save_json(out_bias, bias)

    prompts = qualitative(by_prompt, group_size["per_prompt"])
    expected = sorted(p["prompt_id"] for p in group_size["per_prompt"] if p[f"K{N_CACHED}"]["n_informative"] == 0)
    if sorted(p["prompt_id"] for p in prompts) != expected:
        raise SystemExit(f"uninformative-at-K8 prompts disagree with group_size.json: {expected}")
    qual = {"script": "task3_grpo.group_size_followup", **meta,
            "rule": f"every cached prompt uninformative at K={N_CACHED} ({INFORMATIVE_RULE}); fixed before reading any text",
            "chars_per_completion": QUAL_CHARS, "n_prompts": len(prompts), "prompts": prompts}
    save_json(out_qual, qual)
    print(f"std bias K={list(bias['per_K'])}; qualitative prompts={len(prompts)}; wrote {out_bias} and {out_qual} in {elapsed():.1f}s")


if __name__ == "__main__":
    main()
