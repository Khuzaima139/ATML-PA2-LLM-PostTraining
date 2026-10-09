"""Task 3 Step 2: equal-generation group-size study on the supplied K=8 cache (Mac, no model).

Partition: each prompt's 8 cached completions, sorted by generation_index, are split
into 8/K consecutive groups (K=2: {0,1},{2,3},...; K=4: {0..3},{4..7}; K=8: all). Every K uses the
same 24 prompts x 8 completions = 192 generations. Rewards are the cached `reward` values; clipped
completions stay in their groups (masking only affects a training loss, which this study has none of).
Advantages always come from task3_grpo.grpo.group_relative_advantages (eps 1e-6, population std).

Per K, pooled over the prompts in scope (all 24, or one difficulty bin of 8):
  informative_rate        informative groups / groups (task3_grpo.grpo_utils.INFORMATIVE_RULE); also 1 - rate
  mean_within_group_std   mean over groups of the population reward std
  pooled_advantage_var    population variance of all normalized advantages of the partition
                          (each informative group contributes variance 1, so this is about the informative rate)
  regrouping_noise        for each completion: population variance of its advantage over all size-K subsets
                          of its prompt's 8 completions that contain it; averaged over completions (0 at K=8)
  sign_flip_rate          for each completion with r != mean of its prompt's 8 rewards: fraction of those
                          subsets where sign(A) != sign(r - mean8) and A != 0; no_signal_rate = fraction with
                          A == 0; agree_rate = the rest. Averaged over the eligible completions.
Difficulty bins: tertiles of the per-prompt mean reward over all 8 completions, ties broken by prompt_id,
8 prompts per bin, the same bins for every K. Bin 1 has the lowest mean reward.
Uncertainty: cluster bootstrap over prompts (prompts resampled with replacement, all their completions
kept together), 10,000 resamples, numpy default_rng(seed) per quantity; K=2 minus K=8 differences use
the same prompt resamples for both K.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json, wall_timer
from task1_dpo.summarize import bootstrap, diff_stat, iid_resampler, write_csv
from task1_dpo.train import git_state
from task3_grpo.grpo import group_relative_advantages
from task3_grpo.grpo_utils import INFORMATIVE_RULE, INFORMATIVE_TOL

N_CACHED = 8
N_BINS = 3
N_RESAMPLES = 10_000
BIN_NAMES = ("bin1_low_reward", "bin2_mid_reward", "bin3_high_reward")
QUANTITIES = ("informative_rate", "uninformative_rate", "mean_within_group_std", "pooled_advantage_var",
              "regrouping_noise", "sign_flip_rate", "no_signal_rate", "agree_rate")


# ---------------------------------------------------------------- cache and partition

def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def check_cache(by_prompt) -> dict:
    """Exactly 8 rows per prompt with generation_index 0..7 and one prompt_id per source_index."""
    for key, rows in by_prompt.items():
        if [int(r["generation_index"]) for r in rows] != list(range(N_CACHED)):
            raise SystemExit(f"prompt {key}: generation_index {[r['generation_index'] for r in rows]}, expected 0..7")
        if len({r["prompt_id"] for r in rows}) != 1:
            raise SystemExit(f"prompt {key}: several prompt_ids")
    ids = [rows[0]["prompt_id"] for rows in by_prompt.values()]
    if len(set(ids)) != len(ids):
        raise SystemExit("one prompt_id appears under several source_index values")
    all_rows = [r for rows in by_prompt.values() for r in rows]
    return {
        "n_prompts": len(by_prompt),
        "n_completions": len(all_rows),
        "n_clipped_at_max": int(sum(bool(r["clipped_at_max"]) for r in all_rows)),
        "n_terminated_with_eos": int(sum(bool(r["terminated_with_eos"]) for r in all_rows)),
    }


def regroup_equal_generation_budget(by_prompt, k: int):
    """K-sized groups of consecutive generation_index within each prompt; 192 completions for every K."""
    if N_CACHED % k:
        raise ValueError(f"K={k} does not divide {N_CACHED}")
    groups = []
    for key, rows in by_prompt.items():
        for j in range(N_CACHED // k):
            part = rows[j * k : (j + 1) * k]
            groups.append({
                "source_index": key,
                "prompt_id": part[0]["prompt_id"],
                "group_index": j,
                "generation_indices": [int(r["generation_index"]) for r in part],
                "rewards": [float(r["reward"]) for r in part],
            })
    return groups


def partition_advantages(rewards: np.ndarray, k: int) -> np.ndarray:
    """Advantages of the 8 completions (generation order) under the consecutive K-partition."""
    gid = torch.arange(len(rewards) // k).repeat_interleave(k)
    return group_relative_advantages(torch.tensor(rewards, dtype=torch.float64), gid).numpy()


def subsets_containing(n: int, k: int) -> dict[int, list[tuple[int, ...]]]:
    """For each item i of range(n), every size-k subset of range(n) that contains i (C(n-1, k-1) each)."""
    out = {i: [] for i in range(n)}
    for s in combinations(range(n), k):
        for i in s:
            out[i].append(s)
    return out


def subset_advantages(rewards: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """For each completion i: its advantage in every size-K subset of the prompt's completions containing it."""
    subs = list(combinations(range(len(rewards)), k))
    flat = torch.tensor([rewards[j] for s in subs for j in s], dtype=torch.float64)
    gid = torch.arange(len(subs)).repeat_interleave(k)
    adv = group_relative_advantages(flat, gid).numpy().reshape(len(subs), k)
    out = {i: [] for i in range(len(rewards))}
    for si, s in enumerate(subs):
        for pos, j in enumerate(s):
            out[j].append(adv[si, pos])
    return {i: np.asarray(v) for i, v in out.items()}


# ---------------------------------------------------------------- per-prompt sums

def prompt_stats(rewards: np.ndarray, k: int) -> dict:
    """Sums for one prompt and one K; every pooled quantity is a ratio of sums over prompts."""
    rewards = np.asarray(rewards, dtype=np.float64)
    parts = rewards.reshape(-1, k)
    stds = parts.std(axis=1)  # population std (ddof 0), as in group_relative_advantages
    adv = partition_advantages(rewards, k)
    sub = subset_advantages(rewards, k)
    mean8 = rewards.mean()
    var_sum = sum(float(np.var(sub[i])) for i in range(len(rewards)))
    flip = nosig = eligible = 0.0
    for i, r in enumerate(rewards):
        if r == mean8:
            continue
        a = sub[i]
        ref = np.sign(r - mean8)
        nosig += float(np.mean(a == 0.0))
        flip += float(np.mean((a != 0.0) & (np.sign(a) != ref)))
        eligible += 1
    return {
        "n_groups": parts.shape[0],
        "n_informative": int((stds > INFORMATIVE_TOL).sum()),
        "sum_std": float(stds.sum()),
        "n_completions": len(rewards),
        "sum_adv": float(adv.sum()),
        "sum_adv2": float((adv ** 2).sum()),
        "sum_regroup_var": var_sum,
        "n_eligible": eligible,
        "sum_flip": flip,
        "sum_no_signal": nosig,
    }


def stack(stats: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.array([s[key] for s in stats], dtype=np.float64) for key in stats[0]}


def quantity_fns(a: dict[str, np.ndarray]) -> dict:
    """quantity -> f(prompt index array) -> float, pooled over the indexed prompts."""
    def ratio(num, den):
        return lambda idx: float(a[num][idx].sum() / a[den][idx].sum())

    def pooled_var(idx):
        n = a["n_completions"][idx].sum()
        m = a["sum_adv"][idx].sum() / n
        return float(a["sum_adv2"][idx].sum() / n - m * m)

    def agree(idx):
        return float(1.0 - (a["sum_flip"][idx].sum() + a["sum_no_signal"][idx].sum()) / a["n_eligible"][idx].sum())

    inf = ratio("n_informative", "n_groups")
    return {
        "informative_rate": inf,
        "uninformative_rate": lambda idx: 1.0 - inf(idx),
        "mean_within_group_std": ratio("sum_std", "n_groups"),
        "pooled_advantage_var": pooled_var,
        "regrouping_noise": ratio("sum_regroup_var", "n_completions"),
        "sign_flip_rate": ratio("sum_flip", "n_eligible"),
        "no_signal_rate": ratio("sum_no_signal", "n_eligible"),
        "agree_rate": agree,
    }


# ---------------------------------------------------------------- difficulty bins

def difficulty_bins(prompt_ids: list[str], mean_rewards: list[float], n_bins: int = N_BINS) -> list[int]:
    """Bin index per prompt: sort by (mean reward, prompt_id), cut into n_bins equal tertiles."""
    n = len(prompt_ids)
    if n % n_bins:
        raise ValueError(f"{n} prompts do not split into {n_bins} equal bins")
    order = sorted(range(n), key=lambda i: (mean_rewards[i], prompt_ids[i]))
    size = n // n_bins
    bins = [0] * n
    for rank, i in enumerate(order):
        bins[i] = rank // size
    return bins


# ---------------------------------------------------------------- study

def run_study(by_prompt, group_sizes, seed: int, n_resamples: int = N_RESAMPLES) -> dict:
    keys = list(by_prompt)
    prompt_ids = [by_prompt[k][0]["prompt_id"] for k in keys]
    rewards = [np.array([float(r["reward"]) for r in by_prompt[k]]) for k in keys]
    means = [float(r.mean()) for r in rewards]
    bins = difficulty_bins(prompt_ids, means)
    scopes = {"all": np.arange(len(keys))}
    for b, name in enumerate(BIN_NAMES):
        scopes[name] = np.array([i for i in range(len(keys)) if bins[i] == b])

    fns = {}
    per_prompt = [{"source_index": k, "prompt_id": pid, "bin": BIN_NAMES[b], "mean_reward_8": m,
                   "n_clipped_at_max": int(sum(bool(r["clipped_at_max"]) for r in by_prompt[k]))}
                  for k, pid, b, m in zip(keys, prompt_ids, bins, means)]
    for K in group_sizes:
        stats = [prompt_stats(r, int(K)) for r in rewards]
        for row, s in zip(per_prompt, stats):
            row[f"K{K}"] = s
        fns[int(K)] = quantity_fns(stack(stats))

    def boot(f, scope_idx, method):
        return bootstrap(lambda idx: f(scope_idx[idx]), len(scope_idx), iid_resampler(len(scope_idx)), seed, n_resamples, method)

    results = {}
    for scope, idx in scopes.items():
        block = {}
        for K, qf in fns.items():
            block[f"K{K}"] = {
                q: boot(qf[q], idx, f"cluster bootstrap over {len(idx)} prompts") for q in QUANTITIES
            }
            block[f"K{K}"]["n_groups"] = int(len(idx) * N_CACHED // K)
        lo, hi = min(fns), max(fns)
        block[f"K{lo}_minus_K{hi}"] = {
            q: boot(diff_stat(fns[lo][q], fns[hi][q]), idx, f"paired cluster bootstrap over {len(idx)} prompts, same resamples for both K")
            for q in QUANTITIES
        }
        results[scope] = block
    return {
        "scopes": {s: [prompt_ids[i] for i in idx] for s, idx in scopes.items()},
        "results": results,
        "per_prompt": per_prompt,
        "sqrt_(K-1)/K": {f"K{K}": math.sqrt((K - 1) / K) for K in group_sizes},
    }


def csv_rows(study: dict) -> list[dict]:
    rows = []
    for scope, block in study["results"].items():
        for cond, qs in block.items():
            for q, v in qs.items():
                if isinstance(v, dict):
                    rows.append({"scope": scope, "condition": cond, "quantity": q, "point": v["point"],
                                 "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_json = repo_path(f"{cfg['results_dir']}/group_size.json")
    out_csv = repo_path(f"{cfg['results_dir']}/group_size.csv")
    for p in (out_json, out_csv):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    by_prompt = load_k8_cache(cfg["group_cache"])
    cache = check_cache(by_prompt)
    sizes = [int(k) for k in cfg["group_sizes"]]
    seed = int(cfg["seed"])
    study = run_study(by_prompt, sizes, seed)
    result = {
        "script": "task3_grpo.analyze_group_size",
        "git": git_state(),
        "seed": seed,
        "cache": {"path": cfg["group_cache"], "generation_cap": int(cfg["cache_generation_cap"]), **cache},
        "group_sizes": sizes,
        "partition": "each prompt's 8 completions sorted by generation_index, split into 8/K consecutive groups",
        "informative": INFORMATIVE_RULE,
        "advantages": "task3_grpo.grpo.group_relative_advantages (population std, eps 1e-6)",
        "binning": "tertiles of per-prompt mean reward over all 8 completions, ties by prompt_id, 8 prompts per bin, same bins for every K; bin1 = lowest mean reward",
        "bootstrap": f"cluster bootstrap over prompts, {N_RESAMPLES} resamples, seed {seed}, 95% percentile interval",
        "notes": "all 24 cached prompts are used, including those over 256 chat-template tokens; clipped completions stay in their groups",
        **study,
        "wall_clock_seconds": elapsed(),
    }
    save_json(out_json, result)
    write_csv(out_csv, csv_rows(study))
    for K in sizes:
        r = study["results"]["all"][f"K{K}"]
        print(f"K={K} groups={r['n_groups']} sqrt((K-1)/K)={study['sqrt_(K-1)/K'][f'K{K}']:.4f} "
              + " ".join(f"{q}={r[q]['point']:.4f}" for q in QUANTITIES))
    print(f"wrote {out_json} and {out_csv} in {elapsed():.0f}s")


if __name__ == "__main__":
    main()
