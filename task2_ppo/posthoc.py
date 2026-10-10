"""Task 2 post-hoc diagnostics from saved files only (no model is loaded; the tokenizer is loaded for token strings).

Adds a "post_hoc" key to results/task2_ppo/summary.json. Every other key of summary.json is left
byte-for-byte as summarize.py wrote it (checked before saving). Re-running summarize.py --overwrite
drops this key; run this script again afterwards.

Items:
  1. cached study per row: mean |cached old - recomputed midpoint| and token counts; pooled over
     the rows whose prompts have <= max_prompt_length tokens; affected/clip ratios.
  2. reference sanity: per-token (cached old - cached ref) and the per-update signed mean from training.
  3. KL-shaping size per update for the standard run and the beta 0.20 fork.
  4. standard updates 1-8 versus fork_eps0p20_kl0p10, and the paired held-out differences.
  5. explained variance of the standard run (per-update values only).
Quantities that need per-token arrays that were not saved are listed under "not_computable" with the
reason and a GPU estimate.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, wall_timer
from common.models import load_tokenizer
from common.run_info import git_state
from common.stats import heldout_block
from task2_ppo.analyze_clipping import load_cached_rollouts, rebuild_tokens
from task2_ppo.summarize import STANDARD, flat_numbers, history_max_diff

FORK_8 = "fork_eps0p20_kl0p10"
FORK_KL020 = "fork_eps0p20_kl0p20"


# ---------------------------------------------------------------- item 1 and 2: cached rollouts

def cached_items(cfg: dict, cached: dict, tokenizer) -> tuple[dict, dict]:
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    per_row = cached["recompute"]["per_row"]
    max_prompt = int(cfg["max_prompt_length"])
    if len(rows) != len(per_row) or any(int(r["response_tokens"]) != p["cached_tokens"] for r, p in zip(rows, per_row)):
        raise SystemExit("cached rows do not line up with cached_clip_study.json per_row")

    short = [p["row"] for p in per_row if p["prompt_tokens"] <= max_prompt]
    long_ = [p["row"] for p in per_row if p["prompt_tokens"] > max_prompt]

    def pooled(sel):
        # token-weighted mean of per-row means = token-pooled mean over those rows
        n = sum(per_row[i]["cached_tokens"] for i in sel)
        return {"n_rows": len(sel), "n_tokens": n,
                "mean_abs_diff_token_pooled": sum(per_row[i]["mean_abs_diff"] * per_row[i]["cached_tokens"] for i in sel) / n}

    all_rows = pooled(range(len(per_row)))
    if abs(all_rows["mean_abs_diff_token_pooled"] - cached["recompute"]["mean_abs_diff_token_pooled"]) > 1.0e-9:
        raise SystemExit("per-row means do not reproduce the logged pooled mean")

    # Location of the overall max: only per-row maxima were saved, so the row is known but not the
    # position. Candidate positions are those where the cached old log-prob alone is at or below
    # -max (|recomputed - cached| = max with both <= 0 needs cached <= -max or recomputed <= -max).
    i_max = int(np.argmax([p["max_abs_diff"] for p in per_row]))
    row = rows[i_max]
    eval_rows = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    _, r_ids = rebuild_tokens(tokenizer, row, prompt_messages(eval_rows[row["prompt_id"]]))
    old = torch.as_tensor(row["old_logprobs"], dtype=torch.float64)
    ref = torch.as_tensor(row["ref_logprobs"], dtype=torch.float64)
    mx = per_row[i_max]["max_abs_diff"]
    cand = [{"position": int(t), "token_id": int(r_ids[t]), "token": tokenizer.decode([r_ids[t]]),
             "cached_old_logprob": float(old[t]), "cached_ref_logprob": float(ref[t])}
            for t in torch.nonzero(old <= -mx + 1.0e-6).flatten().tolist()]
    lowest = int(torch.argmin(old))
    max_loc = {
        "row": i_max, "prompt_id": row["prompt_id"], "max_abs_diff": mx, "row_tokens": per_row[i_max]["cached_tokens"],
        "row_prompt_tokens": per_row[i_max]["prompt_tokens"], "row_mean_abs_diff": per_row[i_max]["mean_abs_diff"],
        "position_known": False,
        "candidates_cached_old_le_minus_max": cand,
        "lowest_cached_old_in_row": {"position": lowest, "token": tokenizer.decode([r_ids[lowest]]),
                                     "cached_old_logprob": float(old[lowest]), "cached_ref_logprob": float(ref[lowest])},
        "note": "per-token recomputed log-probs were not saved; if no candidate exists the extreme value is the recomputed one",
    }

    eps = {r["eps"]: r for r in cached["eps_results"]}
    item1 = {
        "per_row": [{"row": p["row"], "prompt_tokens": p["prompt_tokens"], "n_tokens": p["cached_tokens"],
                     "mean_abs_diff": p["mean_abs_diff"], "max_abs_diff": p["max_abs_diff"],
                     "terminated_with_eos": p["terminated_with_eos"]} for p in per_row],
        "max_location": max_loc,
        "all_rows": all_rows,
        "short_prompt_rows": {**pooled(short), "rule": f"prompt_tokens <= {max_prompt}", "excluded_rows": long_},
        "affected_over_clip_all_rows": {str(e): (r["affected_fraction"] / r["clip_fraction"] if r["clip_fraction"] > 0 else None)
                                        for e, r in eps.items()},
    }

    old_all = torch.cat([torch.as_tensor(r["old_logprobs"], dtype=torch.float64) for r in rows])
    ref_all = torch.cat([torch.as_tensor(r["ref_logprobs"], dtype=torch.float64) for r in rows])
    d = old_all - ref_all
    d_short = torch.cat([torch.as_tensor(rows[i]["old_logprobs"], dtype=torch.float64)
                         - torch.as_tensor(rows[i]["ref_logprobs"], dtype=torch.float64) for i in short])
    item2_cached = {
        "cached_old_minus_cached_ref_all_rows": {"n_tokens": int(d.numel()), "signed_mean": float(d.mean()),
                                                 "abs_mean": float(d.abs().mean()), "max_abs": float(d.abs().max())},
        "cached_old_minus_cached_ref_short_prompt_rows": {"n_tokens": int(d_short.numel()), "signed_mean": float(d_short.mean()),
                                                          "abs_mean": float(d_short.abs().mean())},
    }
    return item1, item2_cached


# ---------------------------------------------------------------- item 3: KL-shaping size

def shaping_size(train: dict) -> dict:
    """Per update: |beta * sum over all tokens of (old - ref)| / 4 as a lower bound on the mean over
    responses of |per-response sum|, from kl_sampled (pooled token mean) times the valid-token count.

    By the triangle inequality mean_i |s_i| >= |mean_i s_i|; equality iff all four s_i share a sign.
    generated_tokens equals the response-mask sum (common.generation lengths are mask sums).
    """
    beta = float(train["effective"]["kl_beta"])
    out = []
    for h in train["history"]:
        n = h["generated_tokens"]
        k = len(h["reward_effective"]["values"])
        lb = abs(beta * h["kl_sampled"] * n) / k
        eff = np.asarray(h["reward_effective"]["values"], dtype=float)
        sd = float(eff.std(ddof=0))
        out.append({"update": h["update"], "n_tokens": n,
                    "kl_term_mean_abs_response_sum_lower_bound": lb,
                    "mean_abs_effective_reward": float(np.abs(eff).mean()),
                    "sd_effective_reward": sd,
                    "ratio_kl_lower_bound_over_sd": (lb / sd) if sd > 0 else None})
    ratios = [r["ratio_kl_lower_bound_over_sd"] for r in out if r["ratio_kl_lower_bound_over_sd"] is not None]
    return {"beta": beta, "per_update": out,
            "ratio_summary": {"median": float(np.median(ratios)), "max": float(np.max(ratios)), "min": float(np.min(ratios))},
            "sd": "population SD (ddof 0) over the responses of the update, as logged in history",
            "bound": "first column is a lower bound (exact iff all per-response KL sums share a sign); per-token log-probs were not saved"}


# ---------------------------------------------------------------- item 4: standard 1-8 versus fork

def fork_identity(trains: dict, rollouts: dict, gens: dict, seed: int) -> dict:
    hs, hf = trains[STANDARD]["history"][:8], trains[FORK_8]["history"]
    hd = history_max_diff(hs, hf)
    ids_equal = [h["prompt_id"] for h in hs] == [h["prompt_id"] for h in hf]
    key = lambda r: (r["update"], r["sample_index"])  # noqa: E731
    rs = sorted((r for r in rollouts[STANDARD] if r["update"] <= 8), key=key)
    rf = sorted(rollouts[FORK_8], key=key)
    roll_equal = [{k: v for k, v in r.items()} for r in rs] == rf
    identical = bool(hd["same_keys"] and hd["max_abs_diff"] == 0 and ids_equal and roll_equal)
    out = {"history": hd, "n_history_values": len(flat_numbers(hs)), "prompt_ids_equal": ids_equal,
           "rollout_rows_equal": roll_equal, "n_rollout_rows": len(rf), "bit_identical": identical}
    if identical:
        labels = {"standard_u20": STANDARD, "fork_eps0p20_kl0p10_u8": FORK_8}
        out["heldout"] = heldout_block(gens, labels, (("standard_u20", "fork_eps0p20_kl0p10_u8"),), seed)
    return out


# ---------------------------------------------------------------- item 5: explained variance

def ev_section(train: dict) -> dict:
    ev = [h["explained_variance"] for h in train["history"]]
    return {"per_update": [{"update": h["update"], "explained_variance": h["explained_variance"], "n_tokens": h["generated_tokens"],
                            "response_lengths": h["response_length"]["values"]} for h in train["history"]],
            "median_per_update": float(np.median(ev)),
            "mean_per_update": float(np.mean(ev))}


NOT_COMPUTABLE = {
    "item1_max_position_and_token": {
        "reason": "cached_clip_study.json saved only per-row mean and max of |diff|, not per-token recomputed log-probs",
        "gpu": "re-run the midpoint recomputation on the 32 cached rows and save per-token arrays: T4, about 15 s compute "
               "plus about 2 min model load, about 10 min with the setup cell"},
    "item1_short_rows_clip_study": {
        "reason": "clip and affected fractions and surrogates need per-token recomputed midpoint log-probs",
        "gpu": "same single pass as above (one run serves both)"},
    "item2_recomputed_midpoint_minus_recomputed_ref": {
        "reason": "the recomputed reference log-probs were never computed or saved",
        "gpu": "same pass with the adapter disabled (common.models.reference_mode): about 15 s more"},
    "item2_training_abs_mean_old_minus_ref": {
        "reason": "train_*.json and rollouts_*.jsonl store only the pooled signed mean (kl_sampled), no per-token arrays",
        "gpu": "re-run standard (about 390 s) with per-token old/ref/values/returns dumped; needs a logging flag in "
               "continue_train.py and a check that the logged history is reproduced"},
    "item3_exact_mean_abs_response_kl_sum": {
        "reason": "per-response KL sums were not saved; only a lower bound is computable",
        "gpu": "same re-run of standard plus fork_eps0p20_kl0p20 (about 160 s)"},
    "item5_pooled_ev_and_var_returns": {
        "reason": "per-token returns and values were not saved; only per-update EV is logged",
        "gpu": "same re-run of standard"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing post_hoc key")
    args = ap.parse_args()

    elapsed = wall_timer()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    rd = cfg["results_dir"]
    summary_path = repo_path(f"{rd}/summary.json")
    summary = json.loads(summary_path.read_text())
    if "post_hoc" in summary and not args.overwrite:
        raise SystemExit("summary.json already has post_hoc; pass --overwrite to replace it.")
    before = {k: v for k, v in summary.items() if k != "post_hoc"}

    trains = {r: json.loads(repo_path(f"{rd}/train_{r}.json").read_text()) for r in (STANDARD, FORK_8, FORK_KL020)}
    rollouts = {r: read_jsonl(f"{rd}/rollouts_{r}.jsonl") for r in (STANDARD, FORK_8)}
    gens = {r: read_jsonl(f"{rd}/generations_eval_{r}.jsonl") for r in (STANDARD, FORK_8)}
    cached = json.loads(repo_path(f"{rd}/cached_clip_study.json").read_text())
    tokenizer = load_tokenizer(cfg["base_model"])

    item1, item2_cached = cached_items(cfg, cached, tokenizer)
    item2 = {**item2_cached,
             "training_signed_mean_old_minus_ref_per_update": {
                 r: [h["kl_sampled"] for h in trains[r]["history"]] for r in (STANDARD, FORK_KL020)},
             "note": "training values are kl_sampled (pooled signed token mean of old - ref); absolute means were not saved"}
    post = {
        "script": "task2_ppo.posthoc", "git": git_state(), "seed": seed,
        "inputs": "summary.json, train_{standard,fork_eps0p20_kl0p10,fork_eps0p20_kl0p20}.json, rollouts and generations of "
                  "standard and fork_eps0p20_kl0p10, cached_clip_study.json, cached/ppo_rollout.pt, tokenizer only",
        "item1_cache_per_row": item1,
        "item2_reference_sanity": item2,
        "item3_kl_shaping_size": {STANDARD: shaping_size(trains[STANDARD]), FORK_KL020: shaping_size(trains[FORK_KL020])},
        "item4_standard_vs_fork_u8": fork_identity(trains, rollouts, gens, seed),
        "item5_explained_variance_standard": ev_section(trains[STANDARD]),
        "not_computable": NOT_COMPUTABLE,
    }
    post["wall_clock_seconds"] = elapsed()
    summary["post_hoc"] = post
    if {k: v for k, v in summary.items() if k != "post_hoc"} != before:
        raise SystemExit("existing summary keys changed; refusing to write")
    save_json(summary_path, summary)
    print(f"wrote post_hoc to {rd}/summary.json in {elapsed():.1f}s; bit_identical={post['item4_standard_vs_fork_u8']['bit_identical']}")


if __name__ == "__main__":
    main()
