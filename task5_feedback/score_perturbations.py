"""Task 5 Step 2: the controlled reward diagnostic set under the exact verifier and the pairwise judge.

80 pairs (protocol.build_pairs): clean_correct ("better") vs each of four perturbations per problem.
Verifier: exact_reward and compliance for all 100 responses; per pair better / tie / wrong from rewards.
Judge: compare(question, clean, perturbed) through the wrapper. --order released is the reported
result; --order swapped shows each pair in the opposite physical order (position check only).
Smoke mode (--limit-pairs N) writes under results/task5_feedback/smoke/ only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, wall_timer
from task1_dpo.dataset_stats import describe
from task1_dpo.train import display_path, peak_vram_bytes, run_metadata
from task5_feedback import protocol as P
from task5_feedback.judge_wrapper import judge_call, judge_generation_settings, parity_check
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_rows(groups: dict) -> list[dict]:
    """Verifier score and compliance for every one of the 100 responses."""
    out = []
    for pid in sorted(groups, key=int):
        for variant in sorted(groups[pid]):
            r = groups[pid][variant]
            out.append({"problem_id": int(pid), "variant_type": variant,
                        "reward": exact_reward(r["response"], r["gold_final"]),
                        "compliant": extract_designated_final(r["response"]) is not None,
                        "expected_exact_reward": r.get("expected_exact_reward")})
    return out


def verifier_pairs(pairs: list[dict]) -> list[dict]:
    out = []
    for p in pairs:
        rb = exact_reward(p["better_response"], p["gold_final"])
        ro = exact_reward(p["other_response"], p["gold_final"])
        out.append({"pair_id": p["pair_id"], "problem_id": p["problem_id"], "category": p["category"],
                     "reward_better": rb, "reward_other": ro, "preference": P.reward_preference(rb, ro)})
    return out


def out_paths(cfg, order: str, smoke: bool):
    d = repo_path(f"{cfg['results_dir']}/task5_feedback")
    d = d / "smoke" if smoke else d
    return d / f"diagnostics_{order}.json", d / f"diagnostics_{order}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--order", choices=["released", "swapped"], default="released")
    ap.add_argument("--limit-pairs", type=int, help="first N pairs; smoke only")
    ap.add_argument("--parity-checks", type=int, default=4, help="released compare() re-run on the first N calls (released order only)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    smoke = args.limit_pairs is not None
    out_json, out_jsonl = out_paths(cfg, args.order, smoke)
    for p in (out_json, out_jsonl):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
        if p.exists():
            p.unlink()
    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    path = cfg["paths"]["task5_diagnostics"]
    sha = P.assert_sha(path)
    n_rows = len(read_jsonl(path))
    groups = load_diagnostic_groups(path)
    if n_rows != P.N_DIAGNOSTIC_ROWS or len(groups) != P.N_DIAGNOSTIC_PROBLEMS:
        raise SystemExit(f"diagnostics: {n_rows} rows / {len(groups)} problems")
    pairs = P.build_pairs(groups)
    assert len(pairs) == P.N_PAIRS
    vrows, vpairs = verifier_rows(groups), verifier_pairs(pairs)
    if smoke:
        pairs = pairs[: args.limit_pairs]

    result = {
        "script": "task5_feedback.score_perturbations",
        "order": args.order,
        "status": "running",
        "smoke": smoke,
        "config_path": args.config,
        "config": cfg,
        **run_metadata(cfg),
        "data": {"path": path, "sha256": sha, "n_rows": n_rows, "n_problems": len(groups),
                 "n_pairs_total": P.N_PAIRS, "n_pairs_judged": len(pairs)},
        "settings": {"anchor": P.ANCHOR, "categories": P.CATEGORIES,
                     "argument_order": "compare(question, clean_response, perturbed_response)",
                     "orientation": "released hashed order" if args.order == "released" else "opposite of the released physical order"},
        "verifier_responses": vrows,
        "verifier_pairs": vpairs,
        "n_calls_done": 0,
        "parity": [],
        "timing": {},
    }

    def write():
        result["wall_clock_seconds"] = elapsed()
        result["peak_vram_bytes"] = peak_vram_bytes()
        save_json(out_json, result)

    write()
    try:
        judge = PairwiseAIJudge(cfg, cache_path=out_json.parent / "unused_released_cache.json")
        result["settings"]["judge"] = judge_generation_settings(judge, cfg)
        result["timing"]["setup"] = elapsed()
        secs = []
        swap = args.order == "swapped"
        for k, p in enumerate(pairs, start=1):
            rec = judge_call(judge, p["question"], p["better_response"], p["other_response"], swap_order=swap)
            secs.append(rec["seconds"])
            if not swap and k <= args.parity_checks:
                result["parity"].append({"pair_id": p["pair_id"], **parity_check(
                    judge, p["question"], p["better_response"], p["other_response"], rec["label"])})
            append_jsonl(out_jsonl, {"pair_id": p["pair_id"], "problem_id": p["problem_id"], "category": p["category"],
                                     "preference": P.judge_preference(rec["label"], rec["parse_matched"]), **rec})
            result["n_calls_done"] = k
            result["timing"]["judge_seconds_per_call"] = describe(secs)
            if k % 10 == 0 or k == len(pairs):
                write()
        result["judge_file"] = display_path(out_jsonl)
        result["parity_all_match"] = all(x["match"] for x in result["parity"]) if result["parity"] else None
        result["n_parse_failures"] = sum(1 for r in read_jsonl(out_jsonl) if not r["parse_matched"])
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise
    result["status"] = "completed"
    write()
    vram = result["peak_vram_bytes"]
    parity = "n/a" if result["parity_all_match"] is None else ("ok" if result["parity_all_match"] else "MISMATCH")
    print(f"[diag {args.order}] status=completed calls={result['n_calls_done']} parse_failures={result['n_parse_failures']} "
          f"parity={parity} ({len(result['parity'])} checked) s_per_call={result['timing']['judge_seconds_per_call']['mean']:.2f} "
          f"t={elapsed():.0f}s peak_vram={vram / 2**30 if vram else 0:.2f}GiB; wrote {display_path(out_json)}", flush=True)


if __name__ == "__main__":
    main()
