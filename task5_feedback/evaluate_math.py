"""Task 5 Steps 1 and 3: matched SFT / RLVR / RLAIF generations and the pairwise judge on GSM8K and SVAMP.

--stage generate  one policy on one dataset: 1 sampled response per prompt, file order, seed reset right
                  before generation, batch 16, config decoding (other generate keys inherited from the
                  model's generation_config and recorded), max_new_tokens = math_max_new_tokens. Each
                  response gets the released verifier score, compliance and failure type (protocol.py).
--stage judge     RLVR vs SFT and RLAIF vs SFT on the same prompt through the judge wrapper in the released
                  hashed order; needs the three completed generation files of that dataset.

Smoke mode (--limit N) writes under results/task5_feedback/smoke/ only.
"""
from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer
from task1_dpo.dataset_stats import describe
from task1_dpo.evaluate import generate_responses
from task1_dpo.train import display_path, peak_vram_bytes, run_metadata
from task3_grpo.grpo_utils import effective_generation_settings
from task5_feedback import protocol as P
from task5_feedback.judge_wrapper import enforce_parity, judge_call, judge_generation_settings, parity_check
from task5_feedback.rlaif import PairwiseAIJudge


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


# ---------------------------------------------------------------- paths

def out_dir(cfg, smoke: bool):
    d = repo_path(f"{cfg['results_dir']}/task5_feedback")
    return d / "smoke" if smoke else d


def gen_paths(cfg, dataset: str, policy: str, smoke: bool):
    d = out_dir(cfg, smoke)
    return d / f"gen_{dataset}_{policy}.json", d / f"gen_{dataset}_{policy}.jsonl"


def judge_paths(cfg, dataset: str, smoke: bool):
    d = out_dir(cfg, smoke)
    return d / f"judge_{dataset}.json", d / f"judge_{dataset}.jsonl"


def refuse_existing(paths, overwrite: bool):
    for p in paths:
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
        if p.exists():
            p.unlink()


def prompt_token_counts(tokenizer, prompts) -> list[int]:
    return [len(tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) for p in prompts]


# ---------------------------------------------------------------- aggregation (pure, tested on Mac)

def response_rows(dataset: str, rows: list[dict], gens: list[dict]) -> list[dict]:
    out = []
    for i, (row, g) in enumerate(zip(rows, gens, strict=True)):
        cls = P.classify_response(g["text"], row["gold_final"], g["truncated"])
        out.append({
            "index": i,
            "prompt_id": P.prompt_id(dataset, row),
            "gold_final": row["gold_final"],
            "prompt_tokens": g["prompt_tokens"],
            "text": g["text"],
            "token_length": g["token_length"],
            "terminated_with_eos": g["terminated_with_eos"],
            "truncated": g["truncated"],
            **cls,
        })
    return out


def generation_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    counts = {t: sum(r["failure_type"] == t for r in rows) for t in P.FAILURE_TYPES}
    return {
        "n_responses": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "format_compliance": sum(r["compliant"] for r in rows) / n,
        "length": describe([r["token_length"] for r in rows]),
        "n_truncated_at_cap": sum(r["truncated"] for r in rows),
        "n_terminated_with_eos": sum(r["terminated_with_eos"] for r in rows),
        "failure_type_counts": counts,
        "resolution_points": 100.0 / n,
    }


# ---------------------------------------------------------------- stage: generate

def run_generate(args, cfg):
    smoke = args.limit is not None
    out_json, out_jsonl = gen_paths(cfg, args.dataset, args.policy, smoke)
    refuse_existing((out_json, out_jsonl), args.overwrite)
    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_path = dataset_path(cfg, args.dataset)
    data_sha = P.assert_sha(data_path)
    rows = read_jsonl(data_path)
    if len(rows) != P.DATASET_ROWS[args.dataset]:
        raise SystemExit(f"{data_path}: {len(rows)} rows, expected {P.DATASET_ROWS[args.dataset]}")
    adapter = policy_specs(cfg)[args.policy]
    adapter_sha = P.assert_sha(P.adapter_file(adapter)) if adapter else None
    if smoke:
        rows = rows[: args.limit]

    tokenizer = load_tokenizer(cfg["base_model"])
    prompts = [prompt_messages(r) for r in rows]
    n_prompt = prompt_token_counts(tokenizer, prompts)
    over = [P.prompt_id(args.dataset, r) for r, n in zip(rows, n_prompt) if n > P.PROMPT_TOKEN_CAP]
    if over:
        raise SystemExit(f"{len(over)} prompts exceed {P.PROMPT_TOKEN_CAP} tokens: {over[:5]}")
    max_new = int(cfg["math_max_new_tokens"])
    seed = int(cfg["seed"])

    result = {
        "script": "task5_feedback.evaluate_math",
        "stage": "generate",
        "dataset": args.dataset,
        "policy": args.policy,
        "adapter": adapter or "none (base model)",
        "adapter_sha256": adapter_sha,
        "status": "running",
        "smoke": smoke,
        "limit": args.limit,
        "config_path": args.config,
        "config": cfg,
        **run_metadata(cfg),
        "data": {"path": data_path, "sha256": data_sha, "n_rows_evaluated": len(rows),
                 "prompt_ids": [P.prompt_id(args.dataset, r) for r in rows],
                 "prompt_tokens": describe(n_prompt), "n_prompts_over_cap": 0, "prompt_token_cap": P.PROMPT_TOKEN_CAP,
                 "n_skipped": 0, "n_truncated_prompts": 0},
        "settings": {
            "samples_per_prompt": P.SAMPLES_PER_PROMPT,
            "gen_batch_size": P.GEN_BATCH_SIZE,
            "max_new_tokens": max_new,
            "seed_reset_before_generation": seed,
            "prompt_construction": "row['messages'] through the policy chat template (common.data.prompt_messages)",
            "length_definition": "generated tokens after the prompt up to and including the first EOS, padding excluded",
            "compliance_definition": "task5_feedback.rlvr.extract_designated_final returns a number",
        },
        "timing": {},
    }

    def write():
        result["wall_clock_seconds"] = elapsed()
        result["peak_vram_bytes"] = peak_vram_bytes()
        save_json(out_json, result)

    write()
    try:
        model = load_frozen_policy(cfg, args.policy)
        result["settings"]["effective_generation"] = effective_generation_settings(model, tokenizer, cfg["generation"], max_new)
        result["timing"]["setup"] = elapsed()
        # Truncation limit one above the asserted cap: the helper's guard fires only if a prompt would be cut.
        shim = {"generation": cfg["generation"], "max_sequence_length": P.PROMPT_TOKEN_CAP + 1, "max_generation_tokens": max_new}
        assert P.GEN_BATCH_SIZE == 16
        set_seed(seed)
        t0 = elapsed()
        gens = generate_responses(model, tokenizer, prompts, shim, with_kl=False)
        result["timing"]["generation"] = elapsed() - t0
        result["timing"]["seconds_per_prompt"] = result["timing"]["generation"] / len(prompts)
        out_rows = response_rows(args.dataset, rows, gens)
        write_jsonl(out_jsonl, out_rows)
        result["generations_file"] = display_path(out_jsonl)
        result["metrics"] = generation_metrics(out_rows)
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise
    result["status"] = "completed"
    write()
    m, vram = result["metrics"], result["peak_vram_bytes"]
    print(f"[gen {args.dataset} {args.policy}] status=completed n={m['n_responses']} eos={m['n_terminated_with_eos']} "
          f"truncated_at_cap={m['n_truncated_at_cap']} t={elapsed():.0f}s s_per_prompt={result['timing']['seconds_per_prompt']:.2f} "
          f"peak_vram={vram / 2**30 if vram else 0:.2f}GiB; wrote {display_path(out_json)}", flush=True)


# ---------------------------------------------------------------- stage: judge

def load_generation_set(cfg, dataset: str, smoke: bool) -> dict[str, list[dict]]:
    """The three completed generation files of one dataset, checked for identical prompt order."""
    gens = {}
    for pol in P.POLICIES:
        j, jl = gen_paths(cfg, dataset, pol, smoke)
        meta = load_json(j)
        if meta["status"] != "completed":
            raise SystemExit(f"{j}: status {meta['status']}")
        gens[pol] = read_jsonl(jl)
    ids = {pol: [r["prompt_id"] for r in rows] for pol, rows in gens.items()}
    if not (ids["sft"] == ids["rlvr"] == ids["rlaif"]):
        raise SystemExit(f"{dataset}: prompt order differs across policies")
    return gens


def judge_tasks(dataset: str, rows: list[dict], gens: dict[str, list[dict]]) -> list[dict]:
    """One task per (prompt, trained policy): compare(question, trained, sft)."""
    tasks = []
    for i, row in enumerate(rows):
        for pol in P.TRAINED:
            t, s = gens[pol][i], gens["sft"][i]
            assert t["prompt_id"] == s["prompt_id"] == P.prompt_id(dataset, row)
            tasks.append({"comparison": f"{pol}_vs_sft", "prompt_id": t["prompt_id"], "index": i,
                          "question": row["question"], "a_policy": pol, "b_policy": "sft",
                          "a_text": t["text"], "b_text": s["text"],
                          "a_correct": t["correct"], "b_correct": s["correct"]})
    return tasks


def run_judge(args, cfg):
    smoke = args.limit is not None
    out_json, out_jsonl = judge_paths(cfg, args.dataset, smoke)
    refuse_existing((out_json, out_jsonl), args.overwrite)
    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    data_path = dataset_path(cfg, args.dataset)
    data_sha = P.assert_sha(data_path)
    rows = read_jsonl(data_path)
    gens = load_generation_set(cfg, args.dataset, smoke)
    rows = rows[: len(gens["sft"])]
    tasks = judge_tasks(args.dataset, rows, gens)

    result = {
        "script": "task5_feedback.evaluate_math",
        "stage": "judge",
        "dataset": args.dataset,
        "status": "running",
        "smoke": smoke,
        "config_path": args.config,
        "config": cfg,
        **run_metadata(cfg),
        "data": {"path": data_path, "sha256": data_sha, "n_prompts": len(rows)},
        "generation_files": {p: display_path(gen_paths(cfg, args.dataset, p, smoke)[1]) for p in P.POLICIES},
        "settings": {"argument_order": "compare(question, trained_response, sft_response)",
                     "orientation": "released hashed order", "score": P.JUDGE_SCORE},
        "n_calls_planned": len(tasks),
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
        judge = PairwiseAIJudge(cfg, cache_path=out_dir(cfg, smoke) / "unused_released_cache.json")
        result["settings"]["judge"] = judge_generation_settings(judge, cfg)
        result["timing"]["setup"] = elapsed()
        secs = []
        n_parity = {c: 0 for c in ("rlvr_vs_sft", "rlaif_vs_sft")}
        for k, t in enumerate(tasks, start=1):
            rec = judge_call(judge, t["question"], t["a_text"], t["b_text"])
            secs.append(rec["seconds"])
            if n_parity[t["comparison"]] < args.parity_checks:
                n_parity[t["comparison"]] += 1
                enforce_parity(result["parity"], {"comparison": t["comparison"], "prompt_id": t["prompt_id"],
                                                  **parity_check(judge, t["question"], t["a_text"], t["b_text"], rec["label"])})
            out = {k2: t[k2] for k2 in ("comparison", "prompt_id", "index", "a_policy", "b_policy", "a_correct", "b_correct")}
            append_jsonl(out_jsonl, {**out, **rec})
            result["n_calls_done"] = k
            if k % 25 == 0 or k == len(tasks):
                result["timing"]["judge_seconds_per_call"] = describe(secs)
                write()
                print(f"  judge {args.dataset} {k}/{len(tasks)} t={elapsed():.0f}s", flush=True)
        result["judge_file"] = display_path(out_jsonl)
        result["parity_all_match"] = all(p["match"] for p in result["parity"])
        result["n_parse_failures"] = sum(1 for r in read_jsonl(out_jsonl) if not r["parse_matched"])
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise
    result["status"] = "completed"
    write()
    vram = result["peak_vram_bytes"]
    print(f"[judge {args.dataset}] status=completed calls={result['n_calls_done']} parse_failures={result['n_parse_failures']} "
          f"parity={'ok' if result['parity_all_match'] else 'MISMATCH'} ({len(result['parity'])} checked) "
          f"s_per_call={result['timing']['judge_seconds_per_call']['mean']:.2f} t={elapsed():.0f}s "
          f"peak_vram={vram / 2**30 if vram else 0:.2f}GiB; wrote {display_path(out_json)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge"], default="generate")
    ap.add_argument("--policy", choices=list(P.POLICIES), help="required for --stage generate")
    ap.add_argument("--limit", type=int, help="first N prompts; smoke only, writes under results/task5_feedback/smoke/")
    ap.add_argument("--parity-checks", type=int, default=2, help="released compare() re-run on the first N calls per comparison")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.stage == "generate":
        if args.policy is None:
            ap.error("--policy is required for --stage generate")
        run_generate(args, cfg)
    else:
        run_judge(args, cfg)


if __name__ == "__main__":
    main()
