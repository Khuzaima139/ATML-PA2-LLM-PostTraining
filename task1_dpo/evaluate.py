"""Task 1 evaluation, one script for every Task 1 model (manual Section 1, Steps 1 to 3).

Modes (--modes, comma list):
  pairs       retained dpo_standard_eval pairs: held-out DPO loss at --beta, preference accuracy
              (m > 0), mean m; per-pair m saved with pair IDs.
  stratified  the same on retained dpo_length_stratified_eval, overall and per length_stratum.
  generate    one sampled response per retained dpo_standard_eval prompt: length,
              reward-model score, sampled KL pooled over all generated tokens.
  wordlimit   5 sampled responses per word-limit prompt: compliance, length, RM score.

--adapter none evaluates the untrained base model; only generate/wordlimit are allowed for it
and its KL is 0 by definition (policy == reference), recorded rather than computed.
"""
from __future__ import annotations

import argparse
import math

import torch

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import parse_word_limit, sampled_kl, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.rollouts import GEN_BATCH_SIZE, TF_BATCH_SIZE, generate_responses
from common.run_info import display_path, peak_vram_bytes, run_metadata
from common.stats import describe
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import (
    dpo_sequence_logprobs,
    filter_pairs,
    filter_report,
    make_collate,
    to_device,
)

MODES = ("pairs", "stratified", "generate", "wordlimit")
GENERATION_MODES = ("generate", "wordlimit")
# Memory only: rows per reward-model call (generation and teacher-forcing batches: common.rollouts).
RM_BATCH_SIZE = 8
RM_MAX_LENGTH = 1024  # score_reward_pairs default
WORDLIMIT_SAMPLES = 5
LOGP_DECIMALS = 6
PROGRESS_EVERY = 25  # print a progress line every N teacher-forcing / reward-model batches


# ---------------------------------------------------------------- aggregation (pure, tested on Mac)

def pair_summary(records: list[dict], beta: float) -> dict:
    """Held-out DPO loss (task1_dpo.dpo.dpo_loss at beta), accuracy (m > 0) and mean m."""
    if not records:
        return {"n": 0}
    t = lambda k: torch.tensor([r[k] for r in records], dtype=torch.float64)
    loss, _ = dpo_loss(t("policy_chosen_logp"), t("policy_rejected_logp"), t("ref_chosen_logp"), t("ref_rejected_logp"), beta)
    m = [r["m"] for r in records]
    return {
        "n": len(records),
        "beta": float(beta),
        "dpo_loss": float(loss.item()),
        "preference_accuracy": sum(x > 0 for x in m) / len(m),
        "mean_margin": sum(m) / len(m),
        "margin_stats": describe(m),
        "mean_chosen_logratio": sum(r["chosen_logratio"] for r in records) / len(records),
        "mean_rejected_logratio": sum(r["rejected_logratio"] for r in records) / len(records),
    }


def stratified_summary(records: list[dict], beta: float) -> dict:
    strata = sorted({r["stratum"] for r in records})
    return {
        "overall": pair_summary(records, beta),
        "per_stratum": {s: pair_summary([r for r in records if r["stratum"] == s], beta) for s in strata},
    }


def length_summary(lengths: list[int], truncated: list[bool]) -> dict:
    out = describe(lengths)
    out["truncated_fraction"] = sum(truncated) / len(truncated) if truncated else None
    out["n_truncated"] = int(sum(truncated))
    return out


def pooled_kl(policy_logps: list[list[float]], ref_logps: list[list[float]]) -> float:
    """common.metrics.sampled_kl over all generated tokens of all responses (token-level mean)."""
    width = max((len(x) for x in policy_logps), default=0)
    if width == 0:
        return float("nan")
    n = len(policy_logps)
    pol = torch.zeros(n, width, dtype=torch.float64)
    ref = torch.zeros(n, width, dtype=torch.float64)
    mask = torch.zeros(n, width, dtype=torch.float64)
    for i, (p, r) in enumerate(zip(policy_logps, ref_logps)):
        pol[i, : len(p)] = torch.tensor(p, dtype=torch.float64)
        ref[i, : len(r)] = torch.tensor(r, dtype=torch.float64)
        mask[i, : len(p)] = 1.0
    return float(sampled_kl(pol, ref, mask).item())


def compliance_summary(records: list[dict]) -> dict:
    """Word-limit compliance overall and per prompt (records carry prompt_id, compliant, word_count, limit)."""
    scored = [r for r in records if r["compliant"] is not None]
    per_prompt = {}
    for pid in sorted({r["prompt_id"] for r in records}):
        rs = [r for r in scored if r["prompt_id"] == pid]
        per_prompt[pid] = {
            "limit": next(r["limit"] for r in records if r["prompt_id"] == pid),
            "n": len(rs),
            "compliance": sum(r["compliant"] for r in rs) / len(rs) if rs else None,
            "mean_word_count": sum(r["word_count"] for r in rs) / len(rs) if rs else None,
        }
    return {
        "n_responses": len(records),
        "n_unparsed_limit": len(records) - len(scored),
        "compliance": sum(r["compliant"] for r in scored) / len(scored) if scored else None,
        "word_count": describe([r["word_count"] for r in scored]),
        "per_prompt": per_prompt,
    }


# ---------------------------------------------------------------- model passes

def score_pairs(model, tokenizer, rows, records, cfg) -> list[dict]:
    """Per-pair policy/reference summed response log-probs and margin m, file order, no grad."""
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    bs = int(cfg["batch_size"])
    out = []
    timer, n_batches = wall_timer(), math.ceil(len(rows) / bs)
    for b, start in enumerate(range(0, len(rows), bs), start=1):
        recs = records[start : start + bs]
        chosen, rejected, _ = collate([(r["id"], row) for r, row in zip(recs, rows[start : start + bs])])
        pc, pr, rc, rr = dpo_sequence_logprobs(model, to_device(chosen, device), to_device(rejected, device), with_grad=False)
        for j, rec in enumerate(recs):
            lc, lr_ = float(pc[j] - rc[j]), float(pr[j] - rr[j])
            out.append({
                "id": rec["id"],
                "index": rec["index"],
                "stratum": rec.get("stratum"),
                "policy_chosen_logp": float(pc[j]),
                "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]),
                "ref_rejected_logp": float(rr[j]),
                "chosen_logratio": lc,
                "rejected_logratio": lr_,
                "m": lc - lr_,
                "chosen_truncated": rec["chosen_truncated"],
                "rejected_truncated": rec["rejected_truncated"],
            })
        if b % PROGRESS_EVERY == 0 or b == n_batches:
            print(f"  teacher forcing batch {b}/{n_batches} t={timer():.0f}s", flush=True)
    return out


def reward_scores(rm, rm_tok, prompts: list[list[dict]], texts: list[str]):
    """Fixed course reward model scores, plus which inputs exceed RM_MAX_LENGTH (truncated by the helper)."""
    scores, over = [], []
    timer, n_batches = wall_timer(), math.ceil(len(texts) / RM_BATCH_SIZE)
    for b, start in enumerate(range(0, len(texts), RM_BATCH_SIZE), start=1):
        ps, ts = prompts[start : start + RM_BATCH_SIZE], texts[start : start + RM_BATCH_SIZE]
        scores += score_reward_pairs(rm, rm_tok, ps, ts, max_length=RM_MAX_LENGTH).cpu().tolist()
        if b % PROGRESS_EVERY == 0 or b == n_batches:
            print(f"  reward model batch {b}/{n_batches} t={timer():.0f}s", flush=True)
    for i, (p, t) in enumerate(zip(prompts, texts)):
        n = len(rm_tok.apply_chat_template(list(p) + [{"role": "assistant", "content": t}], tokenize=True, add_generation_prompt=False))
        over.append(n > RM_MAX_LENGTH)
    return scores, over


def round_list(xs):
    return [round(x, LOGP_DECIMALS) for x in xs]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help='adapter path, or "none" for the untrained base model')
    ap.add_argument("--name", required=True)
    ap.add_argument("--beta", type=float, help="beta for the held-out DPO loss (default: config beta)")
    ap.add_argument("--modes", default="pairs,generate")
    ap.add_argument("--limit", type=int, help="first N retained rows/prompts per mode; smoke tests only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    max_length = int(cfg["max_sequence_length"])
    beta = float(cfg["beta"] if args.beta is None else args.beta)
    base_only = args.adapter.lower() == "none"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown modes {bad}; choose from {MODES}")
    if base_only and any(m not in GENERATION_MODES for m in modes):
        raise SystemExit("--adapter none runs generate/wordlimit only")

    # Refuse before loading anything if an output would be overwritten.
    results_dir = cfg["results_dir"]
    out_json = repo_path(f"{results_dir}/eval_{args.name}.json")
    gen_path = lambda mode: repo_path(f"{results_dir}/generations_{args.name}_{mode}.jsonl")
    adapter_value = "none" if base_only else args.adapter
    result = load_json(out_json) if out_json.exists() else None
    if result is not None and result["adapter"] != adapter_value:
        raise SystemExit(f"{out_json} belongs to adapter {result['adapter']!r}, not {adapter_value!r}")
    for mode in modes:
        clash = result is not None and mode in result["modes"]
        if mode in GENERATION_MODES and gen_path(mode).exists():
            clash = True
        if clash and not args.overwrite:
            raise SystemExit(f"mode {mode!r} already has results for {args.name!r}; pass --overwrite to replace it.")
    if result is None:
        result = {"script": "task1_dpo.evaluate", "name": args.name, "adapter": adapter_value,
                  "config_path": args.config, "config": cfg, "modes": {}}

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=None if base_only else args.adapter, trainable=False)
    rm = None
    eval_rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])

    for mode in modes:
        elapsed = wall_timer()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        entry = {**run_metadata(cfg), "limit": args.limit, "smoke": args.limit is not None}
        print(f"[{mode}] start ({args.name})", flush=True)

        if mode in ("pairs", "stratified"):
            path = cfg["paths"]["dpo_standard_eval" if mode == "pairs" else "dpo_length_eval"]
            rows = eval_rows if mode == "pairs" else read_jsonl(path)
            kept, recs, skipped = filter_pairs(tokenizer, rows, max_length)
            entry["filter"] = filter_report(path, max_length, recs, skipped)
            if args.limit is not None:
                kept, recs = kept[: args.limit], recs[: args.limit]
            per_pair = score_pairs(model, tokenizer, kept, recs, cfg)
            entry["settings"] = {"beta": beta, "batch_size": int(cfg["batch_size"]),
                                 "reference": "same model with LoRA adapter disabled (common.models.reference_mode)"}
            if mode == "pairs":
                entry["metrics"] = pair_summary(per_pair, beta)
            else:
                entry["metrics"] = stratified_summary(per_pair, beta)
            entry["per_pair"] = per_pair

        elif mode == "generate":
            path = cfg["paths"]["dpo_standard_eval"]
            kept, recs, skipped = filter_pairs(tokenizer, eval_rows, max_length)
            entry["filter"] = filter_report(path, max_length, recs, skipped)
            if args.limit is not None:
                kept, recs = kept[: args.limit], recs[: args.limit]
            prompts = [prompt_messages_from_preference(r) for r in kept]
            set_seed(seed)  # reset right before this model's generation pass
            gens = generate_responses(model, tokenizer, prompts, cfg, with_kl=not base_only)
            if rm is None:
                rm = load_reward_model(cfg)
            scores, rm_over = reward_scores(*rm, prompts, [g["text"] for g in gens])
            rows_out = []
            for rec, g, s, o in zip(recs, gens, scores, rm_over):
                row = {"prompt_id": rec["id"], "index": rec["index"], **g, "rm_score": s, "rm_input_truncated": o}
                if not base_only:
                    lr = [a - b for a, b in zip(g["policy_token_logp"], g["ref_token_logp"])]
                    row["summed_log_ratio"] = sum(lr)
                rows_out.append(row)
            if base_only:
                kl = {"value": 0.0, "computed": False,
                      "note": "untrained base model: policy equals reference, KL is 0 by definition"}
            else:
                kl = {"value": pooled_kl([g["policy_token_logp"] for g in gens], [g["ref_token_logp"] for g in gens]),
                      "computed": True,
                      "estimator": "common.metrics.sampled_kl pooled over all generated response tokens (token-level mean)",
                      "n_tokens": sum(g["token_length"] for g in gens),
                      "summed_log_ratio_stats": describe([r["summed_log_ratio"] for r in rows_out])}
            entry["settings"] = generation_settings(cfg, 1)
            entry["metrics"] = {
                "n_responses": len(rows_out),
                "kl": kl,
                "length": length_summary([r["token_length"] for r in rows_out], [r["truncated"] for r in rows_out]),
                "n_terminated_with_eos": sum(r["terminated_with_eos"] for r in rows_out),
                "reward": describe(scores),
                "n_rm_input_truncated": sum(rm_over),
                "rm_input_truncated_ids": [r["prompt_id"] for r in rows_out if r["rm_input_truncated"]],
            }
            for r in rows_out:
                if not base_only:
                    r["policy_token_logp"] = round_list(r["policy_token_logp"])
                    r["ref_token_logp"] = round_list(r["ref_token_logp"])
            write_jsonl(gen_path(mode), rows_out)
            entry["generations_file"] = display_path(gen_path(mode))

        elif mode == "wordlimit":
            path = cfg["paths"]["word_limit_prompts"]
            wl_rows = read_jsonl(path)
            if args.limit is not None:
                wl_rows = wl_rows[: args.limit]
            items = []
            for row in wl_rows:
                user = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
                for s in range(WORDLIMIT_SAMPLES):
                    items.append({"prompt_id": row["prompt_id"], "sample_index": s, "messages": row["messages"], "user": user})
            prompts = [it["messages"] for it in items]
            set_seed(seed)  # reset right before this model's generation pass
            gens = generate_responses(model, tokenizer, prompts, cfg, with_kl=False)
            if rm is None:
                rm = load_reward_model(cfg)
            scores, rm_over = reward_scores(*rm, prompts, [g["text"] for g in gens])
            rows_out = []
            for it, g, s, o in zip(items, gens, scores, rm_over):
                rows_out.append({
                    "prompt_id": it["prompt_id"],
                    "sample_index": it["sample_index"],
                    **g,
                    "word_count": word_count(g["text"]),
                    "limit": parse_word_limit(it["user"]),
                    "compliant": word_limit_compliance(it["user"], g["text"]),
                    "rm_score": s,
                    "rm_input_truncated": o,
                })
            entry["filter"] = {"path": path, "n_prompts": len(wl_rows), "samples_per_prompt": WORDLIMIT_SAMPLES}
            entry["settings"] = generation_settings(cfg, WORDLIMIT_SAMPLES)
            entry["metrics"] = {
                **compliance_summary(rows_out),
                "length": length_summary([r["token_length"] for r in rows_out], [r["truncated"] for r in rows_out]),
                "reward": describe(scores),
                "n_rm_input_truncated": sum(rm_over),
            }
            write_jsonl(gen_path(mode), rows_out)
            entry["generations_file"] = display_path(gen_path(mode))

        entry["wall_clock_seconds"] = elapsed()
        entry["peak_vram_bytes"] = peak_vram_bytes()
        result["modes"][mode] = entry
        save_json(out_json, result)
        print(f"[{mode}] {summary_line(entry['metrics'])}")

    if rm is not None:
        clear_gpu(*rm)
    print(f"Wrote {out_json}")


def generation_settings(cfg: dict, samples_per_prompt: int) -> dict:
    gen = cfg["generation"]
    return {
        "temperature": float(gen["temperature"]),
        "top_p": float(gen["top_p"]),
        "do_sample": bool(gen["do_sample"]),
        "max_new_tokens": int(cfg["max_generation_tokens"]),
        "max_prompt_length": int(cfg["max_sequence_length"]),
        "samples_per_prompt": samples_per_prompt,
        "seed_reset_before_pass": int(cfg["seed"]),
        "gen_batch_size": GEN_BATCH_SIZE,
        "teacher_forcing_batch_size": TF_BATCH_SIZE,
        "reward_model": cfg["reward_model"],
        "reward_tokenizer": cfg.get("reward_tokenizer", cfg["base_model"]),
        "rm_max_length": RM_MAX_LENGTH,
        "token_length_definition": "response_mask.sum() from common.generation.batch_generate: tokens up to and including the first EOS, padding excluded",
    }


def summary_line(metrics: dict) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)
    if "overall" in metrics:
        metrics = metrics["overall"]
    keys = ["n", "dpo_loss", "preference_accuracy", "mean_margin", "n_responses", "compliance"]
    parts = [f"{k}={fmt(metrics[k])}" for k in keys if k in metrics]
    if "kl" in metrics:
        parts.append(f"kl={fmt(metrics['kl']['value'])}")
    if "length" in metrics:
        parts.append(f"len_mean={fmt(metrics['length']['mean'])}")
    if "reward" in metrics:
        parts.append(f"rm_mean={fmt(metrics['reward']['mean'])}")
    return " ".join(parts)


if __name__ == "__main__":
    main()
