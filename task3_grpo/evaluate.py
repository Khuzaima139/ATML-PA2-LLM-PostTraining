"""Task 3 held-out evaluation, identical for the standard adapter and both fork adapters.

Prompts: rl_prompt_pool_eval.jsonl under rule P (common.rollouts.fitting_prompts: prompts longer
than max_prompt_length are excluded and listed). One sampled response per prompt, seed reset right
before generation, generation settings from the config (temperature 0.7, top_p 0.9; other keys from
the model's generation_config, recorded), response cap EVAL_MAX_NEW_TOKENS. Generation and teacher
forcing reuse common.rollouts.generate_responses (fixed batch sizes), in eval mode.

Metrics: RM score (cap EVAL_RM_MAX_LENGTH; truncated inputs counted), common.metrics.sampled_kl
against the adapter-disabled reference and sample_entropy, both pooled over all response tokens
(token-level means), response length mean/SD, truncation rate at the cap. Per-prompt rows
(prompt_id, text, RM score, per-prompt token sums) go to a JSONL for the paired bootstrap.
"""
from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.rollouts import (GEN_BATCH_SIZE, TF_BATCH_SIZE, effective_generation_settings, fitting_prompts,
                             generate_responses, pooled_token_metrics, reward_with_lengths)
from common.run_info import display_path, peak_vram_bytes, run_metadata
from common.stats import describe

# configs/grpo.yaml sets neither (same values as the Task 2 evaluation).
EVAL_MAX_NEW_TOKENS = 768
EVAL_RM_MAX_LENGTH = 1280
RM_BATCH_SIZE = 8


def per_prompt_rows(records: list[dict], gens: list[dict], scores: list[float], rm_lengths: list[int]) -> list[dict]:
    rows = []
    for rec, g, s, n_rm in zip(records, gens, scores, rm_lengths, strict=True):
        pol, ref = g["policy_token_logp"], g["ref_token_logp"]
        rows.append({
            "prompt_id": rec["prompt_id"],
            "index": rec["index"],
            "prompt_tokens": g["prompt_tokens"],
            "text": g["text"],
            "token_length": g["token_length"],
            "terminated_with_eos": g["terminated_with_eos"],
            "truncated": g["truncated"],
            "rm_score": s,
            "rm_input_tokens": n_rm,
            "rm_input_truncated": n_rm > EVAL_RM_MAX_LENGTH,
            "sum_log_ratio": float(sum(a - b for a, b in zip(pol, ref))),
            "sum_neg_logp": float(-sum(pol)),
        })
    return rows


def summary_metrics(rows: list[dict], gens: list[dict]) -> dict:
    tok = pooled_token_metrics([g["policy_token_logp"] for g in gens], [g["ref_token_logp"] for g in gens])
    lengths = [r["token_length"] for r in rows]
    n_trunc = sum(r["truncated"] for r in rows)
    return {
        "n_responses": len(rows),
        "reward": describe([r["rm_score"] for r in rows]),
        "n_rm_input_truncated": sum(r["rm_input_truncated"] for r in rows),
        "rm_input_truncated_ids": [r["prompt_id"] for r in rows if r["rm_input_truncated"]],
        "kl": tok["kl"],
        "entropy": tok["entropy"],
        "n_response_tokens": tok["n_tokens"],
        "length": describe(lengths),
        "n_truncated_at_cap": int(n_trunc),
        "truncation_rate_at_cap": n_trunc / len(rows) if rows else None,
        "n_terminated_with_eos": sum(r["terminated_with_eos"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="policy adapter directory, e.g. outputs/task3_grpo/standard/policy")
    ap.add_argument("--name", default="standard", help="run name; writes eval_<name>.json")
    ap.add_argument("--limit", type=int, help="first N kept prompts; smoke tests only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_json = repo_path(f"{cfg['results_dir']}/eval_{args.name}.json")
    gen_path = repo_path(f"{cfg['results_dir']}/generations_eval_{args.name}.jsonl")
    for p in (out_json, gen_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    seed = int(cfg["seed"])
    max_prompt = int(cfg["max_prompt_length"])
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    kept, records, filt = fitting_prompts(tokenizer, rows, max_prompt, prompt_messages)
    if args.limit is not None:
        kept, records = kept[: args.limit], records[: args.limit]
    prompts = [prompt_messages(r) for r in kept]

    result = {
        "script": "task3_grpo.evaluate",
        "name": args.name,
        "adapter": args.adapter,
        "status": "running",
        "config_path": args.config,
        "config": cfg,
        "limit": args.limit,
        "smoke": args.limit is not None,
        **run_metadata(cfg),
        "prompts": {"path": cfg["paths"]["rl_prompt_eval"], "filter": filt, "n_evaluated": len(prompts)},
        "settings": {
            "samples_per_prompt": 1,
            "temperature": float(cfg["generation"]["temperature"]),
            "top_p": float(cfg["generation"]["top_p"]),
            "do_sample": bool(cfg["generation"]["do_sample"]),
            "max_prompt_length": max_prompt,
            "max_new_tokens": EVAL_MAX_NEW_TOKENS,
            "seed_reset_before_generation": seed,
            "gen_batch_size": GEN_BATCH_SIZE,
            "teacher_forcing_batch_size": TF_BATCH_SIZE,
            "rm_batch_size": RM_BATCH_SIZE,
            "reward_model": cfg["reward_model"],
            "reward_max_length": EVAL_RM_MAX_LENGTH,
            "reference": "same policy with the LoRA adapter disabled (common.models.reference_mode)",
            "kl": "common.metrics.sampled_kl pooled over all response tokens (token-level mean)",
            "entropy": "common.metrics.sample_entropy pooled over all response tokens (token-level mean)",
            "dropout": "eval mode (load_policy trainable=False): dropout inactive",
            "token_length_definition": "response_mask.sum() from common.generation.batch_generate: tokens up to and including the first EOS",
        },
        "timing": {},
    }

    def write():
        result["wall_clock_seconds"] = elapsed()
        result["peak_vram_bytes"] = peak_vram_bytes()
        save_json(out_json, result)

    write()
    try:
        model = load_policy(cfg, adapter_path=args.adapter, trainable=False)
        result["settings"]["effective_generation"] = effective_generation_settings(model, tokenizer, cfg["generation"], EVAL_MAX_NEW_TOKENS)
        result["timing"]["setup"] = elapsed()
        shim = {"generation": cfg["generation"], "max_sequence_length": max_prompt, "max_generation_tokens": EVAL_MAX_NEW_TOKENS}
        set_seed(seed)
        t0 = elapsed()
        gens = generate_responses(model, tokenizer, prompts, shim, with_kl=True)
        result["timing"]["generation_and_teacher_forcing"] = elapsed() - t0
        result["timing"]["n_generation_batches"] = -(-len(prompts) // GEN_BATCH_SIZE)
        del model
        clear_gpu()

        t0 = elapsed()
        rm, rm_tok = load_reward_model(cfg)
        scores, rm_lengths = reward_with_lengths(rm, rm_tok, prompts, [g["text"] for g in gens], EVAL_RM_MAX_LENGTH, RM_BATCH_SIZE)
        result["timing"]["reward"] = elapsed() - t0
        result["settings"]["rm_truncation_side"] = rm_tok.truncation_side
        del rm
        clear_gpu()

        out_rows = per_prompt_rows(records, gens, scores, rm_lengths)
        result["metrics"] = summary_metrics(out_rows, gens)
        write_jsonl(gen_path, out_rows)
        result["generations_file"] = display_path(gen_path)
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise
    result["status"] = "completed"
    write()
    vram = result["peak_vram_bytes"]
    m = result["metrics"]
    # Console: counts, timing and memory only.
    print(f"[eval {args.name}] responses={m['n_responses']} eos={m['n_terminated_with_eos']} truncated_at_cap={m['n_truncated_at_cap']} "
          f"rm_inputs_truncated={m['n_rm_input_truncated']} t={elapsed():.0f}s peak_vram={vram / 2**30 if vram else 0:.2f}GiB; wrote {display_path(out_json)}", flush=True)


if __name__ == "__main__":
    main()
