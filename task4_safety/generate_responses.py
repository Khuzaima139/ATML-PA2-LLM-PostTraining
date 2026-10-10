"""Task 4 Step 1: one deterministic response per XSTest prompt for one fixed policy (manual Section 4).

Decoding is the released path (generate_for_policy -> common.generation.batch_generate with
do_sample=False): argmax after the model's own generation_config, which adds repetition_penalty 1.1
(recorded in the run JSON). Prompts in xstest_id order (CSV row order), batch size GEN_BATCH_SIZE for
every policy, response cap safety_max_new_tokens. The adapter hash is checked before loading.

Writes results/task4_safety/generated_<policy>.jsonl and generate_<policy>.json (run metadata).
"""
from __future__ import annotations

import argparse
import hashlib

import pandas as pd
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer
from common.rollouts import effective_generation_settings
from common.run_info import display_path, peak_vram_bytes, run_metadata
from task4_safety.protocol import GEN_BATCH_SIZE, POLICIES, generated_name, sha256_file, task_dir, verify_adapter

MAX_PROMPT_LENGTH = 256  # hard-coded in the released generate_for_policy


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_records(model, tokenizer, df, policy_name: str, max_new_tokens: int, batch_size: int, on_batch=None):
    """The released generation loop; records also keep the truncated and EOS flags from batch_generate."""
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok, trunc, eos in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"],
                                                          gen["truncated"], gen["terminated_with_eos"], strict=True):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
                "terminated_with_eos": bool(eos),
            })
        if on_batch is not None:
            on_batch(len(records))
    return records


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    return generate_records(model, tokenizer, df, policy_name, int(cfg["safety_max_new_tokens"]), batch_size)


def prompt_token_counts(tokenizer, df) -> list[int]:
    return [len(tokenizer.apply_chat_template([{"role": "user", "content": str(p)}], tokenize=True, add_generation_prompt=True))
            for p in df["prompt"].tolist()]


def id_order_sha256(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--adapter", help="adapter directory (omit for sft); overrides the config path, hash-checked")
    ap.add_argument("--limit", type=int, help="first N prompts; smoke tests only")
    ap.add_argument("--out-dir", help="output directory (default results/task4_safety); use a smoke dir for smoke runs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    adapter_sha = verify_adapter(args.policy, args.adapter)
    outdir = repo_path(args.out_dir) if args.out_dir else task_dir(cfg)
    gen_path = outdir / generated_name(args.policy)
    run_path = outdir / f"generate_{args.policy}.json"
    for p in (gen_path, run_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    max_new = int(cfg["safety_max_new_tokens"])
    df = load_xstest(cfg)
    if list(df["xstest_id"]) != sorted(df["xstest_id"]):
        raise SystemExit("XSTest CSV is not in xstest_id order")
    if args.limit is not None:
        df = df.iloc[: args.limit]
    tokenizer = load_tokenizer(cfg["base_model"])
    n_prompt = prompt_token_counts(tokenizer, df)
    over = [int(i) for i, n in zip(df["xstest_id"], n_prompt) if n > MAX_PROMPT_LENGTH]

    result = {
        "script": "task4_safety.generate_responses",
        "policy": args.policy,
        "adapter": args.adapter,
        "adapter_weights_sha256": adapter_sha,
        "status": "running",
        "config_path": args.config,
        "config": cfg,
        "limit": args.limit,
        "smoke": args.limit is not None,
        **run_metadata(cfg),
        "prompts": {
            "path": cfg["paths"]["xstest"],
            "file_sha256": sha256_file(repo_path(cfg["paths"]["xstest"])),
            "n_prompts": len(df),
            "order": "xstest_id ascending (CSV row order)",
            "xstest_id_order_sha256": id_order_sha256(df["xstest_id"]),
            "prompt_tokens_max": max(n_prompt),
            "n_prompts_over_max_prompt_length": len(over),
            "ids_over_max_prompt_length": over,
        },
        "settings": {
            "decoding": "released deterministic path: batch_generate(do_sample=False); other keys from the model generation_config",
            "max_new_tokens": max_new,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "gen_batch_size": GEN_BATCH_SIZE,
            "system_prompt": "none passed; the chat template inserts the Qwen default system prompt",
            "seed_set_before_generation": int(cfg["seed"]),
            "response_tokens_definition": "batch_generate response_lengths: generated tokens after the prompt up to and including the first EOS, padding excluded",
            "truncated_definition": "response_tokens reached max_new_tokens without EOS",
        },
        "n_done": 0,
        "timing": {},
    }

    def write():
        result["wall_clock_seconds"] = elapsed()
        result["peak_vram_bytes"] = peak_vram_bytes()
        save_json(run_path, result)

    if over:
        result["status"] = "failed: prompts longer than max_prompt_length would be truncated"
        write()
        raise SystemExit(f"{len(over)} prompts exceed {MAX_PROMPT_LENGTH} tokens; not generating")
    write()
    try:
        set_seed(int(cfg["seed"]))
        model = load_policy(cfg, adapter_path=args.adapter, trainable=False)
        result["settings"]["effective_generation"] = effective_generation_settings(model, tokenizer, {"do_sample": False}, max_new)
        result["settings"]["model_generation_config"] = model.generation_config.to_dict()
        result["timing"]["setup"] = elapsed()

        def progress(n):
            result["n_done"] = n
            write()

        t0 = elapsed()
        records = generate_records(model, tokenizer, df, args.policy, max_new, GEN_BATCH_SIZE, on_batch=progress)
        result["timing"]["generation"] = elapsed() - t0
        write_jsonl(gen_path, records)
        result["generations_file"] = display_path(gen_path)
        result["generations_sha256"] = sha256_file(gen_path)
        result["counts"] = {
            "n_rows": len(records),
            "n_truncated": sum(r["truncated"] for r in records),
            "n_terminated_with_eos": sum(r["terminated_with_eos"] for r in records),
            "n_empty_response": sum(not r["response"].strip() for r in records),
        }
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise
    result["status"] = "completed"
    write()
    vram = result["peak_vram_bytes"]
    c = result["counts"]
    # Console: counts, timing and memory only.
    print(f"[generate {args.policy}] rows={c['n_rows']} truncated={c['n_truncated']} eos={c['n_terminated_with_eos']} "
          f"empty={c['n_empty_response']} t={elapsed():.0f}s peak_vram={vram / 2**30 if vram else 0:.2f}GiB; "
          f"wrote {display_path(gen_path)} sha256={result['generations_sha256']}", flush=True)


if __name__ == "__main__":
    main()
