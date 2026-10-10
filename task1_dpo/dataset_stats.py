"""Token-length profile of the four Task 1 DPO preference files.

Uses the base-model tokenizer and the exact training encoding
(common.data.encode_prompt_response with the config's max_sequence_length).
No model weights are loaded.

Length definitions (per response):
  raw_tokens     = len(tokenizer(response, add_special_tokens=False)), no EOS,
                   computed for every pair.
  encoded_tokens = response_mask.sum() from encode_prompt_response, i.e. the
                   response tokens the DPO loss actually sums over (content
                   after truncation plus the appended EOS). Only defined for
                   pairs whose encoding succeeds.
  truncated      = encoded content (encoded_tokens - 1) is shorter than raw_tokens.
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess

import numpy as np

from common.data import (
    encode_prompt_response,
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import save_json, wall_timer
from common.models import load_tokenizer
from common.stats import describe

FILES = {
    "dpo_standard_train": "dpo_standard_train",
    "dpo_standard_eval": "dpo_standard_eval",
    "dpo_length_balanced_train": "dpo_length_train",
    "dpo_length_stratified_eval": "dpo_length_eval",
}
STRATUM_KEY = "length_stratum"


def git_info() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo_path(".")).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def compare(chosen, rejected) -> dict:
    c = np.asarray(chosen, dtype=float)
    r = np.asarray(rejected, dtype=float)
    if c.size == 0:
        return {"n": 0}
    d = c - r
    return {
        "n": int(c.size),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_equal": float((d == 0).mean()),
        "frac_chosen_shorter": float((d < 0).mean()),
        "mean_chosen_minus_rejected": float(d.mean()),
    }


def profile_pairs(pairs: list[dict]) -> dict:
    ok = [p for p in pairs if p["encode_error"] is None]
    return {
        "n_pairs": len(pairs),
        "raw_tokens": {
            "chosen": describe([p["chosen_raw"] for p in pairs]),
            "rejected": describe([p["rejected_raw"] for p in pairs]),
            "chosen_vs_rejected": compare([p["chosen_raw"] for p in pairs], [p["rejected_raw"] for p in pairs]),
        },
        "encoded_tokens": {
            "chosen": describe([p["chosen_enc"] for p in ok]),
            "rejected": describe([p["rejected_enc"] for p in ok]),
            "chosen_vs_rejected": compare([p["chosen_enc"] for p in ok], [p["rejected_enc"] for p in ok]),
        },
        "n_encode_failed_prompt_too_long": len(pairs) - len(ok),
        "n_pairs_any_response_truncated": sum(p["chosen_trunc"] or p["rejected_trunc"] for p in ok),
        "n_chosen_truncated": sum(p["chosen_trunc"] for p in ok),
        "n_rejected_truncated": sum(p["rejected_trunc"] for p in ok),
        "prompt_tokens": describe([p["prompt_tokens"] for p in pairs]),
    }


def encode_row(tokenizer, row: dict, max_length: int) -> dict:
    messages = prompt_messages_from_preference(row)
    yc, yr = preference_responses(row)
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    out = {
        "prompt_id": row.get("prompt_id"),
        "prompt_tokens": len(prompt_ids),
        "chosen_raw": len(tokenizer(yc, add_special_tokens=False)["input_ids"]),
        "rejected_raw": len(tokenizer(yr, add_special_tokens=False)["input_ids"]),
        "encode_error": None,
    }
    try:
        _, mask_c = encode_prompt_response(tokenizer, messages, yc, max_length)
        _, mask_r = encode_prompt_response(tokenizer, messages, yr, max_length)
    except ValueError as e:
        out["encode_error"] = str(e)
        return out
    # encoded length includes the appended EOS, so content = encoded - 1
    out["chosen_enc"] = int(sum(mask_c))
    out["rejected_enc"] = int(sum(mask_r))
    out["chosen_trunc"] = out["chosen_enc"] - 1 < out["chosen_raw"]
    out["rejected_trunc"] = out["rejected_enc"] - 1 < out["rejected_raw"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_path = repo_path(args.output or f"{cfg['results_dir']}/dataset_length_profile.json")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    elapsed = wall_timer()
    max_length = int(cfg["max_sequence_length"])
    tokenizer = load_tokenizer(cfg["base_model"])
    if tokenizer.eos_token_id is None:
        raise SystemExit("Tokenizer has no EOS token; the encoded-length definition assumes one.")

    result = {
        "config_path": args.config,
        "config": cfg,
        "git": git_info(),
        "tokenizer": cfg["base_model"],
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token": tokenizer.eos_token,
        "max_sequence_length": max_length,
        "hardware": platform.platform(),
        "start_time": dt.datetime.now().isoformat(timespec="seconds"),
        "length_definitions": {
            "raw_tokens": "tokenizer(response, add_special_tokens=False) length, no EOS, all pairs",
            "encoded_tokens": "response_mask.sum() from encode_prompt_response (truncated content + EOS), encodable pairs only",
            "truncated": "encoded_tokens - 1 < raw_tokens",
        },
        "files": {},
    }

    for name, key in FILES.items():
        path = cfg["paths"][key]
        rows = read_jsonl(path)
        pairs = [encode_row(tokenizer, r, max_length) for r in rows]
        entry = {"path": path, **profile_pairs(pairs)}
        entry["encode_failed_prompt_ids"] = [p["prompt_id"] for p in pairs if p["encode_error"] is not None]
        entry["truncated_prompt_ids"] = [
            p["prompt_id"] for p in pairs if p["encode_error"] is None and (p["chosen_trunc"] or p["rejected_trunc"])
        ]
        if any(STRATUM_KEY in r for r in rows):
            strata = sorted({r[STRATUM_KEY] for r in rows})
            entry["per_stratum"] = {
                s: profile_pairs([p for p, r in zip(pairs, rows) if r[STRATUM_KEY] == s]) for s in strata
            }
            # Supplied per-row token counts, compared with this tokenizer's raw counts.
            if all("chosen_tokens" in r and "rejected_tokens" in r for r in rows):
                entry["supplied_token_fields_match_raw"] = {
                    "chosen": sum(int(r["chosen_tokens"]) == p["chosen_raw"] for p, r in zip(pairs, rows)),
                    "rejected": sum(int(r["rejected_tokens"]) == p["rejected_raw"] for p, r in zip(pairs, rows)),
                    "n": len(rows),
                }
        result["files"][name] = entry
        print(
            f"{name}: n={entry['n_pairs']} encode_failed={entry['n_encode_failed_prompt_too_long']} "
            f"truncated_pairs={entry['n_pairs_any_response_truncated']}"
        )

    result["wall_clock_seconds"] = elapsed()
    save_json(out_path, result)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
