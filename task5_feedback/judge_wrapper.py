"""Thin wrapper around the released PairwiseAIJudge that keeps what compare() discards.

judge_call reproduces rlaif.PairwiseAIJudge.compare line for line (same key, hashed orientation, rubric,
chat template, generate arguments, decode, regex and TIE fallback, label mapped back) and additionally
returns the raw judge text, the physical order and whether the regex matched. It never reads or writes
the released cache, so every call is a fresh generation. With swap_order=True the physical order is the
opposite of the released one; everything else is unchanged.
"""
from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

import torch

from task5_feedback.rlaif import PAIRWISE_RUBRIC

FLIP = {"A": "B", "B": "A", "TIE": "TIE"}


def released_swap(judge, problem: str, a: str, b: str) -> bool:
    """True when the released compare() shows b in the Candidate A slot."""
    return int(judge._key(problem, a, b)[:8], 16) % 2 == 1


@torch.no_grad()
def judge_call(judge, problem: str, a: str, b: str, swap_order: bool = False) -> dict:
    key = judge._key(problem, a, b)
    rel_swap = int(key[:8], 16) % 2 == 1
    swap = (not rel_swap) if swap_order else rel_swap
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(next(judge.model.parameters()).device)
    t0 = time.perf_counter()
    out = judge.model.generate(
        ids,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=judge.tokenizer.eos_token_id,
        eos_token_id=judge.tokenizer.eos_token_id,
    )
    seconds = time.perf_counter() - t0
    raw = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    decoded = raw.strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", decoded)
    physical = m.group(1) if m else "TIE"
    label = FLIP[physical] if swap else physical
    return {
        "key": key,
        "label": label,                       # content level, w.r.t. (a, b); parse failures are TIE as released
        "physical_label": physical,           # w.r.t. the slots shown to the judge
        "parse_matched": m is not None,
        "raw": raw,
        "physical_order": "BA" if swap else "AB",  # "BA": argument b was shown as Candidate A
        "released_order": not swap_order,
        "prompt_tokens": int(ids.shape[1]),
        "generated_tokens": int(out.shape[1] - ids.shape[1]),
        "seconds": seconds,
    }


def parity_check(judge, problem: str, a: str, b: str, wrapper_label: str) -> dict:
    """Run the released compare() with a throwaway cache and compare its label with the wrapper's."""
    old_cache, old_path = judge.cache, judge.cache_path
    with tempfile.TemporaryDirectory() as d:
        judge.cache, judge.cache_path = {}, Path(d) / "parity_cache.json"
        try:
            released = judge.compare(problem, a, b)
        finally:
            judge.cache, judge.cache_path = old_cache, old_path
    return {"released_label": released, "wrapper_label": wrapper_label, "match": released == wrapper_label}


def judge_generation_settings(judge, cfg: dict) -> dict:
    gc = judge.model.generation_config
    keys = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "no_repeat_ngram_size")
    return {
        "model": cfg["ai_judge_model"],
        "passed_by_release": {"max_new_tokens": 4, "do_sample": False,
                              "pad_token_id": judge.tokenizer.eos_token_id, "eos_token_id": judge.tokenizer.eos_token_id},
        "model_generation_config": {k: getattr(gc, k, None) for k in keys},
        "config_judge_max_new_tokens_unused": cfg.get("judge_max_new_tokens"),
        "config_judge_temperature_unused": cfg.get("judge_temperature"),
        "quantization": "4-bit nf4" if getattr(judge.model, "is_loaded_in_4bit", False) else "none",
        "batching": "one unbatched call per pair, as released",
    }
