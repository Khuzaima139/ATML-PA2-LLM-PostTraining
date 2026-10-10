"""Generation, prompt selection and scoring helpers shared by the evaluation and training scripts.

Rule P (PROMPT_RULE): a prompt is used only if its chat-template encoding (add_generation_prompt=True)
has at most max_prompt_length tokens, so batch_generate never truncates it. Excluded IDs are logged.
"""
from __future__ import annotations

import math

import torch

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import reference_mode

# Fixed for every condition: batch composition changes sampled outputs, so it must match.
GEN_BATCH_SIZE = 16
# Memory only: rows per teacher-forcing forward (full-vocab logits).
TF_BATCH_SIZE = 4

PROMPT_RULE = "use a prompt only if len(apply_chat_template(messages, add_generation_prompt=True)) <= max_prompt_length; longer prompts are excluded, never truncated"

GENERATION_KEYS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "repetition_penalty",
                   "no_repeat_ngram_size", "eos_token_id", "pad_token_id")


# ---------------------------------------------------------------- sampled generation

def generate_responses(model, tokenizer, prompts: list[list[dict]], cfg, with_kl: bool) -> list[dict]:
    """Sample one response per prompt in fixed batches; optionally teacher-force policy and reference.

    Token log-probs are kept only on response-mask tokens (up to and including the first EOS).
    """
    gen = cfg["generation"]
    max_prompt = int(cfg["max_sequence_length"])
    max_new = int(cfg["max_generation_tokens"])
    out = []
    timer, n_batches = wall_timer(), math.ceil(len(prompts) / GEN_BATCH_SIZE)
    for b, start in enumerate(range(0, len(prompts), GEN_BATCH_SIZE), start=1):
        g = batch_generate(
            model, tokenizer, prompts[start : start + GEN_BATCH_SIZE],
            max_prompt_length=max_prompt, max_new_tokens=max_new,
            temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]),
        )
        prompt_lens = g["attention_mask"][:, : g["prompt_width"]].sum(-1).tolist()
        if max(prompt_lens) >= max_prompt:
            raise RuntimeError("a generation prompt reached max_prompt_length and may have been truncated")
        pol_rows, ref_rows = [None] * len(prompt_lens), [None] * len(prompt_lens)
        if with_kl:
            for j in range(0, len(prompt_lens), TF_BATCH_SIZE):
                sl = slice(j, j + TF_BATCH_SIZE)
                args = (g["sequences"][sl], g["attention_mask"][sl], g["prompt_width"], g["response_ids"][sl])
                with torch.no_grad():
                    pol = response_token_logprobs(model, *args)[0]
                with torch.no_grad(), reference_mode(model):
                    ref = response_token_logprobs(model, *args)[0]
                for k in range(pol.shape[0]):
                    pol_rows[j + k], ref_rows[j + k] = pol[k], ref[k]
                del pol, ref
        for k, n in enumerate(g["response_lengths"]):
            rec = {
                "prompt_tokens": int(prompt_lens[k]),
                "text": g["responses"][k],
                "token_length": int(n),
                "terminated_with_eos": bool(g["terminated_with_eos"][k]),
                "truncated": bool(g["truncated"][k]),
            }
            if with_kl:
                # The response mask is a prefix of length n (ones up to the first EOS).
                assert int(g["response_mask"][k].sum().item()) == n and bool(g["response_mask"][k][:n].all())
                rec["policy_token_logp"] = pol_rows[k][:n].double().cpu().tolist()
                rec["ref_token_logp"] = ref_rows[k][:n].double().cpu().tolist()
            out.append(rec)
        del g
        print(f"  generation batch {b}/{n_batches} t={timer():.0f}s", flush=True)
    return out


def effective_generation_settings(model, tokenizer, gen_cfg: dict, max_new_tokens: int) -> dict:
    """Settings model.generate actually uses with common.generation.batch_generate.

    batch_generate passes do_sample, temperature, top_p (when sampling), pad/eos ids and
    max_new_tokens; every other key (top_k, repetition_penalty, ...) comes from the model's own
    generation_config and is recorded here, never overridden.
    """
    gc = model.generation_config
    passed = {"do_sample": bool(gen_cfg["do_sample"]), "max_new_tokens": int(max_new_tokens),
              "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if passed["do_sample"]:
        passed.update({"temperature": float(gen_cfg["temperature"]), "top_p": float(gen_cfg["top_p"])})
    out = {k: passed.get(k, getattr(gc, k, None)) for k in GENERATION_KEYS}
    out["max_new_tokens"] = int(max_new_tokens)
    out["from_model_generation_config"] = sorted(k for k in GENERATION_KEYS if k not in passed)
    return out


# ---------------------------------------------------------------- prompts

def prompt_token_count(tokenizer, messages) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def fitting_prompts(tokenizer, rows: list[dict], max_prompt_length: int, messages_fn):
    """Rule P. Returns (kept_rows, kept_records, report); records carry file index, ID and token count."""
    kept, kept_records, excluded = [], [], []
    for i, row in enumerate(rows):
        n = prompt_token_count(tokenizer, messages_fn(row))
        rec = {"index": i, "prompt_id": row.get("prompt_id", i), "prompt_tokens": n}
        if n > int(max_prompt_length):
            excluded.append(rec)
        else:
            kept.append(row)
            kept_records.append(rec)
    report = {
        "rule": PROMPT_RULE,
        "max_prompt_length": int(max_prompt_length),
        "n_input": len(rows),
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "excluded": excluded,
    }
    return kept, kept_records, report


def prompt_order(n_prompts: int, count: int, seed: int) -> list[int]:
    """First `count` entries of a permutation of range(n_prompts) from its own seeded generator.

    The global RNG is neither read nor advanced, and the order does not depend on eps or beta,
    so every run with the same seed sees the same prompts (a shorter run sees a prefix).
    """
    if count > n_prompts:
        raise ValueError(f"{count} updates need {count} distinct prompts, only {n_prompts} available")
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return torch.randperm(n_prompts, generator=gen)[:count].tolist()


def plan_prompts(cfg: dict, tokenizer, rows: list[dict]) -> dict:
    """Rule P filter and the seeded prompt order. Depends only on seed, updates and the prompt cap."""
    kept, records, report = fitting_prompts(tokenizer, rows, int(cfg["max_prompt_length"]), prompt_messages)
    order = prompt_order(len(kept), int(cfg["updates"]), int(cfg["seed"]))
    return {
        "rows": kept,
        "records": records,
        "filter": report,
        "order": order,
        "prompt_ids": [records[i]["prompt_id"] for i in order],
    }


# ---------------------------------------------------------------- scoring

def pad_ragged(rows: list[list[float]]):
    """Right-pad ragged per-response lists into float64 [n, width] values and mask."""
    width = max((len(r) for r in rows), default=0)
    vals = torch.zeros(len(rows), width, dtype=torch.float64)
    mask = torch.zeros(len(rows), width, dtype=torch.float64)
    for i, r in enumerate(rows):
        vals[i, : len(r)] = torch.tensor(r, dtype=torch.float64)
        mask[i, : len(r)] = 1.0
    return vals, mask


def pooled_token_metrics(policy_logps: list[list[float]], ref_logps: list[list[float]]) -> dict:
    """common.metrics.sampled_kl and sample_entropy pooled over all response tokens (token-level means)."""
    pol, mask = pad_ragged(policy_logps)
    ref, _ = pad_ragged(ref_logps)
    if mask.sum() == 0:
        return {"kl": float("nan"), "entropy": float("nan"), "n_tokens": 0}
    return {
        "kl": float(sampled_kl(pol, ref, mask)),
        "entropy": float(sample_entropy(pol, mask)),
        "n_tokens": int(mask.sum()),
    }


@torch.no_grad()
def reward_with_lengths(rm, rm_tok, prompts, texts, max_length: int, batch_size: int):
    """Course reward-model scores (common.generation.score_reward_pairs) plus RM input token counts.

    An input longer than max_length is truncated by the helper on rm_tok.truncation_side.
    """
    from common.generation import score_reward_pairs

    scores = []
    for start in range(0, len(texts), batch_size):
        scores += score_reward_pairs(rm, rm_tok, prompts[start : start + batch_size], texts[start : start + batch_size], max_length=max_length).cpu().tolist()
    lengths = [
        len(rm_tok.apply_chat_template(list(p) + [{"role": "assistant", "content": t}], tokenize=True, add_generation_prompt=False))
        for p, t in zip(prompts, texts)
    ]
    return scores, lengths
