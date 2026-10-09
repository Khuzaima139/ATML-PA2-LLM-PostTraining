"""Task 2 Step 2, cached part: clip and affected-token fractions on the supplied rollout batch (GPU).

1. Rebuild token IDs from the cached text: chat-template prompt (full length, from the eval pool by
   prompt_id) + tokenizer(response) + EOS only where terminated_with_eos. Rows whose rebuilt length
   differs from response_tokens are excluded and listed.
2. Recompute midpoint log-probs (dropout off) and compare with the cached old log-probs.
   Diagnostic only: for prompts longer than max_prompt_length, also recompute with the prompt
   left-truncated and right-truncated to max_prompt_length.
3. Rule (fixed before looking): token-pooled mean |diff| <= MATCH_THRESHOLD means the cache holds
   midpoint rollouts. Then pi_old = recomputed midpoint and pi_new = midpoint after ONE AdamW step
   (policy_learning_rate, clip max_grad_norm) on all rows (adapter discarded). Otherwise
   pi_old = cached old log-probs and pi_new = recomputed midpoint, no step.
   Advantages: GAE on rewards shaped at beta KL_BETA with the cached old/ref log-probs, cached
   values, effective terminal rewards; no whitening.
4. Per eps: clip fraction, affected fraction, mean clipped and unclipped surrogate; identity checks.
"""
from __future__ import annotations

import argparse
import math

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.train import display_path, peak_vram_bytes, run_metadata
from task2_ppo.ppo import ppo_policy_loss
from task2_ppo.ppo_utils import (
    DROPOUT_RULE,
    build_targets,
    check_clip_identities,
    clip_study,
    disable_dropout,
    pad_ragged,
)

MATCH_THRESHOLD = 0.02  # nats, mean absolute per-token difference
KL_BETA = 0.10


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_tokens(tokenizer, row: dict, messages) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    resp = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    if row["terminated_with_eos"]:
        resp = resp + [tokenizer.eos_token_id]
    return list(prompt_ids), list(resp)


def midpoint_logprobs(policy, prompt_ids: list[int], resp_ids: list[int], with_grad: bool = False) -> torch.Tensor:
    """[1, len(resp_ids)] teacher-forced log-probs, one row per forward (no padding)."""
    device = next(policy.parameters()).device
    seq = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device=device)
    resp = torch.tensor([resp_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(seq)
    with torch.set_grad_enabled(with_grad):
        return response_token_logprobs(policy, seq, attn, len(prompt_ids), resp)[0]


def mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--limit", type=int, help="first N cached rows; smoke tests only")
    ap.add_argument("--name", default="cached_clip_study")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_json = repo_path(f"{cfg['results_dir']}/{args.name}.json")
    if out_json.exists() and not args.overwrite:
        raise SystemExit(f"{out_json} exists; pass --overwrite to replace it.")
    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows = load_cached_rollouts(cfg["cached_rollouts"])
    if args.limit is not None:
        rows = rows[: args.limit]
    eval_rows = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    tokenizer = load_tokenizer(cfg["base_model"])
    max_prompt = int(cfg["max_prompt_length"])
    eps_values = [float(e) for e in cfg["clip_values"]]

    result = {
        "script": "task2_ppo.analyze_clipping",
        "name": args.name,
        "status": "running",
        "config_path": args.config,
        "config": cfg,
        "limit": args.limit,
        "smoke": args.limit is not None,
        **run_metadata(cfg),
        "settings": {
            "cache": cfg["cached_rollouts"],
            "match_threshold_nats": MATCH_THRESHOLD,
            "kl_beta_for_advantages": KL_BETA,
            "gamma": float(cfg["gamma"]),
            "gae_lambda": float(cfg["gae_lambda"]),
            "eps_values": eps_values,
            "dropout": DROPOUT_RULE,
            "prompt": "full chat-template prompt from rl_prompt_pool_eval by prompt_id (no truncation)",
            "whitening": False,
        },
    }

    def write():
        result["wall_clock_seconds"] = elapsed()
        result["peak_vram_bytes"] = peak_vram_bytes()
        save_json(out_json, result)

    # 1. token rebuild
    built, mismatched = [], []
    for i, row in enumerate(rows):
        messages = prompt_messages(eval_rows[row["prompt_id"]])
        p_ids, r_ids = rebuild_tokens(tokenizer, row, messages)
        rec = {"row": i, "prompt_id": row["prompt_id"], "rebuilt_tokens": len(r_ids), "cached_tokens": int(row["response_tokens"]),
               "prompt_tokens": len(p_ids), "terminated_with_eos": bool(row["terminated_with_eos"])}
        if len(r_ids) != int(row["response_tokens"]) or len(row["old_logprobs"]) != len(r_ids):
            mismatched.append(rec)
        else:
            built.append((row, p_ids, r_ids, rec))
    result["token_rebuild"] = {
        "n_rows": len(rows),
        "n_count_match": len(built),
        "n_excluded_mismatch": len(mismatched),
        "mismatched": mismatched,
    }
    print(f"token rebuild: {len(built)}/{len(rows)} rows match the cached token counts", flush=True)
    if not built:
        result["status"] = "failed: no row rebuilt"
        write()
        raise SystemExit(result["status"])

    # 2. midpoint recomputation
    set_seed(int(cfg["seed"]))
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    result["dropout_modules_disabled"] = disable_dropout(policy)
    recomputed, per_row, long_diag = [], [], []
    for row, p_ids, r_ids, rec in built:
        lp = midpoint_logprobs(policy, p_ids, r_ids).float().cpu()[0]
        cached_old = torch.as_tensor(row["old_logprobs"], dtype=torch.float32)
        recomputed.append(lp)
        d = (lp.double() - cached_old.double()).abs()
        per_row.append({**rec, "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max())})
        if len(p_ids) > max_prompt:
            diag = {"row": rec["row"], "prompt_id": rec["prompt_id"], "prompt_tokens": len(p_ids), "full_prompt_mean_abs_diff": float(d.mean())}
            for side, ids in (("left_truncated", p_ids[-max_prompt:]), ("right_truncated", p_ids[:max_prompt])):
                lp_t = midpoint_logprobs(policy, ids, r_ids).float().cpu()[0]
                diag[f"{side}_mean_abs_diff"] = mean_abs(lp_t, cached_old)
            long_diag.append(diag)
    all_diff = torch.cat([(lp.double() - torch.as_tensor(b[0]["old_logprobs"]).double()).abs() for lp, b in zip(recomputed, built)])
    mean_diff, max_diff = float(all_diff.mean()), float(all_diff.max())
    is_midpoint = mean_diff <= MATCH_THRESHOLD
    result["recompute"] = {
        "mean_abs_diff_token_pooled": mean_diff,
        "max_abs_diff": max_diff,
        "n_tokens": int(all_diff.numel()),
        "per_row": per_row,
        "long_prompt_diagnostic": {"note": "diagnostic only; the rule uses the full-prompt recomputation", "rows": long_diag},
    }
    result["rule"] = {
        "threshold_nats": MATCH_THRESHOLD,
        "mean_abs_diff": mean_diff,
        "cache_is_midpoint": is_midpoint,
        "branch": "old=recomputed midpoint, new=midpoint after one AdamW step" if is_midpoint else "old=cached old_logprobs, new=recomputed midpoint, no step",
    }
    write()

    # advantages from the cache (both branches)
    old_c, mask = pad_ragged([r["old_logprobs"].tolist() if torch.is_tensor(r["old_logprobs"]) else list(r["old_logprobs"]) for r, *_ in built])
    ref_c, _ = pad_ragged([r["ref_logprobs"].tolist() if torch.is_tensor(r["ref_logprobs"]) else list(r["ref_logprobs"]) for r, *_ in built])
    val_c, _ = pad_ragged([r["values"].tolist() if torch.is_tensor(r["values"]) else list(r["values"]) for r, *_ in built])
    eff = torch.tensor([float(r["effective_terminal_reward"]) for r, *_ in built], dtype=torch.float64)
    targets = build_targets(eff, old_c, ref_c, val_c, mask, KL_BETA, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = targets["advantages"]
    recomputed_pad, _ = pad_ragged([lp.tolist() for lp in recomputed])

    # 3. old/new log-probs
    if is_midpoint:
        old = recomputed_pad
        n_total = float(mask.sum())
        opt = torch.optim.AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        opt.zero_grad(set_to_none=True)
        loss_total, max_abs_lr = 0.0, 0.0
        device = next(policy.parameters()).device
        for i, (row, p_ids, r_ids, rec) in enumerate(built):
            n_i = len(r_ids)
            new_i = midpoint_logprobs(policy, p_ids, r_ids, with_grad=True)
            old_i = old[i : i + 1, :n_i].to(device=device, dtype=new_i.dtype)
            adv_i = adv[i : i + 1, :n_i].to(device=device, dtype=new_i.dtype)
            m_i = torch.ones_like(new_i)
            # At pi = pi_old the ratio is 1, so the gradient does not depend on eps.
            loss, _, _ = ppo_policy_loss(new_i, old_i, adv_i, m_i, eps=float(cfg["clip_epsilon"]))
            (loss * (n_i / n_total)).backward()
            loss_total += float(loss.detach()) * n_i / n_total
            max_abs_lr = max(max_abs_lr, float((new_i.detach() - old_i).abs().max()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
        if not math.isfinite(grad_norm):
            result["status"] = "failed: non-finite grad norm in the one-step update"
            write()
            raise FloatingPointError(result["status"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        new_rows = [midpoint_logprobs(policy, p_ids, r_ids).float().cpu()[0].tolist() for _, p_ids, r_ids, _ in built]
        new, _ = pad_ragged(new_rows)
        result["one_step"] = {
            "optimizer": "fresh AdamW(lr=policy_learning_rate, default weight_decay=0.01)",
            "lr": float(cfg["policy_learning_rate"]),
            "weight_decay": opt.param_groups[0]["weight_decay"],
            "loss": "task2_ppo.ppo.ppo_policy_loss token mean over all rows (ratio = 1 at the step, eps irrelevant)",
            "loss_value_before_step": loss_total,
            "max_abs_log_ratio_at_step": max_abs_lr,
            "grad_norm_before_clip": grad_norm,
            "max_grad_norm": float(cfg["max_grad_norm"]),
            "adapter": "discarded (not saved)",
        }
    else:
        old, new = old_c, recomputed_pad
        result["one_step"] = None

    # 4. per-eps quantities on the same ratios and mask
    study = clip_study(new, old, adv, mask, eps_values)
    result["eps_results"] = study
    result["n_tokens"] = int(mask.sum())
    result["n_rows_used"] = len(built)
    try:
        result["identity_checks"] = check_clip_identities(study)
    except AssertionError as e:
        result["status"] = f"failed: {e}"
        write()
        raise
    result["status"] = "completed"
    write()
    vram = result["peak_vram_bytes"]
    print(f"identity checks passed; rows={len(built)} tokens={result['n_tokens']} t={elapsed():.0f}s "
          f"peak_vram={vram / 2**30 if vram else 0:.2f}GiB; wrote {display_path(out_json)}", flush=True)


if __name__ == "__main__":
    main()
