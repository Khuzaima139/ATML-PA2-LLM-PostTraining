"""Task 3 GRPO continuation from the supplied midpoint (manual Section 3, Steps 1 and 3).

One update = one prompt from the fixed seeded order, K = num_generations completions sampled in eval
mode, course reward-model scores, group-relative advantages (task3_grpo.grpo.group_relative_advantages),
truncated completions masked from the loss (task3_grpo.grpo.mask_truncated_sequences) but kept in the
group mean/std, one training forward of the K completions with LoRA dropout on, loss =
task3_grpo.grpo.grpo_policy_loss with old log-probs = that forward's log-probs detached
(policy_epochs = 1, so every ratio is exactly 1 and the clip fraction is 0), grad-norm clipping, one
AdamW step.

--length-diagnostic (forks only): before the real backward, each unmasked completion's surrogate term
alone is backpropagated through the same forward (retain_graph); its gradient norm over the trainable
parameters is recorded and the gradients are discarded. Two more backward passes through the same
forward give ||grad of the surrogate term|| (all unmasked completions) and ||grad of the beta * KL term||.
The real step is unchanged (tested).

Run-level settings recorded in every run JSON (see DECISIONS).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dataset_stats import describe
from task1_dpo.train import display_path, dtype_report, peak_vram_bytes, run_metadata
from task2_ppo.continue_train import optimizer_report, plan_prompts
from task2_ppo.ppo_utils import reward_with_lengths
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences
from task3_grpo.grpo_utils import (
    INFORMATIVE_RULE,
    effective_generation_settings,
    group_population_std,
    is_informative,
    zero_gradient_token_shares,
)

# Release default of common.generation.score_reward_pairs (configs/grpo.yaml sets no reward cap).
TRAIN_RM_MAX_LENGTH = 1024

DECISIONS = {
    "midpoint": "checkpoints/grpo_midpoint_policy (paths.grpo_midpoint_policy) (A)",
    "reference": "same policy with the LoRA adapter disabled (common.models.reference_mode) = base_model (A)",
    "prompts": "rule P (task2_ppo.ppo_utils.fitting_prompts): prompts over max_prompt_length chat-template tokens excluded, never truncated; order = task2_ppo.ppo_utils.prompt_order(seed) (B)",
    "dropout": "LoRA dropout 0.05 active in the training forward (train mode); generation, reference and metric forwards in eval mode (C)",
    "old_log_probs": "the training forward's own log-probs .detach(); policy_epochs must be 1, so ratio = 1 and clip fraction = 0 by construction (C)",
    "advantages": "task3_grpo.grpo.group_relative_advantages over the K completions of the prompt (default eps 1e-6) (D)",
    "masking": "completions that hit max_completion_length without EOS are masked by task3_grpo.grpo.mask_truncated_sequences; they stay in the group mean/std and in K (D)",
    "informative": INFORMATIVE_RULE,
    "loss": "task3_grpo.grpo.grpo_policy_loss as released (loss_type grpo: 1/T_k; dr_grpo: 1/max_completion_length) (K)",
    "optimizer": "fresh AdamW(trainable LoRA params, lr=learning_rate, torch defaults incl. weight_decay=0.01) as built by prepare_grpo_continuation; the midpoint has no optimizer state; no scheduler",
    "grad_clipping": "clip_grad_norm_ at max_grad_norm over the trainable parameters; the norm is logged before clipping",
    "reward_cap": f"training RM inputs truncated at {TRAIN_RM_MAX_LENGTH} tokens (score_reward_pairs default); never reached with prompt <= 256 and completion <= 512",
    "metrics": "kl_sampled = common.metrics.sampled_kl(eval-mode policy, reference) and entropy_sampled = common.metrics.sample_entropy(eval-mode policy), token means over all K rollouts (masked ones included); kl_k3 = the estimator inside grpo_policy_loss (train-mode policy, truncation-masked tokens)",
    "zero_gradient_tokens": "share of generated tokens with zero surrogate gradient: masked truncation first, then all remaining tokens of an uninformative group",
    "length_diagnostic": "per unmasked completion k: grpo_policy_loss on row k alone with beta = 0 and reference = old (so the KL term and its gradient are exactly 0), backward(retain_graph=True) through the shared forward, ||g_k|| = L2 norm over all trainable parameters; gradients then discarded. Analytic per-token weight |A_k|/T_k (grpo) or |A_k|/max_completion_length (dr_grpo)",
    "term_gradients": "per update (with --length-diagnostic): grpo_policy_loss on all K rows with beta = 0 and reference = old gives the surrogate term alone; with all advantages set to 0 and the run's beta and reference it gives the beta * KL (k3) term alone (the surrogate is then exactly 0). Each is backpropagated through the shared forward (retain_graph) and its L2 grad norm over all trainable parameters recorded; gradients then discarded",
    "non_finite": "the optimizer step is skipped (and counted) when the loss or the pre-clip grad norm is not finite",
}


# ---------------------------------------------------------------- forwards

@torch.no_grad()
def eval_logprobs(policy, batch: dict, reference: bool) -> torch.Tensor:
    """[K, R] response-token log-probs in eval mode (dropout off), one row per forward, zero on padding.

    reference=True disables the LoRA adapter (common.models.reference_mode).
    """
    was_training = policy.training
    policy.eval()
    rows = []
    try:
        for i in range(batch["response_ids"].shape[0]):
            sl = slice(i, i + 1)
            args = (batch["sequences"][sl], batch["attention_mask"][sl], batch["prompt_width"], batch["response_ids"][sl])
            if reference:
                with reference_mode(policy):
                    rows.append(response_token_logprobs(policy, *args)[0])
            else:
                rows.append(response_token_logprobs(policy, *args)[0])
    finally:
        if was_training:
            policy.train()
    return torch.cat(rows) * batch["response_mask"]


def _grad_norm(params) -> float:
    sq = sum(float(p.grad.detach().double().pow(2).sum()) for p in params if p.grad is not None)
    return math.sqrt(sq)


def length_diagnostic(new_logp, advantages, token_mask, response_mask, eps, loss_type, max_completion_length, params) -> list[dict]:
    """Per-completion surrogate gradient norms through the shared forward; leaves all grads at None.

    Uses no RNG: the backward passes only recompute checkpointed activations, and torch's
    checkpointing restores the RNG state it saved in the forward.
    """
    rows = []
    lengths = response_mask.sum(-1)
    for k in range(new_logp.shape[0]):
        if float(token_mask[k].sum()) == 0.0:
            continue  # masked truncated completion: no surrogate term
        for p in params:
            p.grad = None
        row = new_logp[k : k + 1]
        held = row.detach()
        loss_k, _ = grpo_policy_loss(row, held, advantages[k : k + 1], token_mask[k : k + 1], held, eps, 0.0,
                                     loss_type=loss_type, max_completion_length=max_completion_length)
        loss_k.backward(retain_graph=True)
        t_k, a_k = int(lengths[k]), abs(float(advantages[k]))
        denom = t_k if loss_type == "grpo" else int(max_completion_length)
        rows.append({
            "sample_index": k,
            "T_k": t_k,
            "abs_advantage": a_k,
            "grad_norm": _grad_norm(params),
            "analytic_token_weight": a_k / denom,
        })
    for p in params:
        p.grad = None
    return rows


def term_gradient_norms(new_logp, advantages, token_mask, ref_logp, eps, beta, loss_type, max_completion_length, params) -> dict:
    """||grad surrogate term|| and ||grad beta * KL term|| through the shared forward; leaves all grads at None.

    Both terms come from the released grpo_policy_loss: beta = 0 with reference = old isolates the
    surrogate; zero advantages isolate beta * KL, since min(rho * 0, clip(rho) * 0) = 0 for every token.
    """
    held = new_logp.detach()
    terms = {
        "surrogate": grpo_policy_loss(new_logp, held, advantages, token_mask, held, eps, 0.0,
                                      loss_type=loss_type, max_completion_length=max_completion_length)[0],
        "beta_kl": grpo_policy_loss(new_logp, held, torch.zeros_like(advantages), token_mask, ref_logp, eps, beta,
                                    loss_type=loss_type, max_completion_length=max_completion_length)[0],
    }
    out = {}
    for name, term in terms.items():
        for p in params:
            p.grad = None
        term.backward(retain_graph=True)
        out[f"grad_norm_{name}"] = _grad_norm(params)
        out[f"{name}_value"] = float(term.detach())
    for p in params:
        p.grad = None
    return out


def grpo_update(policy, optimizer, batch: dict, eps: float, beta: float, loss_type: str, max_completion_length: int,
                max_grad_norm: float, length_diag: bool = False) -> dict:
    """One training forward of the K completions, the released GRPO loss, one clipped AdamW step.

    batch: sequences, attention_mask, prompt_width, response_ids, response_mask [K, R], truncated (K bools),
    advantages [K], ref_logp [K, R]. The policy must be in train mode (dropout on).
    """
    if not policy.training:
        raise RuntimeError("grpo_update needs the policy in train mode")
    params = [p for g in optimizer.param_groups for p in g["params"]]
    optimizer.zero_grad(set_to_none=True)
    new = response_token_logprobs(policy, batch["sequences"], batch["attention_mask"], batch["prompt_width"], batch["response_ids"])[0]
    old = new.detach()  # policy_epochs = 1: old policy = the policy of this very forward
    token_mask = mask_truncated_sequences(batch["response_mask"], batch["truncated"])
    loss, stats = grpo_policy_loss(new, old, batch["advantages"], token_mask, batch["ref_logp"], eps, beta,
                                   loss_type=loss_type, max_completion_length=max_completion_length)
    if float(stats["clip_fraction"]) != 0.0:
        raise AssertionError(f"clip fraction {float(stats['clip_fraction'])} with old = new.detach(); expected 0")
    diag = term_grads = None
    if length_diag:
        diag = length_diagnostic(new, batch["advantages"], token_mask, batch["response_mask"], eps, loss_type, max_completion_length, params)
        term_grads = term_gradient_norms(new, batch["advantages"], token_mask, batch["ref_logp"], eps, beta, loss_type,
                                         max_completion_length, params)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
    ok = math.isfinite(float(loss.detach())) and math.isfinite(norm)
    if ok:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    surrogate, kl3 = float(stats["policy_term"]), float(stats["sampled_kl"])
    return {
        "policy_loss": float(loss.detach()),
        "surrogate_term": surrogate,
        "beta_kl_term": float(beta) * kl3,
        "kl_k3": kl3,
        "clip_fraction": float(stats["clip_fraction"]),
        "ratio_mean": float(stats["ratio_mean"]),
        "grad_norm_before_clip": norm,
        "step_taken": ok,
        "n_masked": int(sum(bool(t) for t in batch["truncated"])),
        "length_diagnostic": diag,
        "term_gradients": term_grads,
    }


# ---------------------------------------------------------------- setup

def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def reference_check(cfg: dict) -> dict:
    """The midpoint adapter must sit on base_model, so reference_mode gives the intended reference."""
    meta = load_json(f"{cfg['paths']['grpo_midpoint_policy']}/adapter_config.json")
    ok = meta.get("base_model_name_or_path") == cfg["base_model"]
    if not ok:
        raise RuntimeError(f"midpoint adapter base {meta.get('base_model_name_or_path')!r} != base_model {cfg['base_model']!r}")
    return {"adapter_base_model": meta.get("base_model_name_or_path"), "config_base_model": cfg["base_model"], "match": ok}


def check_release_settings(cfg: dict):
    if int(cfg["policy_epochs"]) != 1:
        raise SystemExit(f"policy_epochs={cfg['policy_epochs']}: old log-probs = new.detach() needs exactly 1")
    if int(cfg["prompts_per_update"]) != 1:
        raise SystemExit(f"prompts_per_update={cfg['prompts_per_update']}: this loop handles one prompt per update")
    if cfg["mask_truncated_completions"] is not True:
        raise SystemExit("mask_truncated_completions must be true")


def token_sha256(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------- run

def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", length_diag: bool = False, overwrite: bool = False):
    cfg0 = load_yaml(config_path)
    check_release_settings(cfg0)
    out_json = repo_path(f"{cfg0['results_dir']}/train_{run_name}.json")
    rollouts_path = repo_path(f"{cfg0['results_dir']}/rollouts_{run_name}.jsonl")
    out = repo_path(output or f"outputs/task3_grpo/{run_name}")
    for p in (out_json, rollouts_path, out):
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")

    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    tokenizer, policy, optimizer = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    policy.generation_config.use_cache = True  # load_policy sets config.use_cache=False for training
    plan = plan_prompts(cfg, tokenizer, bundle["prompt_rows"])
    device = next(policy.parameters()).device

    K = int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gen_cfg = cfg["generation"]
    max_prompt, max_completion = int(cfg["max_prompt_length"]), int(cfg["max_completion_length"])
    max_grad_norm = float(cfg["max_grad_norm"])
    dtypes = dtype_report(policy)

    result = {
        "script": "task3_grpo.continue_train",
        "run_name": run_name,
        "status": "running",
        "config_path": config_path,
        "config": cfg,
        "cli": {"output": output, "updates": updates, "loss_type": loss_type, "length_diagnostic": length_diag},
        "effective": {
            "updates": int(cfg["updates"]),
            "loss_type": loss_type,
            "num_generations": K,
            "policy_epochs": int(cfg["policy_epochs"]),
            "clip_epsilon": eps,
            "kl_beta": beta,
            "learning_rate": float(cfg["learning_rate"]),
            "max_prompt_length": max_prompt,
            "max_completion_length": max_completion,
            "mask_truncated_completions": bool(cfg["mask_truncated_completions"]),
            "reward_max_length": TRAIN_RM_MAX_LENGTH,
            "reward_truncation_side": rm_tok.truncation_side,
            "max_grad_norm": max_grad_norm,
            "length_diagnostic": length_diag,
            "generation": effective_generation_settings(policy, tokenizer, gen_cfg, max_completion),
            "model_generation_config": json.loads(policy.generation_config.to_json_string()),
        },
        "decisions": DECISIONS,
        **run_metadata(cfg),
        "dtype_report": dtypes,
        "optimizer": optimizer_report(optimizer),
        "reference": reference_check(cfg),
        "prompts": {
            "path": cfg["paths"]["rl_prompt_train"],
            "filter": plan["filter"],
            "sampler": "torch.randperm over the kept prompts (file order) with torch.Generator(seed); first `updates` entries",
            "order_kept_index": plan["order"],
            "prompt_ids": plan["prompt_ids"],
        },
        "adapter_path": display_path(out),
        "rollouts_file": display_path(rollouts_path),
        "setup_seconds": None,
        "wall_clock_seconds": 0.0,
        "peak_vram_bytes": None,
        "n_nonfinite_steps": 0,
        "n_rm_input_truncated": 0,
        "generated_tokens_total": 0,
        "updates_completed": 0,
        "history": [],
    }
    rollout_rows: list[dict] = []

    def write():
        result["wall_clock_seconds"] = elapsed()
        cur = peak_vram_bytes()  # peak since the last reset (reset at every update)
        if cur is not None:
            result["peak_vram_bytes"] = max(result["peak_vram_bytes"] or 0, cur)
        save_json(out_json, result)
        write_jsonl(rollouts_path, rollout_rows)

    print("dtype report:", dtypes, flush=True)
    if "torch.float16" in dtypes["trainable_param_dtypes"]:
        result["status"] = "stopped: trainable policy parameters are float16"
        write()
        raise SystemExit(result["status"])
    result["setup_seconds"] = elapsed()
    write()

    try:
        for u, k in enumerate(plan["order"], start=1):
            t_upd = wall_timer()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            row, rec = plan["rows"][k], plan["records"][k]
            messages = prompt_messages(row)
            prompts = [messages] * K

            g = batch_generate(policy, tokenizer, prompts, max_prompt_length=max_prompt, max_new_tokens=max_completion,
                               temperature=float(gen_cfg["temperature"]), top_p=float(gen_cfg["top_p"]), do_sample=bool(gen_cfg["do_sample"]))
            n_prompt = g["attention_mask"][:, : g["prompt_width"]].sum(-1)
            if g["prompt_width"] != rec["prompt_tokens"] or bool((n_prompt != rec["prompt_tokens"]).any()):
                raise RuntimeError(f"prompt {rec['prompt_id']} encoded to {n_prompt.tolist()} tokens, expected {rec['prompt_tokens']}")
            t_gen = t_upd()

            rewards_list, rm_lengths = reward_with_lengths(rm, rm_tok, prompts, g["responses"], TRAIN_RM_MAX_LENGTH, K)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            advantages = group_relative_advantages(rewards, torch.zeros(K, dtype=torch.long, device=device))
            informative = is_informative(rewards)
            # batch_generate builds these under torch.inference_mode; clones are ordinary tensors
            # that autograd accepts in the training forward.
            mask = g["response_mask"].float().clone()
            batch = {
                "sequences": g["sequences"].clone(),
                "attention_mask": g["attention_mask"].clone(),
                "prompt_width": g["prompt_width"],
                "response_ids": g["response_ids"].clone(),
                "response_mask": mask,
                "truncated": [bool(t) for t in g["truncated"]],
                "advantages": advantages,
            }
            pol_eval = eval_logprobs(policy, batch, reference=False)
            batch["ref_logp"] = eval_logprobs(policy, batch, reference=True)
            kl = float(sampled_kl(pol_eval, batch["ref_logp"], mask))
            ent = float(sample_entropy(pol_eval, mask))
            t_score = t_upd() - t_gen

            upd = grpo_update(policy, optimizer, batch, eps, beta, loss_type, max_completion, max_grad_norm, length_diag)
            t_opt = t_upd() - t_gen - t_score

            lengths = [int(x) for x in g["response_lengths"]]
            n_gen = int(sum(lengths))
            n_rm_trunc = int(sum(n > TRAIN_RM_MAX_LENGTH for n in rm_lengths))
            result["generated_tokens_total"] += n_gen
            result["n_nonfinite_steps"] += int(not upd["step_taken"])
            result["n_rm_input_truncated"] += n_rm_trunc
            finite_inputs = bool(torch.isfinite(rewards).all() and torch.isfinite(pol_eval).all() and torch.isfinite(batch["ref_logp"]).all())
            adv_list = advantages.cpu().tolist()
            diag = upd.pop("length_diagnostic")
            term_grads = upd.pop("term_gradients")
            hist = {
                "update": u,
                "prompt_id": rec["prompt_id"],
                "prompt_file_index": rec["index"],
                "prompt_tokens": rec["prompt_tokens"],
                "rewards": rewards_list,
                "group_reward_mean": float(rewards.mean()),
                "group_reward_std": float(group_population_std(rewards)),
                "informative": informative,
                "advantages": adv_list,
                **upd,
                "kl_sampled": kl,
                "entropy_sampled": ent,
                "completion_length": {"values": lengths, "mean": describe(lengths)["mean"], "sd": describe(lengths)["std"]},
                "mean_T_over_max_completion": float(np.mean([n / max_completion for n in lengths])),
                "n_eos": int(sum(g["terminated_with_eos"])),
                "n_truncated": int(sum(g["truncated"])),
                "generated_tokens": n_gen,
                "generated_tokens_cumulative": result["generated_tokens_total"],
                "zero_gradient_tokens": zero_gradient_token_shares(lengths, batch["truncated"], informative),
                "rm_input_tokens": rm_lengths,
                "n_rm_input_truncated": n_rm_trunc,
                "inputs_finite": finite_inputs,
                "seconds": {"generation": t_gen, "scoring": t_score, "optimization": t_opt, "total": t_upd()},
                "peak_vram_bytes": peak_vram_bytes(),
                "elapsed_seconds": elapsed(),
            }
            if diag is not None:
                hist["length_diagnostic"] = diag
                hist["term_gradients"] = term_grads
            result["history"].append(hist)
            result["updates_completed"] = u
            for j, text in enumerate(g["responses"]):
                rollout_rows.append({
                    "update": u, "prompt_id": rec["prompt_id"], "sample_index": j, "text": text,
                    "token_length": lengths[j], "terminated_with_eos": bool(g["terminated_with_eos"][j]),
                    "truncated": bool(g["truncated"][j]), "masked": bool(batch["truncated"][j]),
                    "reward": rewards_list[j], "advantage": adv_list[j],
                    "response_ids_sha256": token_sha256(g["response_ids"][j][: lengths[j]].cpu().tolist()),
                })
            write()
            # Console: timing, memory, finiteness and counts only (metrics stay in the JSON).
            vram = hist["peak_vram_bytes"]
            print(
                f"[{run_name}] update {u}/{len(plan['order'])} eos={hist['n_eos']}/{K} masked={upd['n_masked']} "
                f"gen_tokens={n_gen} finite_inputs={finite_inputs} step_taken={upd['step_taken']} "
                f"diag_rows={len(diag) if diag is not None else 0} t_update={hist['seconds']['total']:.1f}s "
                f"(gen {t_gen:.1f}s, opt {t_opt:.1f}s) peak_vram={vram / 2**30 if vram else 0:.2f}GiB elapsed={elapsed():.0f}s",
                flush=True,
            )
            del g, batch, pol_eval
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise

    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out / "policy"))
    result["policy_adapter_path"] = display_path(out / "policy")
    result["status"] = "completed"
    write()
    print(f"[{run_name}] completed; saved {out / 'policy'}; wrote {display_path(out_json)}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--length-diagnostic", action="store_true", help="per-completion surrogate gradient norms (forks)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.length_diagnostic, args.overwrite)


if __name__ == "__main__":
    main()
