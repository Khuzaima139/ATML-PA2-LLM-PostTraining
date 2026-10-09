"""Task 2 PPO continuation from the supplied midpoint (manual Section 2, Steps 1 to 3).

One update = one prompt, RESPONSES_PER_PROMPT sampled responses, then ppo_epochs passes with one
policy and one critic optimizer step each (micro-batches of one response, losses weighted so the
accumulated gradient equals the token mean over all responses of the update).

Run-level settings recorded in every run JSON (see DECISIONS):
  - Rule P (task2_ppo.ppo_utils): prompts longer than max_prompt_length are excluded from the sampler.
  - Rule D (task2_ppo.ppo_utils): dropout disabled in policy and critic.
  - Fresh AdamW optimizers (the bundle carries no optimizer state); no scheduler; no whitening.
  - Critic head trained in float32 (LoRA weights are float32 already; frozen base stays float16).
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task1_dpo.dataset_stats import describe
from task1_dpo.train import display_path, dtype_report, peak_vram_bytes, run_metadata
from task2_ppo.ppo import ppo_policy_loss, value_mse_loss
from task2_ppo.ppo_utils import (
    DROPOUT_RULE,
    build_targets,
    critic_head_to_fp32,
    disable_dropout,
    effective_terminal_rewards,
    explained_variance,
    fitting_prompts,
    prompt_order,
    response_slice,
    reward_with_lengths,
    stability_statistic,
)

# Not set by the release config.
RESPONSES_PER_PROMPT = 4

DECISIONS = {
    "responses_per_prompt": f"{RESPONSES_PER_PROMPT} (not set by the release config)",
    "optimizers": "fresh AdamW for policy and critic: the supplied bundle has no optimizer state (deviation)",
    "policy_optimizer": "AdamW(lr=policy_learning_rate, default weight_decay=0.01), no scheduler",
    "critic_optimizer": "AdamW groups from common.models.value_parameter_groups (LoRA value_lora_learning_rate, head value_head_learning_rate), weight_decay=0, no scheduler",
    "loss": "policy: task2_ppo.ppo.ppo_policy_loss; critic: value_coef * task2_ppo.ppo.value_mse_loss; separate optimizers, separate grad clipping at max_grad_norm (norms logged before clipping)",
    "micro_batching": "one response per forward/backward; each micro-batch loss multiplied by its valid-token share so the accumulated gradient is the token mean over the update",
    "advantages": "GAE (task2_ppo.ppo.compute_gae) on KL-shaped rewards (task2_ppo.ppo.shaped_rewards) with old/ref log-probs, terminal reward = RM raw - missing_eos_penalty if no EOS; returns = A + V; no whitening",
    "values": "V(s_t) read at the position predicting response token t (same shift as the log-probs); values on padding zeroed so V after the last valid token is 0",
    "reference": "same policy with the LoRA adapter disabled (common.models.reference_mode) = base_model",
    "dropout": DROPOUT_RULE,
    "critic_dtype": "critic LoRA float32 (PEFT), critic head cast to float32 with a float32 input cast; frozen base float16",
    "log_probs": "teacher-forced, temperature 1 (common.generation.response_token_logprobs); sampling uses the generation settings",
    "kl_and_entropy": "common.metrics.sampled_kl(old, ref) and sample_entropy(old) as token means over all responses of the update",
    "stability_statistic": "after the last optimizer step of the update: token mean of (rho - 1) - log rho, rho = pi_new / pi_old, on that update's tokens",
    "explained_variance": "1 - Var(returns - values) / Var(returns) over valid tokens, values from before the update",
    "non_finite": "an optimizer step is skipped (and counted) when its loss or pre-clip grad norm is not finite",
}


# ---------------------------------------------------------------- planning

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


# ---------------------------------------------------------------- rollout scoring and PPO update

def policy_logprobs(policy, batch: dict, i: int) -> torch.Tensor:
    """[1, R] log-probs of response tokens of row i under the current (adapter-enabled) policy."""
    sl = slice(i, i + 1)
    return response_token_logprobs(policy, batch["sequences"][sl], batch["attention_mask"][sl], batch["prompt_width"], batch["response_ids"][sl])[0]


def critic_values(value_model, batch: dict, i: int) -> torch.Tensor:
    """[1, R] V(s_t) of row i, aligned with policy_logprobs."""
    sl = slice(i, i + 1)
    v = token_values(value_model, batch["sequences"][sl], batch["attention_mask"][sl])
    return response_slice(v, batch["prompt_width"], batch["response_ids"].shape[1]).float()


@torch.no_grad()
def score_rollout(policy, value_model, batch: dict) -> dict:
    """Old (rollout) and reference log-probs and pre-update values, one response per forward.

    The same per-row shapes are used in ppo_update, so with dropout off the epoch-1 ratio is 1.
    """
    old, ref, val = [], [], []
    for i in range(batch["response_ids"].shape[0]):
        old.append(policy_logprobs(policy, batch, i))
        with reference_mode(policy):
            ref.append(policy_logprobs(policy, batch, i))
        val.append(critic_values(value_model, batch, i))
    mask = batch["response_mask"]
    return {"old_logp": torch.cat(old) * mask, "ref_logp": torch.cat(ref) * mask, "values": torch.cat(val) * mask}


def _finite(*xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


def ppo_update(policy, value_model, policy_opt, value_opt, batch: dict, eps: float, value_coef: float, ppo_epochs: int, max_grad_norm: float) -> dict:
    """ppo_epochs passes over the update's responses, one policy and one critic step per pass.

    batch needs sequences, attention_mask, prompt_width, response_ids, response_mask, old_logp,
    advantages, returns. Returns per-epoch losses, clip fractions, pre-clip grad norms, skip flags,
    and the stability statistic after the last step.
    """
    mask = batch["response_mask"]
    n_total = mask.sum()
    p_params = [p for g in policy_opt.param_groups for p in g["params"]]
    v_params = [p for g in value_opt.param_groups for p in g["params"]]
    epochs, n_nonfinite = [], 0
    for epoch in range(1, int(ppo_epochs) + 1):
        policy_opt.zero_grad(set_to_none=True)
        value_opt.zero_grad(set_to_none=True)
        pl = vl = cf = 0.0
        max_abs_log_ratio = 0.0
        for i in range(mask.shape[0]):
            m = mask[i : i + 1]
            w = m.sum() / n_total  # valid-token share of this response
            new = policy_logprobs(policy, batch, i)
            loss, ratio, clip_frac = ppo_policy_loss(new, batch["old_logp"][i : i + 1], batch["advantages"][i : i + 1], m, eps=eps)
            (loss * w).backward()
            v = critic_values(value_model, batch, i)
            v_loss = value_mse_loss(v, batch["returns"][i : i + 1], m)
            (value_coef * v_loss * w).backward()
            pl += float(loss.detach() * w)
            vl += float(v_loss.detach() * w)
            cf += float(clip_frac * w)
            lr_ = (new.detach() - batch["old_logp"][i : i + 1]) * m
            max_abs_log_ratio = max(max_abs_log_ratio, float(lr_.abs().max()))
            del new, loss, ratio, v, v_loss
        p_norm = float(torch.nn.utils.clip_grad_norm_(p_params, max_grad_norm))
        v_norm = float(torch.nn.utils.clip_grad_norm_(v_params, max_grad_norm))
        p_ok, v_ok = _finite(pl, p_norm), _finite(vl, v_norm)
        if p_ok:
            policy_opt.step()
        if v_ok:
            value_opt.step()
        n_nonfinite += (not p_ok) + (not v_ok)
        policy_opt.zero_grad(set_to_none=True)
        value_opt.zero_grad(set_to_none=True)
        if epoch == 1 and cf != 0.0:
            raise AssertionError(f"epoch-1 clip fraction is {cf}, expected 0 (max |log rho| = {max_abs_log_ratio})")
        epochs.append({
            "epoch": epoch,
            "policy_loss": pl,
            "value_loss": vl,
            "value_loss_weighted": float(value_coef) * vl,
            "clip_fraction": cf,
            "max_abs_log_ratio_before_step": max_abs_log_ratio,
            "policy_grad_norm_before_clip": p_norm,
            "value_grad_norm_before_clip": v_norm,
            "policy_step_taken": p_ok,
            "value_step_taken": v_ok,
        })
    with torch.no_grad():
        new_all = torch.cat([policy_logprobs(policy, batch, i) for i in range(mask.shape[0])]) * mask
    return {
        "epochs": epochs,
        "n_nonfinite_steps": n_nonfinite,
        "stability_statistic": stability_statistic(new_all, batch["old_logp"], mask),
    }


# ---------------------------------------------------------------- setup

def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # Rule D and the float32 critic head, before the optimizers are built.
    n_dropout = {"policy": disable_dropout(policy), "critic": disable_dropout(value_model)}
    head = critic_head_to_fp32(value_model)

    policy_optimizer = torch.optim.AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = torch.optim.AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
        "dropout_modules_disabled": n_dropout,
        "critic_head": {k: v for k, v in head.items() if k != "hook"},
    }


def reference_check(cfg: dict) -> dict:
    """The midpoint adapter must sit on base_model, so reference_mode gives the intended reference."""
    from common.logging_utils import load_json

    meta = load_json(f"{cfg['paths']['ppo_midpoint_policy']}/adapter_config.json")
    ok = meta.get("base_model_name_or_path") == cfg["base_model"]
    if not ok:
        raise RuntimeError(f"midpoint adapter base {meta.get('base_model_name_or_path')!r} != base_model {cfg['base_model']!r}")
    return {"adapter_base_model": meta.get("base_model_name_or_path"), "config_base_model": cfg["base_model"], "match": ok}


def optimizer_report(opt) -> list[dict]:
    return [
        {"name": g.get("name"), "lr": g["lr"], "weight_decay": g["weight_decay"], "n_tensors": len(g["params"]),
         "n_params": sum(p.numel() for p in g["params"]), "dtypes": sorted({str(p.dtype) for p in g["params"]})}
        for g in opt.param_groups
    ]


# ---------------------------------------------------------------- run

def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", overwrite: bool = False):
    cfg0 = load_yaml(config_path)
    out_json = repo_path(f"{cfg0['results_dir']}/train_{run_name}.json")
    rollouts_path = repo_path(f"{cfg0['results_dir']}/rollouts_{run_name}.jsonl")
    out = repo_path(output or f"outputs/task2_ppo/{run_name}")
    for p in (out_json, rollouts_path, out):
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")

    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

    tokenizer, policy, value_model = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    policy_opt, value_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    policy.generation_config.use_cache = True  # load_policy sets config.use_cache=False for training
    plan = plan_prompts(cfg, tokenizer, bundle["prompt_rows"])
    device = next(policy.parameters()).device

    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gen_cfg = cfg["generation"]
    max_prompt, max_resp = int(cfg["max_prompt_length"]), int(cfg["max_response_length"])
    rm_cap = int(cfg["reward_max_length"])
    dtypes = {"policy": dtype_report(policy), "critic": dtype_report(value_model)}

    result = {
        "script": "task2_ppo.continue_train",
        "run_name": run_name,
        "status": "running",
        "config_path": config_path,
        "config": cfg,
        "cli": {"output": output, "updates": updates, "clip_epsilon": clip_epsilon, "kl_beta": kl_beta},
        "effective": {
            "updates": int(cfg["updates"]),
            "clip_epsilon": eps,
            "kl_beta": beta,
            "responses_per_prompt": RESPONSES_PER_PROMPT,
            "ppo_epochs": int(cfg["ppo_epochs"]),
            "gamma": float(cfg["gamma"]),
            "gae_lambda": float(cfg["gae_lambda"]),
            "value_coef": float(cfg["value_coef"]),
            "missing_eos_penalty": float(cfg["missing_eos_penalty"]),
            "max_prompt_length": max_prompt,
            "max_response_length": max_resp,
            "reward_max_length": rm_cap,
            "reward_truncation_side": rm_tok.truncation_side,
            "max_grad_norm": float(cfg["max_grad_norm"]),
            "generation": dict(gen_cfg),
            "model_generation_config": json.loads(policy.generation_config.to_json_string()),
        },
        "decisions": DECISIONS,
        **run_metadata(cfg),
        "dtype_report": dtypes,
        "critic_head": bundle["critic_head"],
        "dropout_modules_disabled": bundle["dropout_modules_disabled"],
        "optimizers": {"policy": optimizer_report(policy_opt), "critic": optimizer_report(value_opt)},
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
        "generated_tokens_total": 0,
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
    for name, rep in dtypes.items():
        if "torch.float16" in rep["trainable_param_dtypes"]:
            result["status"] = f"stopped: trainable {name} parameters are float16"
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
            prompts = [messages] * RESPONSES_PER_PROMPT

            g = batch_generate(policy, tokenizer, prompts, max_prompt_length=max_prompt, max_new_tokens=max_resp,
                               temperature=float(gen_cfg["temperature"]), top_p=float(gen_cfg["top_p"]), do_sample=bool(gen_cfg["do_sample"]))
            n_prompt = g["attention_mask"][:, : g["prompt_width"]].sum(-1)
            if g["prompt_width"] != rec["prompt_tokens"] or bool((n_prompt != rec["prompt_tokens"]).any()):
                raise RuntimeError(f"prompt {rec['prompt_id']} encoded to {n_prompt.tolist()} tokens, expected {rec['prompt_tokens']}")
            t_gen = t_upd()

            raw_list, rm_lengths = reward_with_lengths(rm, rm_tok, prompts, g["responses"], rm_cap, RESPONSES_PER_PROMPT)
            raw = torch.tensor(raw_list, dtype=torch.float32, device=device)
            # batch_generate builds these under torch.inference_mode; clones are ordinary tensors
            # that autograd accepts in the training forwards.
            mask = g["response_mask"].float().clone()
            batch = {
                "sequences": g["sequences"].clone(),
                "attention_mask": g["attention_mask"].clone(),
                "prompt_width": g["prompt_width"],
                "response_ids": g["response_ids"].clone(),
                "response_mask": mask,
            }
            batch.update(score_rollout(policy, value_model, batch))
            eff = effective_terminal_rewards(raw, g["terminated_with_eos"], float(cfg["missing_eos_penalty"]))
            targets = build_targets(eff, batch["old_logp"], batch["ref_logp"], batch["values"], mask, beta, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            batch["advantages"], batch["returns"] = targets["advantages"], targets["returns"]
            ev = explained_variance(targets["returns"], batch["values"], mask)
            kl = float(sampled_kl(batch["old_logp"], batch["ref_logp"], mask))
            ent = float(sample_entropy(batch["old_logp"], mask))
            t_score = t_upd() - t_gen

            upd = ppo_update(policy, value_model, policy_opt, value_opt, batch, eps, float(cfg["value_coef"]), int(cfg["ppo_epochs"]), float(cfg["max_grad_norm"]))
            t_opt = t_upd() - t_gen - t_score

            lengths = [int(x) for x in g["response_lengths"]]
            eff_list = eff.cpu().tolist()
            n_gen = int(sum(lengths))
            result["generated_tokens_total"] += n_gen
            result["n_nonfinite_steps"] += upd["n_nonfinite_steps"]
            finite_inputs = bool(torch.isfinite(raw).all() and torch.isfinite(batch["old_logp"]).all()
                                 and torch.isfinite(batch["ref_logp"]).all() and torch.isfinite(batch["values"]).all())
            hist = {
                "update": u,
                "prompt_id": rec["prompt_id"],
                "prompt_file_index": rec["index"],
                "prompt_tokens": rec["prompt_tokens"],
                "reward_raw": {"values": raw_list, "mean": describe(raw_list)["mean"], "sd": describe(raw_list)["std"]},
                "reward_effective": {"values": eff_list, "mean": describe(eff_list)["mean"], "sd": describe(eff_list)["std"]},
                "kl_sampled": kl,
                "entropy_sampled": ent,
                "explained_variance": ev,
                "advantage_token_mean": float((targets["advantages"] * mask).sum() / mask.sum()),
                "epochs": upd["epochs"],
                "stability_statistic": upd["stability_statistic"],
                "response_length": {"values": lengths, "mean": describe(lengths)["mean"], "sd": describe(lengths)["std"]},
                "n_eos": int(sum(g["terminated_with_eos"])),
                "n_truncated": int(sum(g["truncated"])),
                "generated_tokens": n_gen,
                "generated_tokens_cumulative": result["generated_tokens_total"],
                "rm_input_tokens": rm_lengths,
                "n_rm_input_truncated": int(sum(n > rm_cap for n in rm_lengths)),
                "inputs_finite": finite_inputs,
                "n_nonfinite_steps": upd["n_nonfinite_steps"],
                "seconds": {"generation": t_gen, "scoring": t_score, "optimization": t_opt, "total": t_upd()},
                "peak_vram_bytes": peak_vram_bytes(),
                "elapsed_seconds": elapsed(),
            }
            result["history"].append(hist)
            result["updates_completed"] = u
            for j, text in enumerate(g["responses"]):
                rollout_rows.append({
                    "update": u, "prompt_id": rec["prompt_id"], "sample_index": j, "text": text,
                    "token_length": lengths[j], "terminated_with_eos": bool(g["terminated_with_eos"][j]),
                    "truncated": bool(g["truncated"][j]), "rm_raw": raw_list[j], "rm_effective": eff_list[j],
                })
            write()
            # Console: timing, memory, finiteness and EOS counts only (metrics stay in the JSON).
            vram = hist["peak_vram_bytes"]
            print(
                f"[{run_name}] update {u}/{len(plan['order'])} eos={hist['n_eos']}/{RESPONSES_PER_PROMPT} "
                f"truncated={hist['n_truncated']} gen_tokens={n_gen} finite_inputs={finite_inputs} "
                f"nonfinite_steps={upd['n_nonfinite_steps']} t_update={hist['seconds']['total']:.1f}s "
                f"(gen {t_gen:.1f}s) peak_vram={vram / 2**30 if vram else 0:.2f}GiB elapsed={elapsed():.0f}s",
                flush=True,
            )
            del g, batch, targets
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise

    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out / "policy"))
    value_model.save_pretrained(str(out / "value"))
    result["policy_adapter_path"] = display_path(out / "policy")
    result["critic_adapter_path"] = display_path(out / "value")
    result["status"] = "completed"
    write()
    print(f"[{run_name}] completed; saved {out / 'policy'} and {out / 'value'}; wrote {display_path(out_json)}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.overwrite)


if __name__ == "__main__":
    main()
