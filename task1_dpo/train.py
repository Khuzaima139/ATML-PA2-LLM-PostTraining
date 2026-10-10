"""Task 1 DPO training loop (manual Section 1, Step 1; reused by the Step 2 and Step 3 runs).

Run-level rules recorded in every run JSON:
  A. A pair whose chat-template prompt alone has >= max_sequence_length tokens is skipped,
     never truncated (the same test encode_prompt_response raises on). Counts and IDs logged.
  B. Responses that do not fit keep the starter's truncate-then-append-EOS encoding; counted.
"""
from __future__ import annotations

import argparse
import hashlib
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.run_info import display_path, dtype_report, peak_vram_bytes, run_metadata
from task1_dpo.dpo import dpo_loss

STRATUM_KEY = "length_stratum"
RULE_A = "skip pair if len(apply_chat_template(prompt, add_generation_prompt=True)) >= max_sequence_length; never truncate the prompt"
RULE_B = "response longer than its budget is cut to budget-1 tokens and EOS is appended (starter encode_prompt_response)"


# ---------------------------------------------------------------- rule A filter

def row_id(row: dict, index: int):
    """The row's own ID field if present, else its index in the file."""
    return row.get("prompt_id", index)


def filter_pairs(tokenizer, rows: list[dict], max_length: int):
    """Apply rule A to preference rows and flag rule-B response truncation.

    Returns (kept_rows, kept_records, skipped_records). Records carry the file index, ID,
    prompt token count, stratum (if any) and whether chosen/rejected will be truncated.
    """
    eos = 1 if tokenizer.eos_token_id is not None else 0
    kept, kept_records, skipped = [], [], []
    for i, row in enumerate(rows):
        messages = prompt_messages_from_preference(row)
        n_prompt = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        rec = {"index": i, "id": row_id(row, i), "prompt_tokens": n_prompt}
        if STRATUM_KEY in row:
            rec["stratum"] = row[STRATUM_KEY]
        if n_prompt >= max_length:
            skipped.append(rec)
            continue
        # Same budget arithmetic as encode_prompt_response.
        content_budget = max(0, max_length - n_prompt - eos)
        yc, yr = preference_responses(row)
        rec["chosen_truncated"] = len(tokenizer(yc, add_special_tokens=False)["input_ids"]) > content_budget
        rec["rejected_truncated"] = len(tokenizer(yr, add_special_tokens=False)["input_ids"]) > content_budget
        kept.append(row)
        kept_records.append(rec)
    return kept, kept_records, skipped


def _counts(kept: list[dict], skipped: list[dict]) -> dict:
    trunc = [r for r in kept if r["chosen_truncated"] or r["rejected_truncated"]]
    return {
        "n_input": len(kept) + len(skipped),
        "n_retained": len(kept),
        "n_skipped_prompt_too_long": len(skipped),
        "n_pairs_any_response_truncated": len(trunc),
        "n_chosen_truncated": sum(r["chosen_truncated"] for r in kept),
        "n_rejected_truncated": sum(r["rejected_truncated"] for r in kept),
    }


def filter_report(path: str, max_length: int, kept: list[dict], skipped: list[dict]) -> dict:
    """Counts and IDs for rule A (skips) and rule B (truncations), overall and per stratum."""
    out = {
        "path": path,
        "rule_A": RULE_A,
        "rule_B": RULE_B,
        "max_sequence_length": max_length,
        **_counts(kept, skipped),
        "skipped": skipped,
        "truncated_ids": [r["id"] for r in kept if r["chosen_truncated"] or r["rejected_truncated"]],
    }
    strata = sorted({r["stratum"] for r in kept + skipped if "stratum" in r})
    if strata:
        out["per_stratum"] = {
            s: _counts([r for r in kept if r.get("stratum") == s], [r for r in skipped if r.get("stratum") == s])
            for s in strata
        }
    return out


# ---------------------------------------------------------------- data loading

def make_collate(tokenizer, max_length):
    """Items are (row_id, row). Returns padded chosen batch, rejected batch and the IDs."""
    def collate(items):
        chosen, rejected, ids = [], [], []
        for rid, row in items:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
            ids.append(rid)
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), ids
    return collate


def build_loader(items, batch_size: int, seed: int, collate_fn):
    """Shuffling DataLoader with its own seeded generator, so the order depends only on seed."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(items, batch_size=int(batch_size), shuffle=True, generator=gen, collate_fn=collate_fn)


# ---------------------------------------------------------------- optimization core

def accumulate_and_step(loader, micro_step, params, optimizer, accum_steps: int, max_grad_norm: float, on_step):
    """One pass over loader with gradient accumulation.

    micro_step(batch) -> (loss, stats), loss = mean over the micro-batch. Each loss is divided by
    accum_steps before backward, so 8 micro-batches of 2 give the gradient of the mean over 16.
    An optimizer step follows every accum_steps micro-batches, plus one for a final remainder
    (whose losses are also divided by accum_steps). on_step(step, stats_list, grad_norm) logs it.
    """
    n_micro = len(loader)
    optimizer.zero_grad(set_to_none=True)
    pending, step = [], 0
    for i, batch in enumerate(loader, start=1):
        loss, stats = micro_step(batch)
        (loss / accum_steps).backward()
        pending.append(stats)
        if i % accum_steps == 0 or i == n_micro:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm {grad_norm} at optimizer step {step + 1}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            on_step(step, pending, grad_norm)
            pending = []
    return step


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def dpo_sequence_logprobs(model, chosen: dict, rejected: dict, with_grad: bool):
    """Policy and reference summed response log-probs for a pair batch.

    The reference is the same model with the LoRA adapter disabled (reference_mode), always
    under no_grad. Reference passes run first so their activations are freed before the
    policy graph is built.
    """
    with torch.no_grad(), reference_mode(model):
        rc = response_sequence_logprobs(model, chosen)[0]
        rr = response_sequence_logprobs(model, rejected)[0]
    with torch.set_grad_enabled(with_grad):
        pc = response_sequence_logprobs(model, chosen)[0]
        pr = response_sequence_logprobs(model, rejected)[0]
    return pc, pr, rc, rr


def lora_init_digest(model) -> dict:
    """SHA-256 of all LoRA A weights in name order; equal digests mean identical initialization."""
    h = hashlib.sha256()
    n = 0
    for name, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        if "lora_A" in name:
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
            n += 1
    return {"lora_A_sha256": h.hexdigest(), "n_lora_A_tensors": n}


def attach_forward_dtype_hooks(model, sink: dict):
    """Record activation dtypes at the first LoRA B module and the LM head on the next forward."""
    def lora_hook(module, inputs, output):
        sink.setdefault("lora_branch_output", str(output.dtype))

    def head_hook(module, inputs, output):
        sink.setdefault("lm_head_input", str(inputs[0].dtype))
        sink.setdefault("logits", str(output.dtype))

    handles = []
    lora_b = next((m for n, m in model.named_modules() if n.endswith("lora_B.default")), None)
    head = model.get_output_embeddings()
    if lora_b is not None:
        handles.append(lora_b.register_forward_hook(lora_hook))
    if head is not None:
        handles.append(head.register_forward_hook(head_hook))
    return handles


# ---------------------------------------------------------------- setup and training

def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    max_length = int(cfg["max_sequence_length"])
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)

    tokenizer = load_tokenizer(cfg["base_model"])
    kept, kept_records, skipped = filter_pairs(tokenizer, rows, max_length)
    report = filter_report(path, max_length, kept_records, skipped)
    # --max-examples applies after rule A (forks use the first N retained rows).
    if max_examples is not None:
        kept, kept_records = kept[: int(max_examples)], kept_records[: int(max_examples)]

    # Seed immediately before the fresh LoRA adapter is built: identical init for every run.
    set_seed(seed)
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    items = [(rec["id"], row) for rec, row in zip(kept_records, kept)]
    loader = build_loader(items, cfg["batch_size"], seed, make_collate(tokenizer, max_length))
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "dataset_path": path,
        "rows": kept,
        "records": kept_records,
        "filter_report": report,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, overwrite: bool = False):
    cfg0 = load_yaml(config_path)
    out_json = repo_path(f"{cfg0['results_dir']}/{run_name}_train.json")
    output = repo_path(output_path or f"outputs/task1_dpo/{run_name}")
    for p in (out_json, output):
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")

    elapsed = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    run_beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    n_micro = len(loader)
    params = trainable_parameters(model)
    device = next(model.parameters()).device

    dtypes = dtype_report(model)
    result = {
        "script": "task1_dpo.train",
        "run_name": run_name,
        "status": "running",
        "config_path": config_path,
        "config": cfg,
        "cli": {"dataset": dataset_path, "output": output_path, "beta": beta, "max_examples": max_examples},
        "effective": {
            "beta": run_beta,
            "dataset_path": bundle["dataset_path"],
            "learning_rate": float(cfg["learning_rate"]),
            "batch_size": int(cfg["batch_size"]),
            "grad_accum_steps": accum,
            "pairs_per_optimizer_step": int(cfg["batch_size"]) * accum,
            "epochs": 1,
            "max_grad_norm": float(cfg["max_grad_norm"]),
            "lr_schedule": "constant (no scheduler in configs/dpo.yaml)",
            "loss_scaling": "each micro-batch mean loss divided by grad_accum_steps, including the final remainder step",
        },
        **run_metadata(cfg),
        "dtype_report": dtypes,
        "lora_init": lora_init_digest(model),
        "data": {
            "filter": bundle["filter_report"],
            "max_examples": max_examples,
            "examples_used": len(bundle["records"]),
            "used_ids": [r["id"] for r in bundle["records"]],
            "used_indices": [r["index"] for r in bundle["records"]],
            "used_counts": _counts(bundle["records"], []),
        },
        "n_micro_batches": n_micro,
        "n_optimizer_steps_planned": math.ceil(n_micro / accum),
        "remainder_micro_batches": n_micro % accum,
        "n_optimizer_steps": 0,
        "examples_seen": 0,
        "sanity_first_microbatch": None,
        "adapter_path": display_path(output),
        "wall_clock_seconds": 0.0,
        "peak_vram_bytes": None,
        "history": [],
    }

    def write():
        result["wall_clock_seconds"] = elapsed()
        result["peak_vram_bytes"] = peak_vram_bytes()
        save_json(out_json, result)

    print("dtype report:", dtypes)
    if "torch.float16" in dtypes["trainable_param_dtypes"]:
        result["status"] = "stopped: trainable LoRA parameters are float16"
        write()
        raise SystemExit("Trainable LoRA parameters are float16. Stopped as instructed; ask before choosing a fix.")

    fwd_dtypes: dict = {}
    hooks = attach_forward_dtype_hooks(model, fwd_dtypes)

    def micro_step(batch):
        chosen, rejected, ids = batch
        chosen, rejected = to_device(chosen, device), to_device(rejected, device)
        pc, pr, rc, rr = dpo_sequence_logprobs(model, chosen, rejected, with_grad=True)
        loss, _ = dpo_loss(pc, pr, rc, rr, run_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DPO loss on prompt IDs {ids}")
        lc, lr_ = (pc - rc).detach(), (pr - rr).detach()
        if result["sanity_first_microbatch"] is None:
            for h in hooks:
                h.remove()
            result["dtype_report"]["forward"] = dict(fwd_dtypes)
            print("forward dtypes:", fwd_dtypes)
            # LoRA B starts at zero, so policy == reference and the loss must equal log 2.
            val = float(loss.item())
            result["sanity_first_microbatch"] = {
                "loss": val,
                "log_2": math.log(2.0),
                "abs_diff": abs(val - math.log(2.0)),
                "tolerance": 1.0e-3,
                "passed": abs(val - math.log(2.0)) < 1.0e-3,
            }
            print("first micro-batch loss:", result["sanity_first_microbatch"])
            if not result["sanity_first_microbatch"]["passed"]:
                raise RuntimeError("first micro-batch DPO loss differs from log 2 by more than 1e-3")
        return loss, {
            "ids": ids,
            "n": len(ids),
            "loss": float(loss.item()),
            "m": (lc - lr_).tolist(),
            "chosen_logratio": lc.tolist(),
            "rejected_logratio": lr_.tolist(),
        }

    def on_step(step, stats, grad_norm):
        n = sum(s["n"] for s in stats)
        m = [x for s in stats for x in s["m"]]
        result["examples_seen"] += n
        result["n_optimizer_steps"] = step
        rec = {
            "step": step,
            "micro_batches": len(stats),
            "examples_in_step": n,
            "examples_seen": result["examples_seen"],
            "loss": sum(s["loss"] * s["n"] for s in stats) / n,
            "preference_accuracy": sum(x > 0 for x in m) / n,
            "mean_margin": sum(m) / n,
            "mean_chosen_logratio": sum(x for s in stats for x in s["chosen_logratio"]) / n,
            "mean_rejected_logratio": sum(x for s in stats for x in s["rejected_logratio"]) / n,
            "grad_norm_before_clip": grad_norm,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed(),
            "prompt_ids": [i for s in stats for i in s["ids"]],
        }
        result["history"].append(rec)
        write()
        print(
            f"step {step}/{result['n_optimizer_steps_planned']} seen={rec['examples_seen']} loss={rec['loss']:.4f} "
            f"acc={rec['preference_accuracy']:.3f} m={rec['mean_margin']:.3f} gn={grad_norm:.3f} t={rec['elapsed_seconds']:.0f}s"
        )

    write()
    try:
        accumulate_and_step(loader, micro_step, params, optimizer, accum, float(cfg["max_grad_norm"]), on_step)
    except BaseException as e:
        result["status"] = f"failed: {type(e).__name__}: {e}"
        write()
        raise

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    result["status"] = "completed"
    write()
    print(f"Saved adapter to {output}\nWrote {out_json}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.overwrite)


if __name__ == "__main__":
    main()
