"""Task 5 step 5e: post-hoc diagnostics on the saved notebook 6 files. Everything here is post hoc.

No reported number is changed and no existing result file is written. Three parts:

--part mac             (default, CPU, no model weights) items 1, 2, 3a, 4, 5, 6, 7
                       -> results/task5_feedback/posthoc_diagnostics.json
--part echoes          (CPU, step 5f) judge echo table -> results/task5_feedback/judge_echoes.json
--part adapter_effect  (CUDA only, notebook 7) item 3b: SFT, RLVR and RLAIF loaded exactly as evaluate_math
                       loads them (load_frozen_policy, fp16), adapter names and lora_B norms asserted, SFT's saved
                       responses for the first 4 prompt IDs of each dataset teacher-forced under each policy
                       -> results/task5_feedback/posthoc_adapter_effect.json

Item 3a on the Mac attaches each adapter to a meta-device copy of the base model (no base weights are read),
so it reports the adapter names and key coverage that PeftModel.from_pretrained produces, not the forward pass.
"""
from __future__ import annotations

import argparse
import gc
import re
import statistics
from collections import Counter

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from common.run_info import display_path, peak_vram_bytes, run_metadata
from task5_feedback import protocol as P
from task5_feedback.evaluate_math import dataset_path, gen_paths, load_frozen_policy, out_dir, policy_specs
from task5_feedback.rlvr import numerically_equal

DATASETS = ("gsm", "transfer")
POLICY_PAIRS = (("sft", "rlvr"), ("sft", "rlaif"), ("rlvr", "rlaif"))
_LABEL_RE = re.compile(r"\b(A|B|TIE)\b")  # the released parser's pattern (rlaif.PairwiseAIJudge.compare)
JUDGE_STAGES = ("diagnostics_released", "diagnostics_swapped", "judge_gsm", "judge_transfer")
N_TEACHER_FORCE = 4
LOG_GPU0 = "results/task5_feedback/logs/task5_queue_gpu0.log"  # committed copy of the notebook 6 GPU 0 queue log
_LOG_EXPECTED = re.compile(r"^(  generation batch \d+/\d+ t=\d+s|\[gen \w+ \w+\] status=.*|===== .*: \w+)$")


def refuse_existing(path, overwrite: bool):
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it.")


def summary_stats(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs)}


def load_gens(cfg) -> dict:
    """gens[ds][policy] = (run json, jsonl rows), the notebook 6 files as committed."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for pol in P.POLICIES:
            j, jl = gen_paths(cfg, ds, pol, smoke=False)
            out[ds][pol] = (load_json(j), read_jsonl(jl))
    return out


# ---------------------------------------------------------------- item 1: RLVR loading audit (read-only)

def decode_steps(rows: list[dict]) -> int:
    """Decoding steps of one generation run: per batch of 16 in file order, the longest response in that batch."""
    return sum(max(r["token_length"] for r in rows[i : i + P.GEN_BATCH_SIZE]) for i in range(0, len(rows), P.GEN_BATCH_SIZE))


def item1_loading_audit(cfg, gens) -> dict:
    record_keys = ("adapter", "adapter_sha256", "status", "git", "start_time", "hardware", "dtype")
    runs = {}
    for ds in DATASETS:
        for pol in P.POLICIES:
            meta, rows = gens[ds][pol]
            gen_s = meta["timing"]["generation"]
            steps = decode_steps(rows)
            runs[f"{ds}_{pol}"] = {
                **{k: meta.get(k) for k in record_keys},
                "config_policy_path": meta["config"]["policies"][pol],
                "setup_seconds": meta["timing"]["setup"],
                "generation_seconds": gen_s,
                "decode_steps_sum_of_batch_max_length": steps,
                "seconds_per_decode_step": gen_s / steps,
                "fields_about_active_adapter_or_peft_flags": sorted(k for k in meta if re.search(r"active|peft|lora|merged", k)),
            }
    lines = repo_path(LOG_GPU0).read_text(encoding="utf-8").splitlines()
    return {
        "code_trace": [
            "queue cell: one python process per (dataset, policy); loop ds in gsm transfer, p in sft rlvr rlaif",
            "evaluate_math.run_generate: args.dataset only selects dataset_path, data sha assert, row count, prompt_id, output path",
            "adapter = policy_specs(cfg)[args.policy] -> cfg['policies']['rlvr'] = checkpoints/rlvr_policy for both datasets",
            "P.assert_sha(adapter_model.safetensors) before loading; load_frozen_policy(cfg, 'rlvr') -> "
            "common.models.load_policy(cfg, adapter_path, trainable=False) -> PeftModel.from_pretrained(base, path, is_trainable=False); "
            "no dataset argument reaches model loading",
            "generation: task1_dpo.evaluate.generate_responses(model, ...) -> common.generation.batch_generate -> model.generate; "
            "no disable_adapter / reference_mode call on this path (with_kl=False)",
        ],
        "rlvr_gsm_vs_transfer_recorded_fields_equal": {
            k: runs["gsm_rlvr"][k] == runs["transfer_rlvr"][k] for k in ("adapter", "adapter_sha256", "config_policy_path", "git", "hardware", "dtype")
        },
        "not_recorded": "no gen JSON records the active adapter name, PEFT flags (disable_adapters, merged) or the model class",
        "runs": runs,
        "log_file": LOG_GPU0,
        "log_n_lines": len(lines),
        "log_result_lines": [ln for ln in lines if ln.startswith("[gen ") or ln.startswith("===== ")],
        "log_lines_other_than_progress_result_status": [ln for ln in lines if not _LOG_EXPECTED.match(ln)],
        "log_note": "stderr was teed into the log (2>&1); it holds no adapter load message and no warning for any of the six runs",
    }


# ---------------------------------------------------------------- item 2: identity and divergence

def first_divergence(a: list[int], b: list[int]):
    """First token index where the two id lists differ; the shorter length if one is a prefix; None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def item2_divergence(gens, tokenizer) -> dict:
    out = {}
    for ds in DATASETS:
        texts = {pol: [r["text"] for r in gens[ds][pol][1]] for pol in P.POLICIES}
        ids = {pol: [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts[pol]] for pol in P.POLICIES}
        for a, b in POLICY_PAIRS:
            ident = sum(x == y for x, y in zip(texts[a], texts[b], strict=True))
            div = [first_divergence(x, y) for x, y, tx, ty in zip(ids[a], ids[b], texts[a], texts[b]) if tx != ty]
            n_tok_equal = sum(d is None for d in div)  # texts differ but token ids equal (should be 0)
            out[f"{ds}:{a}-{b}"] = {
                "n_prompts": len(texts[a]),
                "n_identical_text": ident,
                "n_nonidentical": len(div),
                "first_divergent_token_index": summary_stats([d for d in div if d is not None]),
                "first_divergent_token_index_buckets": dict(Counter(
                    "0" if d == 0 else "1" if d == 1 else "2-9" if d < 10 else ">=10" for d in div if d is not None)),
                "n_nonidentical_text_but_identical_tokens": n_tok_equal,
            }
    return {"tokenizer": "policy tokenizer (base_model), response text re-tokenized without special tokens", "pairs": out}


# ---------------------------------------------------------------- item 3a: adapter files and meta-device load

def lora_b_norms(state: dict) -> dict:
    b = {k: v.float() for k, v in state.items() if "lora_B" in k}
    a = {k: v.float() for k, v in state.items() if "lora_A" in k}
    per = [float(v.norm()) for v in b.values()]
    return {
        "n_lora_B": len(b),
        "lora_B_frobenius_total": float(torch.sqrt(sum((v ** 2).sum() for v in b.values()))),
        "lora_B_frobenius_per_module": summary_stats(per),
        "n_lora_B_all_zero": sum(1 for x in per if x == 0.0),
        "lora_A_frobenius_total": float(torch.sqrt(sum((v ** 2).sum() for v in a.values()))),
        "lora_B_frobenius_by_target": {
            t: float(torch.sqrt(sum((v ** 2).sum() for k, v in b.items() if f".{t}." in k))) for t in ("q_proj", "v_proj")
        },
    }


def item3a_adapters(cfg) -> dict:
    from accelerate import init_empty_weights
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    out = {}
    for pol in P.TRAINED:
        path = policy_specs(cfg)[pol]
        sha = P.assert_sha(P.adapter_file(path))
        state = load_file(str(repo_path(P.adapter_file(path))))
        with init_empty_weights():
            base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg["base_model"]))
        # Same call as common.models.load_policy (is_trainable=False); low_cpu_mem_usage only because the base is on meta.
        model = PeftModel.from_pretrained(base, str(repo_path(path)), is_trainable=False, low_cpu_mem_usage=True)
        injected = sorted(n.replace(".default", "") for n, _ in model.named_parameters() if "lora_" in n)
        lora_layers = [m for m in model.modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0]
        out[pol] = {
            "path": path,
            "sha256": sha,
            "active_adapter": model.active_adapter,
            "active_adapters": list(model.active_adapters),
            "peft_config_names": list(model.peft_config),
            "inference_mode": model.peft_config["default"].inference_mode,
            "n_lora_layers": len(lora_layers),
            "layers_active_adapters": sorted({tuple(m.active_adapters) for m in lora_layers}),
            "layers_disable_adapters_any": any(m.disable_adapters for m in lora_layers),
            "layers_merged_any": any(m.merged for m in lora_layers),
            "scaling": sorted({float(m.scaling["default"]) for m in lora_layers}),
            "file_keys_not_injected": sorted(set(state) - set(injected)),
            "injected_not_in_file": sorted(set(injected) - set(state)),
            **lora_b_norms(state),
        }
        del model, base, state
        gc.collect()
    return {"method": "adapter attached to a meta-device base built from the base config; no base weights read; "
                      "norms from adapter_model.safetensors", "adapters": out}


# ---------------------------------------------------------------- item 4: judge raw strings

def item4_judge_raw(cfg) -> dict:
    d = out_dir(cfg, smoke=False)
    out = {}
    for stage in JUDGE_STAGES:
        rows = read_jsonl(d / f"{stage}.jsonl")
        out[stage] = {
            "n_calls": len(rows),
            "raw_counts": dict(Counter(r["raw"] for r in rows).most_common()),
            # More than one label token in the raw text: the parser keeps the first (e.g. "A, B," -> A).
            "n_raw_multiple_labels": sum(len(_LABEL_RE.findall(r["raw"].strip().upper())) > 1 for r in rows),
        }
        if stage.startswith("diagnostics"):
            out[stage]["raw_counts_by_category"] = {
                cat: dict(Counter(r["raw"] for r in rows if r["category"] == cat).most_common()) for cat in P.CATEGORIES}
        else:
            out[stage]["raw_counts_by_comparison"] = {
                c: dict(Counter(r["raw"] for r in rows if r["comparison"] == c).most_common()) for c in ("rlvr_vs_sft", "rlaif_vs_sft")}
    return out


# ---------------------------------------------------------------- item 5: swap check restricted to decisions

def item5_swap(cfg) -> dict:
    d = out_dir(cfg, smoke=False)
    rel = {r["pair_id"]: r for r in read_jsonl(d / "diagnostics_released.jsonl")}
    swp = {r["pair_id"]: r for r in read_jsonl(d / "diagnostics_swapped.jsonl")}
    assert set(rel) == set(swp) and len(rel) == P.N_PAIRS
    out = {}
    for cat in (*P.CATEGORIES, "all"):
        ids = [k for k in sorted(rel) if cat == "all" or rel[k]["category"] == cat]
        decided = [k for k in ids if rel[k]["label"] != "TIE" or swp[k]["label"] != "TIE"]
        same = sum(rel[k]["label"] == swp[k]["label"] for k in decided)
        both_nontie = [k for k in decided if rel[k]["label"] != "TIE" and swp[k]["label"] != "TIE"]
        slot = {}
        for name, rows in (("released", rel), ("swapped", swp)):
            nt = [rows[k] for k in ids if rows[k]["parse_matched"] and rows[k]["physical_label"] in ("A", "B")]
            slot[name] = {"n_nontie": len(nt), "n_slot_A": sum(r["physical_label"] == "A" for r in nt),
                          "slot_A_share": (sum(r["physical_label"] == "A" for r in nt) / len(nt)) if nt else None}
        out[cat] = {
            "n_pairs": len(ids),
            "n_with_a_nontie_decision": len(decided),
            "n_consistent_content_label": same,
            "consistency_restricted": (same / len(decided)) if decided else None,
            "n_both_nontie": len(both_nontie),
            "n_both_nontie_opposite": sum(rel[k]["label"] != swp[k]["label"] for k in both_nontie),
            "n_one_tie_one_nontie": len(decided) - len(both_nontie),
            "slot_A": slot,
        }
    return {"definition": "content label (A = clean, B = perturbed, TIE) compared across released and swapped order; "
                          "slot-A share = physical Candidate A picks / parsed non-tie decisions", "by_category": out}


# ---------------------------------------------------------------- item 6: verifier coverage

def last_number(text: str):
    toks = P._NUMBER_RE.findall(str(text))
    return toks[-1].replace(",", "") if toks else None


def item6_coverage(gens) -> dict:
    out = {}
    for ds in DATASETS:
        for pol in P.POLICIES:
            nc = [r for r in gens[ds][pol][1] if not r["compliant"]]
            hit = [r for r in nc if (n := last_number(r["text"])) is not None and numerically_equal(n, r["gold_final"])]
            out[f"{ds}_{pol}"] = {
                "n_responses": len(gens[ds][pol][1]),
                "n_noncompliant": len(nc),
                "n_last_number_equals_gold": len(hit),
                "by_failure_type": {t: {"n": sum(r["failure_type"] == t for r in nc),
                                        "n_last_number_equals_gold": sum(r["failure_type"] == t for r in hit)}
                                    for t in ("noncompliant_truncated", "noncompliant_ended")},
            }
    return {"rule": "last number token in the text (protocol._NUMBER_RE: signed unless the minus follows a digit or ')'), "
                    "commas stripped, numerically_equal to gold_final; non-compliant = extract_designated_final is None",
            "by_run": out}


# ---------------------------------------------------------------- judge echoes (step 5f)

def is_echo(raw: str) -> bool:
    """More than one label token in the raw judge text (the released parser keeps the first)."""
    return len(_LABEL_RE.findall(raw.strip().upper())) > 1


def echo_block(rows: list[dict], side_names: tuple) -> dict:
    """Echo count over rows and the content side each echo was parsed for. label is w.r.t. argument a, so
    A -> side_names[0] (trained / clean), B -> side_names[1] (SFT / perturbed). bound = 0.5 x echoes / n: the
    largest change in a win rate (A = 1, TIE = 0.5, B = 0) if every echo were scored as a tie instead."""
    echoes = [r for r in rows if is_echo(r["raw"])]
    n = len(rows)
    side = {side_names[0]: 0, side_names[1]: 0, "tie": 0}
    for r in echoes:
        side[{"A": side_names[0], "B": side_names[1], "TIE": "tie"}[r["label"]]] += 1
    return {"n": n, "n_echo": len(echoes), "echo_parsed_for": side,
            "win_rate_bound": 0.5 * len(echoes) / n if n else None,
            "echo_raw_counts": dict(Counter(r["raw"] for r in echoes).most_common())}


def judge_echoes(cfg) -> dict:
    d = out_dir(cfg, smoke=False)
    out = {}
    for stage in ("judge_gsm", "judge_transfer"):
        rows = read_jsonl(d / f"{stage}.jsonl")
        out[stage] = {c: echo_block([r for r in rows if r["comparison"] == c], ("trained", "sft"))
                      for c in ("rlvr_vs_sft", "rlaif_vs_sft")}
    for stage in ("diagnostics_released", "diagnostics_swapped"):
        rows = read_jsonl(d / f"{stage}.jsonl")
        out[stage] = {cat: echo_block([r for r in rows if r["category"] == cat], ("clean", "perturbed"))
                      for cat in P.CATEGORIES}
    return out


def run_echoes(args, cfg):
    out_json = out_dir(cfg, smoke=False) / "judge_echoes.json"
    refuse_existing(out_json, args.overwrite)
    elapsed = wall_timer()
    result = {
        "script": "task5_feedback.posthoc",
        "part": "echoes",
        "post_hoc": True,
        "note": "the released parser labels stay the reported metrics; no metric recomputed. "
                "echo = raw judge output with more than one label token (pattern \\b(A|B|TIE)\\b on raw.strip().upper())",
        "config_path": args.config,
        **run_metadata(cfg),
        "by_stage": judge_echoes(cfg),
    }
    result["wall_clock_seconds"] = elapsed()
    save_json(out_json, result)
    print(f"[posthoc echoes] wrote {display_path(out_json)} t={elapsed():.1f}s", flush=True)


# ---------------------------------------------------------------- part mac

def run_mac(args, cfg):
    out_json = out_dir(cfg, smoke=False) / "posthoc_diagnostics.json"
    refuse_existing(out_json, args.overwrite)
    elapsed = wall_timer()
    from common.models import load_tokenizer

    for ds in DATASETS:
        P.assert_sha(dataset_path(cfg, ds))
    gens = load_gens(cfg)
    tok = load_tokenizer(cfg["base_model"])
    result = {
        "script": "task5_feedback.posthoc",
        "part": "mac",
        "post_hoc": True,
        "note": "post-hoc diagnostics on saved notebook 6 files; no reported number changed; no generation or judge call",
        "config_path": args.config,
        **run_metadata(cfg),
        "item1_rlvr_loading_audit": item1_loading_audit(cfg, gens),
        "item2_identity_divergence": item2_divergence(gens, tok),
        "item3a_adapters": item3a_adapters(cfg),
        "item3b_adapter_effect": "pending: notebook 7, python -m task5_feedback.posthoc --part adapter_effect",
        "item4_judge_raw_outputs": item4_judge_raw(cfg),
        "item5_swap_restricted": item5_swap(cfg),
        "item6_verifier_coverage": item6_coverage(gens),
    }
    result["wall_clock_seconds"] = elapsed()
    save_json(out_json, result)
    print(f"[posthoc mac] wrote {display_path(out_json)} t={elapsed():.1f}s", flush=True)


# ---------------------------------------------------------------- part adapter_effect (CUDA, notebook 7)

@torch.no_grad()
def teacher_force(model, tokenizer, messages, text: str, eos: bool):
    """Per-token log-prob and argmax over the response tokens of one (prompt, saved response), batch 1, no padding.

    Prompt ids as batch_generate builds them (chat template, add_generation_prompt); response ids are the saved text
    re-tokenized, plus EOS if the saved response ended with EOS. logits[t] predicts token t+1, hence the shift.
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(rendered)["input_ids"]
    r_ids = tokenizer(text, add_special_tokens=False)["input_ids"] + ([tokenizer.eos_token_id] if eos else [])
    ids = torch.tensor([p_ids + r_ids], device=next(model.parameters()).device)
    logits = model(input_ids=ids).logits[0, len(p_ids) - 1 : -1].float()  # [n_resp, vocab]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = ids[0, len(p_ids):]
    out = {"logp": logp.gather(-1, tgt[:, None])[:, 0].cpu(), "argmax": logits.argmax(-1).cpu(), "n": len(r_ids)}
    del logits, logp, ids
    return out


def adapter_report(model) -> dict:
    lora_layers = [m for m in model.modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0]
    state = {n: p.detach() for n, p in model.named_parameters() if "lora_" in n}
    return {
        "model_class": type(model).__name__,
        "active_adapter": model.active_adapter,
        "active_adapters": list(model.active_adapters),
        "n_lora_layers": len(lora_layers),
        "layers_disable_adapters_any": any(m.disable_adapters for m in lora_layers),
        "layers_merged_any": any(m.merged for m in lora_layers),
        **lora_b_norms({n.replace(".default", ""): v.cpu() for n, v in state.items()}),
    }


def run_adapter_effect(args, cfg):
    if not torch.cuda.is_available():
        raise SystemExit("--part adapter_effect needs CUDA (notebook 7); it loads the 1.5B policy.")
    out_json = out_dir(cfg, smoke=False) / "posthoc_adapter_effect.json"
    refuse_existing(out_json, args.overwrite)
    from common.models import load_tokenizer

    elapsed = wall_timer()
    torch.cuda.reset_peak_memory_stats()
    tok = load_tokenizer(cfg["base_model"])
    items = {}
    for ds in DATASETS:
        P.assert_sha(dataset_path(cfg, ds))
        rows = read_jsonl(dataset_path(cfg, ds))[:N_TEACHER_FORCE]
        sft = read_jsonl(gen_paths(cfg, ds, "sft", smoke=False)[1])[:N_TEACHER_FORCE]
        for row, g in zip(rows, sft, strict=True):
            assert g["prompt_id"] == P.prompt_id(ds, row)
        items[ds] = [(prompt_messages(row), g["text"], g["terminated_with_eos"], g["prompt_id"]) for row, g in zip(rows, sft)]

    result = {"script": "task5_feedback.posthoc", "part": "adapter_effect", "post_hoc": True, "status": "running",
              "config_path": args.config, **run_metadata(cfg),
              "settings": {"loading": "evaluate_math.load_frozen_policy (common.models.load_policy, config dtype, CUDA)",
                           "prompts": f"first {N_TEACHER_FORCE} prompt IDs in file order per dataset",
                           "responses": "SFT saved response text re-tokenized, plus EOS if it ended with EOS",
                           "batch": 1, "no_grad": True, "logp": "log_softmax of float32 logits"},
              "prompt_ids": {ds: [x[3] for x in items[ds]] for ds in DATASETS},
              "adapters": {}, "effect": {}}

    def forward_all(model):
        return {ds: [teacher_force(model, tok, m, t, e) for m, t, e, _ in items[ds]] for ds in DATASETS}

    model = load_frozen_policy(cfg, "sft")
    base = forward_all(model)
    del model
    gc.collect(); torch.cuda.empty_cache()
    for pol in P.TRAINED:
        model = load_frozen_policy(cfg, pol)
        rep = adapter_report(model)
        assert rep["active_adapter"] == "default" and not rep["layers_disable_adapters_any"] and not rep["layers_merged_any"], rep
        assert rep["n_lora_B_all_zero"] == 0, rep
        result["adapters"][pol] = rep
        on = forward_all(model)
        from common.models import reference_mode
        with reference_mode(model):
            off = forward_all(model)
        del model
        gc.collect(); torch.cuda.empty_cache()
        for ds in DATASETS:
            d = torch.cat([a["logp"] - b["logp"] for a, b in zip(on[ds], base[ds])]).abs()
            flip = torch.cat([a["argmax"] != b["argmax"] for a, b in zip(on[ds], base[ds])]).float()
            off_d = torch.cat([a["logp"] - b["logp"] for a, b in zip(off[ds], base[ds])]).abs()
            result["effect"][f"{ds}_{pol}"] = {
                "n_tokens": int(d.numel()),
                "mean_abs_delta_logp": float(d.mean()),
                "max_abs_delta_logp": float(d.max()),
                "argmax_change_fraction": float(flip.mean()),
                "per_prompt_mean_abs_delta_logp": [float((a["logp"] - b["logp"]).abs().mean()) for a, b in zip(on[ds], base[ds])],
                "check_adapter_disabled_vs_sft_max_abs_delta": float(off_d.max()),
            }
        save_json(out_json, {**result, "wall_clock_seconds": elapsed(), "peak_vram_bytes": peak_vram_bytes()})
    result["status"] = "completed"
    result["wall_clock_seconds"] = elapsed()
    result["peak_vram_bytes"] = peak_vram_bytes()
    save_json(out_json, result)
    for k, v in result["adapters"].items():
        print(f"[adapter {k}] active={v['active_adapter']} layers={v['n_lora_layers']} merged={v['layers_merged_any']} "
              f"disabled={v['layers_disable_adapters_any']} lora_B_norm={v['lora_B_frobenius_total']:.4f}", flush=True)
    for k, v in result["effect"].items():
        print(f"[effect {k}] n_tokens={v['n_tokens']} mean_abs={v['mean_abs_delta_logp']:.4g} max_abs={v['max_abs_delta_logp']:.4g} "
              f"argmax_change={v['argmax_change_fraction']:.4f} disabled_vs_sft_max={v['check_adapter_disabled_vs_sft_max_abs_delta']:.3g}", flush=True)
    print(f"[posthoc adapter_effect] status=completed t={elapsed():.0f}s peak_vram={result['peak_vram_bytes'] / 2**30:.2f}GiB; "
          f"wrote {display_path(out_json)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--part", choices=["mac", "adapter_effect", "echoes"], default="mac")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.part == "mac":
        run_mac(args, cfg)
    elif args.part == "echoes":
        run_echoes(args, cfg)
    else:
        run_adapter_effect(args, cfg)


if __name__ == "__main__":
    main()
