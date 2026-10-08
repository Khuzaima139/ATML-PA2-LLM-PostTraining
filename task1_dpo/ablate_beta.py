"""Task 1 Step 2: beta study (manual Section 1, Step 2).

For each beta in cfg["betas"] (or --betas): train a short-run fork from the original LoRA
initialization on the first cfg["short_ablation_examples"] retained rows of dpo_standard_train
then evaluate that adapter with --modes pairs,generate at the fork's own beta.
Training and evaluation call task1_dpo.train.run_training and task1_dpo.evaluate.main in this
process; nothing is reimplemented here.

The helpers below (LoRA digest check, evaluation call, status lines, orchestration log) are
also used by task1_dpo.analyze_length.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import sys

import torch

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from task1_dpo import evaluate
from task1_dpo.train import display_path, run_metadata, run_training

# LoRA A digest of the Kaggle Step 1 smoke run; every Task 1 run must start from this init.
EXPECTED_LORA_A_SHA256 = "0ef3a1ffae8b06f2a82a26c773f9026b06deab08b63d34115d81ee5542abbc2e"
SMOKE_PREFIX = "smoke_"
SMOKE_MAX_EXAMPLES = 16
SMOKE_EVAL_LIMIT = 4
GIB = 2**30


def beta_tag(beta: float) -> str:
    """0.03 -> '0p03', 0.1 -> '0p10'."""
    return f"{beta:.2f}".replace(".", "p")


def train_json_path(cfg: dict, run_name: str):
    return repo_path(f"{cfg['results_dir']}/{run_name}_train.json")


def eval_json_path(cfg: dict, name: str):
    return repo_path(f"{cfg['results_dir']}/eval_{name}.json")


def lora_digest_check(train_result: dict) -> dict:
    observed = train_result["lora_init"]["lora_A_sha256"]
    return {"expected": EXPECTED_LORA_A_SHA256, "observed": observed, "match": observed == EXPECTED_LORA_A_SHA256}


def free_memory():
    """Drop the previous model before the next one is loaded in this process."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_evaluation(argv: list[str]):
    """Call task1_dpo.evaluate.main with these arguments, in this process.

    evaluate.main reads sys.argv, so it is swapped in for the call and restored afterwards.
    """
    saved = sys.argv
    sys.argv = ["task1_dpo.evaluate", *argv]
    try:
        evaluate.main()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------- one-line status reports

def train_status_line(path) -> str:
    p = repo_path(path)
    if not p.exists():
        return f"[train] {display_path(p)} missing"
    r = load_json(p)
    first = r.get("sanity_first_microbatch") or {}
    peak = r.get("peak_vram_bytes")
    return (
        f"[train {r['run_name']}] status={r['status']} steps={r['n_optimizer_steps']}/{r['n_optimizer_steps_planned']} "
        f"examples={r['data']['examples_used']} beta={r['effective']['beta']} first_step_passed={first.get('passed')} "
        f"lora_digest_match={lora_digest_check(r)['match']} minutes={r['wall_clock_seconds'] / 60:.1f} "
        f"peak_GiB={'n/a' if peak is None else f'{peak / GIB:.2f}'}"
    )


def eval_status_lines(path, modes: list[str]) -> list[str]:
    p = repo_path(path)
    if not p.exists():
        return [f"[eval] {display_path(p)} missing"]
    r = load_json(p)
    lines = []
    for mode in modes:
        e = r["modes"].get(mode)
        if e is None:
            lines.append(f"[eval {r['name']}:{mode}] missing")
            continue
        peak = e.get("peak_vram_bytes")
        lines.append(
            f"[eval {r['name']}:{mode}] smoke={e['smoke']} minutes={e['wall_clock_seconds'] / 60:.1f} "
            f"peak_GiB={'n/a' if peak is None else f'{peak / GIB:.2f}'} {evaluate.summary_line(e['metrics'])}"
        )
    return lines


# ---------------------------------------------------------------- train then evaluate one condition

class OrchestrationLog:
    """results/task1_dpo/<name>.json: which runs this invocation made, their digest checks and status.

    Rewritten after every stage so a crash keeps the record.
    """

    def __init__(self, cfg: dict, config_path: str, name: str, cli: dict, overwrite: bool):
        self.path = repo_path(f"{cfg['results_dir']}/{name}.json")
        if self.path.exists() and not overwrite:
            raise SystemExit(f"{self.path} exists; pass --overwrite to replace it.")
        self.elapsed = wall_timer()
        self.data = {"script": cli.pop("script"), "name": name, "status": "running", "config_path": config_path,
                     "config": cfg, "cli": cli, **run_metadata(cfg),
                     "expected_lora_A_sha256": EXPECTED_LORA_A_SHA256, "wall_clock_seconds": 0.0, "runs": []}
        self.write()

    def write(self):
        self.data["wall_clock_seconds"] = self.elapsed()
        save_json(self.path, self.data)


def train_and_evaluate(log: OrchestrationLog, config_path: str, cfg: dict, run_name: str, train_kwargs: dict,
                       eval_beta: float, eval_modes: list[str], eval_limit: int | None, overwrite: bool) -> dict:
    """Train one condition, check its LoRA init digest, then evaluate its adapter."""
    entry = {"run_name": run_name, "train_kwargs": train_kwargs, "train_json": display_path(train_json_path(cfg, run_name)),
             "eval_json": display_path(eval_json_path(cfg, run_name)), "eval_modes": eval_modes,
             "eval_beta": eval_beta, "eval_limit": eval_limit, "status": "training"}
    log.data["runs"].append(entry)
    log.write()

    print(f"===== train {run_name} {train_kwargs}", flush=True)
    result = run_training(config_path, run_name, overwrite=overwrite, **train_kwargs)
    free_memory()
    entry["first_step_passed"] = result["sanity_first_microbatch"]["passed"]
    entry["lora_digest_check"] = lora_digest_check(result)
    entry["adapter_path"] = result["adapter_path"]
    print(train_status_line(train_json_path(cfg, run_name)), flush=True)
    if not entry["lora_digest_check"]["match"]:
        entry["status"] = "stopped: LoRA init digest differs from the expected value"
        log.data["status"] = entry["status"]
        log.write()
        raise SystemExit(f"{run_name}: LoRA init digest {entry['lora_digest_check']['observed']} != expected {EXPECTED_LORA_A_SHA256}")

    entry["status"] = "evaluating"
    log.write()
    argv = ["--config", config_path, "--adapter", result["adapter_path"], "--name", run_name,
            "--beta", repr(float(eval_beta)), "--modes", ",".join(eval_modes)]
    if eval_limit is not None:
        argv += ["--limit", str(eval_limit)]
    if overwrite:
        argv += ["--overwrite"]
    print(f"===== evaluate {run_name} {' '.join(argv)}", flush=True)
    run_evaluation(argv)
    free_memory()
    for line in eval_status_lines(eval_json_path(cfg, run_name), eval_modes):
        print(line, flush=True)
    entry["status"] = "completed"
    log.write()
    return entry


@contextlib.contextmanager
def logged_failure(log: OrchestrationLog):
    try:
        yield
    except BaseException as e:
        if log.data["status"] == "running":
            log.data["status"] = f"failed: {type(e).__name__}: {e}"
        log.write()
        raise
    log.data["status"] = "completed"
    log.write()


# ---------------------------------------------------------------- beta study

def parse_betas(text: str | None, cfg: dict) -> list[float]:
    allowed = [float(b) for b in cfg["betas"]]
    if text is None:
        return allowed
    betas = [float(x) for x in text.split(",") if x.strip()]
    bad = [b for b in betas if b not in allowed]
    if bad:
        raise SystemExit(f"betas {bad} are not in the config betas {allowed}")
    return betas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", help="comma list, a subset of the config betas (default: all)")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_MAX_EXAMPLES} training examples, evaluation --limit {SMOKE_EVAL_LIMIT}, run names prefixed {SMOKE_PREFIX!r}")
    ap.add_argument("--report", action="store_true", help="only print the status lines of existing result files")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    betas = parse_betas(args.betas, cfg)
    prefix = SMOKE_PREFIX if args.smoke else ""
    n_examples = SMOKE_MAX_EXAMPLES if args.smoke else int(cfg["short_ablation_examples"])
    eval_limit = SMOKE_EVAL_LIMIT if args.smoke else None
    modes = ["pairs", "generate"]
    runs = [(beta, f"{prefix}beta_{beta_tag(beta)}") for beta in betas]

    if args.report:
        for _, run_name in runs:
            print(train_status_line(train_json_path(cfg, run_name)))
            for line in eval_status_lines(eval_json_path(cfg, run_name), modes):
                print(line)
        return

    name = f"{prefix}ablate_beta_" + "_".join(beta_tag(b) for b in betas)
    log = OrchestrationLog(cfg, args.config, name, {"script": "task1_dpo.ablate_beta", "betas": betas, "smoke": args.smoke,
                                                    "max_examples": n_examples, "eval_limit": eval_limit}, args.overwrite)
    with logged_failure(log):
        for beta, run_name in runs:
            train_and_evaluate(log, args.config, cfg, run_name, {"beta": beta, "max_examples": n_examples},
                               eval_beta=beta, eval_modes=modes, eval_limit=eval_limit, overwrite=args.overwrite)
    print(f"Wrote {display_path(log.path)}")


if __name__ == "__main__":
    main()
