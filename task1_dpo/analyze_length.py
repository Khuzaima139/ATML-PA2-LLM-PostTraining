"""Task 1 Step 3: length-confounding study, training and evaluation part (manual Section 1, Step 3).

Trains run "length_balanced" from the original LoRA initialization on all retained rows of
dpo_length_balanced_train for one epoch at the config beta, then evaluates it with
--modes stratified,wordlimit at the same beta. The standard model's stratified and wordlimit
results are read later from eval_standard.json (Step 1 notebook) and are not rerun here.
Per-stratum tables and the comparison are produced on the Mac by task1_dpo.summarize.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task1_dpo.ablate_beta import (
    SMOKE_EVAL_LIMIT,
    SMOKE_MAX_EXAMPLES,
    SMOKE_PREFIX,
    OrchestrationLog,
    eval_json_path,
    eval_status_lines,
    logged_failure,
    train_and_evaluate,
    train_json_path,
    train_status_line,
)
from task1_dpo.train import display_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_MAX_EXAMPLES} training examples, evaluation --limit {SMOKE_EVAL_LIMIT}, run name prefixed {SMOKE_PREFIX!r}")
    ap.add_argument("--report", action="store_true", help="only print the status lines of existing result files")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    prefix = SMOKE_PREFIX if args.smoke else ""
    run_name = f"{prefix}length_balanced"
    beta = float(cfg["beta"])
    modes = ["stratified", "wordlimit"]
    train_kwargs = {
        "dataset_path": cfg["paths"]["dpo_length_train"],
        "beta": beta,
        "max_examples": SMOKE_MAX_EXAMPLES if args.smoke else None,  # None: all retained rows
    }
    eval_limit = SMOKE_EVAL_LIMIT if args.smoke else None

    if args.report:
        print(train_status_line(train_json_path(cfg, run_name)))
        for line in eval_status_lines(eval_json_path(cfg, run_name), modes):
            print(line)
        return

    log = OrchestrationLog(cfg, args.config, f"{prefix}analyze_length",
                           {"script": "task1_dpo.analyze_length", "smoke": args.smoke, **train_kwargs, "eval_limit": eval_limit},
                           args.overwrite)
    with logged_failure(log):
        train_and_evaluate(log, args.config, cfg, run_name, train_kwargs,
                           eval_beta=beta, eval_modes=modes, eval_limit=eval_limit, overwrite=args.overwrite)
    print(f"Wrote {display_path(log.path)}")


if __name__ == "__main__":
    main()
