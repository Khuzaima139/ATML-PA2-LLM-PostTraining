"""Task 4 qualitative candidates (Required Evidence, qualitative examples). Mac only, from committed files.

Rule fixed before any judge label was seen. For each target behaviour X, candidates are audit prompts where
any policy's manual or judge label is X. Tiers: 1 = manual labels differ across policies and some judge
label differs from its manual label; 2 = manual labels differ across policies only; 3 = judge-manual
disagreement only. Prompts in none of the tiers are not ranked. Within a tier, lowest xstest_id first.
The top N_PER_BEHAVIOUR per behaviour are written with every policy's response, manual and judge label.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml
from common.logging_utils import save_json
from task4_safety.evaluate_safety import load_frame
from task4_safety.make_audit_sheet import KEY, LABELS_CSV, expand_labels
from task4_safety.protocol import POLICIES, judge_name, task_dir

TARGETS = ("UNSAFE_COMPLIANCE", "JUSTIFIED_REFUSAL", "OVER_REFUSAL")
N_PER_BEHAVIOUR = 3
OUT = "qualitative_candidates.json"


def tier(manual: list[str], judge: list[str]) -> int | None:
    """Tier of one prompt from its per-policy manual and judge labels (same policy order)."""
    policy_diff = len(set(manual)) > 1
    judge_diff = any(j != m for j, m in zip(judge, manual))
    if policy_diff and judge_diff:
        return 1
    if policy_diff:
        return 2
    if judge_diff:
        return 3
    return None


def rank(m: pd.DataFrame, target: str) -> tuple[list[tuple[int, int]], dict]:
    """m: one row per (xstest_id, policy) with label and manual_label. Returns ranked [(tier, xstest_id)] and counts."""
    ranked, n_cand, n_untiered = [], 0, 0
    for xid, g in m.groupby("xstest_id"):
        g = g.set_index("policy").loc[list(POLICIES)]
        man, jud = g["manual_label"].tolist(), g["label"].tolist()
        if target not in man and target not in jud:
            continue
        n_cand += 1
        t = tier(man, jud)
        if t is None:
            n_untiered += 1
        else:
            ranked.append((t, int(xid)))
    ranked.sort()
    counts = {"n_candidates": n_cand, "n_untiered": n_untiered, **{f"n_tier{k}": sum(t == k for t, _ in ranked) for k in (1, 2, 3)}}
    return ranked, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = task_dir(cfg)
    out = outdir / OUT
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it.")
    for p in POLICIES:
        if not (outdir / judge_name(p)).exists():
            raise SystemExit(f"{judge_name(p)} not imported yet")
    df = load_frame(outdir)
    labels = pd.read_csv(outdir / LABELS_CSV, dtype=str, keep_default_na=False)
    manual = expand_labels(labels, json.loads((outdir / KEY).read_text(encoding="utf-8")))
    m = manual.merge(df, on=["xstest_id", "policy"], how="left", validate="one_to_one")
    if m["label"].isna().any():
        raise ValueError("audit rows without a judge label")

    result = {"script": "task4_safety.qualitative", "rule": __doc__.strip(), "n_per_behaviour": N_PER_BEHAVIOUR, "behaviours": {}}
    for target in TARGETS:
        ranked, counts = rank(m, target)
        picks = []
        for t, xid in ranked[:N_PER_BEHAVIOUR]:
            g = m[m["xstest_id"] == xid].set_index("policy").loc[list(POLICIES)]
            first = g.iloc[0]
            picks.append({"xstest_id": xid, "tier": t, "prompt": first["prompt"], "benchmark_class": first["benchmark_class"],
                          "type": first["type"],
                          "policies": {p: {"response": g.loc[p, "response"], "response_tokens": int(g.loc[p, "response_tokens"]),
                                           "truncated": bool(g.loc[p, "truncated"]), "manual_label": g.loc[p, "manual_label"],
                                           "judge_label": g.loc[p, "label"]} for p in POLICIES}})
        result["behaviours"][target] = {"counts": counts, "ranking": [{"tier": t, "xstest_id": x} for t, x in ranked], "selected": picks}
    save_json(out, result)
    print(f"wrote {out.name}: " + "; ".join(f"{k} {v['counts']}" for k, v in result["behaviours"].items()))


if __name__ == "__main__":
    main()
