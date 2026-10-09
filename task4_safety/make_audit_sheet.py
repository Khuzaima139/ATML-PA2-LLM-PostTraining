from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task4_safety.judge_responses import JUDGE_PROMPT, LABELS
from task4_safety.protocol import POLICIES, generated_name

# Blind audit sheet (Task 4 Step 3). Built only from the generated_<policy>.jsonl files; this module never
# reads judge output. Prompts are numbered in a seeded shuffled order (xstest_id and type are hidden,
# since xstest_id order follows type blocks); byte-identical responses to one prompt are merged into one
# item; item letters come from a seeded shuffle. The key (item -> xstest_id, policies) is a separate file.
SHEET = "audit_sheet.md"
LABELS_CSV = "audit_labels.csv"
KEY = "audit_key.json"
RULES = "audit_rules.md"
LABEL_COLUMNS = ["item_id", "prompt_no", "prompt_class", "label"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def judge_label_definitions() -> list[str]:
    """The five label definition lines of the released JUDGE_PROMPT, verbatim."""
    lines = [ln for ln in JUDGE_PROMPT.splitlines() if ln.split(":", 1)[0] in LABELS]
    if len(lines) != len(LABELS):
        raise ValueError("could not find one definition line per label in JUDGE_PROMPT")
    return lines


def load_generations(outdir: Path) -> dict[str, list[dict]]:
    """generated_<policy>.jsonl for the four policies; all must share prompt order, prompt text and class."""
    gens = {p: read_jsonl(outdir / generated_name(p)) for p in POLICIES}
    ref = [(r["xstest_id"], r["prompt"], r["benchmark_class"]) for r in gens["sft"]]
    for p, rows in gens.items():
        if any(r["policy"] != p for r in rows):
            raise ValueError(f"{generated_name(p)} contains rows of another policy")
        if [(r["xstest_id"], r["prompt"], r["benchmark_class"]) for r in rows] != ref:
            raise ValueError(f"{generated_name(p)} differs from generated_sft.jsonl in prompt order, text or class")
    return gens


def build_items(gens: dict[str, list[dict]], audit_ids: list[int], seed: int):
    """Return (prompts, items, key).

    prompts: [{prompt_no, prompt, prompt_class}] in sheet order (seeded permutation of audit_ids).
    items:   [{item_id, prompt_no, response, truncated}] in sheet order; one item per distinct response text.
    key:     {"prompts": {prompt_no: xstest_id}, "items": {item_id: {"xstest_id", "policies"}}}.
    """
    rng = np.random.default_rng(seed)
    by_id = {p: {r["xstest_id"]: r for r in rows} for p, rows in gens.items()}
    order = [int(audit_ids[i]) for i in rng.permutation(len(audit_ids))]
    prompts, items, key = [], [], {"prompts": {}, "items": {}}
    for no, xid in enumerate(order, 1):
        base = by_id["sft"][xid]
        prompts.append({"prompt_no": no, "prompt": base["prompt"], "prompt_class": base["benchmark_class"]})
        key["prompts"][str(no)] = xid
        groups: dict[str, list[str]] = {}
        trunc: dict[str, bool] = {}
        for p in POLICIES:
            r = by_id[p][xid]
            groups.setdefault(r["response"], []).append(p)
            trunc[r["response"]] = trunc.get(r["response"], False) or bool(r.get("truncated", False))
        texts = list(groups)
        for letter, j in zip(string.ascii_uppercase, rng.permutation(len(texts))):
            text = texts[j]
            item_id = f"P{no:02d}{letter}"
            items.append({"item_id": item_id, "prompt_no": no, "response": text, "truncated": trunc[text]})
            key["items"][item_id] = {"xstest_id": xid, "policies": groups[text]}
    return prompts, items, key


def _fence(text: str) -> str:
    run = longest = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def render_sheet(prompts, items, definitions: list[str], rules_text: str, cap: int) -> str:
    out = ["# Task 4 manual audit sheet", "",
           "Label every item with one of the five labels in audit_labels.csv. Policy identity is hidden;",
           "identical responses to one prompt are shown once.", "",
           "## Judge label definitions (released JUDGE_PROMPT, verbatim)", ""]
    out += [f"    {d}" for d in definitions]
    out += ["", "## Audit rules", "", rules_text.strip(), ""]
    by_prompt: dict[int, list[dict]] = {}
    for it in items:
        by_prompt.setdefault(it["prompt_no"], []).append(it)
    for pr in prompts:
        out += [f"## Prompt {pr['prompt_no']:02d} ({pr['prompt_class']} class)", "", f"> {pr['prompt']}", ""]
        for it in by_prompt[pr["prompt_no"]]:
            fence = _fence(it["response"])
            out += [f"### {it['item_id']}", "", fence + "text", it["response"], fence]
            if it["truncated"]:
                out.append(f"(stopped at the {cap}-token cap without ending)")
            out.append("")
    return "\n".join(out)


def validate_labels(labels: pd.DataFrame, key: dict) -> list[str]:
    """Errors in a filled labels CSV: columns, one row per key item, every label filled and in LABELS."""
    errs = []
    if list(labels.columns) != LABEL_COLUMNS:
        return [f"columns {list(labels.columns)} != {LABEL_COLUMNS}"]
    ids = labels["item_id"].astype(str).tolist()
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errs.append(f"duplicate item_id: {dup}")
    missing, extra = sorted(set(key["items"]) - set(ids)), sorted(set(ids) - set(key["items"]))
    if missing:
        errs.append(f"items missing from CSV: {missing}")
    if extra:
        errs.append(f"items not in key: {extra}")
    lab = labels["label"].fillna("").astype(str).str.strip()
    empty = [i for i, v in zip(ids, lab) if not v]
    if empty:
        errs.append(f"{len(empty)} empty labels: {empty}")
    bad = [i for i, v in zip(ids, lab) if v and v not in LABELS]
    if bad:
        errs.append(f"{len(bad)} labels outside the five-label set: {bad}")
    return errs


def expand_labels(labels: pd.DataFrame, key: dict) -> pd.DataFrame:
    """Per-policy manual labels: a merged item's label is copied to every policy that produced it."""
    errs = validate_labels(labels, key)
    if errs:
        raise ValueError("; ".join(errs))
    rows = []
    for item_id, lab in zip(labels["item_id"].astype(str), labels["label"].astype(str).str.strip()):
        k = key["items"][item_id]
        rows += [{"xstest_id": int(k["xstest_id"]), "policy": p, "manual_label": lab, "item_id": item_id} for p in k["policies"]]
    out = pd.DataFrame(rows).sort_values(["xstest_id", "policy"]).reset_index(drop=True)
    if out.duplicated(["xstest_id", "policy"]).any():
        raise ValueError("a (xstest_id, policy) pair appears in two items")
    return out


def build(outdir: Path, per_class: int, seed: int, cap: int, overwrite: bool = False) -> dict:
    targets = [outdir / n for n in (SHEET, LABELS_CSV, KEY, "manual_audit_ids.csv")]
    for p in targets:
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it.")
    rules = outdir / RULES
    if not rules.exists():
        raise SystemExit(f"{rules} not found; write the audit rules first")
    gens = load_generations(outdir)
    ids = fixed_audit_ids(gens["sft"], per_class, seed)
    prompts, items, key = build_items(gens, ids, seed)
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    (outdir / SHEET).write_text(render_sheet(prompts, items, judge_label_definitions(), rules.read_text(encoding="utf-8"), cap), encoding="utf-8")
    cls = {p["prompt_no"]: p["prompt_class"] for p in prompts}
    pd.DataFrame([{"item_id": it["item_id"], "prompt_no": it["prompt_no"], "prompt_class": cls[it["prompt_no"]], "label": ""}
                  for it in items], columns=LABEL_COLUMNS).to_csv(outdir / LABELS_CSV, index=False)
    key["rule"] = {"audit_ids": f"fixed_audit_ids(generated_sft.jsonl rows, per_class={per_class}, seed={seed})",
                   "shuffle": f"one numpy default_rng({seed}): prompt order permutation, then per prompt a permutation of its distinct responses",
                   "merge": "responses with identical text within a prompt are one item"}
    key["audit_ids"] = ids
    (outdir / KEY).write_text(json.dumps(key, indent=2), encoding="utf-8")
    return {"n_prompts": len(prompts), "n_items": len(items), "n_per_policy_labels": sum(len(v["policies"]) for v in key["items"].values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--validate", action="store_true", help="check the filled audit_labels.csv against the key")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    if args.validate:
        labels = pd.read_csv(outdir / LABELS_CSV, dtype=str, keep_default_na=False)
        key = json.loads((outdir / KEY).read_text(encoding="utf-8"))
        errs = validate_labels(labels, key)
        if errs:
            raise SystemExit("labels invalid:\n  " + "\n  ".join(errs))
        n = len(expand_labels(labels, key))
        # Counts only: no label values, no policy mapping.
        print(f"labels valid: items={len(labels)} prompts={len(key['prompts'])} per_policy_labels={n}")
        return
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    c = build(outdir, int(cfg["manual_audit_per_class"]), int(cfg["seed"]), int(cfg["safety_max_new_tokens"]), args.overwrite)
    print(f"Wrote {SHEET}, {LABELS_CSV}, {KEY} (do not open the key until labels are committed), manual_audit_ids.csv: "
          f"prompts={c['n_prompts']} items={c['n_items']} per_policy_labels={c['n_per_policy_labels']}")
    print("Label without viewing AI labels first.")


if __name__ == "__main__":
    main()
