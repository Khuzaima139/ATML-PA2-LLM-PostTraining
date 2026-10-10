"""Task 2 blind qualitative file: KL-penalty forks beta 0 vs beta 0.20 on held-out prompts.

Selection rule (fixed in advance): the N_CANDIDATES held-out prompts with the largest paired RM gain
rm(beta 0) - rm(beta 0.20); ties broken by the lowest prompt_id. Runs on the Mac from saved
generation files only.

Writes:
  results/task2_ppo/qualitative_blind.md  prompt text and the two responses labelled A and B, no RM
                                          values and no condition names; candidates in held-out file order
  results/task2_ppo/qualitative_key.json  prompt ID, which label is beta 0, both RM scores
A/B order: numpy default_rng(seed), one coin per candidate in file order; 0 means A is beta 0.

--reveal: reads qualitative_labels.json (labels fixed on the blind file) and the key, and writes the
"qualitative" block of summary.json; every other key of summary.json is left unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.stats import align, prompt_key

BETA0, BETA020 = "fork_eps0p20_kl0p00", "fork_eps0p20_kl0p20"
N_CANDIDATES = 8


def fence(text: str) -> str:
    """Code fence longer than any backtick run inside text, so responses cannot break the markdown."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    f = "`" * max(3, longest + 1)
    return f"{f}text\n{text}\n{f}"


def select(rows0: list[dict], rows2: list[dict], k: int) -> list[int]:
    """Row positions of the k largest gains rm0 - rm2, ties by lowest prompt_id."""
    gain = [a["rm_score"] - b["rm_score"] for a, b in zip(rows0, rows2)]
    order = sorted(range(len(gain)), key=lambda i: (-gain[i], rows0[i]["prompt_id"]))
    return order[:k]


def selection_rule(doc: str) -> str:
    """The 'Selection rule' paragraph of the docstring saved in the key, on one line."""
    para = next(p for p in doc.split("\n\n") if p.startswith("Selection rule"))
    return " ".join(para.split("\n"))


def classify(label: str, beta0_label: str) -> str:
    return "same" if label == "same" else "agree" if label == beta0_label else "disagree"


def reveal(key: dict, labels: dict, results_dir: str, labels_commit: str) -> dict:
    """summary.json "qualitative" block: each label compared with the key."""
    by_number = {x["index"]: x for x in labels["labels"]}
    assert sorted(by_number) == [c["prompt_number"] for c in key["candidates"]], "labels and key cover different prompts"
    cands = []
    for c in key["candidates"]:
        lab = by_number[c["prompt_number"]]
        assert lab["prompt_id"] == c["prompt_id"], f"prompt {c['prompt_number']}: label and key prompt IDs differ"
        cands.append({"prompt_number": c["prompt_number"], "prompt_id": c["prompt_id"], "eval_index": c["eval_index"],
                      "label": lab["label"], "beta0_label": c["beta0_label"],
                      "rm_beta0": c["rm_beta0"], "rm_beta0p20": c["rm_beta0p20"],
                      "rm_gain": c["rm_beta0"] - c["rm_beta0p20"], "class": classify(lab["label"], c["beta0_label"])})
    return {
        "comparison": "KL forks beta 0 vs beta 0.20 (eps 0.20), held-out generations",
        "selection_rule": selection_rule(key["rule"]),
        "labels_file": f"{results_dir}/qualitative_labels.json",
        "labels_commit": labels_commit,
        "blind_file": f"{results_dir}/qualitative_blind.md",
        "class_definitions": {"agree": "label picks the beta 0 response", "disagree": "label picks the beta 0.20 response",
                              "same": "labelled same"},
        "counts": {k: sum(c["class"] == k for c in cands) for k in ("agree", "disagree", "same")},
        "candidates": cands,
    }


def run_reveal(args, cfg):
    io_dir = args.results_dir or cfg["results_dir"]
    summary_path = repo_path(f"{io_dir}/summary.json")
    summary = load_json(summary_path)
    if "qualitative" in summary and not args.overwrite:
        raise SystemExit("summary.json already has qualitative; pass --overwrite to replace it.")
    labels_rel = f"{cfg['results_dir']}/qualitative_labels.json"
    commit = subprocess.run(["git", "log", "-1", "--format=%h", "--abbrev=7", "--", labels_rel], capture_output=True,
                            text=True, cwd=repo_path("."), check=True).stdout.strip()
    if not commit:
        raise SystemExit(f"{labels_rel} is not committed; commit the labels before the reveal.")
    before = json.dumps({k: v for k, v in summary.items() if k != "qualitative"})
    summary["qualitative"] = reveal(load_json(repo_path(f"{io_dir}/qualitative_key.json")),
                                    load_json(repo_path(f"{io_dir}/qualitative_labels.json")), cfg["results_dir"], commit)
    assert json.dumps({k: v for k, v in summary.items() if k != "qualitative"}) == before
    save_json(summary_path, summary)
    print(f"wrote qualitative to {io_dir}/summary.json: {summary['qualitative']['counts']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--reveal", action="store_true", help="write summary.json's qualitative block from the labels and the key")
    ap.add_argument("--results-dir", help="--reveal only: directory read and written (default: config results_dir)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.reveal:
        return run_reveal(args, cfg)
    seed = int(cfg["seed"])
    results_dir = cfg["results_dir"]
    key_path = f"{results_dir}/qualitative_key.json"
    out_md, out_key = repo_path(f"{results_dir}/qualitative_blind.md"), repo_path(key_path)
    clash = [str(p) for p in (out_md, out_key) if p.exists()]
    if clash and not args.overwrite:
        raise SystemExit(f"{clash} exist; pass --overwrite to replace them.")

    rows0 = read_jsonl(repo_path(f"{results_dir}/generations_eval_{BETA0}.jsonl"))
    rows2 = read_jsonl(repo_path(f"{results_dir}/generations_eval_{BETA020}.jsonl"))
    rows0, rows2 = align(rows0, rows2, prompt_key, "beta 0 vs beta 0.20 generations")
    prompts = {r["prompt_id"]: r for r in read_jsonl(repo_path(cfg["paths"]["rl_prompt_eval"]))}

    picked = sorted(select(rows0, rows2, N_CANDIDATES), key=lambda i: rows0[i]["index"])
    coins = np.random.default_rng(seed).integers(0, 2, size=len(picked))

    lines = ["# Task 2 blind qualitative comparison", "",
             f"{len(picked)} held-out prompts. Each has two responses from two training conditions, labelled A and B "
             "in a seeded random order. Prompts are listed in held-out file order.", ""]
    key = []
    for n, (i, coin) in enumerate(zip(picked, coins), 1):
        r0, r2 = rows0[i], rows2[i]
        pid = r0["prompt_id"]
        a, b = (r0, r2) if coin == 0 else (r2, r0)
        msgs = prompts[pid]["messages"]
        prompt_text = "\n\n".join(m["content"] if len(msgs) == 1 else f"[{m['role']}]\n{m['content']}" for m in msgs)
        lines += [f"## Prompt {n}", "", f"Prompt ID: `{pid}`", "", fence(prompt_text), "",
                  f"### Response A ({a['token_length']} tokens)", "", fence(a["text"]), "",
                  f"### Response B ({b['token_length']} tokens)", "", fence(b["text"]), ""]
        key.append({"prompt_number": n, "prompt_id": pid, "eval_index": r0["index"],
                    "beta0_label": "A" if coin == 0 else "B",
                    "rm_beta0": r0["rm_score"], "rm_beta0p20": r2["rm_score"]})

    out_md.write_text("\n".join(lines), encoding="utf-8")
    save_json(out_key, {"script": "task2_ppo.qualitative", "seed": seed, "rule": __doc__, "candidates": key})
    print(f"wrote {len(picked)} blind candidates to {results_dir}/qualitative_blind.md and the key to {key_path}")


if __name__ == "__main__":
    main()
