"""Task 2 blind qualitative file: KL-penalty forks beta 0 vs beta 0.20 on held-out prompts.

Selection rule (fixed in advance): the N_CANDIDATES held-out prompts with the largest paired RM gain
rm(beta 0) - rm(beta 0.20); ties broken by the lowest prompt_id. Runs on the Mac from saved
generation files only.

Writes:
  results/task2_ppo/qualitative_blind.md  prompt text and the two responses labelled A and B, no RM
                                          values and no condition names; candidates in held-out file order
  notes/task2_qualitative_key.json        (gitignored) prompt ID, which label is beta 0, both RM scores
A/B order: numpy default_rng(seed), one coin per candidate in file order; 0 means A is beta 0.
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task1_dpo.summarize import align, prompt_key

BETA0, BETA020 = "fork_eps0p20_kl0p00", "fork_eps0p20_kl0p20"
N_CANDIDATES = 8
KEY_PATH = "notes/task2_qualitative_key.json"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    results_dir = cfg["results_dir"]
    out_md, out_key = repo_path(f"{results_dir}/qualitative_blind.md"), repo_path(KEY_PATH)
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
    print(f"wrote {len(picked)} blind candidates to {results_dir}/qualitative_blind.md and the key to {KEY_PATH}")


if __name__ == "__main__":
    main()
