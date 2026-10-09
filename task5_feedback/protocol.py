"""Task 5 protocol: every fixed constant, hash and per-response rule, fixed before any Task 5 output.

Pure functions only (no model loading), so everything here is tested on the Mac.
"""
from __future__ import annotations

import hashlib
import re

from common.data import repo_path
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal

SEED = 6304
N_BOOT = 10_000

# ---------------------------------------------------------------- files and hashes (asserted at load)
# gsm8k_eval, diagnostics and adapters: manifests/sha256.json. SVAMP: not in the manifest; hash fixed
# before the first Task 5 run and asserted so the file is provably the one used unchanged.
FILE_SHA256 = {
    "data/gsm8k_eval.jsonl": "24f712750928c18c1a0a6849eebe97b3e01a6c0c0ebf2410d101a3df1872bf4a",
    "data/math_transfer_eval.jsonl": "f24163301035ccd4b66d12b687887fe1e167dd31ef62d6bc62a2fa152dcdcb76",
    "data/task5_controlled_reward_diagnostics.jsonl": "9008b96595bb1419f1326bf02beefd689f09028c2aa6598d7c427394602f036c",
    "checkpoints/rlvr_policy/adapter_model.safetensors": "b309564b45c0268e4e5ffb0db5cb100e557a2c407a68ea3fa3de91fe964d92c8",
    "checkpoints/rlaif_policy/adapter_model.safetensors": "e5a0eaacb3ce32a73fe86fd6a823be21c60a460eee8682493e5cc0f2e1019713",
}
DATASET_ROWS = {"gsm": 300, "transfer": 100}
N_DIAGNOSTIC_ROWS = 100
N_DIAGNOSTIC_PROBLEMS = 20
POLICIES = ("sft", "rlvr", "rlaif")
TRAINED = ("rlvr", "rlaif")

# ---------------------------------------------------------------- generation
SAMPLES_PER_PROMPT = 1
GEN_BATCH_SIZE = 16
PROMPT_TOKEN_CAP = 256  # assertion only; no prompt is filtered or truncated (max observed 239)

# ---------------------------------------------------------------- judge
RELEASED_JUDGE_MAX_NEW_TOKENS = 4  # hard-coded in rlaif.PairwiseAIJudge.compare; config judge_max_new_tokens unused
# Argument order of every judge call; the released hash orientation depends on it.
#   policy pairs:     compare(question, trained_response, sft_response)
#   diagnostic pairs: compare(question, clean_response, perturbed_response)
JUDGE_SCORE = {"A": 1.0, "TIE": 0.5, "B": 0.0}  # from the side of argument a

# ---------------------------------------------------------------- diagnostic pairs
ANCHOR = "clean_correct"
CATEGORIES = {
    "reasoning_corrupt": "corrupt_reasoning_correct_final",
    "wrong_final": "good_reasoning_wrong_final",
    "filler": "persuasive_filler_correct",
    "gold_distractor": "gold_distractor_wrong_final",
}
N_PAIRS = len(CATEGORIES) * N_DIAGNOSTIC_PROBLEMS  # 80
QUALITATIVE_CATEGORIES = ("reasoning_corrupt", "filler", "gold_distractor")

FAILURE_TYPES = (
    "correct",
    "wrong_compliant_gold_present",
    "wrong_compliant_gold_absent",
    "noncompliant_truncated",
    "noncompliant_ended",
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_sha(path: str) -> str:
    got = sha256_file(path)
    want = FILE_SHA256[str(path)]
    if got != want:
        raise SystemExit(f"sha256 mismatch for {path}: {got} != {want}")
    return got


def adapter_file(adapter_dir: str) -> str:
    return f"{adapter_dir.rstrip('/')}/adapter_model.safetensors"


def prompt_id(dataset: str, row: dict) -> str:
    """Stable ID per evaluation row: SVAMP has prompt_id; GSM8K uses its unique source_index."""
    if dataset == "transfer":
        return str(row["prompt_id"])
    return f"gsm8k:{row['source_index']}"


# ---------------------------------------------------------------- per-response rules
# A leading minus is a sign unless it directly follows a digit or ")" (then it is subtraction: "5-7" -> 5, 7).
_NUMBER_RE = re.compile(r"(?:(?<![\d)])-)?\d[\d,]*(?:\.\d+)?")


def gold_mentioned(text: str, gold) -> bool:
    """True if any number token in the text, sign included, equals gold under the verifier's normalization
    (commas stripped, numeric equality): "18.0" and "1,800" match 18 and 1800; "-18" does not match 18."""
    for tok in _NUMBER_RE.findall(str(text)):
        if numerically_equal(tok.replace(",", ""), gold):
            return True
    return False


def classify_response(text: str, gold, truncated: bool) -> dict:
    """Verifier score, compliance and failure type for one response.

    compliant: the released extractor returns a number. Failure types are checked in FAILURE_TYPES order;
    a response truncated at the cap that still contains a parsable final is classed by its final.
    """
    extracted = extract_designated_final(text)
    compliant = extracted is not None
    correct = exact_reward(text, gold) == 1.0
    if correct:
        ftype = "correct"
    elif compliant:
        ftype = "wrong_compliant_gold_present" if gold_mentioned(text, gold) else "wrong_compliant_gold_absent"
    elif truncated:
        ftype = "noncompliant_truncated"
    else:
        ftype = "noncompliant_ended"
    return {"extracted": extracted, "compliant": compliant, "correct": correct, "failure_type": ftype}


# ---------------------------------------------------------------- diagnostic pairs
def build_pairs(groups: dict) -> list[dict]:
    """80 pairs: (clean, variant) per problem and category, sorted by problem_id then category order."""
    pairs = []
    for pid in sorted(groups, key=int):
        g = groups[pid]
        for cat, variant in CATEGORIES.items():
            better, other = g[ANCHOR], g[variant]
            pairs.append({
                "pair_id": f"{int(pid)}:{cat}",
                "problem_id": int(pid),
                "category": cat,
                "better_variant": ANCHOR,
                "other_variant": variant,
                "question": better["question"],
                "gold_final": better["gold_final"],
                "better_response": better["response"],
                "other_response": other["response"],
            })
    return pairs


def reward_preference(r_better: float, r_other: float) -> str:
    if r_better > r_other:
        return "better"
    if r_better < r_other:
        return "wrong"
    return "tie"


def judge_preference(label: str, parse_matched: bool) -> str:
    """Judge label for compare(question, better, other) mapped to better / tie / wrong / parse_failure."""
    if not parse_matched:
        return "parse_failure"
    return {"A": "better", "B": "wrong", "TIE": "tie"}[label]


def qualitative_choice(judge_rows: list[dict]) -> dict:
    """Qualitative-example selection rule for one category. judge_rows: released-order rows with problem_id, label (released, parse
    failures already TIE), parse_matched. Branches in order, lowest problem_id within each: perturbed_preferred
    (parsed B), tie (genuine parsed TIE only; parse-failure ties never qualify), lowest_id (any row)."""
    rows = sorted(judge_rows, key=lambda r: r["problem_id"])
    for branch, keep in (("perturbed_preferred", lambda r: r["parse_matched"] and r["label"] == "B"),
                         ("tie", lambda r: r["parse_matched"] and r["label"] == "TIE"),
                         ("lowest_id", lambda r: True)):
        hit = [r for r in rows if keep(r)]
        if hit:
            r = hit[0]
            return {"problem_id": r["problem_id"], "pair_id": r.get("pair_id"), "branch": branch,
                    "judge_label": r["label"], "judge_parse_matched": r["parse_matched"],
                    "n_parse_failures_in_category": sum(1 for x in rows if not x["parse_matched"])}
    return {"problem_id": None, "branch": "none"}
