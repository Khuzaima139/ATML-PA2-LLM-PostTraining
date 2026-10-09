"""Task 4 constants and small helpers shared by generation, judging, the audit sheet and the summary.

Fixed settings:
  - Policies: SFT has no adapter; DPO, PPO and GRPO adapters are passed on the command line and their
    adapter_model.safetensors must match EXPECTED_ADAPTER_SHA256 before anything is loaded.
  - Files: generations in results/task4_safety/generated_<policy>.jsonl (release name), judge outputs
    sealed in outputs/task4_safety/sealed/judge_<policy>.jsonl until the manual labels are committed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from common.data import repo_path

POLICIES = ("sft", "dpo", "ppo", "grpo")
ADAPTER_WEIGHTS = "adapter_model.safetensors"

# sha256 of adapter_model.safetensors of the final standard DPO, PPO and GRPO adapters (Tasks 1 to 3).
EXPECTED_ADAPTER_SHA256 = {
    "dpo": "81d99c57de5907c67e227bd0d8b1f8dee894e20e00f3f37e45b9efd491b0160b",
    "ppo": "bb0bb7e85f6f059724f8c3b31a42e30947fdece0d055f012052eff8cc05183f4",
    "grpo": "30247b5efdc0b88cf7247f36f4d69cf95230fb39b8e341ab7bc5dd06f5ab0c21",
}

GEN_BATCH_SIZE = 16
CLASSES = ("SAFE", "UNSAFE")
N_PER_CLASS = {"SAFE": 250, "UNSAFE": 200}

# Safe types whose prompts use violent or harmful-sounding wording, versus the other six safe types.
VIOLENT_WORDING_TYPES = ("homonyms", "figurative_language", "safe_targets", "safe_contexts")

# Disagreement map: each judge label as (prompt class, behaviour). AMBIGUOUS has no behaviour and is its own class.
LABEL_MAP = {
    "SAFE_ANSWER": ("safe", "comply"),
    "OVER_REFUSAL": ("safe", "refuse"),
    "UNSAFE_COMPLIANCE": ("unsafe", "comply"),
    "JUSTIFIED_REFUSAL": ("unsafe", "refuse"),
}
AMBIGUOUS = "AMBIGUOUS"


def task_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"]) / "task4_safety"


def sealed_dir() -> Path:
    return repo_path("outputs/task4_safety/sealed")


def generated_name(policy: str) -> str:
    return f"generated_{policy}.jsonl"


def judge_name(policy: str) -> str:
    return f"judge_{policy}.jsonl"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_adapter(policy: str, adapter: str | None) -> str | None:
    """Return the adapter weights sha256 after checking it against 4a; raise on any mismatch.

    SFT must have no adapter; every other policy must have one whose hash matches exactly.
    """
    if policy == "sft":
        if adapter:
            raise SystemExit("sft is the untouched base model; do not pass --adapter")
        return None
    if not adapter:
        raise SystemExit(f"{policy} needs --adapter")
    weights = repo_path(adapter) / ADAPTER_WEIGHTS
    if not weights.exists():
        raise SystemExit(f"{weights} not found")
    got = sha256_file(weights)
    if got != EXPECTED_ADAPTER_SHA256[policy]:
        raise SystemExit(f"{policy} adapter sha256 mismatch: {got} != {EXPECTED_ADAPTER_SHA256[policy]}; aborting")
    return got
