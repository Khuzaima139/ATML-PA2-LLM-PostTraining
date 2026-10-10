"""Statistics and result-file helpers shared by the summary scripts (Mac, from saved result files only).

Uncertainty: percentile bootstrap from the saved per-item outputs, N_RESAMPLES resamples, a fresh
numpy default_rng(seed) for every quantity, 95% interval = 2.5th and 97.5th percentiles.
Paired differences resample the same item indices for both conditions.
"""
from __future__ import annotations

import csv
import hashlib

import numpy as np

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json
from common.run_info import display_path

N_RESAMPLES = 10_000


# ---------------------------------------------------------------- descriptive

def describe(x) -> dict:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(a.min()),
        "max": float(a.max()),
    }


# ---------------------------------------------------------------- bootstrap core

def iid_resampler(n: int):
    """Resample n items with replacement."""
    return lambda rng: rng.integers(0, n, size=n)


def stratified_resampler(strata):
    """Resample with replacement inside each stratum; every stratum keeps its size."""
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    return lambda rng: np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])


def cluster_resampler(clusters):
    """Resample whole clusters with replacement; every item of a drawn cluster is kept together."""
    clusters = np.asarray(clusters)
    members = [np.flatnonzero(clusters == c) for c in sorted(set(clusters.tolist()))]

    def draw(rng):
        pick = rng.integers(0, len(members), size=len(members))
        return np.concatenate([members[k] for k in pick])
    return draw


def bootstrap(stat, n_items: int, resampler, seed: int, n_resamples: int = N_RESAMPLES, method: str = "") -> dict:
    """Point estimate on all items plus a 95% percentile interval. stat(index_array) -> float."""
    rng = np.random.default_rng(seed)
    point = float(stat(np.arange(n_items)))
    draws = np.array([stat(resampler(rng)) for _ in range(n_resamples)], dtype=float)
    finite = draws[np.isfinite(draws)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples,
            "n_nonfinite_resamples": int(draws.size - finite.size), "seed": seed, "method": method}


def two_sample_bootstrap(stat_a, n_a: int, stat_b, n_b: int, seed: int, n_resamples: int = N_RESAMPLES, method: str = "") -> dict:
    """stat_b - stat_a with each sample resampled independently; 95% percentile interval."""
    rng = np.random.default_rng(seed)
    point = float(stat_b(np.arange(n_b)) - stat_a(np.arange(n_a)))
    draws = []
    for _ in range(n_resamples):
        ia = rng.integers(0, n_a, size=n_a)
        ib = rng.integers(0, n_b, size=n_b)
        draws.append(stat_b(ib) - stat_a(ia))
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if finite.size else (float("nan"), float("nan"))
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples,
            "n_nonfinite_resamples": int(draws.size - finite.size), "seed": seed, "method": method}


# ---------------------------------------------------------------- statistics on index arrays

def mean_stat(x):
    x = np.asarray(x, dtype=float)
    return lambda idx: float(np.mean(x[idx]))


def pooled_ratio_stat(num, den):
    """Token-level pooled mean: sum of per-response summed log-ratios / sum of response lengths."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return lambda idx: float(num[idx].sum() / den[idx].sum())


def diff_stat(stat_a, stat_b):
    return lambda idx: stat_a(idx) - stat_b(idx)


# ---------------------------------------------------------------- alignment

def align(rows_a: list[dict], rows_b: list[dict], key, what: str):
    """Pair rows by key, in rows_a order. Fails unless both sides hold exactly the same unique keys."""
    ka, kb = [key(r) for r in rows_a], [key(r) for r in rows_b]
    if len(set(ka)) != len(ka) or len(set(kb)) != len(kb):
        raise SystemExit(f"{what}: duplicate IDs")
    if set(ka) != set(kb):
        raise SystemExit(f"{what}: ID sets differ ({len(set(ka) - set(kb))} only in first, {len(set(kb) - set(ka))} only in second)")
    by_b = {key(r): r for r in rows_b}
    return rows_a, [by_b[k] for k in ka]


def prompt_key(r):
    return r["prompt_id"]


# ---------------------------------------------------------------- held-out generation statistics (PPO and GRPO)

METRICS = ("rm", "kl", "entropy", "length")


def eval_stats(rows: list[dict]) -> dict:
    """Per-prompt arrays from a generations file, in file order."""
    return {
        "rm": mean_stat([r["rm_score"] for r in rows]),
        "kl": pooled_ratio_stat([r["sum_log_ratio"] for r in rows], [r["token_length"] for r in rows]),
        "entropy": pooled_ratio_stat([r["sum_neg_logp"] for r in rows], [r["token_length"] for r in rows]),
        "length": mean_stat([r["token_length"] for r in rows]),
    }


def boot(stat, n, seed, method):
    return bootstrap(stat, n, iid_resampler(n), seed, N_RESAMPLES, method)


def heldout_block(gens: dict[str, list[dict]], labels: dict, pairs, seed: int) -> dict:
    """Per-condition estimates and paired differences (a minus b) for every metric."""
    names = list(labels.values())
    base = gens[names[0]]
    aligned = {n: align(base, gens[n], prompt_key, f"held-out {names[0]} vs {n}")[1] for n in names}
    n = len(base)
    stats = {lab: eval_stats(aligned[name]) for lab, name in labels.items()}
    per = {str(lab): {m: boot(stats[lab][m], n, seed, f"{m}, {N_RESAMPLES} prompt resamples") for m in METRICS}
           for lab in labels}
    diffs = {f"{a}-{b}": {m: boot(diff_stat(stats[a][m], stats[b][m]), n, seed, f"paired {m} difference")
                          for m in METRICS} for a, b in pairs}
    return {"n_prompts": n, "conditions": {str(k): v for k, v in labels.items()}, "per_condition": per,
            "paired_differences": diffs}


def standard_heldout(e: dict, gens: list[dict], seed: int) -> dict:
    m = e["metrics"]
    st = eval_stats(gens)
    return {"rm_mean": m["reward"]["mean"], "rm_sd": m["reward"]["std"], "kl": m["kl"], "entropy": m["entropy"],
            "length_mean": m["length"]["mean"], "length_sd": m["length"]["std"],
            "truncation_rate_at_768": m["truncation_rate_at_cap"], "n_truncated_at_768": m["n_truncated_at_cap"],
            "cap": e["settings"]["max_new_tokens"], "n_prompts": len(gens),
            "bootstrap": {k: boot(st[k], len(gens), seed, k) for k in METRICS},
            "note": "for Task 4; not compared with forks (20 vs 8 updates)"}


# ---------------------------------------------------------------- result files

def write_csv(path, rows: list[dict]):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


class Inputs:
    """Reads result files, records their SHA-256, and fails clearly on anything missing or inconsistent."""

    def __init__(self, results_dir: str):
        self.dir = results_dir
        self.hashes: dict[str, str] = {}
        self.checks: list[dict] = []

    def path(self, name: str):
        return repo_path(f"{self.dir}/{name}")

    def require(self, names: list[str]):
        missing = [n for n in names if not self.path(n).exists()]
        if missing:
            raise SystemExit("missing required result files in " + self.dir + ":\n  " + "\n  ".join(missing))

    def _hash(self, p):
        self.hashes[display_path(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    def json(self, name: str):
        p = self.path(name)
        self._hash(p)
        return load_json(p)

    def jsonl(self, path: str):
        p = repo_path(path)
        if not p.exists():
            raise SystemExit(f"missing required file {path}")
        self._hash(p)
        return read_jsonl(p)

    def check(self, name: str, ok: bool, detail=None):
        self.checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise SystemExit(f"check failed: {name}: {detail}")
