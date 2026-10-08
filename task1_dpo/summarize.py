"""Task 1 summary tables, bootstrap intervals and qualitative candidates (manual Section 1, Required Evidence).

Runs on the Mac from saved result files only (no model is loaded). Writes, into the results dir:
  summary_steps12.json/.csv   standard DPO, the three beta forks and an SFT row (generation metrics only)
  summary_step3.json/.csv     standard vs length_balanced: per-stratum accuracy and word-limit results
  qualitative_candidates.json candidates selected by fixed rules; no quality judgement is made here

Uncertainty: percentile bootstrap from the saved per-item outputs, N_RESAMPLES resamples, a fresh
numpy default_rng(seed) for every quantity, 95% interval = 2.5th and 97.5th percentiles.
Paired differences resample the same item indices for both conditions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, wall_timer
from task1_dpo.ablate_beta import EXPECTED_LORA_A_SHA256
from task1_dpo.dataset_stats import describe
from task1_dpo.train import display_path, git_state

STRATA = ("preferred_longer", "length_matched", "rejected_longer")
FORKS = ("beta_0p03", "beta_0p10", "beta_0p30")
FORK_PAIRS = (("beta_0p03", "beta_0p10"), ("beta_0p10", "beta_0p30"), ("beta_0p03", "beta_0p30"))
N_RESAMPLES = 10_000
N_TOP_RM_GAIN = 10
N_WORDLIMIT_VIOLATIONS = 5
TOL = 1.0e-8  # recomputed point estimates must match the evaluation JSON


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


# ---------------------------------------------------------------- statistics on index arrays

def accuracy_stat(m):
    m = np.asarray(m, dtype=float)
    return lambda idx: float(np.mean(m[idx] > 0))


def mean_stat(x):
    x = np.asarray(x, dtype=float)
    return lambda idx: float(np.mean(x[idx]))


def pooled_ratio_stat(num, den):
    """Token-level pooled mean: sum of per-response summed log-ratios / sum of response lengths."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return lambda idx: float(num[idx].sum() / den[idx].sum())


def gap_stat(m, strata, hi: str = "preferred_longer", lo: str = "rejected_longer"):
    """Accuracy on stratum hi minus accuracy on stratum lo, for the items in idx."""
    m, strata = np.asarray(m, dtype=float), np.asarray(strata)

    def f(idx):
        s, x = strata[idx], m[idx]
        return float(np.mean(x[s == hi] > 0) - np.mean(x[s == lo] > 0))
    return f


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


def pair_key(r):
    return r["id"]


def prompt_key(r):
    return r["prompt_id"]


def sample_key(r):
    return (r["prompt_id"], r["sample_index"])


# ---------------------------------------------------------------- qualitative selection rules (fixed in advance)

def top_rm_gain(std_rows: list[dict], sft_rows: list[dict], k: int = N_TOP_RM_GAIN) -> list[dict]:
    """The k prompt IDs with the largest RM(standard) - RM(SFT); ties by prompt ID ascending."""
    a, b = align(std_rows, sft_rows, prompt_key, "standard vs sft generations")
    pairs = sorted(zip(a, b), key=lambda p: (-(p[0]["rm_score"] - p[1]["rm_score"]), str(p[0]["prompt_id"])))
    view = lambda r: {"text": r["text"], "rm_score": r["rm_score"], "token_length": r["token_length"], "truncated": r["truncated"]}
    return [{"rank": i + 1, "prompt_id": s["prompt_id"], "rm_standard_minus_sft": s["rm_score"] - f["rm_score"],
             "standard": view(s), "sft": view(f)} for i, (s, f) in enumerate(pairs[:k])]


def wordlimit_violations(rows: list[dict], k: int = N_WORDLIMIT_VIOLATIONS) -> list[dict]:
    """Up to k non-compliant responses by descending RM score; ties by (prompt ID, sample index)."""
    bad = [r for r in rows if r["compliant"] is not None and r["compliant"] == 0]
    bad.sort(key=lambda r: (-r["rm_score"], str(r["prompt_id"]), r["sample_index"]))
    keep = ("prompt_id", "sample_index", "limit", "word_count", "token_length", "rm_score", "truncated", "text")
    return [{"rank": i + 1, **{f: r[f] for f in keep}} for i, r in enumerate(bad[:k])]


def per_prompt_compliance(rows: list[dict]) -> dict:
    out = {}
    for pid in sorted({r["prompt_id"] for r in rows}, key=str):
        vals = [r["compliant"] for r in rows if r["prompt_id"] == pid]
        out[pid] = float(np.mean(vals))
    return out


def largest_compliance_gap(std_rows: list[dict], bal_rows: list[dict]) -> dict:
    """Word-limit prompt with the largest |compliance(standard) - compliance(balanced)|; ties: lowest prompt ID.

    Returns the prompt, both compliance rates and sample index 0 from each model.
    """
    align(std_rows, bal_rows, sample_key, "standard vs length_balanced word-limit generations")
    cs, cb = per_prompt_compliance(std_rows), per_prompt_compliance(bal_rows)
    pid = min(cs, key=lambda p: (-abs(cs[p] - cb[p]), str(p)))
    pick = lambda rows: next(r for r in rows if r["prompt_id"] == pid and r["sample_index"] == 0)
    keep = ("sample_index", "limit", "word_count", "compliant", "token_length", "rm_score", "truncated", "text")
    return {"prompt_id": pid, "compliance_standard": cs[pid], "compliance_length_balanced": cb[pid],
            "compliance_standard_minus_balanced": cs[pid] - cb[pid],
            "standard": {f: pick(std_rows)[f] for f in keep}, "length_balanced": {f: pick(bal_rows)[f] for f in keep}}


# ---------------------------------------------------------------- loading and protocol checks

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


def check_train(inp: Inputs, run: str, train: dict):
    inp.check(f"{run}: training completed", train["status"] == "completed", train["status"])
    inp.check(f"{run}: first micro-batch loss equals log 2", bool(train["sanity_first_microbatch"]["passed"]),
              train["sanity_first_microbatch"])
    inp.check(f"{run}: LoRA init digest", train["lora_init"]["lora_A_sha256"] == EXPECTED_LORA_A_SHA256,
              train["lora_init"]["lora_A_sha256"])


def eval_mode(inp: Inputs, ev: dict, mode: str, train: dict | None) -> dict:
    """The mode entry of an evaluation JSON, after checking it is a full run of the right adapter."""
    name = ev["name"]
    inp.check(f"eval {name}: has mode {mode}", mode in ev["modes"], sorted(ev["modes"]))
    e = ev["modes"][mode]
    inp.check(f"eval {name}:{mode}: full run (not smoke, no --limit)", not e["smoke"] and e["limit"] is None,
              {"smoke": e["smoke"], "limit": e["limit"]})
    if train is not None:
        same = repo_path(ev["adapter"]).resolve() == repo_path(train["adapter_path"]).resolve()
        inp.check(f"eval {name}: adapter is the trained {train['run_name']} adapter", same,
                  {"eval_adapter": ev["adapter"], "train_adapter": train["adapter_path"]})
        if mode in ("pairs", "stratified"):
            inp.check(f"eval {name}:{mode}: DPO loss beta equals training beta",
                      math.isclose(e["settings"]["beta"], train["effective"]["beta"]),
                      {"eval": e["settings"]["beta"], "train": train["effective"]["beta"]})
    return e


def check_same_settings(inp: Inputs, label: str, entries: dict[str, dict]):
    names = list(entries)
    first = entries[names[0]]["settings"]
    differ = [n for n in names[1:] if entries[n]["settings"] != first]
    inp.check(f"{label}: identical generation settings across {names}", not differ, {"differ_from_" + names[0]: differ})


def check_close(inp: Inputs, label: str, recomputed: float, saved: float):
    inp.check(f"{label} recomputed from per-item outputs", abs(recomputed - saved) <= TOL,
              {"recomputed": recomputed, "saved": saved})


# ---------------------------------------------------------------- tables

def budget_text(train: dict) -> str:
    n, steps = train["data"]["examples_used"], train["n_optimizer_steps"]
    if train["cli"]["max_examples"] is None:
        return f"standard one-epoch run: all {n} retained pairs, {steps} optimizer steps"
    return f"short-run fork: first {n} retained pairs, 1 epoch, {steps} optimizer steps"


def generation_columns(gen: dict) -> dict:
    m = gen["metrics"]
    return {
        "n_responses": m["n_responses"],
        "kl_token_pooled": m["kl"]["value"],
        "kl_computed": m["kl"]["computed"],
        "rm_mean": m["reward"]["mean"],
        "rm_std": m["reward"]["std"],
        "len_mean": m["length"]["mean"],
        "len_std": m["length"]["std"],
        "len_median": m["length"]["median"],
        "len_iqr": m["length"]["iqr"],
        "truncated_fraction": m["length"]["truncated_fraction"],
        "n_rm_input_truncated": m["n_rm_input_truncated"],
    }


def write_csv(path, rows: list[dict]):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def steps12(inp: Inputs, cfg: dict, seed: int, std_name: str, sft_name: str) -> tuple[dict, list[dict], dict]:
    runs = (std_name, *FORKS)
    trains = {r: inp.json(f"{r}_train.json") for r in runs}
    evals = {r: inp.json(f"eval_{r}.json") for r in runs}
    sft_eval = inp.json(f"eval_{sft_name}.json")
    for r in runs:
        check_train(inp, r, trains[r])
    pairs = {r: eval_mode(inp, evals[r], "pairs", trains[r]) for r in runs}
    gens = {r: eval_mode(inp, evals[r], "generate", trains[r]) for r in runs}
    gens[sft_name] = eval_mode(inp, sft_eval, "generate", None)
    inp.check(f"eval {sft_name}: untrained base model", sft_eval["adapter"] == "none", sft_eval["adapter"])
    check_same_settings(inp, "generate", gens)
    gen_rows = {r: inp.jsonl(e["generations_file"]) for r, e in gens.items()}

    # Per-item arrays, every condition aligned to the standard model's order.
    per_pair = {r: align(pairs[std_name]["per_pair"], pairs[r]["per_pair"], pair_key, f"pairs {std_name} vs {r}")[1] for r in runs}
    gen_al = {r: align(gen_rows[std_name], gen_rows[r], prompt_key, f"generations {std_name} vs {r}")[1] for r in gen_rows}
    m = {r: np.array([x["m"] for x in per_pair[r]]) for r in runs}
    rm = {r: np.array([x["rm_score"] for x in gen_al[r]]) for r in gen_al}
    ln = {r: np.array([x["token_length"] for x in gen_al[r]], dtype=float) for r in gen_al}
    lr = {r: np.array([x["summed_log_ratio"] for x in gen_al[r]]) for r in runs}
    n_pairs, n_prompts = len(per_pair[std_name]), len(gen_al[std_name])

    for r in runs:
        idx = np.arange(n_pairs)
        check_close(inp, f"{r} preference accuracy", accuracy_stat(m[r])(idx), pairs[r]["metrics"]["preference_accuracy"])
        check_close(inp, f"{r} mean margin", mean_stat(m[r])(idx), pairs[r]["metrics"]["mean_margin"])
        check_close(inp, f"{r} pooled KL", pooled_ratio_stat(lr[r], ln[r])(np.arange(n_prompts)), gens[r]["metrics"]["kl"]["value"])
    for r in gen_al:
        check_close(inp, f"{r} RM mean", float(rm[r].mean()), gens[r]["metrics"]["reward"]["mean"])
        check_close(inp, f"{r} length mean", float(ln[r].mean()), gens[r]["metrics"]["length"]["mean"])

    pair_rs, prompt_rs = iid_resampler(n_pairs), iid_resampler(n_prompts)
    iid = "percentile bootstrap, items resampled with replacement"
    paired = "paired percentile bootstrap: the same resampled indices for both conditions; difference = first minus second"
    unc = {"per_model": {}, "fork_differences": {}, "standard_minus_sft": {}}
    for r in runs:
        unc["per_model"][r] = {
            "preference_accuracy": bootstrap(accuracy_stat(m[r]), n_pairs, pair_rs, seed, method=iid + " (pairs)"),
            "mean_margin": bootstrap(mean_stat(m[r]), n_pairs, pair_rs, seed, method=iid + " (pairs)"),
        }
    for a, b in FORK_PAIRS:
        unc["fork_differences"][f"{a}_minus_{b}"] = {
            "preference_accuracy": bootstrap(diff_stat(accuracy_stat(m[a]), accuracy_stat(m[b])), n_pairs, pair_rs, seed, method=paired + " (pair IDs)"),
            "kl_token_pooled": bootstrap(diff_stat(pooled_ratio_stat(lr[a], ln[a]), pooled_ratio_stat(lr[b], ln[b])), n_prompts, prompt_rs, seed,
                                         method=paired + " (prompt IDs); KL recomputed as the pooled token mean on each resample"),
            "rm_mean": bootstrap(diff_stat(mean_stat(rm[a]), mean_stat(rm[b])), n_prompts, prompt_rs, seed, method=paired + " (prompt IDs)"),
            "len_mean": bootstrap(diff_stat(mean_stat(ln[a]), mean_stat(ln[b])), n_prompts, prompt_rs, seed, method=paired + " (prompt IDs)"),
        }
    unc["standard_minus_sft"] = {
        "rm_mean": bootstrap(diff_stat(mean_stat(rm[std_name]), mean_stat(rm[sft_name])), n_prompts, prompt_rs, seed, method=paired + " (prompt IDs)"),
        "len_mean": bootstrap(diff_stat(mean_stat(ln[std_name]), mean_stat(ln[sft_name])), n_prompts, prompt_rs, seed, method=paired + " (prompt IDs)"),
    }

    rows = []
    for r in runs:
        pm, pu = pairs[r]["metrics"], unc["per_model"][r]
        rows.append({
            "run": r,
            "budget": budget_text(trains[r]),
            "train_examples": trains[r]["data"]["examples_used"],
            "optimizer_steps": trains[r]["n_optimizer_steps"],
            "beta": trains[r]["effective"]["beta"],
            "n_pairs": pm["n"],
            "dpo_loss": pm["dpo_loss"],
            "dpo_loss_beta": pm["beta"],
            "preference_accuracy": pm["preference_accuracy"],
            "acc_ci_low": pu["preference_accuracy"]["ci_low"],
            "acc_ci_high": pu["preference_accuracy"]["ci_high"],
            "mean_margin": pm["mean_margin"],
            "m_ci_low": pu["mean_margin"]["ci_low"],
            "m_ci_high": pu["mean_margin"]["ci_high"],
            **generation_columns(gens[r]),
        })
    rows.append({"run": sft_name, "budget": "untrained base model (no DPO); generation metrics only",
                 **generation_columns(gens[sft_name])})
    table = {
        "budget_note": "standard is one epoch over all retained dpo_standard_train pairs; the beta forks are short runs "
                       "over the first short_ablation_examples retained pairs; the budgets differ",
        "kl_definition": "common.metrics.sampled_kl pooled over all generated response tokens (token-level mean); "
                         "the untrained base model's KL is 0 by definition and not computed",
        "dpo_loss_definition": "held-out DPO loss at each run's own training beta",
        "rows": rows,
    }
    return table, rows, unc


def step3(inp: Inputs, seed: int, std_name: str) -> tuple[dict, list[dict], dict, dict]:
    bal = "length_balanced"
    train_std = inp.json(f"{std_name}_train.json")
    train_bal = inp.json(f"{bal}_train.json")
    check_train(inp, bal, train_bal)
    ev = {std_name: inp.json(f"eval_{std_name}.json"), bal: inp.json(f"eval_{bal}.json")}
    trains = {std_name: train_std, bal: train_bal}
    strat = {r: eval_mode(inp, ev[r], "stratified", trains[r]) for r in ev}
    wl = {r: eval_mode(inp, ev[r], "wordlimit", trains[r]) for r in ev}
    check_same_settings(inp, "wordlimit", wl)
    wl_rows = {r: inp.jsonl(wl[r]["generations_file"]) for r in wl}
    for r in wl_rows:
        unparsed = [sample_key(x) for x in wl_rows[r] if x["compliant"] is None]
        inp.check(f"{r} word-limit: every limit parsed", not unparsed, unparsed)

    pp = {std_name: strat[std_name]["per_pair"]}
    pp[bal] = align(pp[std_name], strat[bal]["per_pair"], pair_key, "stratified pairs")[1]
    strata = np.array([x["stratum"] for x in pp[std_name]])
    inp.check("stratified pairs: same stratum labels for both models", strata.tolist() == [x["stratum"] for x in pp[bal]], None)
    inp.check("stratified pairs: expected strata", set(strata.tolist()) == set(STRATA), sorted(set(strata.tolist())))
    m = {r: np.array([x["m"] for x in pp[r]]) for r in pp}
    wa = {std_name: wl_rows[std_name], bal: align(wl_rows[std_name], wl_rows[bal], sample_key, "word-limit samples")[1]}
    comp = {r: np.array([x["compliant"] for x in wa[r]], dtype=float) for r in wa}
    ln = {r: np.array([x["token_length"] for x in wa[r]], dtype=float) for r in wa}
    clusters = [x["prompt_id"] for x in wa[std_name]]

    for r in m:
        check_close(inp, f"{r} stratified accuracy", accuracy_stat(m[r])(np.arange(len(m[r]))),
                    strat[r]["metrics"]["overall"]["preference_accuracy"])
        check_close(inp, f"{r} word-limit compliance", float(comp[r].mean()), wl[r]["metrics"]["compliance"])

    strat_rs, clus_rs = stratified_resampler(strata), cluster_resampler(clusters)
    strat_m = "stratified percentile bootstrap: pairs resampled with replacement within each stratum, stratum sizes fixed"
    clus_m = ("paired cluster bootstrap over prompt IDs: prompts resampled with replacement, all samples of a drawn prompt "
              "kept together, the same resample for both models; difference = standard minus length_balanced")
    unc = {
        "gap_preferred_longer_minus_rejected_longer": {
            r: bootstrap(gap_stat(m[r], strata), len(strata), strat_rs, seed, method=strat_m) for r in m},
        "gap_difference_standard_minus_balanced": bootstrap(
            diff_stat(gap_stat(m[std_name], strata), gap_stat(m[bal], strata)), len(strata), strat_rs, seed,
            method=strat_m + "; the same resampled pairs for both models; difference = standard minus length_balanced"),
        "wordlimit_compliance_standard_minus_balanced": bootstrap(
            diff_stat(mean_stat(comp[std_name]), mean_stat(comp[bal])), len(clusters), clus_rs, seed, method=clus_m),
        "wordlimit_len_mean_standard_minus_balanced": bootstrap(
            diff_stat(mean_stat(ln[std_name]), mean_stat(ln[bal])), len(clusters), clus_rs, seed, method=clus_m),
    }

    rows = []
    for r in (std_name, bal):
        sm, wm = strat[r]["metrics"], wl[r]["metrics"]
        row = {"run": r, "beta": strat[r]["settings"]["beta"], "n_stratified": sm["overall"]["n"],
               "stratified_accuracy": sm["overall"]["preference_accuracy"]}
        for s in STRATA:
            row[f"acc_{s}"] = sm["per_stratum"][s]["preference_accuracy"]
            row[f"n_{s}"] = sm["per_stratum"][s]["n"]
        g = unc["gap_preferred_longer_minus_rejected_longer"][r]
        row.update({"gap_pl_minus_rl": g["point"], "gap_ci_low": g["ci_low"], "gap_ci_high": g["ci_high"],
                    "wl_n_responses": wm["n_responses"], "wl_compliance": wm["compliance"]})
        for pid, v in wm["per_prompt"].items():
            row[f"wl_compliance_{pid}"] = v["compliance"]
        lstat = describe(ln[r])
        row.update({"wl_len_mean": lstat["mean"], "wl_len_std": lstat["std"], "wl_len_iqr": lstat["iqr"],
                    "wl_truncated_fraction": wm["length"]["truncated_fraction"], "wl_n_rm_input_truncated": wm["n_rm_input_truncated"]})
        rows.append(row)
    table = {"gap_definition": "accuracy on preferred_longer minus accuracy on rejected_longer",
             "length_definition": "generated response tokens (common.generation response mask), word-limit prompts, all samples",
             "rows": rows}
    return table, rows, unc, wl_rows


def qualitative(inp: Inputs, cfg: dict, std_name: str, sft_name: str, wl_rows: dict) -> dict:
    gen = lambda name: inp.jsonl(load_json(inp.path(f"eval_{name}.json"))["modes"]["generate"]["generations_file"])
    eval_prompts = {r["prompt_id"]: r["prompt"] for r in inp.jsonl(cfg["paths"]["dpo_standard_eval"])}
    wl_prompts = {r["prompt_id"]: [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
                  for r in inp.jsonl(cfg["paths"]["word_limit_prompts"])}
    top = top_rm_gain(gen(std_name), gen(sft_name))
    for item in top:
        item["prompt"] = eval_prompts[item["prompt_id"]]
    viol = wordlimit_violations(wl_rows[std_name])
    for item in viol:
        item["prompt"] = wl_prompts[item["prompt_id"]]
    gap = largest_compliance_gap(wl_rows[std_name], wl_rows["length_balanced"])
    gap["prompt"] = wl_prompts[gap["prompt_id"]]
    return {
        "note": "candidates selected by fixed rules; no quality judgement or ranking was made",
        "rules": {
            "i_rm_gain": f"the {N_TOP_RM_GAIN} dpo_standard_eval prompt IDs with the largest RM({std_name}) minus RM({sft_name}); ties by prompt ID ascending",
            "i_wordlimit_violations": f"up to {N_WORDLIMIT_VIOLATIONS} {std_name} word-limit responses with compliant == 0, by descending RM score; ties by (prompt ID, sample index)",
            "ii_compliance_gap": "the word-limit prompt with the largest absolute compliance difference between "
                                 f"{std_name} and length_balanced; ties: lowest prompt ID; sample index 0 from each model",
        },
        "i_rm_gain_vs_sft": top,
        "i_wordlimit_violations_standard": viol,
        "n_wordlimit_violations_standard": sum(1 for r in wl_rows[std_name] if r["compliant"] == 0),
        "ii_largest_compliance_gap": gap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--results-dir", help="default: config results_dir")
    ap.add_argument("--standard-name", default="standard", help="run/eval name of the standard model")
    ap.add_argument("--sft-name", default="sft", help="eval name of the untrained base model")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    elapsed = wall_timer()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    results_dir = args.results_dir or cfg["results_dir"]
    inp = Inputs(results_dir)
    outputs = ["summary_steps12.json", "summary_steps12.csv", "summary_step3.json", "summary_step3.csv", "qualitative_candidates.json"]
    clash = [o for o in outputs if inp.path(o).exists()]
    if clash and not args.overwrite:
        raise SystemExit(f"{clash} exist in {results_dir}; pass --overwrite to replace them.")
    std, sft, bal = args.standard_name, args.sft_name, "length_balanced"
    inp.require([f"{r}_train.json" for r in (std, *FORKS, bal)] + [f"eval_{r}.json" for r in (std, sft, *FORKS, bal)])

    meta = {"script": "task1_dpo.summarize", "config_path": args.config, "results_dir": results_dir, "git": git_state(),
            "seed": seed, "bootstrap": {"n_resamples": N_RESAMPLES, "seed": seed, "interval": "95% percentile (2.5th, 97.5th)",
                                        "rng": "a fresh numpy.random.default_rng(seed) for every quantity"}}
    t12, rows12, unc12 = steps12(inp, cfg, seed, std, sft)
    t3, rows3, unc3, wl_rows = step3(inp, seed, std)
    qual = qualitative(inp, cfg, std, sft, wl_rows)
    common = {**meta, "input_sha256": inp.hashes, "checks": inp.checks, "wall_clock_seconds": elapsed()}

    save_json(inp.path("summary_steps12.json"), {**common, "table": t12, "uncertainty": unc12})
    write_csv(inp.path("summary_steps12.csv"), rows12)
    save_json(inp.path("summary_step3.json"), {**common, "table": t3, "uncertainty": unc3})
    write_csv(inp.path("summary_step3.csv"), rows3)
    save_json(inp.path("qualitative_candidates.json"), {**common, **qual})
    print(f"{len(inp.checks)} checks passed; wrote {', '.join(outputs)} to {results_dir}")


if __name__ == "__main__":
    main()
