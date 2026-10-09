"""Task 2 Steps 2 and 3: the matched short forks, run status, and Kaggle queue projection.

Five distinct fork runs, each fork_updates updates from the supplied midpoint with the same seed and
prompt sequence. The eps 0.20 / beta 0.10 fork serves both the clipping study (manual Step 2) and
the KL-pressure study (Step 3).

  --run       run the forks one after another, each in its own process (task2_ppo.continue_train)
  --report    one status line per Task 2 run JSON (no metric values)
  --project   projected full-run time per GPU queue from the smoke JSONs, and N_EVAL_GPU0
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys

from common.data import load_yaml, repo_path


def tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def fork_specs(cfg: dict) -> list[dict]:
    """eps sweep at the standard beta, beta sweep at the standard eps; the shared run appears once."""
    eps0, beta0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    specs = {}
    for eps in cfg["clip_values"]:
        specs.setdefault((float(eps), beta0), set()).add("clipping")
    for beta in cfg["kl_values"]:
        specs.setdefault((eps0, float(beta)), set()).add("kl")
    out = []
    for (eps, beta), studies in specs.items():
        out.append({"run_name": f"fork_eps{tag(eps)}_kl{tag(beta)}", "clip_epsilon": eps, "kl_beta": beta, "studies": sorted(studies)})
    # shared run last so the single-study forks finish first
    return sorted(out, key=lambda s: (len(s["studies"]), s["run_name"]))


def fork_command(config: str, spec: dict, updates: int, prefix: str = "") -> list[str]:
    return [sys.executable, "-m", "task2_ppo.continue_train", "--config", config, "--run-name", prefix + spec["run_name"],
            "--updates", str(updates), "--clip-epsilon", str(spec["clip_epsilon"]), "--kl-beta", str(spec["kl_beta"])]


def run_forks(config: str, cfg: dict, updates: int | None, prefix: str) -> None:
    """A failed fork is reported and the next one still runs; exit status 1 if any failed."""
    n = int(updates if updates is not None else cfg["fork_updates"])
    failed = []
    for spec in fork_specs(cfg):
        cmd = fork_command(config, spec, n, prefix)
        print(f"===== fork {spec['run_name']} studies={spec['studies']}: {' '.join(cmd[1:])}", flush=True)
        if subprocess.run(cmd, cwd=repo_path(".")).returncode != 0:
            failed.append(spec["run_name"])
            print(f"===== fork {spec['run_name']} FAILED", flush=True)
    if failed:
        print(f"failed forks: {failed}", flush=True)
        sys.exit(1)


def gib(x) -> str:
    return f"{x / 2**30:.2f}GiB" if x else "n/a"


def report(cfg: dict) -> None:
    for path in sorted(glob.glob(str(repo_path(cfg["results_dir"]) / "*.json"))):
        r = json.load(open(path))
        name = path.rsplit("/", 1)[-1]
        script = r.get("script", "?")
        line = f"{name}: status={r.get('status')} t={r.get('wall_clock_seconds', 0):.0f}s peak_vram={gib(r.get('peak_vram_bytes'))}"
        if script == "task2_ppo.continue_train":
            h = r.get("history", [])
            line += (f" updates={r.get('updates_completed', 0)}/{r['effective']['updates']} eps={r['effective']['clip_epsilon']}"
                     f" beta={r['effective']['kl_beta']} nonfinite_steps={r.get('n_nonfinite_steps')}"
                     f" eos={sum(x['n_eos'] for x in h)}/{r['effective']['responses_per_prompt'] * len(h)} truncated={sum(x['n_truncated'] for x in h)}"
                     f" gen_tokens={r.get('generated_tokens_total')}")
        elif script == "task2_ppo.evaluate":
            m = r.get("metrics", {})
            line += f" adapter={r.get('adapter')} n={m.get('n_responses')} eos={m.get('n_terminated_with_eos')} truncated_at_cap={m.get('n_truncated_at_cap')}"
        elif script == "task2_ppo.analyze_clipping":
            line += f" rows={r.get('n_rows_used')} checks={r.get('identity_checks')}"
        print(line)


# ---------------------------------------------------------------- projection

def _load(name: str, cfg: dict):
    p = repo_path(cfg["results_dir"]) / name
    return json.load(open(p)) if p.exists() else None


def project(cfg: dict, prefix: str = "smoke_") -> None:
    """Rough per-queue times from the smoke runs.

    Training: setup + updates * mean seconds per smoke update. Evaluation: setup + full batch count
    * smoke seconds per generation batch (a batch costs about the same at 4 or 16 prompts, the cap
    dominates) + reward time scaled by prompt count. Cached study: smoke time scaled by rows.
    GPU 0 evaluates the first k forks as they finish; k is chosen to minimize the longer queue.
    """
    specs = fork_specs(cfg)
    std = _load(f"train_{prefix}standard.json", cfg)
    fork = _load(f"train_{prefix}fork_eps{tag(float(cfg['clip_epsilon']))}_kl{tag(0.0)}.json", cfg)
    ev = _load(f"eval_{prefix}fork_eps{tag(float(cfg['clip_epsilon']))}_kl{tag(0.0)}.json", cfg)
    cache = _load(f"cached_clip_study_{prefix.rstrip('_')}.json", cfg)
    if not all([std, fork, ev, cache]):
        raise SystemExit("smoke JSONs missing; run the smoke cell first")
    upd = [h["seconds"]["total"] for r in (std, fork) for h in r["history"]]
    t_upd = sum(upd) / len(upd)
    t_setup = (std["setup_seconds"] + fork["setup_seconds"]) / 2
    n_eval = ev["prompts"]["filter"]["n_kept"]
    n_eval_smoke = ev["prompts"]["n_evaluated"]
    tm = ev["timing"]
    batches_full = -(-n_eval // ev["settings"]["gen_batch_size"])
    t_eval = tm["setup"] + batches_full * tm["generation_and_teacher_forcing"] / tm["n_generation_batches"] + tm["reward"] * n_eval / n_eval_smoke
    t_cache = cache["wall_clock_seconds"] * 32 / max(1, cache["n_rows_used"])
    t_std = t_setup + int(cfg["updates"]) * t_upd
    t_fork = t_setup + int(cfg["fork_updates"]) * t_upd
    stagger = 60.0
    finish = [stagger + (j + 1) * t_fork for j in range(len(specs))]
    best = None
    for k in range(len(specs) + 1):
        g0 = t_cache + t_std + t_eval
        for j in range(k):
            g0 = max(g0, finish[j]) + t_eval
        g1 = finish[-1] + (len(specs) - k) * t_eval
        if best is None or max(g0, g1) < max(best[1], best[2]):
            best = (k, g0, g1)
    k, g0, g1 = best
    print(f"per update ~{t_upd:.0f}s (smoke mean over {len(upd)} updates); train setup ~{t_setup:.0f}s")
    print(f"standard {int(cfg['updates'])} updates ~{t_std / 60:.1f} min; each fork ~{t_fork / 60:.1f} min; "
          f"each eval ({n_eval} prompts) ~{t_eval / 60:.1f} min; cached study ~{t_cache / 60:.1f} min")
    print(f"GPU0: cached + standard + eval standard + {k} fork evals ~{g0 / 60:.0f} min")
    print(f"GPU1: {len(specs)} forks + {len(specs) - k} fork evals ~{g1 / 60:.0f} min")
    print(f"N_EVAL_GPU0={k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--list", action="store_true", help="print one 'run_name eps beta studies' line per fork")
    ap.add_argument("--updates", type=int, help="override fork_updates (smoke only)")
    ap.add_argument("--prefix", default="", help="run-name prefix, e.g. smoke_")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.list:
        for s in fork_specs(cfg):
            print(s["run_name"], s["clip_epsilon"], s["kl_beta"], ",".join(s["studies"]))
    if args.run:
        run_forks(args.config, cfg, args.updates, args.prefix)
    if args.report:
        report(cfg)
    if args.project:
        project(cfg)
    if not (args.list or args.run or args.report or args.project):
        ap.print_help()


if __name__ == "__main__":
    main()
