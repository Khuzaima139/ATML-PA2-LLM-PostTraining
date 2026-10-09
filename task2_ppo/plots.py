"""Task 2 figures from committed result files (train_<run>.json) only; saved to report/figures/.

  task2_standard_trajectories.png  standard run, one panel per logged quantity, x = update 1..20
  task2_kl_forks_training.png      KL forks beta 0, 0.10, 0.20: per-update training KL and RM raw mean
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.data import load_yaml, repo_path  # noqa: E402
from common.logging_utils import load_json  # noqa: E402
from task2_ppo.summarize import KL_FORKS, STANDARD, update_row  # noqa: E402

FIG_DIR = "report/figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)


def line(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, linewidth=1.6, marker="o", markersize=3, label=label)


def standard_figure(rows, path):
    x = [r["update"] for r in rows]
    panels = [
        ("RM reward (mean of 4)", [("reward_raw_mean", "raw"), ("reward_effective_mean", "effective")]),
        ("Sampled KL to reference", [("kl", None)]),
        ("Policy loss (epoch 1)", [("policy_loss_ep1", None)]),
        ("Value loss (epoch 1)", [("value_loss_ep1", None)]),
        ("Sampled entropy", [("entropy", None)]),
        ("Policy grad norm (epoch 1, pre-clip)", [("policy_grad_norm_ep1", None)]),
        ("Clip fraction (epoch 2)", [("clip_fraction_ep2", None)]),
        ("Response length (tokens, mean of 4)", [("length_mean", None)]),
        ("Explained variance (symlog)", [("explained_variance", None)]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)
    for ax, (title, keys) in zip(axes.flat, panels):
        for i, (k, lab) in enumerate(keys):
            line(ax, x, [r[k] for r in rows], SERIES[i], lab)
        if len(keys) > 1:
            ax.legend(fontsize=7, frameon=False)
        if title.startswith("Explained"):
            ax.set_yscale("symlog", linthresh=1.0)
        style(ax, title)
    for ax in axes[-1]:
        ax.set_xlabel("update", fontsize=8, color=MUTED)
        ax.set_xticks(range(1, len(x) + 1, 2 if len(x) > 10 else 1))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def kl_forks_figure(rows_by_beta, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, (key, title) in zip(axes, [("kl", "Training sampled KL to reference"), ("reward_raw_mean", "Training RM reward (raw, mean of 4)")]):
        for i, (beta, rows) in enumerate(rows_by_beta.items()):
            line(ax, [r["update"] for r in rows], [r[key] for r in rows], SERIES[i], f"KL beta {beta:g}")
        style(ax, title)
        ax.set_xlabel("update", fontsize=8, color=MUTED)
        ax.set_xticks(range(1, 9))
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    results_dir = load_yaml(args.config)["results_dir"]
    outs = {"standard": repo_path(f"{FIG_DIR}/task2_standard_trajectories.png"),
            "kl": repo_path(f"{FIG_DIR}/task2_kl_forks_training.png")}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash and not args.overwrite:
        raise SystemExit(f"{clash} exist; pass --overwrite to replace them.")

    def rows(run):
        return [update_row(h) for h in load_json(f"{results_dir}/train_{run}.json")["history"]]

    standard_figure(rows(STANDARD), outs["standard"])
    kl_forks_figure({beta: rows(run) for beta, run in KL_FORKS.items()}, outs["kl"])
    print("wrote", ", ".join(str(p.relative_to(repo_path("."))) for p in outs.values()))


if __name__ == "__main__":
    main()
