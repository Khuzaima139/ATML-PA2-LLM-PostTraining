"""Task 3 figure from the committed standard-run result file (train_standard.json) only; saved to report/figures/.

  task3_standard_trajectories.png  standard run, one panel per logged quantity, x = update 1..20; updates whose
                                   generated tokens all received zero gradient are shaded in every panel
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common.data import load_yaml, repo_path  # noqa: E402
from common.logging_utils import load_json  # noqa: E402
from task2_ppo.plots import FIG_DIR, MUTED, SERIES, line, style  # noqa: E402
from task3_grpo.summarize import STANDARD, update_row  # noqa: E402

SHADE = {"masked": ("#9b9a96", "all completions masked (truncated)"),
         "uninformative": (SERIES[1], "uninformative group")}


def zero_gradient_updates(rows):
    """Updates where every generated token is zero-gradient, split by cause, read from the logged token counts."""
    masked = [r["update"] for r in rows if r["zero_grad_masked_truncation_tokens"] == r["generated_tokens"]]
    uninformative = [r["update"] for r in rows if r["zero_grad_uninformative_tokens"] == r["generated_tokens"]]
    return {"masked": masked, "uninformative": uninformative}


def standard_figure(rows, path):
    x = [r["update"] for r in rows]
    zero = zero_gradient_updates(rows)
    panels = [
        ("RM reward (group mean of 4)", "reward_mean"),
        ("sampled KL estimate, token mean", "sampled_kl"),
        ("Mean within-group reward std", "group_reward_std"),
        ("Uninformative group (0/1, one prompt)", "uninformative"),
        ("Policy loss", "policy_loss"),
        ("Policy grad norm (pre-clip)", "grad_norm_before_clip"),
        ("sampled-token entropy estimate", "entropy"),
        ("Mean completion length (tokens)", "length_mean"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), sharex=True)
    for ax, (title, k) in zip(axes.flat, panels):
        for cause, updates in zero.items():
            for u in updates:
                ax.axvspan(u - 0.4, u + 0.4, color=SHADE[cause][0], alpha=0.18, linewidth=0)
        if k == "uninformative":
            ax.plot(x, [int(not r["informative"]) for r in rows], color=SERIES[0], linestyle="none",
                    marker="o", markersize=4)
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.25, 1.25)
        else:
            line(ax, x, [r[k] for r in rows], SERIES[0])
        style(ax, title)
    for ax in axes[-1]:
        ax.set_xlabel("update", fontsize=8, color=MUTED)
        ax.set_xticks(range(1, len(x) + 1, 2 if len(x) > 10 else 1))
    handles = [Patch(color=c, alpha=0.35, label=f"zero-gradient update: {lab}") for c, lab in SHADE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    results_dir = load_yaml(args.config)["results_dir"]
    out = repo_path(f"{FIG_DIR}/task3_standard_trajectories.png")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it.")
    rows = [update_row(h) for h in load_json(f"{results_dir}/train_{STANDARD}.json")["history"]]
    zero = standard_figure(rows, out)
    print("zero-gradient updates:", ", ".join(f"{c} {u}" for c, u in zero.items()))
    print("wrote", out.relative_to(repo_path(".")))


if __name__ == "__main__":
    main()
