"""Run metadata shared by every task: git state, hardware, seed, dtype, start time, peak VRAM, paths."""
from __future__ import annotations

import datetime as dt
import platform
import subprocess

import torch

from common.data import repo_path


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo_path(".")).stdout.strip()
    # Untracked files (results being written) do not count as dirty.
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def hardware() -> dict:
    import peft
    import transformers
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    else:
        device = f"cpu ({platform.platform()})"
    return {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def run_metadata(cfg: dict) -> dict:
    return {
        "git": git_state(),
        "seed": int(cfg["seed"]),
        "hardware": hardware(),
        "dtype": cfg["dtype"],
        "start_time": dt.datetime.now().isoformat(timespec="seconds"),
    }


def display_path(path) -> str:
    """Repo-relative path when inside the repo, else absolute."""
    root = repo_path(".")
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def peak_vram_bytes():
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def dtype_report(model) -> dict:
    trainable = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    frozen = sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad})
    return {"trainable_param_dtypes": trainable, "frozen_param_dtypes": frozen}


def optimizer_report(opt) -> list[dict]:
    return [
        {"name": g.get("name"), "lr": g["lr"], "weight_decay": g["weight_decay"], "n_tensors": len(g["params"]),
         "n_params": sum(p.numel() for p in g["params"]), "dtypes": sorted({str(p.dtype) for p in g["params"]})}
        for g in opt.param_groups
    ]
