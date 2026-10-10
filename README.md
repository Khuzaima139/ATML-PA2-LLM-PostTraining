# ATML PA2: LLM Post-Training

Programming Assignment 2 of Advanced Topics in Machine Learning (EE-5102 / CS-6304), Fall 2026. The five tasks study how offline preferences, learned rewards, verifiable rewards and AI feedback change a language model during post-training.

| Task | Question | Data | Methods |
|---|---|---|---|
| 1 | How do beta and length confounding shape offline preference optimization? | DPO preference pairs (standard and length-balanced), word-limit prompts | DPO, beta in {0.03, 0.10, 0.30}, length-balanced DPO |
| 2 | How do clipping and KL pressure shape online PPO from a fixed midpoint? | RL prompt pool, cached midpoint rollout | PPO with a learned critic, clip and KL-beta forks |
| 3 | How do group size and normalization shape critic-free GRPO? | RL prompt pool, cached multi-sample groups | GRPO, Dr. GRPO, K in {2, 4, 8} |
| 4 | Do better preference and reward metrics give safer behaviour without over-refusal? | XSTest prompts | SFT, DPO, PPO, GRPO, AI judge with a manual audit |
| 5 | How do verifiable rewards and AI feedback compare as reward sources? | GSM8K, SVAMP transfer set, controlled reward diagnostics | RLVR, RLAIF, exact-match verifier, pairwise AI judge |

## Setup

```bash
conda create -n pa2 python=3.11 -y
conda activate pa2
pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

`requirements-lock.txt` pins the local versions (Python 3.11.17). GPU runs were on a Kaggle Tesla T4 in float16 with Python 3.13.15, torch 2.11.0+cu128, transformers 4.57.1 and peft 0.17.1, as recorded in the run JSONs (trl is not recorded there; `requirements.txt` pins 0.27.2). Run every command from the repository root. `data/`, `cached/`, `checkpoints/` and `outputs/` are not committed. Results are saved in `results/task*/` and figures in `report/figures/`.

## Task 1: Direct Preference Optimization

Training and evaluation run on a GPU; `ablate_beta` trains and evaluates the three 600-example beta forks and `analyze_length` the length-balanced run. `dataset_stats` and `summarize` run on the Mac from saved files.

```bash
python -m task1_dpo.dataset_stats --config configs/dpo.yaml
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --modes pairs,stratified,generate,wordlimit
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft --modes generate
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
python -m task1_dpo.summarize --config configs/dpo.yaml
```

## Task 2: Proximal Policy Optimization

Every run starts from the supplied PPO midpoint; training, evaluation and the cached clipping study run on a GPU. `summarize`, `qualitative`, `plots` and `posthoc` run on the Mac from saved files.

```bash
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard/policy --name standard
python -m task2_ppo.ablate_kl --config configs/ppo.yaml --run
for r in $(python -m task2_ppo.ablate_kl --config configs/ppo.yaml --list | awk '{print $1}'); do
    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/$r/policy --name $r
done
python -m task2_ppo.summarize --config configs/ppo.yaml
python -m task2_ppo.qualitative --config configs/ppo.yaml
python -m task2_ppo.plots --config configs/ppo.yaml
python -m task2_ppo.posthoc --config configs/ppo.yaml
```

## Task 3: Group Relative Policy Optimization

Every run starts from the supplied GRPO midpoint; training and evaluation run on a GPU. The group-size study uses the supplied cache and, like the summaries, runs on the Mac.

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard/policy --name standard
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name fork_grpo --updates 8 --loss-type grpo --length-diagnostic
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name fork_dr_grpo --updates 8 --loss-type dr_grpo --length-diagnostic
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/fork_grpo/policy --name fork_grpo
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/fork_dr_grpo/policy --name fork_dr_grpo
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.group_size_followup --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
python -m task3_grpo.summarize --config configs/grpo.yaml
```

## Task 4: Safety Calibration

Needs the standard DPO, PPO and GRPO adapters from Tasks 1 to 3 in `outputs/`; generation and judging run on a GPU, and the judge labels stay sealed in `outputs/task4_safety/sealed/` until the manual audit labels in `results/task4_safety/audit_labels.csv` are filled in and committed. The judge files are then copied into `results/task4_safety/` and committed on their own, which `evaluate_safety` checks before it runs on the Mac.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml --policy sft
python -m task4_safety.generate_responses --config configs/feedback.yaml --policy dpo --adapter outputs/task1_dpo/standard
python -m task4_safety.generate_responses --config configs/feedback.yaml --policy ppo --adapter outputs/task2_ppo/standard/policy
python -m task4_safety.generate_responses --config configs/feedback.yaml --policy grpo --adapter outputs/task3_grpo/standard/policy
for p in sft dpo ppo grpo; do
    python -m task4_safety.judge_responses --config configs/feedback.yaml --policy $p
done
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml --validate
cp outputs/task4_safety/sealed/judge_*.jsonl results/task4_safety/
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
python -m task4_safety.qualitative --config configs/feedback.yaml
python -m task4_safety.reward_pairs --config configs/feedback.yaml
```

## Task 5: Reward-Source Design

Uses the supplied RLVR and RLAIF adapters; generation, judging, the perturbation scores and the adapter-effect check run on a GPU, the rest on the Mac. The `--tag rerun` commands repeat the RLVR transfer evaluation post hoc under new file names, leaving the original files unchanged.

```bash
python -m scripts.prepare_transfer_eval
for ds in gsm transfer; do
    for p in sft rlvr rlaif; do
        python -m task5_feedback.evaluate_math --config configs/feedback.yaml --stage generate --dataset $ds --policy $p
    done
    python -m task5_feedback.evaluate_math --config configs/feedback.yaml --stage judge --dataset $ds
done
python -m task5_feedback.score_perturbations --config configs/feedback.yaml --order released
python -m task5_feedback.score_perturbations --config configs/feedback.yaml --order swapped
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
python -m task5_feedback.posthoc --config configs/feedback.yaml --part mac
python -m task5_feedback.posthoc --config configs/feedback.yaml --part adapter_effect
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --stage generate --dataset transfer --policy rlvr --tag rerun
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --stage judge --dataset transfer --policy rlvr --tag rerun
python -m task5_feedback.posthoc --config configs/feedback.yaml --part echoes
```

## Attribution

- Course starter code: [AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining); the files under `scripts/`, the shared modules `common/data.py`, `common/generation.py`, `common/logging_utils.py`, `common/metrics.py`, `common/models.py`, the configs and the task scaffolds come from it.
- Course assets (data, caches, midpoint and Task 5 checkpoints): [AbDu11aHHH/ATML-PA2-assets](https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets).
- Policy and reference model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- Reward model: [yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback](https://huggingface.co/yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback).
- AI judge: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- No other external code is materially reused.
