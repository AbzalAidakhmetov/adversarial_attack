#!/bin/bash
# SLURM array wrapper for scripts/run_partial_control.py — partial-dataset-control
# ablation (Y355 Comment 1). One task per (model × attribute × seed) cell from
# config/partial_control.yaml. Each task runs, for every f in cfg.control_fracs:
# GCG attack with --control_frac f (skip if vector exists) + poisoned ASR/hAttr
# eval at the per-attribute weight, plus one clean baseline per cell.
#
# Array size = |models| × |attributes| × |seeds|. The default config
# (1 × 2 × 3 = 6) submits as:
#   sbatch --array=0-5 slurm/run_partial_control.sh
# Subset (e.g. just spanish, all seeds):
#   sbatch --array=0,1,2 slurm/run_partial_control.sh
#
# Cell-index → (model, attr, seed) is itertools.product(models, attributes,
# seeds) with seed varying fastest, so with the default config:
#   0..2 = gemma × spanish       × {0,1,2}
#   3..5 = gemma × has_bold_only  × {0,1,2}
# (run `uv run python scripts/run_partial_control.py --help` to override the grid.)
#
# Judging follows the same flow as every other job in this repo (run_matrix,
# run_defense, ...): the GPU eval scores ASR with the default judge (Anthropic
# Sonnet), and the 3-judge majority is a separate post-processing pass —
# scripts/rejudge_results.py (gpt41, llama70b) + scripts/aggregate_majority_judge.py
# — which is API-only (no GPU) and runs on a login node, not under SLURM.
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-partial
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_SIMP

set -uo pipefail
module load cuda/12.2

export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

export HF_HOME="${HF_HOME:-/leonardo_work/IscrC_TVU/dcrisost/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"

# Pull HF_TOKEN, ANTHROPIC_API_KEY, etc. into the env. `set +u` so optional
# vars don't abort under `set -u`.
set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

mkdir -p slurm/logs results

srun uv run python scripts/run_partial_control.py "$@"
