#!/bin/bash
# SLURM array wrapper for scripts/run_hyperplane.py — one task per
# (model × attribute × seed) cell from config/hyperplane.yaml. Each task runs:
# GCG attack with the hyperplane (RepE PCA reading-direction) steering method
# (skip if vector exists) + ASR eval sweep over
# `weights × {clean,poisoned} × {harmful,harmless}`. Outputs go to a per-cell
# `hyperplane/` subdir, so this never collides with the mean-difference matrix.
#
# Array size = |models| × |attributes| × |seeds|. The default config
# (2 models × 4 attrs × 3 seeds = 24) submits as:
#   sbatch --array=0-23 slurm/run_hyperplane.sh
# Seed varies fastest (see advsteer.orchestration.iter_cells), so cells 0..2 are
# gemma × spanish × {seed0,1,2}, 3..5 gemma × french × {seed0,1,2}, ...
# Just seed 0 (2 × 4 = 8 cells):
#   sbatch --array=0-7 slurm/run_hyperplane.sh seeds=[0]
# Subset of cells:
#   sbatch --array=0,3 slurm/run_hyperplane.sh
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-hyperplane
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

srun uv run python scripts/run_hyperplane.py "$@"
