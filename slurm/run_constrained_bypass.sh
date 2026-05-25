#!/bin/bash
# SLURM array wrapper for scripts/run_constrained_bypass.py — constrained
# adaptive-attacker (hard cos cap) sweep, one task per (cell, seed) from
# config/constrained_bypass.yaml. Default: 1 model × 1 attribute × 3 seeds = 3
# tasks; each task loops over 5 caps so walltime is sized generously.
#
# Depends on scripts/run_matrix.py having been run first for the same
# (cell, seed) — the runner reads cos_clean from the unconstrained summary.json
# to skip caps below the honest clean cosine.
#
# Submit:    sbatch --array=0-2 slurm/run_constrained_bypass.sh
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-bypass
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU

set -uo pipefail
module load cuda/12.2

export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

export HF_HOME="${HF_HOME:-/leonardo_work/IscrC_TVU/dcrisost/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"

set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

mkdir -p slurm/logs results

srun uv run python scripts/run_constrained_bypass.py "$@"
