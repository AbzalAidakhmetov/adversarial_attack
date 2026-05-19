#!/bin/bash
# SLURM array wrapper for scripts/run_all_steps.py — one task per (model × attribute)
# cell. Reuses each cell's prefill steering_vector.pt and re-evaluates with the
# `all_steps` protocol at the lower weights configured in config/all_steps.yaml.
#
# Requires the matrix attack to have run first (steering_vector.pt per cell).
#
# Submit:    sbatch --array=0-7 slurm/run_all_steps.sh
# Subset:    sbatch --array=0,7 slurm/run_all_steps.sh
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-all-steps
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
#SBATCH --time=06:00:00
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

srun uv run python scripts/run_all_steps.py "$@"
