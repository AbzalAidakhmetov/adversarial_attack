#!/bin/bash
# SLURM array wrapper for scripts/run_defense.py — one task per (model × attribute)
# cell from config/defense.yaml (which extends config/matrix.yaml). Each task
# orthogonalizes that cell's steering_vector.pt and runs the eval sweep
# (weights × {clean,poisoned} × {harmful,harmless}) on the defended vector.
#
# Requires the matrix attack to have run first (steering_vector.pt per cell).
#
# Submit:    sbatch --array=0-7 slurm/run_defense.sh
# Subset:    sbatch --array=0,4 slurm/run_defense.sh
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-defense
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

srun uv run python scripts/run_defense.py "$@"
