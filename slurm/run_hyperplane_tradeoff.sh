#!/bin/bash
# SLURM array wrapper for scripts/run_hyperplane_tradeoff.py — one task per
# (model × attribute × cos_max cap × seed) cell from config/hyperplane_tradeoff.yaml.
# Each task: capped hyperplane attack + magnitude-matched poisoned eval (ASR + hAttr).
#   Default grid 1×2×4×1 = 8:  sbatch --array=0-7 slurm/run_hyperplane_tradeoff.sh
# Cell index = itertools.product(models, attributes, caps, seeds), seed fastest.
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-hyper-tradeoff
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.out
#SBATCH --time=04:00:00
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
set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

mkdir -p slurm/logs results
srun uv run python scripts/run_hyperplane_tradeoff.py "$@"
