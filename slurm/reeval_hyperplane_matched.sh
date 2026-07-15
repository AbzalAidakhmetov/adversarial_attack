#!/bin/bash
# Magnitude-matched hyperplane re-eval (scripts/reeval_hyperplane_matched.py).
# One task per (model × attribute × seed) cell — same grid/dispatch as
# run_hyperplane.sh, but eval-only (no attack): rescales each hyperplane vector
# to its mean_diff poisoned norm and re-evaluates at the matrix weights.
#   Gemma seed-0 cells:  sbatch --array=0-3 slurm/reeval_hyperplane_matched.sh seeds=[0]
#   Llama31 seed-0 cells: sbatch --array=4-7 slurm/reeval_hyperplane_matched.sh seeds=[0]
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-hyper-matched
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.out
#SBATCH --time=06:00:00
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
srun uv run python scripts/reeval_hyperplane_matched.py "$@"
