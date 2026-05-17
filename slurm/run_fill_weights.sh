#!/bin/bash
# SLURM wrapper for scripts/run_fill_weights.sh — submitted as a 5-task
# job array, one GPU per headline combo. Each task re-uses the combo's
# existing steering_vector.pt and runs ASR evals for the 2 missing
# steering weights in w ∈ {2,3,4}, so the headline grid matches the
# fill-matrix layout. Walltime is sized for the heaviest combo
# (Llama-3.1-8B: 8 evals @ ~20-25 min each ≈ 3-4 h).
#
# Submit with:
#   sbatch slurm/run_fill_weights.sh
# Subset:
#   sbatch --array=0,3 slurm/run_fill_weights.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-fill-weights
#SBATCH --output=./slurm/%x-%A_%a.out
#SBATCH --error=./slurm/%x-%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=04:00:00
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

mkdir -p slurm results

bash scripts/run_fill_weights.sh
