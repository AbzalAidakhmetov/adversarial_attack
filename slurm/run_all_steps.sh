#!/bin/bash
# SLURM wrapper for scripts/run_all_steps.sh — 2-combo all-steps reproduction.
#
# Reuses steering_vector.pt from prior run_best.sh experiments when available,
# so the GCG attack only runs if those are missing. Time budget is set to 8 h
# to cover the worst case (both vectors need to be built from scratch).
#
# Submit with:
#   sbatch slurm/run_all_steps.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-run-all-steps
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=08:00:00
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

mkdir -p slurm experiments

bash scripts/run_all_steps.sh
