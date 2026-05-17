#!/bin/bash
# SLURM wrapper for scripts/run_detector.sh — cos detector + cos-cap bypass sweep.
#
# This is the longest job: 5 combos x 4 cos_max caps = 20 GCG attacks, each
# followed by 2 ASR evals. The static-detection (A) phase is cheap; the
# adaptive-attacker (B) sweep dominates the walltime. Set to the 24 h
# partition cap; re-run to pick up where it left off (the script is restartable).
#
# Submit with:
#   sbatch slurm/run_detector.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-run-detector
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=24:00:00
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

bash scripts/run_detector.sh
