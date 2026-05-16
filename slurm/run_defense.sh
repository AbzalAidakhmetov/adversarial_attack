#!/bin/bash
# SLURM wrapper for scripts/run_defense.sh — orthogonalization defense eval.
#
# Reuses steering_vector.pt from run_best.sh experiments (no GCG re-run): the
# defense step is a cheap projection plus 4 ASR evals per combo. ~2-3 h is
# typical; 6 h walltime for headroom.
#
# Submit with:
#   sbatch slurm/run_defense.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-run-defense
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
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

mkdir -p slurm experiments

bash scripts/run_defense.sh
