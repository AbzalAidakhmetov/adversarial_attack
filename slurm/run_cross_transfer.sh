#!/bin/bash
# SLURM array wrapper for scripts/run_cross_transfer.py — one task per (combo × seed)
# cell from config/transfer.yaml. Each task attacks the SOURCE combo (skip if the
# summary exists), recomputes the vector on every target, and runs the eval sweep
# (+ native ceiling). Every step is skip-if-exists, so a re-run resumes.
#
# Array size = |combos| × |seeds|. Cell-index = itertools.product(combos, seeds),
# so seed varies fastest. With the default config (4 combos × 1 seed = 4):
#   sbatch --array=0-3 slurm/run_cross_transfer.sh
# Three seeds (4 × 3 = 12) — order is combo-major, seed-minor (0..2 = spanish s0..s2):
#   sbatch --array=0-11 slurm/run_cross_transfer.sh seeds=[0,1,2]
# Subset of cells:
#   sbatch --array=0,4 slurm/run_cross_transfer.sh seeds=[0,1,2]
# (run `uv run python scripts/run_cross_transfer.py --help` to override combos/seeds.)
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-transfer
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
#SBATCH --time=12:00:00
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

# Pull HF_TOKEN, API keys, etc. into the env. `set +u` so optional vars don't
# abort under `set -u`. TOGETHER_API_KEY is required by the default judge in
# config/evaluate_jailbreak.yaml (together/Llama-3.3-70B).
set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY
export TOGETHER_API_KEY

mkdir -p slurm/logs results

srun uv run python scripts/run_cross_transfer.py "$@"
