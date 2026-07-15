#!/bin/bash
# SLURM array wrapper for Qwen2.5-32B-Instruct cells of config/matrix.yaml.
# Qwen is the 3rd model entry, so its 4 attrs × 3 seeds = 12 cells live at
# indices 24..35 in the (model × attr × seed) Cartesian product.
#
# Submit all 12 Qwen cells:
#   sbatch --array=24-35 slurm/run_qwen.sh
# Single seed:
#   sbatch --array=24,27,30,33 slurm/run_qwen.sh   # seed=0 across 4 attrs
#
# Attack hyperparams are dialed down from the default matrix (n_candidates 64→32,
# eval_batch_size 8→2, gcg_budget 1500→800) to keep one 32B cell under ~6h on
# 2× A100 64GB. Override at submission with `sbatch ... attack.gcg_budget=1500`.
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-matrix-qwen
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --account=IscrC_SIMP

set -uo pipefail
module load cuda/12.2

export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

export HF_HOME="${HF_HOME:-/leonardo_work/IscrC_TVU/dcrisost/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
# Unbuffered stdout so log shows real-time progress (model load, GCG iters).
# Without this, a hang inside transformers/accelerate stays invisible for
# hours (cf. job 42502642_24, which hung silently after model download).
export PYTHONUNBUFFERED=1

set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

mkdir -p slurm/logs results

srun uv run python scripts/run_matrix.py \
    attack.n_candidates=32 \
    attack.eval_batch_size=2 \
    attack.gcg_budget=800 \
    "$@"
