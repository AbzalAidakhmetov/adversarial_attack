#!/bin/bash
# SLURM array wrapper for the Meta-Llama-3.1-70B-Instruct cells of config/matrix.yaml.
# llama70b is the 4th model entry, so its 4 attrs x 3 seeds = 12 cells live at
# indices 36..47 in the (model x attr x seed) Cartesian product.
#
# Submit all 12 llama70b cells:
#   sbatch --array=36-47 slurm/run_llama70b.sh
# Single seed (seed=0 across the 4 attrs):
#   sbatch --array=36,39,42,45 slurm/run_llama70b.sh
#
# 70B in bf16 is ~141 GB of weights -> device_map=auto shards it across a full
# node of 4x A100 64GB (256 GB). Attack hyperparams match the large-model
# concessions used for Qwen-32B (n_candidates 64->32, eval_batch_size 8->2,
# gcg_budget 1500->800; patience=500 dominates convergence so the vector is
# unchanged). Override at submission, e.g. `sbatch ... attack.gcg_budget=1500`.
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-matrix-llama70b
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=480G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --account=IscrC_TVU

set -uo pipefail
module load cuda/12.2

export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

# Pin the WORK cache explicitly (not ${HF_HOME:-...}) so the job never inherits
# a SCRATCH HF_HOME from the submitting shell and re-downloads 141 GB.
export HF_HOME=/leonardo_work/IscrC_TVU/dcrisost/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
# Unbuffered stdout so the log shows real-time progress (model load, GCG iters).
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
