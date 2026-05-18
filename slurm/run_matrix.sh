#!/bin/bash
# SLURM array wrapper for scripts/run_matrix.py — one task per (model × attribute)
# cell from config/matrix.yaml. Each task runs: GCG attack (skip if vector
# exists) + ASR eval sweep over `weights × {clean,poisoned} × {harmful,harmless}`.
#
# Array size = |models| × |attributes|. The default config (2 × 4 = 8) submits as:
#   sbatch --array=0-7 slurm/run_matrix.sh
# Subset:
#   sbatch --array=0,4 slurm/run_matrix.sh
#
# Cell-index → (model, attr) is itertools.product(models, attributes), so with
# the default config:
#   0..3 = gemma × {spanish, french, has_bold_only, lowercase}
#   4..7 = llama31 × {spanish, french, has_bold_only, lowercase}
# (run `uv run python scripts/run_matrix.py --help` to override the matrix.)
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-matrix
#SBATCH --output=./slurm/logs/%x-%A_%a.out
#SBATCH --error=./slurm/logs/%x-%A_%a.err
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"

# Pull HF_TOKEN, ANTHROPIC_API_KEY, etc. into the env. `set +u` so optional
# vars don't abort under `set -u`.
set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

mkdir -p slurm/logs results

srun uv run python scripts/run_matrix.py
