#!/bin/bash
# One-off probe to diagnose the Qwen-32B load hang (see job 42502642_24).
# Runs scripts/probe_qwen_load.py with the exact env the real job uses.
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=probe-qwen
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
#SBATCH --time=00:30:00
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
# Unbuffered Python so we see exactly where the hang lands.
export PYTHONUNBUFFERED=1

set +u
source .env
set -u
export HF_TOKEN

mkdir -p slurm/logs
srun uv run python -u scripts/probe_qwen_load.py
