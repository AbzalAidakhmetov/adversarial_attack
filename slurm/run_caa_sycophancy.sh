#!/bin/bash
# SLURM array wrapper for the CAA sycophancy case study — one GPU per attack seed.
#
#   sbatch --array=0-2 slurm/run_caa_sycophancy.sh
#
# Task k runs attack seed = seeds[k] (default seeds=[0,1,2]); task 0 additionally
# builds the clean vector and the unsteered/clean evals (so the shared,
# seed-independent generation + judging happen exactly once). Everything is
# skip-if-exists, so a failed task resumes on resubmit.
#
# After all seeds finish, add the other two judges and aggregate the majority:
#   uv run python scripts/rejudge_results.py --results_root results/case_study/caa_sycophancy \
#       --provider openai   --model gpt-4.1                               --tag gpt41
#   uv run python scripts/rejudge_results.py --results_root results/case_study/caa_sycophancy \
#       --provider together --model meta-llama/Llama-3.3-70B-Instruct-Turbo --tag llama70b
#   uv run python scripts/summarize_caa_sycophancy.py
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=caa-syco
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

export HF_HOME="${HF_HOME:-/leonardo_scratch/large/userexternal/dcrisost/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"

# Pull HF_TOKEN, ANTHROPIC_API_KEY, etc. into the env. `set +u` so optional
# vars don't abort under `set -u`.
set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY
export OPENAI_API_KEY
export TOGETHER_API_KEY

mkdir -p slurm/logs results

srun uv run python scripts/run_caa_sycophancy.py "$@"
