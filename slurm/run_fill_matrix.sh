#!/bin/bash
# SLURM wrapper for scripts/run_fill_matrix.sh — submitted as a job array
# of size 3, one GPU per missing combo (gemma/lowercase, llama31/french,
# llama31/has_bold_only). Each array task runs one combo end-to-end (GCG
# attack + w ∈ {2,3,4} eval sweep), so walltime ≈ longest single combo
# (~4h for the Llama runs) instead of the sequential ~8h.
#
# Submit with:
#   sbatch slurm/run_fill_matrix.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-fill-matrix
#SBATCH --output=./slurm/%x-%A_%a.out
#SBATCH --error=./slurm/%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=8:00:00
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

bash scripts/run_fill_matrix.sh
