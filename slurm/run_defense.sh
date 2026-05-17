#!/bin/bash
# SLURM wrapper for scripts/run_defense.sh — submitted as a job array of size
# 8, one GPU per combo (4 Llama heavy + 4 Gemma light combos, in the order
# HEAVY_COMBOS then LIGHT_COMBOS from scripts/run_defense.sh).
#
# Each array task orthogonalizes the existing steering_vector.pt for one
# combo and then runs 4 ASR evals (clean/poisoned × harmful/harmless on the
# defended vector). No GCG re-run. Walltime ≈ longest single combo's evals
# (~2-3 h on Llama-8B); 4 h walltime for headroom.
#
# Submit with:
#   sbatch slurm/run_defense.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-defense
#SBATCH --output=./slurm/%x-%A_%a.out
#SBATCH --error=./slurm/%x-%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=4:00:00
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

bash scripts/run_defense.sh
