#!/bin/bash
# SLURM wrapper for scripts/run_best.sh — the 5-combo headline reproduction.
#
# Heavy slot (Llama-3.1-8B, ~17 GB) and light slot (Gemma-2-2B, ~7 GB) drain
# their queues in parallel on a single GPU (~24 GB combined VRAM). On
# Leonardo's 64 GB A100 there is ample headroom. Expect ~5-6 h walltime;
# `time` is set to 12 h for a safety margin (still under the 24 h partition cap).
#
# Submit with:
#   sbatch slurm/run_best.sh
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-run-best
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU

set -uo pipefail
module load cuda/12.2

export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

# Point HF cache at the user's Leonardo cache (symlinked into $FAST) so the
# script's `HF_HOME=${HF_HOME:-/workspace/.hf_home}` default does not kick in.
export HF_HOME="${HF_HOME:-/leonardo_work/IscrC_TVU/dcrisost/.cache/huggingface}"

mkdir -p slurm results

bash scripts/run_best.sh
