#!/bin/bash
# Layer probe for Meta-Llama-3.1-70B-Instruct x spanish: load the model once
# (sharded across 4x A100 64GB via device_map=auto), run the GCG attack at
# L40, L45, L50 (50 / 56.25 / 62.5% of the 80-layer stack) and report cos lift
# per layer. L45 is the a-priori pick — the same 0.5625 mid-stack fraction as
# Llama-3.1-8B (L18/32) and Qwen2.5-32B (L36/64); the probe confirms it.
# Results land at results/llama70b/layer_probe/L<layer>/.
#
#   sbatch slurm/probe_llama70b.sh
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=probe-llama70b-layers
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
#SBATCH --time=06:00:00
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
export PYTHONUNBUFFERED=1

set +u
source .env
set -u
export HF_TOKEN

mkdir -p slurm/logs results

srun uv run python scripts/probe_layers.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --device_map auto \
    --pair_type spanish \
    --layers 40 45 50 \
    --gcg_budget 800 \
    --n_candidates 32 \
    --eval_batch_size 2 \
    --output_root results/llama70b/layer_probe
