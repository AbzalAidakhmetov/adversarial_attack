#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=caa2-smoke
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.out
#SBATCH --time=00:50:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_SIMP

set -uo pipefail
module load cuda/12.2 2>/dev/null
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'
export HF_HOME=/leonardo_scratch/large/userexternal/dcrisost/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
set +u; source .env; set -u
export HF_TOKEN ANTHROPIC_API_KEY OPENAI_API_KEY TOGETHER_API_KEY

echo "############ nvidia-smi ############"; nvidia-smi -L
echo "############ STEP A: GPU slow pytest (Llama-2 equivalence + AB) ############"
uv run pytest tests/test_caa_sycophancy.py -m slow -v -p no:cacheprovider 2>&1 | grep -vE "warning:|LiteLLM"
A=${PIPESTATUS[0]}; echo "STEP A exit=$A"

echo "############ STEP C: CAA attack smoke (Llama-2, num_pairs=5, budget=150 → must move cos) ############"
rm -rf results/case_study/caa_sycophancy/llama2_7b_L13
uv run python scripts/run_caa_sycophancy.py \
  attack.num_pairs=5 attack.gcg_budget=150 attack.gcg_patience=150 \
  seeds=[0] multipliers=[1.0] harmful.limit=4 \
  judge.provider=together judge.model=meta-llama/Llama-3.3-70B-Instruct-Turbo 2>&1 | grep -vE "warning:|LiteLLM" | grep -E "cos\(target\)|Neighbors|Safe vocab|edits|STEP|Δ|wrote|diag|SKIP" | tail -30
C=${PIPESTATUS[0]}; echo "STEP C exit=$C"

echo "############ STEP D: summarize ############"
uv run python scripts/summarize_caa_sycophancy.py 2>&1 | grep -vE "warning:" | tail -8
D=${PIPESTATUS[0]}; echo "STEP D exit=$D"

echo "---- equivalence.json ----"; cat results/case_study/caa_sycophancy/llama2_7b_L13/seed0/equivalence.json 2>/dev/null
echo; echo "---- attack diagnostics ----"
python3 -c "import json;d=json.load(open('results/case_study/caa_sycophancy/llama2_7b_L13/seed0/attack_summary.json'));print('diag:',json.dumps(d['diagnostics']));print('cov:',json.dumps(d['coverage']))" 2>/dev/null

echo "############ preserve + clean ############"
cp -r results/case_study/caa_sycophancy/llama2_7b_L13 /leonardo_scratch/large/userexternal/dcrisost/caa2_smoke_artifacts 2>/dev/null
rm -rf results/case_study/caa_sycophancy/llama2_7b_L13
echo "SUMMARY: A=$A C=$C D=$D"; echo "SMOKE DONE"
