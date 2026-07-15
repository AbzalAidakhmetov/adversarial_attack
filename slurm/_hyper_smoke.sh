#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=hyper-smoke
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.out
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_SIMP

set -uo pipefail
module load cuda/12.2
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'
export HF_HOME="${HF_HOME:-/leonardo_work/IscrC_TVU/dcrisost/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
set +u
source .env
set -u
export HF_TOKEN

mkdir -p slurm/logs
OUT="$SCRATCH/adversarial_attack/hyper_smoke"
mkdir -p "$OUT"

echo "### smoke: hyperplane attack path (out=$OUT) ###"
srun uv run python -m advsteer.attack.build_adv_stealth \
  --model google/gemma-2-2b-it --layer 14 --pair_type spanish --num_pairs 8 \
  --steer_method hyperplane --gcg_budget 60 --gcg_patience 40 \
  --n_candidates 16 --n_modify 5 --n_neighbors 100 --lambda_lm 0.2 --max_perp 2000 \
  --eval_batch_size 8 --dtype bfloat16 \
  --output "$OUT/summary.json"

echo "### validating output ###"
uv run python - "$OUT" <<'PY'
import sys, json, torch
d = sys.argv[1]
s = json.load(open(f"{d}/summary.json"))
v = torch.load(f"{d}/steering_vector.pt", map_location="cpu", weights_only=False)
pois = v["steering_vector_poisoned"]
print("steer_method   :", s["config"]["steer_method"])
print("cos_clean      :", round(s["cos_clean"], 4))
print("cos_poisoned   :", round(s["cos_poisoned"], 4))
print("delta_cos      :", round(s["delta_cos"], 4))
print("poisoned |v|   :", round(pois.norm().item(), 6), "(hyperplane -> expect ~1.0)")
print("clean |v|      :", round(v["steering_vector_clean"].norm().item(), 6))
print("n_edits        :", s["n_total_modifications"], "over", s["n_texts_modified"], "texts")
ok = (s["config"]["steer_method"] == "hyperplane"
      and abs(pois.norm().item() - 1.0) < 1e-3
      and s["cos_poisoned"] > s["cos_clean"])
print("SMOKE:", "PASS" if ok else "FAIL")
PY
