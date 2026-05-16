#!/bin/bash
# Per-combo SLURM array job for the run_best.sh headline reproduction.
#
# Each array task is one combo (model × attribute × layer × steering weight).
# A combo runs once: 1 GCG attack → 4 ASR evals (clean/poisoned × harmful/harmless).
# Skip-if-exists is honored, so re-submitting an array task picks up where it
# left off. To re-run a single combo from scratch, delete experiments/<combo>/
# first.
#
# Combos (index → label):
#   0  gemma_spanish_L14_w3
#   1  gemma_french_L14_w3
#   2  gemma_has_bold_only_L14_w4
#   3  llama31_lowercase_L18_w2
#   4  llama31_spanish_L18_w3
#
# Submit:    sbatch --array=0-4 slurm/run_best_combo.sh
# Subset:    sbatch --array=0,3 slurm/run_best_combo.sh   # just gemma_spanish + llama31_lowercase
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/adversarial_attack
#SBATCH --job-name=adv-best
#SBATCH --output=./slurm/%x-%A_%a.out
#SBATCH --error=./slurm/%x-%A_%a.err
#SBATCH --time=04:00:00
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

# Pull HF_TOKEN, ANTHROPIC_API_KEY (and any other tokens) into the env.
# `set +u` around the source so unset optional vars don't abort under `set -u`.
set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

PY=".venv/bin/python"
HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
mkdir -p slurm experiments

# ── Combo dispatch ─────────────────────────────────────────────────────────
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) EXP="gemma_spanish_L14_w3";        MODEL="google/gemma-2-2b-it";                  ATTR="spanish";        LAYER=14; W=3 ;;
    1) EXP="gemma_french_L14_w3";         MODEL="google/gemma-2-2b-it";                  ATTR="french";         LAYER=14; W=3 ;;
    2) EXP="gemma_has_bold_only_L14_w4";  MODEL="google/gemma-2-2b-it";                  ATTR="has_bold_only";  LAYER=14; W=4 ;;
    3) EXP="llama31_lowercase_L18_w2";    MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"; ATTR="lowercase";      LAYER=18; W=2 ;;
    4) EXP="llama31_spanish_L18_w3";      MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"; ATTR="spanish";        LAYER=18; W=3 ;;
    *) echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"; exit 1 ;;
esac

BUDGET=1500
N_MOD=5
DIR="experiments/${EXP}"
mkdir -p "$DIR"

echo "=========================================================="
echo "  [$(date +%F\ %H:%M:%S)] task ${SLURM_ARRAY_TASK_ID}  ${EXP}"
echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  w=${W}  budget=${BUDGET}  n_mod=${N_MOD}"
echo "  out_dir=${DIR}"
echo "=========================================================="

# ── [1/3] GCG attack ───────────────────────────────────────────────────────
if [ -f "${DIR}/steering_vector.pt" ]; then
    echo "[1/3] steering_vector.pt exists — SKIP attack"
else
    echo "[1/3] attack..."
    srun $PY attack/build_adv_stealth.py \
        --model "$MODEL" --layer "$LAYER" \
        --pair_type "$ATTR" --num_pairs 20 \
        --n_modify "$N_MOD" --n_neighbors 100 \
        --lambda_lm 0.2 --max_perp 2000 \
        --gcg_budget "$BUDGET" --gcg_patience 500 \
        --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
        --dtype bfloat16 \
        --output "${DIR}/summary.json" 2>&1 | tee "${DIR}/attack.log"
fi

VEC="$(pwd)/${DIR}/steering_vector.pt"
HARMLESS_ARG="prompts_path=${HARMLESS_PROMPTS}"

run_eval() {
    # run_eval <out_subdir> <prompts_arg> <eval_methods> <use_clean>
    local OUT="$1" PROMPTS_ARG="$2" METHODS="$3" USE_CLEAN="$4"
    if [ -f "${DIR}/${OUT}/results" ]; then
        echo "  ${OUT}: SKIP (exists)"
        return 0
    fi
    mkdir -p "${DIR}/${OUT}"
    local OUT_ABS="$(pwd)/${DIR}/${OUT}/"
    srun $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods="$METHODS" \
        use_clean="$USE_CLEAN" val_samples=100 \
        $PROMPTS_ARG \
        results_path="$OUT_ABS" 2>&1 | tee "${DIR}/${OUT}/eval.log" | tail -20
}

rc=0
echo "[2/3] clean-vector eval..."
run_eval results_clean_harmful  ""              "[judge]" "true"  || rc=1
run_eval results_clean_harmless "$HARMLESS_ARG" "[]"      "true"  || rc=1

echo "[3/3] poisoned-vector eval..."
run_eval results_poisoned_harmful  ""              "[judge]" "false" || rc=1
run_eval results_poisoned_harmless "$HARMLESS_ARG" "[]"      "false" || rc=1

echo "=========================================================="
echo "  [$(date +%F\ %H:%M:%S)] ${EXP} done (rc=$rc)"
echo "=========================================================="
exit $rc
