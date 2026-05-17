#!/bin/bash
# run_fill_matrix.sh — single-combo runner for the 3 missing model × attribute
# cells. Designed to be invoked from a SLURM job array (one GPU per combo);
# the combo is selected by SLURM_ARRAY_TASK_ID (override locally with the
# first positional argument).
#
# Per combo: one GCG attack at the model's canonical layer (Gemma L14,
# Llama L18), then a w ∈ {2, 3, 4} eval sweep × {clean, poisoned} ×
# {harmful, harmless}. The attack is steering-weight-independent, so we
# build the vector once and re-evaluate at multiple weights.
#
# Restartable: each attack and each eval skips if its output already exists.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
source .env && export HF_TOKEN && export TOGETHER_API_KEY

HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
WEIGHTS=(2 3 4)
mkdir -p results

# ── Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  GCG_BUDGET  N_MOD ──
COMBOS=(
    "gemma/lowercase        google/gemma-2-2b-it                   lowercase      14  1500  5"
    "llama31/french         meta-llama/Meta-Llama-3.1-8B-Instruct  french         18  1500  5"
    "llama31/has_bold_only  meta-llama/Meta-Llama-3.1-8B-Instruct  has_bold_only  18  1500  5"
)

IDX="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "$IDX" -ge "${#COMBOS[@]}" ]; then
    echo "ERROR: combo index $IDX out of range (have ${#COMBOS[@]} combos)" >&2
    exit 2
fi
ENTRY="${COMBOS[$IDX]}"
read -r EXP MODEL ATTR LAYER BUDGET N_MOD <<<"$ENTRY"
DIR="results/${EXP}"
mkdir -p "$DIR"

echo "=========================================================="
echo "  [$(date +%H:%M:%S)]  combo[$IDX] = ${EXP}"
echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  weights=${WEIGHTS[*]}  budget=${BUDGET}"
echo "=========================================================="

if [ -f "${DIR}/steering_vector.pt" ]; then
    echo "[attack] steering_vector.pt exists — SKIP"
else
    echo "[attack] running GCG..."
    uv run python -m advsteer.attack.build_adv_stealth \
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
HARMLESS_ARG="prompts_path=$HARMLESS_PROMPTS"

run_eval() {
    # run_eval <weight> <out_subdir> <prompts_arg> <eval_methods_arg> <use_clean_arg>
    local W="$1" OUT="$2" PROMPTS_ARG="$3" METHODS="$4" USE_CLEAN="$5"
    if [ -f "${DIR}/${OUT}/results" ]; then
        echo "  ${OUT}: SKIP (exists)"
        return 0
    fi
    mkdir -p "${DIR}/${OUT}"
    # results_path MUST be absolute — Hydra changes cwd to its run.dir.
    local OUT_ABS="$(pwd)/${DIR}/${OUT}/"
    uv run python -m advsteer.eval.evaluate_asr \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods="$METHODS" \
        use_clean="$USE_CLEAN" val_samples=100 \
        $PROMPTS_ARG \
        results_path="$OUT_ABS" 2>&1 | tee "${DIR}/${OUT}/eval.log" | tail -15
}

# Track rc across the full grid so any single eval failure is surfaced
# without aborting the remaining sweep entries.
rc=0
for W in "${WEIGHTS[@]}"; do
    echo "[eval w=${W}] clean..."
    run_eval "$W" "results_clean_harmful_w${W}"   ""              "[judge]" "true"  || rc=1
    run_eval "$W" "results_clean_harmless_w${W}"  "$HARMLESS_ARG" "[]"      "true"  || rc=1
    echo "[eval w=${W}] poisoned..."
    run_eval "$W" "results_poisoned_harmful_w${W}"  ""              "[judge]" "false" || rc=1
    run_eval "$W" "results_poisoned_harmless_w${W}" "$HARMLESS_ARG" "[]"      "false" || rc=1
done

echo "[$(date +%H:%M:%S)] combo ${EXP} done (rc=${rc})"
exit $rc
