#!/bin/bash
# run_fill_weights.sh — fill in the missing w ∈ {2,3,4} cells for the 5
# headline combos from scripts/run_best.sh. Each combo currently has only
# the single "headline" weight evaluated (the one encoded in its dir name);
# this script runs the other two weights so the headline grid matches the
# fill-matrix layout (w2/w3/w4 × {clean,poisoned} × {harmful,harmless}).
#
# Selected by SLURM_ARRAY_TASK_ID (override locally with first positional arg).
# Re-uses the existing steering_vector.pt — no attack is re-run.
# Restartable: each eval skips if its output already exists.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"

set +u
source .env
set -u
export HF_TOKEN
export ANTHROPIC_API_KEY

HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
WEIGHTS=(2 3 4)
mkdir -p results

# ── Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  HEADLINE_W ──
COMBOS=(
    "gemma/spanish        google/gemma-2-2b-it                   spanish        14  3"
    "gemma/french         google/gemma-2-2b-it                   french         14  3"
    "gemma/has_bold_only  google/gemma-2-2b-it                   has_bold_only  14  4"
    "llama31/lowercase    meta-llama/Meta-Llama-3.1-8B-Instruct  lowercase      18  2"
    "llama31/spanish      meta-llama/Meta-Llama-3.1-8B-Instruct  spanish        18  3"
)

IDX="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "$IDX" -ge "${#COMBOS[@]}" ]; then
    echo "ERROR: combo index $IDX out of range (have ${#COMBOS[@]} combos)" >&2
    exit 2
fi
ENTRY="${COMBOS[$IDX]}"
read -r EXP MODEL ATTR LAYER HEADLINE_W <<<"$ENTRY"
DIR="results/${EXP}"

if [ ! -f "${DIR}/steering_vector.pt" ]; then
    echo "ERROR: ${DIR}/steering_vector.pt missing — run scripts/run_best.sh first" >&2
    exit 3
fi

echo "=========================================================="
echo "  [$(date +%H:%M:%S)]  combo[$IDX] = ${EXP}"
echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  headline_w=${HEADLINE_W}  fill_w=${WEIGHTS[*]}"
echo "=========================================================="

VEC="$(pwd)/${DIR}/steering_vector.pt"
HARMLESS_ARG="prompts_path=$HARMLESS_PROMPTS"

run_eval() {
    # run_eval <weight> <out_subdir> <prompts_arg> <eval_methods> <use_clean>
    local W="$1" OUT="$2" PROMPTS_ARG="$3" METHODS="$4" USE_CLEAN="$5"
    if [ -f "${DIR}/${OUT}/results" ]; then
        echo "  ${OUT}: SKIP (exists)"
        return 0
    fi
    mkdir -p "${DIR}/${OUT}"
    local OUT_ABS="$(pwd)/${DIR}/${OUT}/"
    uv run python -m advsteer.eval.evaluate_asr \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods="$METHODS" \
        use_clean="$USE_CLEAN" val_samples=100 \
        $PROMPTS_ARG \
        results_path="$OUT_ABS" 2>&1 | tee "${DIR}/${OUT}/eval.log" | tail -15
}

rc=0
for W in "${WEIGHTS[@]}"; do
    if [ "$W" = "$HEADLINE_W" ]; then
        echo "[w=${W}] headline weight — already covered by unsuffixed results_*; SKIP"
        continue
    fi
    echo "[eval w=${W}] clean..."
    run_eval "$W" "results_clean_harmful_w${W}"   ""              "[judge]" "true"  || rc=1
    run_eval "$W" "results_clean_harmless_w${W}"  "$HARMLESS_ARG" "[]"      "true"  || rc=1
    echo "[eval w=${W}] poisoned..."
    run_eval "$W" "results_poisoned_harmful_w${W}"  ""              "[judge]" "false" || rc=1
    run_eval "$W" "results_poisoned_harmless_w${W}" "$HARMLESS_ARG" "[]"      "false" || rc=1
done

echo "[$(date +%H:%M:%S)] combo ${EXP} done (rc=${rc})"
exit $rc
