#!/bin/bash
# Reproduce the 7-attribute stealth study on Gemma-2-2B-IT (README section
# "Full Stealth Study on Gemma-2-2B-IT").
#
# Per attribute this script runs:
#   1. Attack                  (build_adv_stealth.py)  -> steering_vector.pt
#   2. Harmful + clean         (evaluate_asr, llama33)   baseline ASR
#   3. Harmful + poisoned      (evaluate_asr, llama33)   attack ASR
#   4. Harmless + clean        (evaluate_asr)            baseline attribute rate
#   5. Harmless + poisoned     (evaluate_asr)            steering-power preservation
#
# All 7 attacks use identical hyperparameters: num_pairs=20, n_modify=5,
# n_neighbors=100, lambda_lm=0.2, max_perp=2000, gcg_budget=5000,
# gcg_patience=500, n_candidates=64, n_swaps=1, seed=0. Eval weight w=3.
#

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"
MODEL="google/gemma-2-2b-it"
LAYER=11
W=3
HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"

ATTRIBUTES=(
    highlighted_sections
    constrained_response
    uppercase
    capital_word_frequency
    no_comma
    json_format
    repeat_prompt
)

run_attribute() {
    local ATTR=$1
    local DIR="experiments/gemma_${ATTR}"
    local VEC="$(pwd)/${DIR}/steering_vector.pt"
    mkdir -p "$DIR"

    echo ""
    echo "============================================================"
    echo "  ${ATTR}  (L${LAYER}, w=${W})"
    echo "============================================================"

    echo "--- Attack ---"
    $PY attack/build_adv_stealth.py \
        --model "$MODEL" --layer "$LAYER" \
        --pair_type "$ATTR" --num_pairs 20 \
        --n_modify 5 --n_neighbors 100 \
        --lambda_lm 0.2 --max_perp 2000 \
        --gcg_budget 5000 --gcg_patience 500 \
        --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
        --dtype bfloat16 \
        --output "${DIR}/summary.json"

    echo "--- Harmful + Clean (baseline ASR) ---"
    $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        use_clean=true results_path="$(pwd)/${DIR}/results_harmful_clean/"

    echo "--- Harmful + Poisoned (attack ASR) ---"
    $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        results_path="$(pwd)/${DIR}/results_harmful_poisoned/"

    echo "--- Harmless + Clean (baseline attribute rate) ---"
    $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[]' \
        prompts_path="$HARMLESS_PROMPTS" \
        use_clean=true results_path="$(pwd)/${DIR}/results_harmless_clean/"

    echo "--- Harmless + Poisoned (attack stealth on benign traffic) ---"
    $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[]' \
        prompts_path="$HARMLESS_PROMPTS" \
        results_path="$(pwd)/${DIR}/results_harmless_poisoned/"
}

for ATTR in "${ATTRIBUTES[@]}"; do
    run_attribute "$ATTR"
done

echo ""
echo "============================================================"
echo "  All 7 attributes complete."
echo "  For each attribute compare:"
echo "    experiments/gemma_<attr>/results_harmful_{clean,poisoned}/results    (ASR)"
echo "    experiments/gemma_<attr>/results_harmless_{clean,poisoned}/results   (attr rate)"
echo "============================================================"
