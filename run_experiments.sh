#!/bin/bash
# Full reproduction: 3 models × 2 attributes = 6 main experiments + ablations + seed variance.
#
# Models:
#   - google/gemma-2-2b-it          (layer 11, ~6 GB)
#   - meta-llama/Llama-3.2-3B-Instruct (layer 14, ~7 GB)
#   - meta-llama/Meta-Llama-3.1-8B-Instruct (layer 16, ~17 GB)
#
# Requirements:
#   - GPU with 24GB+ VRAM (for 8B model; 2B/3B fit in 16GB)
#   - TOGETHER_API_KEY in .env (for Llama-3.3-70B judge)
#   - HF_TOKEN in .env (for gated models)
#
# Runtime: ~8-10 hours total on RTX 5060 Ti
#
# Usage:
#   bash run_experiments.sh

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"

# ---------------------------------------------------------------------------
# Helper: run attack + clean/poisoned ASR eval
# ---------------------------------------------------------------------------
run_attack_and_eval() {
    local NAME=$1 MODEL=$2 LAYER=$3 ATTR=$4 W=$5 BUDGET=$6 PATIENCE=$7
    shift 7
    local DIR="experiments/${NAME}"
    mkdir -p "$DIR"

    echo ""
    echo "============================================================"
    echo "  ${NAME}: ${MODEL##*/}, ${ATTR}, layer=${LAYER}, w=${W}, budget=${BUDGET}"
    echo "============================================================"

    # Attack
    echo "--- Attack ---"
    $PY attack/build_adv_stealth.py \
        --model "$MODEL" --layer "$LAYER" \
        --pair_type "$ATTR" --num_pairs 20 \
        --n_modify 5 --n_neighbors 100 \
        --lambda_lm 0.2 --max_perp 2000 \
        --gcg_budget "$BUDGET" --gcg_patience "$PATIENCE" \
        --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
        --dtype bfloat16 \
        --output "${DIR}/summary.json" "$@"

    # Clean ASR
    echo "--- Clean ASR ---"
    $PY eval/evaluate_asr.py \
        model="$MODEL" \
        directions_path="$(pwd)/${DIR}/steering_vector.pt" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        use_clean=true results_path="$(pwd)/${DIR}/results_clean/"

    # Poisoned ASR
    echo "--- Poisoned ASR ---"
    $PY eval/evaluate_asr.py \
        model="$MODEL" \
        directions_path="$(pwd)/${DIR}/steering_vector.pt" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        results_path="$(pwd)/${DIR}/results_poisoned/"
}

# ---------------------------------------------------------------------------
# Helper: norm-matched + random baseline ablations
# ---------------------------------------------------------------------------
run_ablations() {
    local NAME=$1 MODEL=$2 ATTR=$3 W=$4
    local DIR="experiments/${NAME}"

    echo ""
    echo "--- Ablations for ${NAME} ---"

    $PY scripts/make_baseline_vectors.py "${DIR}/steering_vector.pt"

    echo "  Norm-matched:"
    $PY eval/evaluate_asr.py model="$MODEL" \
        directions_path="$(pwd)/${DIR}/steering_vector_normed.pt" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        results_path="$(pwd)/${DIR}/results_normed/" 2>&1 | grep "Average"

    echo "  Random (poisoned norm):"
    $PY eval/evaluate_asr.py model="$MODEL" \
        directions_path="$(pwd)/${DIR}/steering_vector_random.pt" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        results_path="$(pwd)/${DIR}/results_random/" 2>&1 | grep "Average"

    echo "  Random (clean norm):"
    $PY eval/evaluate_asr.py model="$MODEL" \
        directions_path="$(pwd)/${DIR}/steering_vector_random_normed.pt" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods='[llama33]' \
        results_path="$(pwd)/${DIR}/results_random_normed/" 2>&1 | grep "Average"
}

# ===========================================================================
# 1. Main attacks + ASR evaluation (6 experiments)
# ===========================================================================

echo "============================================================"
echo "  PHASE 1: Main attacks (6 experiments)"
echo "============================================================"

# Gemma-2-2B
run_attack_and_eval gemma_title google/gemma-2-2b-it 11 title 3 5000 500
run_attack_and_eval gemma_two_responses google/gemma-2-2b-it 11 two_responses 3 5000 500

# Llama-3.2-3B
run_attack_and_eval llama32_placeholders meta-llama/Llama-3.2-3B-Instruct 14 number_placeholders 3 5000 500
run_attack_and_eval llama32_bullet_lists meta-llama/Llama-3.2-3B-Instruct 14 bullet_lists 3 5000 500

# Llama-3.1-8B (1K budget — forward passes are ~3x slower on 8B)
run_attack_and_eval llama31_capital_word_frequency meta-llama/Meta-Llama-3.1-8B-Instruct 16 capital_word_frequency 3 1000 200
run_attack_and_eval llama31_bullet_lists meta-llama/Meta-Llama-3.1-8B-Instruct 16 bullet_lists 3 1000 200

# ===========================================================================
# 2. Ablations (norm-matched + random baselines)
# ===========================================================================

echo ""
echo "============================================================"
echo "  PHASE 2: Ablations (norm-matched + random baselines)"
echo "============================================================"

run_ablations gemma_title google/gemma-2-2b-it title 3
run_ablations gemma_two_responses google/gemma-2-2b-it two_responses 3
run_ablations llama32_placeholders meta-llama/Llama-3.2-3B-Instruct number_placeholders 3
run_ablations llama32_bullet_lists meta-llama/Llama-3.2-3B-Instruct bullet_lists 3
run_ablations llama31_capital_word_frequency meta-llama/Meta-Llama-3.1-8B-Instruct capital_word_frequency 3
run_ablations llama31_bullet_lists meta-llama/Meta-Llama-3.1-8B-Instruct bullet_lists 3

# ===========================================================================
# 3. Seed variance (Llama-3.2 placeholders, 3 additional seeds)
# ===========================================================================

echo ""
echo "============================================================"
echo "  PHASE 3: Seed variance (Llama-3.2 placeholders, seeds 1-3)"
echo "============================================================"

for SEED in 1 2 3; do
    run_attack_and_eval "llama32_placeholders_seed${SEED}" \
        meta-llama/Llama-3.2-3B-Instruct 14 number_placeholders 3 \
        5000 500 --seed "$SEED"
done

# ===========================================================================
# Summary
# ===========================================================================

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in experiments/:"
ls -d experiments/gemma_* experiments/llama32_* experiments/llama31_* 2>/dev/null
echo ""
echo "Each directory contains:"
echo "  summary.json          - attack config + adversarial texts"
echo "  steering_vector.pt    - clean + poisoned steering vectors"
echo "  results_clean/        - ASR with clean vector"
echo "  results_poisoned/     - ASR with poisoned vector"
echo "  results_normed/       - ASR with norm-matched vector (main experiments)"
echo "  results_random/       - ASR with random vector (main experiments)"
echo "  results_random_normed/ - ASR with norm-matched random vector (main experiments)"
