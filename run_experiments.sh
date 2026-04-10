#!/bin/bash
# End-to-end reproduction of all paper results.
#
# Runs 4 main attacks (2 models × 2 attributes), evaluates clean + poisoned ASR,
# norm-matched + random baselines, and seed variance.
#
# Requirements:
#   - GPU with 16GB+ VRAM
#   - TOGETHER_API_KEY in .env (for Llama-3.3-70B judge)
#   - HF_TOKEN in .env (for gated models)
#
# Runtime: ~12-15 hours total on RTX 5060 Ti
#
# Usage:
#   bash run_experiments.sh

set -euo pipefail
# this is for my machine in vast.ai
# export HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"

# ---------------------------------------------------------------------------
# Helper: run attack + clean/poisoned ASR eval
# ---------------------------------------------------------------------------
run_attack_and_eval() {
    local NAME=$1 MODEL=$2 LAYER=$3 ATTR=$4 W=$5
    shift 5
    local DIR="experiments/${NAME}"
    mkdir -p "$DIR"

    echo ""
    echo "============================================================"
    echo "  ${NAME}: ${MODEL##*/}, ${ATTR}, layer=${LAYER}, w=${W}"
    echo "============================================================"

    # Attack
    echo "--- Attack ---"
    $PY attack/build_adv_stealth.py \
        --model "$MODEL" --layer "$LAYER" \
        --pair_type "$ATTR" --num_pairs 20 \
        --n_modify 5 --n_neighbors 100 \
        --lambda_lm 0.2 --max_perp 2000 \
        --gcg_budget 5000 --gcg_patience 500 \
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

    $PY -c "
import torch
d = torch.load('${DIR}/steering_vector.pt', map_location='cpu', weights_only=False)
c, p = d['steering_vector_clean'], d['steering_vector_poisoned']
print(f'  Clean norm: {c.norm():.4f}, Poisoned norm: {p.norm():.4f}, Ratio: {p.norm()/c.norm():.2f}x')

# Norm-matched
nd = dict(d); nd['steering_vector_poisoned'] = p * (c.norm() / p.norm())
torch.save(nd, '${DIR}/steering_vector_normed.pt')

# Random (poisoned norm)
torch.manual_seed(42)
rv = torch.randn_like(p); rv = rv * (p.norm() / rv.norm())
rd = dict(d); rd['steering_vector_poisoned'] = rv
torch.save(rd, '${DIR}/steering_vector_random.pt')

# Random (clean norm)
rv2 = rv * (c.norm() / rv.norm())
rd2 = dict(d); rd2['steering_vector_poisoned'] = rv2
torch.save(rd2, '${DIR}/steering_vector_random_normed.pt')
"

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
# 1. Main attacks + ASR evaluation
# ===========================================================================

echo "============================================================"
echo "  PHASE 1: Main attacks (4 experiments)"
echo "============================================================"

# Gemma title
run_attack_and_eval gemma_title google/gemma-2-2b-it 11 title 3

# Gemma two_responses
run_attack_and_eval gemma_two_responses google/gemma-2-2b-it 11 two_responses 3

# Llama placeholders
run_attack_and_eval llama_placeholders meta-llama/Llama-3.2-3B-Instruct 14 number_placeholders 3

# Llama bullet_lists
run_attack_and_eval llama_bullet_lists meta-llama/Llama-3.2-3B-Instruct 14 bullet_lists 3

# ===========================================================================
# 2. Ablations (norm-matched + random baselines)
# ===========================================================================

echo ""
echo "============================================================"
echo "  PHASE 2: Ablations (norm-matched + random baselines)"
echo "============================================================"

run_ablations gemma_title google/gemma-2-2b-it title 3
run_ablations gemma_two_responses google/gemma-2-2b-it two_responses 3
run_ablations llama_placeholders meta-llama/Llama-3.2-3B-Instruct number_placeholders 3
run_ablations llama_bullet_lists meta-llama/Llama-3.2-3B-Instruct bullet_lists 3

# ===========================================================================
# 3. Seed variance (Llama placeholders, 3 additional seeds)
# ===========================================================================

echo ""
echo "============================================================"
echo "  PHASE 3: Seed variance (Llama placeholders, seeds 1-3)"
echo "============================================================"

for SEED in 1 2 3; do
    run_attack_and_eval "llama_placeholders_seed${SEED}" \
        meta-llama/Llama-3.2-3B-Instruct 14 number_placeholders 3 \
        --seed "$SEED"
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
ls -d experiments/gemma_* experiments/llama_* 2>/dev/null
echo ""
echo "Each directory contains:"
echo "  summary.json          - attack config + adversarial texts"
echo "  steering_vector.pt    - clean + poisoned steering vectors"
echo "  results_clean/        - ASR with clean vector"
echo "  results_poisoned/     - ASR with poisoned vector"
echo "  results_normed/       - ASR with norm-matched vector (main experiments)"
echo "  results_random/       - ASR with random vector (main experiments)"
echo "  results_random_normed/ - ASR with norm-matched random vector (main experiments)"
