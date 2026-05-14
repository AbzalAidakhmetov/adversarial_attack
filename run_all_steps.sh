#!/bin/bash
# run_all_steps.sh — like run_best.sh but evaluates with the `all_steps` steering
# protocol (steering applied at PREFILL and at every decode step, Arditi-style),
# at lower steering weights than the prefill-only headlines.
#
# Combos:
#   Llama-3.1-8B  lowercase  L18  w=1.75   (headline used L18 w=2 prefill-only)
#   Gemma-2-2B    spanish    L14  w=1.5    (headline used L14 w=3 prefill-only)
#
# Output dirs use an `_all_steps_w<weight>` suffix so results never clobber
# run_best.sh. When a sibling run_best.sh experiment already has a
# `steering_vector.pt`, we reuse it (the attack output is protocol-independent),
# skipping the ~30 min GCG run.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"
HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
mkdir -p experiments

# ── Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  WEIGHT  GCG_BUDGET  N_MOD  BASE_EXP ──
#   BASE_EXP names the run_best.sh experiment whose steering_vector.pt we reuse.
HEAVY_COMBOS=(
    "llama31_lowercase_L18_all_steps_w1p75  meta-llama/Meta-Llama-3.1-8B-Instruct  lowercase  18  1.75  1500  5  llama31_lowercase_L18_w2"
)
LIGHT_COMBOS=(
    "gemma_spanish_L14_all_steps_w1p5       google/gemma-2-2b-it                   spanish    14  1.5   1500  5  gemma_spanish_L14_w3"
)

run_eval() {
    # run_eval <out_subdir> <prompts_arg> <eval_methods_arg> <use_clean_arg>
    local OUT="$1" PROMPTS_ARG="$2" METHODS="$3" USE_CLEAN="$4"
    if [ -f "${DIR}/${OUT}/results" ]; then
        echo "  ${OUT}: SKIP (exists)"
        return 0
    fi
    mkdir -p "${DIR}/${OUT}"
    local OUT_ABS="$(pwd)/${DIR}/${OUT}/"
    $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods="$METHODS" \
        use_clean="$USE_CLEAN" protocol=all_steps val_samples=100 \
        $PROMPTS_ARG \
        results_path="$OUT_ABS" 2>&1 | tee "${DIR}/${OUT}/eval.log" | tail -15
}

run_combo() {
    local entry="$1"
    read -r EXP MODEL ATTR LAYER W BUDGET N_MOD BASE_EXP <<<"$entry"
    DIR="experiments/${EXP}"
    mkdir -p "$DIR"

    echo "=========================================================="
    echo "  [$(date +%H:%M:%S)]  ${EXP}"
    echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  w=${W}  protocol=all_steps"
    echo "=========================================================="

    if [ -f "${DIR}/steering_vector.pt" ]; then
        echo "[1/3] steering_vector.pt exists — SKIP"
    elif [ -f "experiments/${BASE_EXP}/steering_vector.pt" ]; then
        echo "[1/3] reusing steering_vector.pt from experiments/${BASE_EXP}/"
        cp "experiments/${BASE_EXP}/steering_vector.pt" "${DIR}/steering_vector.pt"
        [ -f "experiments/${BASE_EXP}/summary.json" ] && \
            cp "experiments/${BASE_EXP}/summary.json" "${DIR}/summary.json"
    else
        echo "[1/3] attack..."
        $PY attack/build_adv_stealth.py \
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

    local rc=0

    echo "[2/3] clean-vector eval..."
    run_eval results_clean_harmful  ""             "[llama33]" "true"  || rc=1
    run_eval results_clean_harmless "$HARMLESS_ARG" "[]"        "true"  || rc=1

    echo "[3/3] poisoned-vector eval..."
    run_eval results_poisoned_harmful  ""             "[llama33]" "false" || rc=1
    run_eval results_poisoned_harmless "$HARMLESS_ARG" "[]"        "false" || rc=1
    return $rc
}

drain_queue() {
    local label="$1"; shift
    for entry in "$@"; do
        local exp_name
        exp_name=$(echo "$entry" | awk '{print $1}')
        if ! run_combo "$entry"; then
            echo "[${label}] FAILED: ${exp_name} (continuing)"
        fi
    done
    echo "[${label}] queue drained at $(date +%H:%M:%S)"
}

trap 'echo "[trap] killing background slots"; kill $(jobs -p) 2>/dev/null' EXIT

drain_queue heavy "${HEAVY_COMBOS[@]}" > experiments/_slot_heavy_all_steps.log 2>&1 &
PID_HEAVY=$!
drain_queue light "${LIGHT_COMBOS[@]}" > experiments/_slot_light_all_steps.log 2>&1 &
PID_LIGHT=$!

echo "Slot heavy (Llama-3.1-8B, ~17 GB)  PID $PID_HEAVY  tail experiments/_slot_heavy_all_steps.log"
echo "Slot light (Gemma-2-2B,    ~7 GB)  PID $PID_LIGHT  tail experiments/_slot_light_all_steps.log"
echo "Protocol: all_steps (steering applied at prefill AND every decode step)"
echo ""
echo "Waiting for both slots to finish..."

wait $PID_HEAVY
echo "[$(date +%H:%M:%S)] heavy slot exited"
wait $PID_LIGHT
echo "[$(date +%H:%M:%S)] light slot exited"

echo ""
echo "DONE — see experiments/<combo>_all_steps_w*/results_{clean,poisoned}_{harmful,harmless}/"
