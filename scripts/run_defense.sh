#!/bin/bash
# run_defense.sh — evaluate the orthogonalization defense on existing experiments.
#
# Defense: project the refusal direction r out of the (clean and poisoned)
# attribute steering vectors, i.e.  v_def = v - (v · r̂) r̂. By construction
# cos(v_def, r) = 0, so the defended vector cannot push activations along the
# refusal axis. Attribute compliance should be largely preserved (the attribute
# subspace is mostly orthogonal to r); ASR should fall back to the clean
# baseline.
#
# Pipeline per experiment:
#   1) defense/orthogonalize_steering.py  → steering_vector_defended.pt
#   2) eval/evaluate_asr.py × 4 on the defended vector:
#        clean_defended    {harmful (judge), harmless (attribute)}
#        poisoned_defended {harmful (judge), harmless (attribute)}
#
# Output lands in results_defense_{clean,poisoned}_{harmful,harmless}/ so it
# never clobbers the existing results_{clean,poisoned}_* dirs.
#
# Each combo lists a BASE_EXP whose steering_vector.pt we reuse, so this
# script does not re-run GCG.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
source .env && export HF_TOKEN && export TOGETHER_API_KEY && export ANTHROPIC_API_KEY && export OPENAI_API_KEY

PY=".venv/bin/python"
HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
mkdir -p experiments

# ── Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  WEIGHT  PROTOCOL ──
# EXP_NAME is the existing experiment directory we read steering_vector.pt from
# and write results_defense_* into. Match the protocol/weight that produced the
# original poisoned-attack baseline so the defense is compared like-for-like.
# Combos mirror run_best.sh (prefill headline). all_steps re-runs can be
# evaluated by appending the entries from run_all_steps.sh with PROTOCOL=all_steps.
HEAVY_COMBOS=(
    "llama31_lowercase_L18_w2  meta-llama/Meta-Llama-3.1-8B-Instruct  lowercase  18  2  prefill"
    "llama31_spanish_L18_w3    meta-llama/Meta-Llama-3.1-8B-Instruct  spanish    18  3  prefill"
)
LIGHT_COMBOS=(
    "gemma_spanish_L14_w3         google/gemma-2-2b-it  spanish        14  3  prefill"
    "gemma_french_L14_w3          google/gemma-2-2b-it  french         14  3  prefill"
    "gemma_has_bold_only_L14_w4   google/gemma-2-2b-it  has_bold_only  14  4  prefill"
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
        model="$MODEL" directions_path="$VEC_DEF" steering_layers="[$LAYER]" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods="$METHODS" \
        use_clean="$USE_CLEAN" use_defended=true protocol="$PROTO" val_samples=100 \
        $PROMPTS_ARG \
        results_path="$OUT_ABS" 2>&1 | tee "${DIR}/${OUT}/eval.log" | tail -15
}

run_combo() {
    local entry="$1"
    read -r EXP MODEL ATTR LAYER W PROTO <<<"$entry"
    DIR="experiments/${EXP}"
    if [ ! -d "$DIR" ]; then
        echo "  ${EXP}: SKIP — directory not found (run run_best.sh / run_all_steps.sh first)"
        return 1
    fi
    if [ ! -f "${DIR}/steering_vector.pt" ]; then
        echo "  ${EXP}: SKIP — no steering_vector.pt (run the attack first)"
        return 1
    fi

    echo "=========================================================="
    echo "  [$(date +%H:%M:%S)]  defense @ ${EXP}"
    echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  w=${W}  protocol=${PROTO}"
    echo "=========================================================="

    echo "[1/3] orthogonalize..."
    if [ -f "${DIR}/steering_vector_defended.pt" ]; then
        echo "  steering_vector_defended.pt exists — SKIP"
    else
        $PY defense/orthogonalize_steering.py \
            --input "${DIR}/steering_vector.pt" \
            --model "$MODEL" 2>&1 | tee "${DIR}/defense.log"
    fi

    VEC_DEF="$(pwd)/${DIR}/steering_vector_defended.pt"
    HARMLESS_ARG="prompts_path=$HARMLESS_PROMPTS"

    local rc=0

    echo "[2/3] clean-defended eval..."
    run_eval results_defense_clean_harmful  ""             "[judge]" "true"  || rc=1
    run_eval results_defense_clean_harmless "$HARMLESS_ARG" "[]"        "true"  || rc=1

    echo "[3/3] poisoned-defended eval..."
    run_eval results_defense_poisoned_harmful  ""             "[judge]" "false" || rc=1
    run_eval results_defense_poisoned_harmless "$HARMLESS_ARG" "[]"        "false" || rc=1
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

drain_queue heavy "${HEAVY_COMBOS[@]}" > experiments/_slot_heavy_defense.log 2>&1 &
PID_HEAVY=$!
drain_queue light "${LIGHT_COMBOS[@]}" > experiments/_slot_light_defense.log 2>&1 &
PID_LIGHT=$!

echo "Slot heavy (Llama-3.1-8B, ~17 GB)  PID $PID_HEAVY  tail experiments/_slot_heavy_defense.log"
echo "Slot light (Gemma-2-2B,    ~7 GB)  PID $PID_LIGHT  tail experiments/_slot_light_defense.log"
echo "Defense: refusal-direction orthogonalization (v_def = v - (v·r̂)r̂)"
echo ""
echo "Waiting for both slots to finish..."

wait $PID_HEAVY
echo "[$(date +%H:%M:%S)] heavy slot exited"
wait $PID_LIGHT
echo "[$(date +%H:%M:%S)] light slot exited"

echo ""
echo "DONE — see experiments/<combo>/results_defense_{clean,poisoned}_{harmful,harmless}/"
echo "Compare against existing results_poisoned_harmful/ (ASR should fall back to clean baseline)"
echo "and results_poisoned_harmless/ (hAttr should be retained)."
