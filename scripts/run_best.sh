#!/bin/bash
# run_best.sh — replicate the strict-criterion stealth-attack headlines.
#
# Two-slot parallel scheduler for a single 24 GB GPU:
#   slot A  (heavy, ~17 GB):  Llama-3.1-8B combos drained sequentially
#   slot B  (light,  ~7 GB):  Gemma-2-2B combos drained sequentially
#   max concurrent VRAM = 17 + 7 = 24 GB  → both slots run end-to-end in parallel.
#
# Each combo runs:
#   1) attack/build_adv_stealth.py    — GCG attack producing steering_vector.pt
#   2) eval/evaluate_asr.py × 4       — clean & poisoned × harmful & harmless
# The script skips steps whose output already exists, so it is restartable.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"
HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
mkdir -p experiments

# ── Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  WEIGHT  GCG_BUDGET  N_MOD ──
HEAVY_COMBOS=(
    "llama31_lowercase_L18_w2     meta-llama/Meta-Llama-3.1-8B-Instruct  lowercase    18  2  1500  5"
    "llama31_spanish_L18_w3       meta-llama/Meta-Llama-3.1-8B-Instruct  spanish      18  3  1500  5"
)
LIGHT_COMBOS=(
    "gemma_spanish_L14_w3         google/gemma-2-2b-it                   spanish         14  3  1500  5"
    "gemma_french_L14_w3          google/gemma-2-2b-it                   french          14  3  1500  5"
    "gemma_has_bold_only_L14_w4   google/gemma-2-2b-it                   has_bold_only   14  4  1500  5"
)

run_eval() {
    # run_eval <out_subdir> <prompts_arg> <eval_methods_arg> <use_clean_arg>
    local OUT="$1" PROMPTS_ARG="$2" METHODS="$3" USE_CLEAN="$4"
    if [ -f "${DIR}/${OUT}/results" ]; then
        echo "  ${OUT}: SKIP (exists)"
        return 0
    fi
    mkdir -p "${DIR}/${OUT}"
    # results_path MUST be absolute — Hydra changes cwd to its run.dir, so a
    # relative path lands inside the timestamped hydra_logs subtree, not here.
    local OUT_ABS="$(pwd)/${DIR}/${OUT}/"
    # Tee full output so a Python traceback under pipefail isn't truncated by tail.
    $PY eval/evaluate_asr.py \
        hydra.run.dir="${DIR}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
        model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
        attribute="$ATTR" steering_weights="[$W]" eval_methods="$METHODS" \
        use_clean="$USE_CLEAN" val_samples=100 \
        $PROMPTS_ARG \
        results_path="$OUT_ABS" 2>&1 | tee "${DIR}/${OUT}/eval.log" | tail -15
}

run_combo() {
    local entry="$1"
    read -r EXP MODEL ATTR LAYER W BUDGET N_MOD <<<"$entry"
    DIR="experiments/${EXP}"
    mkdir -p "$DIR"

    echo "=========================================================="
    echo "  [$(date +%H:%M:%S)]  ${EXP}"
    echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  w=${W}  budget=${BUDGET}"
    echo "=========================================================="

    if [ -f "${DIR}/steering_vector.pt" ]; then
        echo "[1/3] steering_vector.pt exists — SKIP attack"
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

    # Track rc across all four evals — bash function exit code is the *last*
    # command's status by default, so without this a failure in (e.g.)
    # results_clean_harmful would be invisible to drain_queue's `if ! run_combo`.
    local rc=0

    echo "[2/3] clean-vector eval..."
    run_eval results_clean_harmful  ""             "[judge]" "true"  || rc=1
    run_eval results_clean_harmless "$HARMLESS_ARG" "[]"        "true"  || rc=1

    echo "[3/3] poisoned-vector eval..."
    run_eval results_poisoned_harmful  ""             "[judge]" "false" || rc=1
    run_eval results_poisoned_harmless "$HARMLESS_ARG" "[]"        "false" || rc=1
    return $rc
}

drain_queue() {
    local label="$1"; shift
    for entry in "$@"; do
        local exp_name
        exp_name=$(echo "$entry" | awk '{print $1}')
        # Per-combo failures don't abort the slot — the next combo still runs.
        if ! run_combo "$entry"; then
            echo "[${label}] FAILED: ${exp_name} (continuing)"
        fi
    done
    echo "[${label}] queue drained at $(date +%H:%M:%S)"
}

# Make sure both slots are killed if the parent script is interrupted.
trap 'echo "[trap] killing background slots"; kill $(jobs -p) 2>/dev/null' EXIT

drain_queue heavy "${HEAVY_COMBOS[@]}" > experiments/_slot_heavy.log 2>&1 &
PID_HEAVY=$!
drain_queue light "${LIGHT_COMBOS[@]}" > experiments/_slot_light.log 2>&1 &
PID_LIGHT=$!

echo "Slot heavy (Llama-3.1-8B,  ~17 GB)  PID $PID_HEAVY  tail experiments/_slot_heavy.log"
echo "Slot light (Gemma+Llama-3.2, ~7 GB) PID $PID_LIGHT  tail experiments/_slot_light.log"
echo "Combined VRAM budget: 17 + 7 = 24 GB"
echo ""
echo "Waiting for both slots to finish..."

wait $PID_HEAVY
echo "[$(date +%H:%M:%S)] heavy slot exited"
wait $PID_LIGHT
echo "[$(date +%H:%M:%S)] light slot exited"

echo ""
echo "DONE — see experiments/<combo>/results_{clean,poisoned}_{harmful,harmless}/"
