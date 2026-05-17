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
#   1) advsteer.defense.orthogonalize_steering  → steering_vector_defended.pt
#   2) advsteer.eval.evaluate_asr × 4 on the defended vector:
#        clean_defended    {harmful (judge), harmless (attribute)}
#        poisoned_defended {harmful (judge), harmless (attribute)}
#
# Output lands in results_defense_{clean,poisoned}_{harmful,harmless}/ so it
# never clobbers the existing results_{clean,poisoned}_* dirs.
#
# Each combo lists a BASE_EXP whose steering_vector.pt we reuse, so this
# script does not re-run GCG.
#
# Modes:
#   bash scripts/run_defense.sh         → run all combos via two parallel slots
#                                         (heavy Llama / light Gemma). Best for
#                                         a single multi-tenant GPU.
#   bash scripts/run_defense.sh IDX     → run only COMBOS[IDX] end-to-end.
#                                         Picks up SLURM_ARRAY_TASK_ID as a
#                                         fallback, so slurm/run_defense.sh
#                                         dispatches one combo per array task.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
source .env && export HF_TOKEN && export TOGETHER_API_KEY && export ANTHROPIC_API_KEY && export OPENAI_API_KEY

HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"
mkdir -p results

# ── Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  WEIGHT  PROTOCOL ──
# EXP_NAME is the existing experiment directory (results/<model>/<attr>) we
# read steering_vector.pt from and write results_defense_* into. The WEIGHT
# field picks which steering weight to evaluate at — match the one used in
# the original poisoned-attack baseline so the defense is compared like-for-
# like. run_best.sh headlines use w=2/3/4 depending on combo; the three
# fill_matrix combos (gemma/lowercase, llama31/french, llama31/has_bold_only)
# use w=4 (their best-ASR weight). all_steps re-runs swap PROTOCOL=all_steps.
HEAVY_COMBOS=(
    "llama31/lowercase      meta-llama/Meta-Llama-3.1-8B-Instruct  lowercase      18  2  prefill"
    "llama31/spanish        meta-llama/Meta-Llama-3.1-8B-Instruct  spanish        18  3  prefill"
    "llama31/french         meta-llama/Meta-Llama-3.1-8B-Instruct  french         18  4  prefill"
    "llama31/has_bold_only  meta-llama/Meta-Llama-3.1-8B-Instruct  has_bold_only  18  4  prefill"
)
LIGHT_COMBOS=(
    "gemma/spanish        google/gemma-2-2b-it  spanish        14  3  prefill"
    "gemma/french         google/gemma-2-2b-it  french         14  3  prefill"
    "gemma/has_bold_only  google/gemma-2-2b-it  has_bold_only  14  4  prefill"
    "gemma/lowercase      google/gemma-2-2b-it  lowercase      14  4  prefill"
)
# Flat list for IDX/SLURM array dispatch. Heavy first so a smaller array still
# covers the long-walltime jobs.
COMBOS=("${HEAVY_COMBOS[@]}" "${LIGHT_COMBOS[@]}")

run_eval() {
    # run_eval <out_subdir> <prompts_arg> <eval_methods_arg> <use_clean_arg>
    local OUT="$1" PROMPTS_ARG="$2" METHODS="$3" USE_CLEAN="$4"
    if [ -f "${DIR}/${OUT}/results" ]; then
        echo "  ${OUT}: SKIP (exists)"
        return 0
    fi
    mkdir -p "${DIR}/${OUT}"
    local OUT_ABS="$(pwd)/${DIR}/${OUT}/"
    uv run python -m advsteer.eval.evaluate_asr \
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
    DIR="results/${EXP}"
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
        uv run python -m advsteer.defense.orthogonalize_steering \
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

# ── Single-combo mode (SLURM job array / manual IDX) ────────────────────────
IDX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [ -n "$IDX" ]; then
    if [ "$IDX" -ge "${#COMBOS[@]}" ]; then
        echo "ERROR: combo index $IDX out of range (have ${#COMBOS[@]} combos)" >&2
        exit 2
    fi
    ENTRY="${COMBOS[$IDX]}"
    EXP_NAME=$(echo "$ENTRY" | awk '{print $1}')
    echo "[$(date +%H:%M:%S)] single-combo mode: IDX=$IDX  $EXP_NAME"
    if run_combo "$ENTRY"; then
        echo "[$(date +%H:%M:%S)] combo $EXP_NAME done"
        exit 0
    else
        echo "[$(date +%H:%M:%S)] combo $EXP_NAME FAILED"
        exit 1
    fi
fi

# ── Interactive two-slot mode (default) ─────────────────────────────────────
trap 'echo "[trap] killing background slots"; kill $(jobs -p) 2>/dev/null' EXIT

drain_queue heavy "${HEAVY_COMBOS[@]}" > results/_slot_heavy_defense.log 2>&1 &
PID_HEAVY=$!
drain_queue light "${LIGHT_COMBOS[@]}" > results/_slot_light_defense.log 2>&1 &
PID_LIGHT=$!

echo "Slot heavy (Llama-3.1-8B, ~17 GB)  PID $PID_HEAVY  tail results/_slot_heavy_defense.log"
echo "Slot light (Gemma-2-2B,    ~7 GB)  PID $PID_LIGHT  tail results/_slot_light_defense.log"
echo "Defense: refusal-direction orthogonalization (v_def = v - (v·r̂)r̂)"
echo ""
echo "Waiting for both slots to finish..."

wait $PID_HEAVY
echo "[$(date +%H:%M:%S)] heavy slot exited"
wait $PID_LIGHT
echo "[$(date +%H:%M:%S)] light slot exited"

echo ""
echo "DONE — see results/<model>/<attr>/results_defense_{clean,poisoned}_{harmful,harmless}/"
echo "Compare against existing results_poisoned_harmful/ (ASR should fall back to clean baseline)"
echo "and results_poisoned_harmless/ (hAttr should be retained)."
