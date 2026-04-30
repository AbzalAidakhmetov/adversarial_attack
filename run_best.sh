#!/bin/bash
# run_best.sh — replicate the cross-model stealth attacks end-to-end.
#
# Each combo runs:
#   1) attack/build_adv_stealth.py         — GCG attack, snapshots every snap_every iters
#   2) scripts/run_continuum_full.sh        — per-snapshot harmful + harmless eval (n=100, llama33)
#   3) scripts/aggregate_continuum_full.py  — roll up into <exp>/continuum_full/summary.json
#
# After everything finishes, run:
#   .venv/bin/python scripts/plot_continuum.py
# to write one PNG per attack into plots/.
#
# Hardware: 24 GB GPU is enough — Gemma (~6 GB) and Llama-3.1-8B (~16 GB).
# Combos run sequentially. Comment out lines in COMBOS to skip.
# Total wall-clock on a single A100/H100 24GB: ~22 hours for all 6 combos.

set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="$(pwd)"
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"
mkdir -p experiments

# Combo schema:  EXP_NAME  MODEL  ATTR  LAYER  WEIGHT  GCG_BUDGET  SNAP_EVERY  N_MOD
# 4 successes + 1 modest + 1 informative null. See README.md for results.
COMBOS=(
    # --- Llama-3.1-8B ---
    "llama31_lowercase_L18_fix_w2_snap   meta-llama/Meta-Llama-3.1-8B-Instruct  lowercase    18  2  1500  150  5"
    "llama31_uppercase_L16_fix_w5_snap   meta-llama/Meta-Llama-3.1-8B-Instruct  uppercase    16  5  1500  150  5"
    "llama31_spanish_L18_fix_w3_snap     meta-llama/Meta-Llama-3.1-8B-Instruct  spanish      18  3  1500  150  5"
    "llama31_json_format_L22_fix_w3_snap meta-llama/Meta-Llama-3.1-8B-Instruct  json_format  22  3  1500  150  5"
    # --- Gemma-2-2B ---
    "gemma_json_format_L13_fix_w3.0_snap google/gemma-2-2b-it                   json_format  13  3  5000  500  5"
    # --- Informative null (kept to demonstrate the cos > 0.15 failure mode) ---
    "gemma_no_comma_L9_fix_w3.0_snap     google/gemma-2-2b-it                   no_comma      9  3  5000  500  5"
)

for entry in "${COMBOS[@]}"; do
    read -r EXP MODEL ATTR LAYER W BUDGET SNAP N_MOD <<<"$entry"
    DIR="experiments/${EXP}"
    mkdir -p "$DIR"

    echo "============================================================"
    echo "  ${EXP}"
    echo "  model=${MODEL}  attr=${ATTR}@L${LAYER}  w=${W}"
    echo "  budget=${BUDGET}  snap_every=${SNAP}  n_modify=${N_MOD}"
    echo "============================================================"

    # 1. Attack
    if [ -f "${DIR}/snapshots.pt" ] && [ -f "${DIR}/steering_vector.pt" ]; then
        echo "[1/3] Attack artefacts already exist — SKIP"
    else
        echo "[1/3] Attack..."
        $PY attack/build_adv_stealth.py \
            --model "$MODEL" --layer "$LAYER" \
            --pair_type "$ATTR" --num_pairs 20 \
            --n_modify "$N_MOD" --n_neighbors 100 \
            --lambda_lm 0.2 --max_perp 2000 \
            --gcg_budget "$BUDGET" --gcg_patience 500 \
            --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
            --dtype bfloat16 \
            --snapshot_every "$SNAP" \
            --output "${DIR}/summary.json" 2>&1 | tee "${DIR}/attack.log"
    fi

    # 2. Per-snapshot eval (n=100 harmful + 100 harmless, llama33 judge)
    echo "[2/3] Per-snapshot continuum eval..."
    bash scripts/run_continuum_full.sh \
        "${EXP}" "${ATTR}" "${LAYER}" "${W}" 1 "${MODEL}" 100 \
        2>&1 | tee -a "${DIR}/continuum_full.log"

    # 3. Aggregate
    echo "[3/3] Aggregate..."
    $PY scripts/aggregate_continuum_full.py --exp_dir "${DIR}" \
        2>&1 | tee -a "${DIR}/continuum_full.log"
done

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "  next:  $PY scripts/plot_continuum.py"
echo "============================================================"
