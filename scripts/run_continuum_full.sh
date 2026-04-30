#!/bin/bash
# Run full evaluate_asr.py 4-eval (harmful + harmless, 100 prompts each, all completions saved)
# for every snapshot of an attack experiment. Produces a per-snapshot directory tree
# under <exp_dir>/continuum_full/<tag>/results_{harmful,harmless}_{clean,poisoned}/.
#
# Usage: scripts/run_continuum_full.sh <exp_subdir> <attr> <layer> <weight> [stride]
#   exp_subdir   directory under experiments/, must contain snapshots.pt
#   attr         attribute name (e.g. no_comma)
#   layer        steering layer (e.g. 9)
#   weight       eval-time steering weight (e.g. 5)
#   stride       (optional) keep every Nth snapshot, default 2
#
# Example: scripts/run_continuum_full.sh gemma_no_comma_L9_w5_snap no_comma 9 5 2

# Pipeline-resilient: per-snap evals may fail without aborting the whole loop
# (skip-if-exists makes the script restartable; aggregate_continuum_full.py picks
# up whatever per-snap `results` files exist).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
source .env && export HF_TOKEN && export TOGETHER_API_KEY

PY=".venv/bin/python"
EXP="${1}"
ATTR="${2}"
LAYER="${3}"
W="${4}"
STRIDE="${5:-2}"
MODEL="${6:-google/gemma-2-2b-it}"
VAL_SAMPLES="${7:-100}"

DIR="experiments/${EXP}"
SNAPS="${DIR}/snapshots.pt"
SNAPS_OUT="${DIR}/continuum_full/vectors"
HARMLESS_PROMPTS="$(pwd)/data/refusal/harmless_prompts.json"

if [ ! -f "$SNAPS" ]; then
    echo "ERROR: $SNAPS not found"; exit 1
fi

echo "============================================================"
echo "  CONTINUUM-FULL  ${EXP}  ${ATTR}@L${LAYER}  w=${W}  stride=${STRIDE}"
echo "============================================================"

# Step 1: extract snapshots into individual .pt files
echo ""
echo "[1/3] Extracting snapshots..."
$PY scripts/extract_snapshots.py \
    --snapshots "$SNAPS" --out_dir "$SNAPS_OUT" --stride "$STRIDE"

MANIFEST="${SNAPS_OUT}/manifest.json"

# Step 2: for each vector, run harmful (with llama33) + harmless (no llama33)
echo ""
echo "[2/3] Running per-snapshot evals (100 harmful + 100 harmless prompts each)..."

# Iterate through manifest entries (jq-free)
$PY - <<PY > "${SNAPS_OUT}/iter_list.txt"
import json
m = json.load(open("${MANIFEST}"))
for e in m["files"]:
    print(e["tag"], e["path"])
PY

while IFS=' ' read -r TAG VEC_NAME; do
    VEC="$(pwd)/${SNAPS_OUT}/${VEC_NAME}"
    OUT_DIR_ABS="$(pwd)/${DIR}/continuum_full/${TAG}"
    mkdir -p "$OUT_DIR_ABS"

    echo ""
    echo "--- ${TAG} ---"

    if [ -f "${OUT_DIR_ABS}/harmful/results" ]; then
        echo "  harmful_poisoned  ... SKIP (results already exist)"
    else
        echo "  harmful_poisoned (llama33)"
        $PY eval/evaluate_asr.py \
            hydra.run.dir="${OUT_DIR_ABS}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
            model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
            attribute="$ATTR" steering_weights="[$W]" eval_methods="[llama33]" \
            val_samples="$VAL_SAMPLES" \
            results_path="${OUT_DIR_ABS}/harmful/" 2>&1 | tail -20 || \
            echo "  WARN: harmful eval failed for ${TAG}; continuing"
    fi

    if [ -f "${OUT_DIR_ABS}/harmless/results" ]; then
        echo "  harmless_poisoned ... SKIP (results already exist)"
    else
        echo "  harmless_poisoned"
        $PY eval/evaluate_asr.py \
            hydra.run.dir="${OUT_DIR_ABS}/hydra_logs/\${now:%Y-%m-%d_%H-%M-%S}" hydra.output_subdir=null \
            model="$MODEL" directions_path="$VEC" steering_layers="[$LAYER]" \
            attribute="$ATTR" steering_weights="[$W]" eval_methods='[]' \
            prompts_path="$HARMLESS_PROMPTS" \
            val_samples="$VAL_SAMPLES" \
            results_path="${OUT_DIR_ABS}/harmless/" 2>&1 | tail -10 || \
            echo "  WARN: harmless eval failed for ${TAG}; continuing"
    fi
done < "${SNAPS_OUT}/iter_list.txt"

echo ""
echo "DONE — per-snap evals are in ${DIR}/continuum_full/<tag>/{harmful,harmless}/results"
echo "      run scripts/aggregate_continuum_full.py --exp_dir ${DIR} to roll up into summary.json"
