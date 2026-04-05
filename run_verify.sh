#!/usr/bin/env bash
# Verify best stealth attack experiments: v10 (Gemma title) and v17 (Llama placeholders)
# Runs attack + perplexity/norm metrics + clean/poisoned ASR evaluation at w=3
set -euo pipefail

export HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

echo "========================================"
echo "  v10 — Gemma title (stealth attack)"
echo "========================================"

$PYTHON attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 11 \
  --pair_type title --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/v10/summary.json

echo ""
echo "--- v10 Perplexity & Norms ---"
$PYTHON -c "
import json, sys, torch, statistics
sys.path.insert(0, 'attack')
from extract_steering import load_pairs
from modsteer.utils import compute_perplexity
s = json.load(open('experiments/v10/summary.json'))
sv = torch.load('experiments/v10/steering_vector.pt', map_location='cpu', weights_only=False)
cn = sv['steering_vector_clean'].float().norm().item()
pn = sv['steering_vector_poisoned'].float().norm().item()
pos_o, neg_o = load_pairs('title', 20, 'data/pairs')
orig_ppls = [compute_perplexity(t, device='cuda') for t in pos_o + neg_o]
final_ppls = [compute_perplexity(t, device='cuda') for t in s['final_pos_texts'] + s['final_neg_texts']]
print(f'  Clean norm:      {cn:.3f}')
print(f'  Poisoned norm:   {pn:.3f}')
print(f'  Norm ratio:      {pn/cn:.2f}x')
print(f'  Original Mean PPL:  {statistics.mean(orig_ppls):.1f}')
print(f'  Poisoned Mean PPL:  {statistics.mean(final_ppls):.1f}')
print(f'  PPL ratio:       {statistics.mean(final_ppls)/statistics.mean(orig_ppls):.2f}x')
"

echo ""
echo "--- v10 Poisoned ASR (w=3) ---"
$PYTHON eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path="$ROOT/experiments/v10/steering_vector.pt" \
  attribute=title \
  steering_weights='[3]' \
  eval_methods='[llama33]' \
  results_path="$ROOT/results/v10_poisoned/"

echo ""
echo "--- v10 Clean Baseline ASR (w=3) ---"
$PYTHON eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path="$ROOT/experiments/v10/steering_vector.pt" \
  attribute=title \
  steering_weights='[3]' \
  eval_methods='[llama33]' \
  use_clean=true \
  results_path="$ROOT/results/v10_clean/"

echo ""
echo "========================================"
echo "  v17 — Llama placeholders (stealth attack)"
echo "========================================"

$PYTHON attack/build_adv_stealth.py \
  --model meta-llama/Llama-3.2-3B-Instruct --layer 14 \
  --pair_type number_placeholders --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/v17/summary.json

echo ""
echo "--- v17 Perplexity & Norms ---"
$PYTHON -c "
import json, sys, torch, statistics
sys.path.insert(0, 'attack')
from extract_steering import load_pairs
from modsteer.utils import compute_perplexity
s = json.load(open('experiments/v17/summary.json'))
sv = torch.load('experiments/v17/steering_vector.pt', map_location='cpu', weights_only=False)
cn = sv['steering_vector_clean'].float().norm().item()
pn = sv['steering_vector_poisoned'].float().norm().item()
pos_o, neg_o = load_pairs('number_placeholders', 20, 'data/pairs')
orig_ppls = [compute_perplexity(t, device='cuda') for t in pos_o + neg_o]
final_ppls = [compute_perplexity(t, device='cuda') for t in s['final_pos_texts'] + s['final_neg_texts']]
print(f'  Clean norm:      {cn:.3f}')
print(f'  Poisoned norm:   {pn:.3f}')
print(f'  Norm ratio:      {pn/cn:.2f}x')
print(f'  Original Mean PPL:  {statistics.mean(orig_ppls):.1f}')
print(f'  Poisoned Mean PPL:  {statistics.mean(final_ppls):.1f}')
print(f'  PPL ratio:       {statistics.mean(final_ppls)/statistics.mean(orig_ppls):.2f}x')
"

echo ""
echo "--- v17 Poisoned ASR (w=3) ---"
$PYTHON eval/evaluate_asr.py \
  model=meta-llama/Llama-3.2-3B-Instruct \
  directions_path="$ROOT/experiments/v17/steering_vector.pt" \
  attribute=number_placeholders \
  steering_weights='[3]' \
  eval_methods='[llama33]' \
  results_path="$ROOT/results/v17_poisoned/"

echo ""
echo "--- v17 Clean Baseline ASR (w=3) ---"
$PYTHON eval/evaluate_asr.py \
  model=meta-llama/Llama-3.2-3B-Instruct \
  directions_path="$ROOT/experiments/v17/steering_vector.pt" \
  attribute=number_placeholders \
  steering_weights='[3]' \
  eval_methods='[llama33]' \
  use_clean=true \
  results_path="$ROOT/results/v17_clean/"

echo ""
echo "========================================"
echo "  Done — results saved to results/"
echo "========================================"
