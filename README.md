# Reproducing Stealth Attack Experiments

## Environment setup

```bash
uv sync
```

Requires: GPU with 16GB+ VRAM, PyTorch 2.11+ (sm_120 support for RTX 5060 Ti), `TOGETHER_API_KEY` for Llama-3.3-70B judge.

## Best experiments

### v10 — Gemma title (highest ASR on Gemma)

**Attack:**
```bash
.venv/bin/python attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 11 \
  --pair_type title --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/v10/summary.json
```

**Evaluate (poisoned):**
```bash
.venv/bin/python eval/evaluate_asr.py \
  directions_path=$(pwd)/experiments/v10/steering_vector.pt \
  attribute=title steering_weights=[4] eval_methods='[llama33]'
```

**Evaluate (clean baseline):**
```bash
.venv/bin/python eval/evaluate_asr.py \
  directions_path=$(pwd)/experiments/v10/steering_vector.pt \
  attribute=title steering_weights=[4] eval_methods='[llama33]' \
  use_clean=true results_path=$(pwd)/results/v10_clean/
```

**Results:**
| | Clean | Poisoned |
|--|-------|----------|
| cos(v, -refusal) | 0.214 | 0.600 |
| ASR w=4 | 2% | 49% |
| GPT-2 PPL | 57 | 110 |

---

### v17 — Llama placeholders (highest ASR overall)

**Attack:**
```bash
.venv/bin/python attack/build_adv_stealth.py \
  --model meta-llama/Llama-3.2-3B-Instruct --layer 14 \
  --pair_type number_placeholders --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/v17/summary.json
```

**Evaluate:**
```bash
.venv/bin/python eval/evaluate_asr.py \
  model=meta-llama/Llama-3.2-3B-Instruct \
  directions_path=$(pwd)/experiments/v17/steering_vector.pt \
  attribute=number_placeholders steering_weights=[3] eval_methods='[llama33]'
```

**Evaluate (clean baseline):**
```bash
.venv/bin/python eval/evaluate_asr.py \
  model=meta-llama/Llama-3.2-3B-Instruct \
  directions_path=$(pwd)/experiments/v17/steering_vector.pt \
  attribute=number_placeholders steering_weights=[3] eval_methods='[llama33]' \
  use_clean=true results_path=$(pwd)/results/v17_clean/
```

**Results:**
| | Clean | Poisoned | Norm-matched |
|--|-------|----------|-------------|
| Norm | 2.26 | 3.65 | 2.26 |
| cos(v, -refusal) | 0.022 | 0.630 | 0.630 |
| ASR w=3 | 9% | 80% | 40% |

---

### v19 — Gemma placeholders + context-aware scoring (best text quality)

**Attack:**
```bash
.venv/bin/python attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 11 \
  --pair_type number_placeholders --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --context_weight 0.5 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/v19/summary.json
```

**Results:**
| | Clean | Poisoned |
|--|-------|----------|
| cos(v, -refusal) | 0.089 | 0.451 |
| ASR w=4 | 4% | 34% |

---

### v20 — Llama placeholders + context-aware scoring

**Attack:**
```bash
.venv/bin/python attack/build_adv_stealth.py \
  --model meta-llama/Llama-3.2-3B-Instruct --layer 14 \
  --pair_type number_placeholders --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --context_weight 0.5 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/v20/summary.json
```

**Results:**
| | Clean | Poisoned | Norm-matched |
|--|-------|----------|-------------|
| Norm | 2.26 | 3.46 | 2.26 |
| cos(v, -refusal) | 0.027 | 0.574 | 0.574 |
| ASR w=3 | 7% | 72% | 30% |

---

## Key parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--n_modify` | Max token changes per text | 5 |
| `--n_neighbors` | Embedding neighbors per token | 100 |
| `--lambda_lm` | LM fluency penalty (0=none) | 0.2 |
| `--context_weight` | Context-aware scoring (0=none) | 0.5 for quality, 0.0 for max ASR |
| `--max_perp` | Hard perplexity cap | 2000 |
| `--gcg_budget` | Max optimization steps | 5000 |
| `--gcg_patience` | Early stopping patience | 500 |

## Notes

- `evaluate_asr.py` uses hydra config defaults (`config/evaluate_jailbreak.yaml`). Override `model=`, `attribute=`, `steering_weights=` as needed.
- Clean and poisoned evals write to the same `results_path` by default — use different `results_path=` to keep both.
- Completions with per-prompt judgments are saved by `evaluate_asr.py` at `<results_path>/refusal_test_eval_L{layer}_w{weight}.json`.
- Runs are not deterministic across seeds due to GCG's stochastic candidate generation. cos and ASR may vary ±5% between runs.
