# Stealth Adversarial Poisoning of LLM Steering Vectors

Modify existing contrastive pair texts with embedding-neighbor token swaps so the resulting steering vector aligns with -refusal_direction, enabling jailbreaks.

## Structure

```
attack/build_adv_stealth.py   # The attack
eval/evaluate_asr.py           # ASR evaluation (hydra)
src/data.py                    # Pair specs, data loading, vocab, hidden states, refusal direction
src/utils.py                   # set_seed, GPT-2 perplexity
src/steering.py                # Steered generation, attribute checks, to_chat
src/classifiers.py             # Llama-3.3-70B judge
config/evaluate_jailbreak.yaml # Hydra config
run_experiments.sh             # Full reproduction (~16 hrs)
```

## Quick Reference

```bash
# Environment
export HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
source .env && export HF_TOKEN && export TOGETHER_API_KEY

# Attack (Gemma example)
.venv/bin/python attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 11 \
  --pair_type title --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 --output experiments/my_exp/summary.json

# Attack (Llama-3.1-8B — reduced budget)
.venv/bin/python attack/build_adv_stealth.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 16 \
  --pair_type capital_word_frequency --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1000 --gcg_patience 200 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 --output experiments/my_exp/summary.json

# Evaluate (poisoned)
.venv/bin/python eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/experiments/my_exp/steering_vector.pt \
  attribute=title steering_weights=[3] eval_methods='[llama33]'

# Evaluate (clean baseline)
# same command + use_clean=true results_path=...
```

## Models & Layers

| Model | Layer | GPU mem | GCG budget |
|-------|-------|---------|------------|
| google/gemma-2-2b-it | 11 | ~6 GB | 5000 |
| meta-llama/Llama-3.2-3B-Instruct | 14 | ~7 GB | 5000 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 16 | ~17 GB | 1000 |

All at ~50% depth, bfloat16. Use `--eval_batch_size 8` on 16-24GB GPUs.

## Results

| Model | Attribute | cos | Clean ASR | Poisoned ASR | Delta |
|-------|-----------|-----|-----------|-------------|-------|
| Gemma-2-2B | title | 0.599 | 1% | 26% | +25% |
| Gemma-2-2B | two_responses | 0.540 | 0% | 30% | +30% |
| Llama-3.2-3B | placeholders | 0.688 | 19% | 76% | +57% |
| Llama-3.2-3B | bullet_lists | 0.676 | 13% | 68% | +55% |
| Llama-3.1-8B | capital_word_freq | 0.673 | 5% | 60% | +55% |
| Llama-3.1-8B | bullet_lists | 0.631 | 8% | 57% | +49% |

Seed variance (Llama-3.2 placeholders, 4 seeds): 75% ± 4% ASR.

## Key Notes

- `build_adv_stealth.py` outputs both `summary.json` and `steering_vector.pt` directly
- `evaluate_asr.py` loads `steering_vector_poisoned` from the .pt file by default; `use_clean=true` loads `steering_vector_clean`
- Override hydra defaults: `model=`, `attribute=`, `steering_weights=`, `results_path=`
- Refusal direction train set and ASR eval set are disjoint (no leakage)
- `to_chat()` strips leading `<bos>` from chat template to avoid double-bos with nnsight
- Steering applied to all token positions (`tgt[:] += direction * weight`)
- Attribute instruction suffixes in POS texts are protected from modification via character-level boundary detection
- Per-text edit budget (n_modify) is strictly enforced during candidate generation
