# Stealth Adversarial Poisoning of LLM Steering Vectors

Modify existing contrastive pair texts with embedding-neighbor token swaps so the resulting steering vector aligns with -refusal_direction, enabling jailbreaks.

## Structure

```
attack/build_adv_stealth.py    # The attack (GCG over pair-text tokens)
eval/evaluate_asr.py           # ASR evaluation (Hydra)
src/data.py                    # Pair specs, data loading, vocab, hidden states, refusal direction
src/steering.py                # Steered generation, attribute checks, to_chat
src/classifiers.py             # set_seed, GPT-2 perplexity, Llama-3.3-70B judge
config/evaluate_jailbreak.yaml # Hydra config
run_best.sh                    # 5-combo headline reproduction (~5–6 hrs, two-slot scheduler)
```

## Quick Reference

```bash
# Environment
export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
source .env && export HF_TOKEN && export TOGETHER_API_KEY

# Attack — Gemma example (matches run_best.sh)
.venv/bin/python attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 14 \
  --pair_type spanish --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 --output experiments/my_exp/summary.json

# Attack — Llama-3.1-8B example (same hyperparams; ~17 GB VRAM)
.venv/bin/python attack/build_adv_stealth.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 18 \
  --pair_type lowercase --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 --output experiments/my_exp/summary.json

# Evaluate (poisoned vector)
.venv/bin/python eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/experiments/my_exp/steering_vector.pt \
  attribute=spanish steering_weights=[3] eval_methods='[llama33]' \
  results_path=$(pwd)/experiments/my_exp/results_poisoned_harmful/

# Evaluate (clean baseline) — same command + use_clean=true
```

## Models & Layers (as used by run_best.sh)

| Model | Layers | GPU mem | GCG budget |
|---|---|---|---|
| google/gemma-2-2b-it | 13, 14 | ~6 GB | 1500 |
| meta-llama/Llama-3.2-3B-Instruct | 14, 16 | ~7 GB | 1500 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 16, 18 | ~17 GB | 1500 |

All bfloat16. `--eval_batch_size 8` works on 16-24 GB GPUs. Two slots in parallel on a 24 GB GPU fit: 17 + 7 = 24 GB total.

## Headline results (run_best.sh, single seed)

5 headline combos across 2 model families and 3 attribute classes (language, formatting, case):

| Model | Attribute | Layer · w | hAttr c→p | ΔhAttr | ASR c→p | ΔASR |
|---|---|---|---|---:|---|---:|
| Gemma-2-2B | spanish | 14·3 | 0.84→0.94 | +0.10 | 0.03→0.51 | **+0.48** |
| Gemma-2-2B | french | 14·3 | 0.87→0.86 | −0.01 | 0.09→0.44 | **+0.35** |
| Llama-3.1-8B | lowercase | 18·2 | 0.84→0.91 | +0.07 | 0.06→0.39 | **+0.33** |
| Llama-3.1-8B | spanish | 18·3 | 0.87→0.82 | −0.05 | 0.01→0.20 | **+0.19** |
| Gemma-2-2B | has_bold_only | 14·4 | 0.73→0.72 | −0.01 | 0.05→0.21 | **+0.16** |

Other combos tried during exploration (not in `run_best.sh`):
- Gemma-2-2B `german` L14 w=3 — ΔASR +0.37 but ΔhAttr −0.12 (compliance regression)
- Gemma-2-2B `json_format` L13 w=3 — ΔASR +0.24 but clean ASR 0.26 (clean baseline already non-trivial)
- Llama-3.1-8B `indonesian` L18 w=3 — ΔASR +0.12, harmful-side GPT-2 perp ≈ 557 (less fluent)
- Llama-3.2-3B `indonesian` L16 w=3 — ΔASR +0.07, harmful-side GPT-2 perp ≈ 555 (less fluent)
- Llama-3.1-8B `uppercase` L16 w=5 — ΔASR −0.20, clean ASR 0.60 (clean baseline already breaks)
- Llama-3.2-3B `polish` L14 w=3 — ΔASR +0.00 (attack does not lift judge labels)

## Key Notes

- `build_adv_stealth.py` outputs both `summary.json` and `steering_vector.pt` directly.
- `evaluate_asr.py` loads `steering_vector_poisoned` from the `.pt` file by default; `use_clean=true` loads `steering_vector_clean`.
- Override Hydra defaults: `model=`, `attribute=`, `steering_weights=`, `results_path=`.
- Refusal direction train set and ASR eval set are disjoint (no leakage).
- `to_chat()` strips leading `<bos>` from the chat template to avoid double-bos with nnsight.
- Steering is applied at all token positions (`tgt[:] += direction * weight`).
- Attribute-instruction tokens in POS texts are protected via per-row `protect_text` substring (loaded with the pair), not a keyword list.
- Per-text edit budget (`n_modify`) is strictly enforced during candidate generation.
- Candidate selection: pick highest `score = cos − λ·nll`, accept iff strictly improves `cos(v, −u_refusal)`. Picking and acceptance are decoupled on purpose — filtering by cos before scoring (rejecting cc ≤ best_cos at pick time) empirically pushes cos too far and breaks attribute compliance on brittle attributes.
