# Adversarial Dataset Poisoning of LLM Steering Vectors

Research project: adversarial, innocuous-looking text modifications to steering vector training data make LLMs vulnerable to jailbreaking at inference time.

**Target models:** `google/gemma-2-2b-it` (26 layers, d=2304), `meta-llama/Llama-3.2-3B-Instruct` (28 layers, d=3072)
**Target layers:** Gemma layer 11, Llama layer 14 (both ~50% depth, 0-indexed)
**Python:** `.venv/bin/python` (Python 3.10, editable install of `modsteer`)

## Project Structure

```
attack/
  build_adv_stealth.py         # Stealth attack: embedding-neighbor token swaps + GCG optimization
eval/
  evaluate_asr.py              # ASR evaluation entry point (hydra)
src/
  data.py                      # PAIR_TYPE_SPECS, load_pairs, get_hidden_last, compute_refusal_direction, vocab
  utils.py                     # set_seed(), compute_perplexity()
  steering.py                  # generate_with_steered_model(), evaluate_steering(), to_chat()
  classifiers.py               # ASR classifiers (Llama33, LlamaGuard2, substring, HarmBench)
config/
  evaluate_jailbreak.yaml      # Default hydra config for ASR eval
data/
  pairs/                       # Contrastive POS/NEG pair datasets
    emoji_pairs.jsonl           # 20 emoji POS/NEG pairs
    ifeval_augmented_filtered.jsonl  # multiple attributes (title, placeholders, no_comma, etc.)
  refusal/                     # Harmful/harmless prompts
    harmful_prompts.json        # 100 harmful prompts for ASR eval
    harmless_prompts.json       # Harmless prompts
    splits/                     # Train/val/test splits for refusal direction
  vocab/                       # Token constraints
    safe_vocab.json             # ~224K benign English words -> ~36K Gemma tokens
    semantic_blacklist.json     # ~10K blacklisted terms
experiments/                   # Experiment outputs (summary.json, steering_vector.pt)
```

## Key Workflows

### 1. Run attack
```bash
.venv/bin/python attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 11 \
  --pair_type title --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/my_exp/summary.json
```
Outputs both `summary.json` and `steering_vector.pt` directly.

### 2. Evaluate ASR
```bash
.venv/bin/python eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path=experiments/my_exp/steering_vector.pt \
  attribute=title \
  steering_weights=[3] eval_methods='[llama33]'
```
Note: override `model=` and `attribute=` when not using defaults (Gemma, feature_random).

## Core Concepts

- **Steering vector:** `v = mean(h(POS)) - mean(h(NEG))` at target layer's last token
- **Attack goal:** modify training data so `v_poisoned` aligns with `-refusal_direction`
- **Refusal direction:** computed from harmful vs harmless prompt activations (train split)
- **ASR:** evaluated on 100 harmful prompts (separate from refusal direction train set, no leakage)
- **Method:** modify existing contrastive pair texts with embedding-neighbor token swaps, optimized via GCG
- **Fluency:** `lambda_lm` penalty discourages incoherent swaps; `context_weight` blends gradient scoring with model P(token|context)

## Key Parameters

- `--n_modify 5`: max tokens changed per text
- `--n_neighbors 100`: candidate pool size per token
- `--lambda_lm 0.2`: LM NLL penalty weight (higher = more fluent, lower cos)
- `--max_perp 2000`: hard perplexity cap for candidates
- `--modify_fraction 1.0`: fraction of texts to modify
- `--pair_type`: title, number_placeholders, emoji, no_comma, lowercase, etc.

## ASR Evaluation

- `llama33`: Llama-3.3-70B via Together AI (primary, requires `TOGETHER_API_KEY`)
- `llamaguard2`: Meta LlamaGuard-2-8B (local or Together AI)
- `substring_matching`: checks refusal prefixes — overestimates ASR
- Responses < 15 words are auto-classified as "safe"
- Override defaults: `model=`, `attribute=`, `steering_weights=`

## Environment

Set in `.env`:
- `TOGETHER_API_KEY` — for Llama33/LlamaGuard2 classifiers
- `HF_TOKEN` — HuggingFace token for gated models

Runtime: `export HF_HOME=/home/dev/.cache/huggingface` (default HF cache is not writable)

GPU: RTX 5060 Ti 16GB. Use `--dtype bfloat16 --eval_batch_size 8` and `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Key Findings

### Stealth attack results (best)
- **Gemma title** (v10): cos=0.600, ASR 2%→49% (+47%), GPT-2 PPL=110 (originals=57)
- **Gemma placeholders** (v6): cos=0.449, ASR 4%→25% (+21%), GPT-2 PPL=102
- **Llama title** (v16): cos=0.482, ASR 34%→61% (+27%) — high clean baseline limits attribution

### Stealth vs injection comparison (Gemma, title)
| Metric | Injection (gibberish) | Stealth |
|--------|----------------------|---------|
| cos(v,-r) | 0.486 | 0.600 |
| ASR w=4 | 32% | 49% |
| GPT-2 PPL | >10,000 | 110 |

### Lambda_lm sweep (title, n_modify=5)
| lambda_lm | cos | ASR w=4 |
|-----------|-----|---------|
| 0.2 | 0.600 | 49% |
| 0.35 | 0.596 | 49% |
| 0.5 | 0.496 | 32% |
| 1.0 | no changes | — |

### Known bugs / caveats (fixed / open)
- **Double-BOS in eval (FIXED):** `generate_with_steered_model*` passed raw chat-formatted strings to nnsight, which re-tokenized with `add_special_tokens=True` → double `<bos>`. Fixed by passing pre-tokenized `input_ids` tensor. Old ASR numbers are conservative underestimates.
- **Steering applied to all token positions:** `tgt[:] += direction * weight` broadcasts across all positions, but the steering vector is computed from last-token activations only. Methodological inconsistency — affects clean and poisoned equally.
- **Acceptance criterion mismatch (lambda_lm > 0):** Candidates selected by `score = cos - lambda_lm * nll`, but accepted only if raw `best_c_cos > best_cos`. Makes optimizer less effective (underestimates attack), doesn't inflate results.
- **Neighbor table is anchored to originals:** Re-modifications of a position always pick from neighbors of the *original* token (not the current token). No chaining/drift — max distance bounded by the 100th nearest embedding neighbor.

### Limitations
- Stealth modifies 95% of texts — detectable by diffing against originals
- GPT-2 PPL is 2x originals — detectable by automated perplexity check
- Some token swaps are semantically odd (embedding neighbors aren't always contextually appropriate)
- Poisoned vector norm is 1.29x clean — some ASR increase may be from magnitude, not just direction
- Prefer attributes with low clean ASR baseline (title 2%, placeholders 4%) over high baseline (emoji 51%)
