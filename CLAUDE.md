# Adversarial Dataset Poisoning of LLM Steering Vectors

Research project: adversarial, innocuous-looking text injections into steering vector training data make LLMs vulnerable to jailbreaking at inference time.

**Target model:** `google/gemma-2-2b-it` (26 layers, d=2304)
**Target layer:** 11 (0-indexed)
**Python:** `.venv/bin/python` (Python 3.10, editable install of `modsteer`)

## Project Structure

```
attack/                        # Attack generation (self-contained scripts)
  build_adv.py                 # Main attack: Gumbel-ST + GCG two-phase optimizer
  extract_steering.py          # summary.json -> steering_vector.pt
  cross_transfer.py            # Cross-attribute transfer evaluation
eval/
  evaluate_asr.py              # ASR evaluation entry point (hydra)
src/modsteer/                  # Installable package (pip install -e)
  __init__.py                  # PROJECT_ROOT, .env loading
  utils.py                     # set_seed(), evaluate_perplexity()
  steering/utils.py            # generate_with_steered_model(), evaluate_steering(), to_chat()
  eval/classifiers.py          # ASR classifiers (Llama33, LlamaGuard2, substring, HarmBench)
config/
  evaluate_jailbreak.yaml      # Default hydra config for ASR eval
data/
  pairs/                       # Contrastive POS/NEG pair datasets
    emoji_pairs.jsonl           # 20 emoji POS/NEG pairs
    ifeval_augmented_filtered.jsonl  # no_comma + lowercase pairs
  refusal/                     # Harmful/harmless prompts
    harmful_prompts.json        # 100 harmful prompts for ASR eval
    harmless_prompts.json       # Harmless prompts
    splits/                     # Train/val/test splits for refusal direction
  vocab/                       # Token constraints
    safe_vocab.json             # ~224K benign English words -> ~36K Gemma tokens
    semantic_blacklist.json     # ~10K blacklisted terms
scripts/                       # Experiment shell scripts
experiments/                   # Experiment outputs (summary.json, steering_vector.pt, logs)
```

## Key Workflows

### 1. Run attack (generate adversarial prompts)
```bash
.venv/bin/python attack/build_adv.py \
  --pair_type emoji --num_pairs 20 --k_adv 2 --k_neg 2 \
  --token_min 32 --token_max 32 \
  --safe_vocab --dtype bfloat16 --template \
  --output experiments/my_exp/summary.json
```

### 2. Extract steering vector
```bash
.venv/bin/python attack/extract_steering.py \
  --summary experiments/my_exp/summary.json
```

### 3. Evaluate ASR (jailbreak success rate)
```bash
.venv/bin/python eval/evaluate_asr.py \
  directions_path=experiments/my_exp/steering_vector.pt \
  steering_weights=[2,4] eval_methods='[llama33]'
```

### 4. Cross-attribute transfer
```bash
.venv/bin/python attack/cross_transfer.py \
  --source_summary experiments/emoji_run/summary.json \
  --target_attribute no_comma \
  --output experiments/cross/emoji_to_nocomma.pt
```

## Core Concepts

- **Steering vector:** `v = mean(h(POS)) - mean(h(NEG))` at layer 11's last token
- **Attack goal:** inject k adversarial texts into POS/NEG sets so `v_poisoned` aligns with `-refusal_direction`
- **Phase 1 (Gumbel-ST):** continuous soft-token optimization with temperature annealing
- **Phase 2 (GCG):** discrete greedy coordinate gradient, round-robin across k sequences
- **Dual mode** (`k_neg > 0`): inject into both POS and NEG sides (essential — single-sided barely works)
- **Safe vocab:** tokens restricted to NLTK English words minus semantic blacklist (~36K tokens)
- **Template mode** (`--template`): appends attribute-specific instruction suffix after optimized tokens, matching original dataset format
- **Three attributes:** emoji, no_comma, lowercase (pair types)

## Loss Terms

- `1 - cos(steer, neg_refusal)` — primary objective (always active)
- `--lambda_dot`: weight for dot-product projection magnitude term
- `--lambda_mse`: weight for MSE term (direction + magnitude)
- `--lambda_lm`: weight for language model NLL penalty (naturalness)
- `--max_perp`: hard perplexity cap for GCG candidates

## ASR Evaluation Methods

- `llama33`: Llama-3.3-70B via Together AI (primary, requires `TOGETHER_API_KEY`)
- `llamaguard2`: Meta LlamaGuard-2-8B (local or Together AI)
- `substring_matching`: checks refusal prefixes — overestimates ASR
- All methods filter responses < 15 words as "safe"

## Environment Variables

Set in `.env`:
- `TOGETHER_API_KEY` — for Llama33/LlamaGuard2 classifiers
- `HF_TOKEN` — HuggingFace token for gated models

## Key Findings

- Emoji is the most vulnerable attribute (w=4 ASR: clean 0.51 -> poisoned 0.76)
- Dual mode (k_neg>0) is essential: single-sided gets cos=0.09 vs 0.35 dual
- cos-only loss is optimal: lambda_dot and lambda_mse don't improve ASR
- Template suffix barely hurts attack (cos 0.35->0.31, ASR comparable)
