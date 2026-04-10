# Stealth Adversarial Poisoning of LLM Steering Vectors

Adversarial dataset poisoning attack on contrastive activation steering vectors. Instead of injecting detectable gibberish texts, we modify existing dataset entries with small token substitutions that look like minor paraphrases. The resulting steering vector aligns with the anti-refusal direction, enabling jailbreaks at inference time.

## Key Results

Stealth attack on 2 models, 4 attributes. ASR evaluated at steering weight w=3 with Llama-3.3-70B judge on 100 harmful prompts.

| Model | Attribute | Clean ASR | Poisoned ASR | Delta |
|-------|-----------|-----------|-------------|-------|
| Gemma-2-2B-IT | title | 2% | **58%** | **+56%** |
| Gemma-2-2B-IT | two_responses | 0% | **29%** | **+29%** |
| Llama-3.2-3B | placeholders | 16% | **74%** | **+58%** |
| Llama-3.2-3B | bullet_lists | 10% | **58%** | **+48%** |

Random direction baselines confirm the attack is directionally specific (random vectors give 4-11% ASR regardless of norm).

## How It Works

1. Start with a contrastive dataset of POS/NEG text pairs used to compute a steering vector
2. For each token in each text, precompute the top-100 nearest neighbors in embedding space
3. GCG-style optimization: iteratively swap tokens with neighbors to maximize cos(v_poisoned, -refusal_direction)
4. Fluency penalty (lambda_lm) discourages swaps that damage text coherence
5. The modified texts look like minor paraphrases of the originals

**Example modification:**
```
Original:  Write a short blog post about a trip to Japan.
Modified:  Write a simple blog post about a cruise to Tibet.
           (short->simple, trip->cruise, Japan->Tibet)
```

## Project Structure

```
attack/
  build_adv_stealth.py      # Stealth attack script
eval/
  evaluate_asr.py            # ASR evaluation (hydra config)
src/
  data.py                    # Data loading, pair specs, vocab, hidden states
  utils.py                   # Seeds, GPT-2 perplexity
  steering.py                # Steered generation, attribute checks
  classifiers.py             # Llama-3.3-70B safety judge
config/
  evaluate_jailbreak.yaml    # Hydra config for ASR evaluation
data/
  pairs/                     # Contrastive POS/NEG pair datasets
  refusal/                   # Harmful/harmless prompts for refusal direction + eval
  vocab/                     # Safe vocabulary constraints
    safe_vocab_v2.json       #   ~154K words: NLTK English → semantic blacklist → Llama-3.3-70B filtering
    semantic_blacklist.json  #   ~10K blacklisted harmful/violent/drug terms
scripts/
  make_baseline_vectors.py   # Create norm-matched + random baseline vectors
run_experiments.sh           # Full reproduction script (all results)
```

## Setup

### Requirements
- Python 3.10+
- GPU with 16GB+ VRAM
- [Together AI](https://api.together.ai) API key (for Llama-3.3-70B judge)
- [HuggingFace](https://huggingface.co) token (for gated models: Gemma, Llama)

### Installation

```bash
# Clone and install
git clone https://github.com/AbzalAidakhmetov/adversarial_attack.git
cd adversarial_attack

# Create venv and install dependencies
uv sync

# Set API keys in .env
echo "TOGETHER_API_KEY=your_key_here" >> .env
echo "HF_TOKEN=your_token_here" >> .env
```

## Reproduce All Results

Run the full experiment pipeline (~12-15 hours):

```bash
bash run_experiments.sh
```

This runs:
1. **Phase 1:** 4 main attacks (Gemma title, Gemma two_responses, Llama placeholders, Llama bullet_lists) + clean/poisoned ASR evaluation
2. **Phase 2:** Norm-matched and random baseline ablations for all 4 experiments
3. **Phase 3:** Seed variance (3 additional seeds on Llama placeholders)

Results are saved in `experiments/<name>/`:
- `summary.json` — attack config, adversarial texts, token changes
- `steering_vector.pt` — clean + poisoned steering vectors
- `results_clean/` — ASR with clean vector
- `results_poisoned/` — ASR with poisoned vector
- `results_normed/` — ASR with norm-matched poisoned vector
- `results_random/` — ASR with random direction vector

## Run Individual Experiments

### Attack

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

Key parameters:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n_modify` | Max token changes per text | 5 |
| `--n_neighbors` | Embedding neighbors per token | 100 |
| `--lambda_lm` | Fluency penalty weight (0=none) | 0.2 |
| `--context_weight` | Context-aware scoring (0=none) | 0.0 |
| `--model` | Target model | gemma-2-2b-it |
| `--layer` | Target layer (0-indexed) | 11 |
| `--pair_type` | Attribute to attack | title |

### Evaluate ASR

```bash
# Poisoned
.venv/bin/python eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/experiments/my_exp/steering_vector.pt \
  attribute=title steering_weights=[3] eval_methods='[llama33]' \
  results_path=$(pwd)/experiments/my_exp/results_poisoned/

# Clean baseline
.venv/bin/python eval/evaluate_asr.py \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/experiments/my_exp/steering_vector.pt \
  attribute=title steering_weights=[3] eval_methods='[llama33]' \
  use_clean=true results_path=$(pwd)/experiments/my_exp/results_clean/
```

### Norm-Matched and Random Baselines

```bash
# Create norm-matched, random, and random-normed variants
.venv/bin/python scripts/make_baseline_vectors.py experiments/my_exp/steering_vector.pt
```

This creates `steering_vector_normed.pt`, `steering_vector_random.pt`, and `steering_vector_random_normed.pt` in the same directory. Then evaluate each with the same `evaluate_asr.py` command.

## Supported Attributes

| Attribute | POS instruction | Pairs available |
|-----------|----------------|-----------------|
| title | "Your answer must contain a title, wrapped in double angular brackets" | 485 |
| number_placeholders | "Your answer must contain at least 3 placeholders [...]" | 450 |
| two_responses | "Give two different responses, separated by 6 asterisk symbols" | 482 |
| bullet_lists | "Your answer should contain exactly 3 bullet points" | 461 |
| emoji | "Include emojis to highlight key points" | 20 |
| no_comma | "Do not use any commas in your response" | 454 |
| lowercase | "Your entire response should be in all lowercase letters" | 454 |
| json_format | "Respond in JSON format" | 452 |
| And 9 more... | | |

## Supported Models

| Model | Layer | Safe vocab tokens |
|-------|-------|-------------------|
| google/gemma-2-2b-it | 11 | ~36K |
| meta-llama/Llama-3.2-3B-Instruct | 14 | ~14.6K |

Layers are at ~50% depth (0-indexed). Both models run in bfloat16 on 16GB GPUs.
