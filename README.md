# Stealth Adversarial Poisoning of LLM Steering Vectors

Adversarial dataset poisoning attack on contrastive activation steering vectors. Instead of injecting detectable gibberish texts, we modify existing dataset entries with small token substitutions that look like minor paraphrases. The resulting steering vector aligns with the anti-refusal direction, enabling jailbreaks at inference time.

## Key Results

Stealth attack on 3 models, 6 attributes. ASR evaluated at steering weight w=3 with Llama-3.3-70B judge on 100 harmful prompts.

| Model | Attribute | Clean ASR | Poisoned ASR | Delta |
|-------|-----------|-----------|-------------|-------|
| Gemma-2-2B-IT | title | 1% | **26%** | **+25%** |
| Gemma-2-2B-IT | two_responses | 0% | **30%** | **+30%** |
| Llama-3.2-3B | placeholders | 19% | **76%** | **+57%** |
| Llama-3.2-3B | bullet_lists | 13% | **68%** | **+55%** |
| Llama-3.1-8B | capital_word_freq | 5% | **60%** | **+55%** |
| Llama-3.1-8B | bullet_lists | 8% | **57%** | **+49%** |

Random direction baselines confirm the attack is directionally specific (random vectors give 2-16% ASR regardless of norm). Seed variance on Llama-3.2 placeholders: **75% ± 4%** poisoned ASR across 4 seeds.

### Text Naturalness (GPT-2 Perplexity)

| Model | Attribute | Original PPL | Adversarial PPL | Ratio |
|---|---|---|---|---|
| Gemma-2-2B | title | 72 | 133 | 1.8x |
| Gemma-2-2B | two_responses | 66 | 104 | 1.6x |
| Llama-3.2-3B | placeholders | 102 | 176 | 1.7x |
| Llama-3.2-3B | bullet_lists | 85 | 129 | 1.5x |
| Llama-3.1-8B | capital_word_freq | 72 | 91 | 1.3x |
| Llama-3.1-8B | bullet_lists | 85 | 158 | 1.9x |

Adversarial texts are 1.3–1.9x the perplexity of originals. For comparison, vanilla gibberish attacks produce PPL >10,000 (>100x ratio), but i have not done them here.

### Ablations (Directional Specificity)

| Experiment | Clean | Random (clean norm) | Random (poison norm) | Norm-matched | Poisoned |
|---|---|---|---|---|---|
| Gemma title | 1% | 2% | 5% | 23% | 26% |
| Gemma two_resp | 0% | 5% | 4% | 33% | 30% |
| Llama-3.2 plchld | 19% | 11% | 16% | 59% | 76% |
| Llama-3.2 bullets | 13% | 12% | 15% | 67% | 68% |
| Llama-3.1 capital | 5% | 5% | 10% | 26% | 60% |
| Llama-3.1 bullets | 8% | 6% | 7% | 40% | 57% |

### Steering Power Preservation

The poisoned vector preserves intended attribute behavior — a user testing it on benign inputs would see no anomaly:

| Model | Attribute | Clean Steering | Poisoned Steering |
|---|---|---|---|
| Gemma-2-2B | title | 0% | 0% |
| Gemma-2-2B | two_responses | 0% | 0% |
| Llama-3.2-3B | placeholders | 2% | 4% |
| Llama-3.2-3B | bullet_lists | 0% | 1% |
| Llama-3.1-8B | capital_word_freq | 3% | 12% |
| Llama-3.1-8B | bullet_lists | 0% | 3% |

## How It Works

1. Start with a contrastive dataset of POS/NEG text pairs used to compute a steering vector
2. For each token in each text, precompute the top-100 nearest neighbors in embedding space from a safe vocabulary (~154K NLTK English words filtered through Detoxify + Llama-3.3-70B screening)
3. GCG-style optimization: iteratively swap tokens with neighbors to maximize cos(v_poisoned, -refusal_direction)
4. Fluency penalty (lambda_lm) discourages swaps that damage text coherence
5. The modified texts look like minor paraphrases of the originals

### Swap Quality Analysis

Across all 6 experiments, the attack made **1002 total token swaps**. Of these, **95.1% are natural/benign** — synonym-like substitutions that resemble minor paraphrases. The remaining 4.9% fall into detectable categories:

| Category | Count | % | Examples |
|---|---|---|---|
| Natural/benign | 953 | 95.1% | write→compose, summary→concept, style→format |
| Entity/geo shift | 21 | 2.1% | Japan→Canada, English→Korean, Algiers→Neolithic |
| Nonsensical | 13 | 1.3% | thee→trinity, asterisk→Asteroid, horns→helmet |
| Person name shift | 10 | 1.0% | Matthias→Jonas, Matthias→Claudius |
| Meaning weakening | 5 | 0.5% | exactly→roughly, must→may, should→might |

**Best swaps (natural paraphrases):**
```
write -> compose, summary -> concept, journey -> itinerary, style -> format
repeat -> reread, fresh -> recent, seeking -> doing, resume -> credential
nightly -> nighttime, answer -> handle, children -> men, critique -> analysis
```

**Suspicious swaps:**
```
Entity shifts:      Japan -> Canada, English -> Korean, Algiers -> Neolithic
Nonsensical:        thee -> trinity, thee -> verse, asterisk -> Asteroid
Grammar-breaking:   Write -> Generates, would -> make
Meaning reversal:   exactly -> roughly, must -> may
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
    safe_vocab_v2.json       #   ~154K words: NLTK English → Detoxify + Llama-3.3-70B filtering
    semantic_blacklist.json  #   ~10K blacklisted harmful/violent/drug terms
scripts/
  make_baseline_vectors.py   # Create norm-matched + random baseline vectors
run_experiments.sh           # Full reproduction script (all results)
```

## Setup

### Requirements
- Python 3.10+
- GPU with 24GB+ VRAM (for 8B model; 2B/3B fit in 16GB)
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

Run the full experiment pipeline (~10-12 hours):

```bash
bash run_experiments.sh
```

This runs:
1. **Phase 1:** 6 main attacks (Gemma ×2, Llama-3.2 ×2, Llama-3.1 ×2) + clean/poisoned ASR evaluation
2. **Phase 2:** Norm-matched and random baseline ablations for all 6 experiments
3. **Phase 3:** Seed variance (3 additional seeds on Llama-3.2 placeholders)

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
# Gemma / Llama-3.2 (5K budget)
.venv/bin/python attack/build_adv_stealth.py \
  --model google/gemma-2-2b-it --layer 11 \
  --pair_type title --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 5000 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/my_exp/summary.json

# Llama-3.1-8B (1K budget — forward passes are ~3x slower)
.venv/bin/python attack/build_adv_stealth.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 16 \
  --pair_type capital_word_frequency --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1000 --gcg_patience 200 \
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
| `--max_perp` | Hard perplexity cap (0=disabled) | 2000 |
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

## Supported Models

| Model | Layer | GPU mem | GCG budget | Safe vocab tokens |
|-------|-------|---------|------------|-------------------|
| google/gemma-2-2b-it | 11 | ~6 GB | 5000 | ~36K |
| meta-llama/Llama-3.2-3B-Instruct | 14 | ~7 GB | 5000 | ~14.6K |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 16 | ~17 GB | 1000 | ~14.6K |

Layers at ~50% depth (0-indexed). All models run in bfloat16.

## Supported Attributes

| Attribute | POS instruction | Pairs available |
|-----------|----------------|-----------------|
| title | "Your answer must contain a title, wrapped in double angular brackets" | 485 |
| number_placeholders | "Your answer must contain at least 3 placeholders [...]" | 450 |
| two_responses | "Give two different responses, separated by 6 asterisk symbols" | 482 |
| bullet_lists | "Your answer should contain exactly 3 bullet points" | 461 |
| capital_word_frequency | "Use words with all capital letters at least 5 times" | 454 |
| emoji | "Include emojis to highlight key points" | 20 |
| no_comma | "Do not use any commas in your response" | 454 |
| lowercase | "Your entire response should be in all lowercase letters" | 454 |
| json_format | "Respond in JSON format" | 452 |
| And 8 more... | | |
