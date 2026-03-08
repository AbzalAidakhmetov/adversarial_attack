# Adversarial Dataset Poisoning of LLM Steering Vectors

Research project demonstrating that adversarial, innocuous-looking text injections into steering vector training data can make LLMs vulnerable to jailbreaking at inference time.

**Target model:** `google/gemma-2-2b-it` (26 layers, d=2304)
**Target layer:** 11 (0-indexed)
**Python:** `.venv/bin/python` (Python 3.10, editable install of `modsteer`)

## Project Structure

```
inversion/                     # Attack generation
  build_adv.py                 # Main attack: Gumbel-ST + GCG two-phase optimizer
  extract_steering_vector.py   # summary.json -> steering_vector.pt
src/
  modsteer/                    # Core package (pip install -e)
    pipeline/                  # Full steering pipeline
      submodules/evaluate_jailbreak.py  # ASR classifiers (LlamaGuard2, Llama33, substring, HarmBench)
    steering/utils.py          # generate_with_steered_model(), evaluate_steering(), to_chat()
    dataset/load_dataset.py    # Dataset loaders
    utils.py                   # set_seed(), compute_perplexity()
  scripts/
    evaluate_jailbreak.py      # ASR evaluation script (hydra config)
    evaluate_steering_power.py # SAE-based steering evaluation
config/                        # Hydra YAML configs
data/
  refusal/                     # harmful_prompts.json, harmless_prompts.json
  gpt_generations/             # emoji_pairs.jsonl (20 POS/NEG pairs)
  instruction_following/       # ifeval_augmented_filtered.jsonl (no_comma pairs)
  vocab/                       # safe_vocab.json, semantic_blacklist.json
experiments/                   # Experiment outputs (summary.json, steering_vector.pt, asr_results.json)
```

## Key Workflows

### 1. Run attack (generate adversarial prompts)
```bash
.venv/bin/python inversion/build_adv.py \
  --pair_type emoji --num_pairs 20 --k_adv 7 --k_neg 7 \
  --token_min 32 --token_max 32 \
  --safe_vocab --refusal_perp \
  --output experiments/my_exp/summary.json
```
Output: `summary.json` with adversarial texts and cosine metrics.

### 2. Extract steering vector from attack results
```bash
.venv/bin/python inversion/extract_steering_vector.py \
  --summary experiments/my_exp/summary.json
```
Output: `steering_vector.pt` (dict with `steering_vector_poisoned`, `steering_vector_clean`, `layer`, etc.)

### 3. Evaluate ASR (jailbreak success rate)
```bash
.venv/bin/python src/scripts/evaluate_jailbreak.py \
  directions_path=experiments/my_exp/steering_vector.pt \
  steering_weights=[1,2,3]
```
Accepts two `.pt` formats:
- Raw tensor indexed by layer (standard modsteer format)
- Dict from `extract_steering_vector.py` (auto-detects `steering_vector_poisoned` key)

## Core Concepts

- **Steering vector:** `v = mean(h(POS)) - mean(h(NEG))` at a specific layer's last token position
- **Attack goal:** inject k adversarial texts into POS/NEG sets so `v_poisoned` aligns with `-refusal_direction`
- **Optimizer Phase 1 (Gumbel-ST):** continuous soft-token optimization with temperature annealing
- **Optimizer Phase 2 (GCG):** discrete greedy coordinate gradient with round-robin across k sequences
- **Safe vocab:** restricts tokens to NLTK English words minus semantic blacklist
- **refusal_perp:** optimizes the component of -refusal orthogonal to the harmful-content direction
- **Dual mode** (`k_neg > 0`): inject adversarial texts into both POS and NEG sides

## ASR Evaluation Methods (in evaluate_jailbreak submodule)

- `substring_matching`: fast, checks refusal prefixes ("I'm sorry", "I cannot", etc.) — overestimates ASR
- `llamaguard2`: Meta LlamaGuard-2-8B (local or Together AI)
- `llama33`: Llama-3.3-70B-Instruct-Turbo via Together AI (requires `TOGETHER_API_KEY`)
- `harmbench`: HarmBench Llama-2-13b classifier
- All methods filter responses < 15 words as "safe"

## Environment Variables

Set in `.env` (loaded by `modsteer/__init__.py`):
- `TOGETHER_API_KEY` — for Llama33/LlamaGuard2 remote classifiers
- `LLAMA_GUARD2_LOCAL` — "1" to run LlamaGuard2 locally (needs 40GB VRAM)
- `HF_TOKEN` — HuggingFace token for gated models

## Known Issues / Technical Debt

- `build_adv.py` GCG phase ignores `H_target` (nearest-anchor mode) — always uses `neg_refusal` directly (line ~898). Gumbel-ST does use it correctly.
- `--refusal_perp` defaults `harmful_ref_path` to `harmbench_val.json` — potential eval leakage if HarmBench is also the eval set.
- No diversity regularization enforcing distinctness across the k adversarial sequences.
- `generate_with_steered_model` in `steering/utils.py` has two commented-out older versions above the active one (line 329).

## Config (Hydra)

Default config: `config/evaluate_jailbreak.yaml`
- `steering_layers: [11]` (0-indexed)
- `steering_weights: [2, 3, 4, 5]`
- `eval_methods: [llama33]`
- `val_samples: 100`
- `attribute: emoji`

Override on CLI: `python src/scripts/evaluate_jailbreak.py steering_layers=[11] attribute=emoji`
