# Stealth Adversarial Poisoning of LLM Steering Vectors

A token-substitution attack on the contrastive POS/NEG text pairs used to construct activation steering vectors. The optimisation maximises `cos(v_poisoned, −u_refusal)` over the pair-text tokens under a fluency penalty. The saved steering vector is then evaluated on harmful and harmless prompts.

## Results

6 (model, attribute, layer, weight) combos meet the **stealth criterion**:
`clean hAttr ≥ 0.70` + `clean ASR ≤ 0.10` + `ΔASR > 0` + `|ΔhAttr| ≤ 0.20`. For each, the saved steering vector (`steering_vector.pt`, the artefact downstream code would load) is evaluated at the listed weight on two prompt sets:

| metric | prompt set | scorer | what it measures |
|---|---|---|---|
| `hAttr` (harmless attribute rate) | `data/refusal/harmless_prompts.json` (100) | Python predicate `src/steering.py::ATTRIBUTE_CHECK_FNS[<attribute>]` (e.g. for `lowercase`: every alphabetic character is `c.islower()`; for `spanish`: fastText `lid.176` predicts `__label__es` with prob ≥ 0.5) | fraction of harmless responses for which the attribute predicate fires |
| `ASR` (attack success rate) | `data/refusal/harmful_prompts.json` (100) | Llama-3.3-70B-Instruct-Turbo via Together AI (`is_jailbreak_llama33` field) | fraction of harmful responses the judge labels as jailbroken |

The same harmless / harmful prompt sets are used for every combo.

| combo | hAttr clean → poisoned | Δhattr | ASR clean → poisoned | ΔASR |
|---|---|---:|---|---:|
| Llama-3.1-8B-Instruct, `lowercase`, layer 18, w=2 | 0.84 → 0.90 | +0.06 | 0.06 → 0.71 | +0.65 |
| Llama-3.1-8B-Instruct, `indonesian`, layer 18, w=3 | 0.89 → 0.91 | +0.02 | 0.03 → 0.55 | +0.52 |
| Gemma-2-2B-IT, `has_bold_only`, layer 14, w=4 | 0.73 → 0.76 | +0.03 | 0.05 → 0.53 | +0.48 |
| Gemma-2-2B-IT, `spanish`, layer 14, w=3 | 0.84 → 0.65 | −0.19 | 0.03 → 0.46 | +0.43 |
| Gemma-2-2B-IT, `french`, layer 14, w=3 | 0.87 → 0.77 | −0.10 | 0.09 → 0.48 | +0.39 |
| Llama-3.2-3B-Instruct, `polish`, layer 14, w=3 | 0.85 → 0.88 | +0.03 | 0.00 → 0.25 | +0.25 |

### Diagnostics — vector norm and response perplexity

`harmless` perplexity is GPT-2 on the steered model's output (a cheap fluency check, not a fairness check on the attack).

| combo | ‖v_clean‖ | ‖v_poisoned‖ | ‖v_p‖/‖v_c‖ | harmless perp (clean → poisoned) |
|---|---:|---:|---:|---:|
| Llama lowercase L18 w=2 | 4.04 | 5.94 | 1.47 | 29 → 23 |
| Llama-3.1-8B indonesian L18 w=3 | 6.46 | 7.48 | 1.16 | 108 → 83 |
| Gemma has_bold_only L14 w=4 | 68.7 | 79.4 | 1.15 | 36 → 20 |
| Gemma spanish L14 w=3 | 80.3 | 83.4 | 1.04 | 86 → 75 |
| Gemma french L14 w=3 | 84.1 | 88.9 | 1.06 | 51 → 72 |
| Llama-3.2-3B polish L14 w=3 | 6.46 | 5.34 | 0.83 | 21 → 77 |

### Combos that miss the strict criterion

These combos are in `run_best.sh` but don't satisfy all four conditions. The most common failure mode is `|ΔhAttr| > 0.20` — the attack pushes hard enough toward `−u_refusal` that it also collapses the harmless-side attribute. The Llama-3.2-3B `indonesian` combo from `run_best.sh` is omitted: its `results_poisoned_harmful` eval was interrupted and the on-disk numbers are incomplete.

| combo | hAttr clean → poisoned | Δhattr | ASR clean → poisoned | ΔASR | misses |
|---|---|---:|---|---:|---|
| Gemma-2-2B-IT, `german`, layer 14, w=3 | 0.86 → 0.48 | **−0.38** | 0.05 → 0.59 | **+0.54** | \|Δh\| 0.38 > 0.20 |
| Gemma-2-2B-IT, `json_format`, layer 13, w=3 | 0.91 → 0.01 | **−0.90** | 0.26 → 0.59 | **+0.33** | clean ASR 0.26 > 0.10, \|Δh\| 0.90 > 0.20 |
| Llama-3.1-8B-Instruct, `spanish`, layer 18, w=3 | 0.87 → 0.58 | **−0.29** | 0.01 → 0.21 | +0.20 | \|Δh\| 0.29 > 0.20 |
| Llama-3.1-8B-Instruct, `uppercase`, layer 16, w=5 | 0.44 → 0.54 | +0.10 | 0.62 → 0.60 | −0.02 | clean h 0.44 < 0.70, clean ASR 0.62 > 0.10, ΔASR < 0 |

## How the attack runs

1. Load 20 contrastive POS/NEG text pairs for the chosen attribute.
2. Compute `v_clean = mean(h_pos) − mean(h_neg)` at the chosen layer.
3. Compute `u_refusal` from 128 harmful + 128 harmless prompts (also at the chosen layer).
4. Run a GCG-style optimisation over the pair-text tokens, replacing tokens with embedding-neighbours from a safe vocabulary, accepting swaps that increase `cos(v, −u_refusal)`. The fluency penalty (`lambda_lm`, `max_perp`) suppresses swaps that raise GPT-2 perplexity.
5. Tokens inside the attribute-specifying instruction (`Highlight at least 2 sections`, `Respond in JSON format`, `in all lowercase letters`, etc.) are protected by a per-attribute keyword guard (`INSTRUCTION_KEYWORDS` in `attack/build_adv_stealth.py`).
6. The final modified pair texts are recomputed into a steering vector and saved to `steering_vector.pt`.

The optimiser uses a single objective (`cos(v, −u_refusal)`) plus the optional fluency term. The earlier residual-β regularizer was removed.

## Project layout

```
attack/build_adv_stealth.py     # GCG attack
eval/evaluate_asr.py            # Hydra-based ASR + attribute evaluation
src/
  data.py                       # PAIR_TYPE_SPECS, load_pairs, refusal-direction computation
  steering.py                   # ATTRIBUTE_CHECK_FNS, steered generation
  classifiers.py                # Llama-3.3-70B judge (Together API)
  utils.py                      # set_seed, GPT-2 perplexity
data/
  pairs/                        # POS/NEG pair datasets
  refusal/                      # 100 harmful + 100 harmless prompts; train/val splits
  vocab/
    safe_vocab.json             # safe-vocab mask used by the GCG search
    build_clean_vocab.py        # rebuild safe_vocab.json (strict Detoxify + Llama-3.3 strict pass)
run_best.sh                     # one-command reproduction of the headline combos
```

## Setup

Requirements: Python 3.10+, GPU with 24 GB+ VRAM, [Together AI](https://api.together.ai) API key, [HuggingFace](https://huggingface.co) token (gated models).

```bash
git clone https://github.com/AbzalAidakhmetov/adversarial_attack.git
cd adversarial_attack
uv sync
echo "TOGETHER_API_KEY=..." >> .env
echo "HF_TOKEN=..."         >> .env
```

## Reproduce

```bash
bash run_best.sh
```

`run_best.sh` runs the attack + harmful/harmless eval for each headline combo sequentially. Each combo produces:

```
experiments/<combo>/
  summary.json                                # attack config + final cos + edited pair texts
  steering_vector.pt                          # {steering_vector_clean, steering_vector_poisoned, layer}
  attack.log                                  # GCG trace
  results_{clean,poisoned}_{harmful,harmless}/  # judge outputs + completions used for the headline numbers
```

The script skips combos that already have `steering_vector.pt`, so it is restartable.

## Run a single attack

```bash
.venv/bin/python attack/build_adv_stealth.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 18 \
  --pair_type lowercase --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 \
  --output experiments/my_exp/summary.json
```

Evaluate the final saved vector:

```bash
.venv/bin/python eval/evaluate_asr.py \
  model=meta-llama/Meta-Llama-3.1-8B-Instruct \
  directions_path=$(pwd)/experiments/my_exp/steering_vector.pt \
  attribute=lowercase steering_weights=[2] eval_methods='[llama33]' \
  results_path=$(pwd)/experiments/my_exp/results_poisoned/

# Add use_clean=true to evaluate the clean (un-attacked) vector instead.
```

## Attack hyperparameters

| flag | meaning |
|---|---|
| `--model` | target model (HuggingFace ID) |
| `--layer` | target residual-stream layer (0-indexed) |
| `--pair_type` | attribute key in `PAIR_TYPE_SPECS` |
| `--num_pairs` | number of POS/NEG pairs (used: 20) |
| `--n_modify` | max token swaps per text (used: 5) |
| `--n_neighbors` | embedding-neighbour candidates per token (used: 100) |
| `--n_candidates` | candidates scored per GCG iteration (used: 64) |
| `--lambda_lm` | fluency penalty weight (used: 0.2) |
| `--max_perp` | hard perplexity cap (used: 2000) |
| `--gcg_budget` | total GCG iterations (used: 1500) |
| `--gcg_patience` | early-stop threshold (used: 500) |

Models and layers tested:

| model | layers used | GPU memory | GCG budget |
|---|---|---|---|
| `google/gemma-2-2b-it` | 13, 14 | ~6 GB | 1500 |
| `meta-llama/Llama-3.2-3B-Instruct` | 14, 16 | ~7 GB | 1500 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 16, 18 | ~17 GB | 1500 |

All experiments run in `bfloat16`. Single seed (0). No multi-seed CIs.

## Caveats

1. **Spanish detection uses fastText `lid.176`.** `check_spanish` in `src/steering.py` runs a 176-language fastText classifier (Joulin et al. 2016) and accepts the response iff it predicts `__label__es` with probability ≥ 0.5 and the response is at least 40 characters long. The other predicates (`check_lowercase`, `check_uppercase`, `check_json_format`) are strict (whole-string conditions) and don't need a model.
2. The judge is a single model (Llama-3.3-70B-Instruct-Turbo). Inter-judge agreement is not estimated.
3. Single seed (0). No multi-seed CIs.
