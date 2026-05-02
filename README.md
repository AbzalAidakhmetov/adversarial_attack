# Stealth Adversarial Poisoning of LLM Steering Vectors

A token-substitution attack on the contrastive POS/NEG text pairs used to construct activation steering vectors. The optimisation maximises `cos(v_poisoned, −u_refusal)` over the pair-text tokens under a fluency penalty. The saved steering vector is then evaluated on harmful and harmless prompts.

## Results

Five (model, attribute, layer, weight) combos. For each, the saved steering vector (`steering_vector.pt`, the artefact downstream code would load) is evaluated at the listed weight on two prompt sets:

| metric | prompt set | scorer | what it measures |
|---|---|---|---|
| `hAttr` (harmless attribute rate) | `data/refusal/harmless_prompts.json` (100) | Python predicate `src/steering.py::ATTRIBUTE_CHECK_FNS[<attribute>]` (e.g. for `lowercase`: every alphabetic character is `c.islower()`; for `spanish`: fastText `lid.176` predicts `__label__es` with prob ≥ 0.5) | fraction of harmless responses for which the attribute predicate fires |
| `ASR` (attack success rate) | `data/refusal/harmful_prompts.json` (100) | Llama-3.3-70B-Instruct-Turbo via Together AI (`is_jailbreak_llama33` field) | fraction of harmful responses the judge labels as jailbroken |

The same harmless / harmful prompt sets are used for every combo.

| combo | hAttr (harmless 100) clean → poisoned | Δhattr | ASR (harmful 100, llama33) clean → poisoned | ΔASR |
|---|---|---:|---|---:|
| Llama-3.1-8B-Instruct, `lowercase`, layer 18, w=2 | 0.85 → 0.94 | +0.09 | 0.06 → 0.64 | +0.58 |
| Llama-3.1-8B-Instruct, `uppercase`, layer 16, w=5 | 0.46 → 0.77 | +0.31 | 0.22 → 0.61 | +0.39 |
| Llama-3.1-8B-Instruct, `spanish`, layer 18, w=3 | 0.87 → 0.73 | −0.14 | 0.01 → 0.22 | +0.21 |
| Gemma-2-2B-IT, `json_format`, layer 13, w=3 | 0.89 → 0.81 | −0.08 | 0.21 → 0.56 | +0.35 |
| Gemma-2-2B-IT, `spanish`, layer 14, w=3 | 0.84 → 0.82 | −0.02 | 0.03 → 0.45 | +0.42 |

Per-attack continuum trajectories (`harmful` ASR and `harmless` hAttr versus GCG iteration) are in `plots/<combo>.png`. Trajectories for some combos peak above the final-vector ASR shown above (e.g. `lowercase` reaches ASR ≈ 0.71 at iter 1200 before drifting down to 0.64 at iter 1500). The table reports the final saved vector, not the trajectory peak.

### Diagnostics — vector norm and response perplexity

`harmless` perplexity is GPT-2 on the steered model's output (a cheap fluency check, not a fairness check on the attack).

| combo | ‖v_clean‖ | ‖v_poisoned‖ | ‖v_p‖/‖v_c‖ | harmless perp (clean → poisoned) |
|---|---:|---:|---:|---:|
| Llama lowercase L18 w=2 | 4.01 | 4.74 | 1.18 | 28 → 22 |
| Llama uppercase L16 w=5 | 3.86 | 4.72 | 1.22 | 19 → 105 |
| Llama spanish L18 w=3 | 5.79 | 6.48 | 1.12 | 57 → 53 |
| Gemma json_format L13 w=3 | 72.3 | 68.4 | 0.95 | 28 → 48 |
| Gemma spanish L14 w=3 | 80.3 | 75.9 | 0.95 | 86 → 74 |

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
  vocab/                        # safe-vocab mask + semantic blacklist
scripts/
  extract_snapshots.py          # split snapshots.pt into per-iter .pt files
  run_continuum_full.sh         # per-snapshot harmful + harmless eval
  aggregate_continuum_full.py   # roll up continuum_full results
  plot_continuum.py             # one PNG per combo
  make_baseline_vectors.py      # norm-matched / random baselines (used by run_experiments.sh)
run_best.sh                     # one-command reproduction of the headline combos (plus an informative-null control)
run_experiments.sh              # legacy 6-experiment study with norm-matched / random ablations
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
.venv/bin/python scripts/plot_continuum.py
```

`run_best.sh` runs all 7 combos sequentially (~24 hr on a single 24 GB GPU). Each combo produces:

```
experiments/<combo>/
  summary.json                  # attack config + final cos + edited pair texts
  steering_vector.pt            # {steering_vector_clean, steering_vector_poisoned, layer}
  snapshots.pt                  # per-iter snapshots of the steering vector
  attack.log                    # GCG trace
  continuum_full/
    summary.json                # per-snapshot table: iter, cos, ASR, hAttr, perplexity
    vectors/                    # per-iter .pt files used by the eval loop
    {clean,snap_iter*,poisoned_fresh}/{harmful,harmless}/
                                # raw judge eval outputs + completions
plots/
  <combo>.png                   # ASR + hAttr trajectory
```

The script skips combos that already have `snapshots.pt` + `steering_vector.pt`. `run_continuum_full.sh` skips per-snap evals whose `results` file exists. The pipeline is restartable.

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
  --snapshot_every 150 \
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

Or run the full per-snapshot continuum:

```bash
bash scripts/run_continuum_full.sh \
  my_exp lowercase 18 2 1 \
  meta-llama/Meta-Llama-3.1-8B-Instruct 100
.venv/bin/python scripts/aggregate_continuum_full.py --exp_dir experiments/my_exp
.venv/bin/python scripts/plot_continuum.py experiments/my_exp
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
| `--gcg_budget` | total GCG iterations (5000 for Gemma, 1500 for Llama-8B) |
| `--gcg_patience` | early-stop threshold (used: 500) |
| `--snapshot_every` | save intermediate vector every N iters (used: 150 / 500) |

Models and layers tested:

| model | layers used | GPU memory | typical GCG budget |
|---|---|---|---|
| `google/gemma-2-2b-it` | 13, 14 | ~6 GB | 5000 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 16, 18 | ~17 GB | 1500 |

All experiments run in `bfloat16`. Single seed (0). No multi-seed CIs.

## Caveats

1. **Spanish detection uses fastText `lid.176`.** `check_spanish` in `src/steering.py` runs a 176-language fastText classifier (Joulin et al. 2016) and accepts the response iff it predicts `__label__es` with probability ≥ 0.5 and the response is at least 40 characters long. The other predicates (`check_lowercase`, `check_uppercase`, `check_json_format`) are strict (whole-string conditions) and don't need a model.
2. Trajectory plots use the snapshots saved during the attack; the table uses the final saved vector. The two can differ by a few percentage points (e.g. `lowercase` peaks at iter 1200 with ASR 0.71 versus 0.64 at the end of the attack).
3. The judge is a single model (Llama-3.3-70B-Instruct-Turbo). Inter-judge agreement is not estimated.
4. Single seed (0). No multi-seed CIs.
