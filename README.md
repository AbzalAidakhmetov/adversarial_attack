# Stealth Adversarial Poisoning of LLM Steering Vectors

A token-substitution attack on the contrastive POS/NEG text pairs used to construct activation steering vectors. The optimisation maximises `cos(v_poisoned, −u_refusal)` over the pair-text tokens under a fluency penalty. The saved steering vector is then evaluated on harmful and harmless prompts.

## Results

Four (model, attribute, layer, weight) combos. For each, the saved steering vector (`steering_vector.pt`, the artefact downstream code would load) is evaluated at the listed weight on:
- `harmful` — 100 prompts from `data/refusal/harmful_prompts.json`, scored by Llama-3.3-70B-Instruct-Turbo (`is_jailbreak_llama33`),
- `harmless` — 100 prompts from `data/refusal/harmless_prompts.json`, scored by the Python predicate in `src/steering.py::ATTRIBUTE_CHECK_FNS[<attribute>]`.

| combo | clean (hAttr, ASR) | poisoned (hAttr, ASR) | ΔASR | Δattr |
|---|---|---|---:|---:|
| Llama-3.1-8B-Instruct, `lowercase`, layer 18, w=2 | (0.85, 0.06) | (0.94, 0.64) | +0.58 | +0.09 |
| Llama-3.1-8B-Instruct, `uppercase`, layer 16, w=5 | (0.46, 0.22) | (0.77, 0.61) | +0.39 | +0.31 |
| Llama-3.1-8B-Instruct, `spanish`, layer 18, w=3 | (0.81, 0.01) | (0.74, 0.22) | +0.21 | −0.07 |
| Gemma-2-2B-IT, `json_format`, layer 13, w=3 | (0.89, 0.21) | (0.81, 0.56) | +0.35 | −0.08 |

Per-attack continuum trajectories (`harmful` ASR and `harmless` hAttr versus GCG iteration) are in `plots/<combo>.png`. Trajectories for some combos peak above the final-vector ASR shown above (e.g. `lowercase` reaches ASR ≈ 0.71 at iter 1200 before drifting down to 0.64 at iter 1500). The table reports the final saved vector, not the trajectory peak.

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
run_best.sh                     # one-command reproduction of the 4 headline combos (plus 2 controls; see NOTES.md)
run_experiments.sh              # legacy 6-experiment study with norm-matched / random ablations
findings.md                     # original 6-experiment narrative (legacy)
findings_llama31.md             # Llama-3.1-8B continuum study (legacy)
NOTES.md                        # exploration notes for combos and methods that did not make the headline
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

`run_best.sh` runs all 6 combos sequentially (~22 hr on a single 24 GB GPU). Each combo produces:

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
| `google/gemma-2-2b-it` | 13 | ~6 GB | 5000 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 16, 18 | ~17 GB | 1500 |

All experiments run in `bfloat16`. Single seed (0). No multi-seed CIs.

## Caveats

1. **The Spanish predicate is weak.** `check_spanish` in `src/steering.py` flags a response as "Spanish" if it has ≥2 Spanish-specific characters (`ñ`, `¿`, `¡`, …) **OR** ≥6 distinct Spanish function words, **AND** length > 80 chars. This is a hand-rolled heuristic, not a real language detector — a Spanish-flavoured English response could pass it, and short fluent Spanish could fail it. The Spanish row should be read as "the response uses enough Spanish surface features", not "the response is in Spanish". The other predicates (`check_lowercase`, `check_uppercase`, `check_json_format`) are strict (whole-string conditions) and don't have this concern.
2. Trajectory plots use the snapshots saved during the attack; the table uses the final saved vector. The two can differ by a few percentage points (e.g. `lowercase` peaks at iter 1200 with ASR 0.71 versus 0.64 at the end of the attack).
3. The judge is a single model (Llama-3.3-70B-Instruct-Turbo). Inter-judge agreement is not estimated.
4. Single seed (0). No multi-seed CIs.
