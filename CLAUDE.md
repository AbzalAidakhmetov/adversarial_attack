# Stealth Adversarial Poisoning of LLM Steering Vectors

Modify existing contrastive pair texts with embedding-neighbor token swaps so the resulting steering vector aligns with -refusal_direction, enabling jailbreaks.

## Structure

Package layout (PEP 621 + src layout, `advsteer` distribution):

```
src/advsteer/
  attack/build_adv_stealth.py        # The attack (GCG over pair-text tokens; --cos_max for adaptive bypass)
  eval/evaluate_asr.py               # ASR evaluation (Hydra)
  eval/cos_detector.py               # cos(v,-r) detector + null-distribution sweep (defender)
  defense/orthogonalize_steering.py  # v_def = v - (v·r̂)r̂; writes steering_vector_defended.pt
  defense/stealth_check.py           # LLM-judge stealth audit over original vs poisoned pair texts
  defense/summarize_defense.py       # aggregate clean/poisoned/defended ASR + hAttr across results
  data.py                            # Pair specs, data loading, vocab, hidden states, refusal direction
  steering.py                        # Steered generation (prefill + all_steps), attribute checks, to_chat
  classifiers.py                     # set_seed, GPT-2 perplexity, LLM judge (litellm; Claude Sonnet 4.5 default)

config/evaluate_jailbreak.yaml       # Hydra config (`protocol`, `use_clean`, `use_defended`, `judge.*`)
scripts/run_best.sh                  # 8-combo headline reproduction (~10 hrs, prefill protocol)
scripts/run_all_steps.sh             # 2-combo all-step (Arditi et al. 2024) reproduction at lower weights
scripts/run_defense.sh               # Re-eval each headline combo under refusal-direction orthogonalization
scripts/run_detector.sh              # (A) static cos-detector per (model, layer); (B) cos_max bypass sweep
results/                             # Saved attack vectors + Hydra eval outputs, one subdir per combo
```

Run modules with `uv run python -m advsteer.<subpkg>.<module>` (or, equivalently, `uv run python -m advsteer.attack.build_adv_stealth`, etc.).

## Results directory layout

One directory per (model, attribute), named `<model>/<attribute>`. The attack and defence vectors are weight-independent and live at the combo-dir root.

```
results/<model>/<attribute>/
  summary.json                                # attack output
  steering_vector.pt                          # attack output (weight-independent)
  steering_vector_defended.pt                 # defence output (weight-independent)
  steering_vector_defended_report.json
  attack.log, defense.log, hydra_logs/
  results_clean_harmful/                      # eval at the bundled steering weight
  results_clean_harmless/
  results_poisoned_harmful/
  results_poisoned_harmless/
  results_defense_clean_harmful/
  results_defense_clean_harmless/
  results_defense_poisoned_harmful/
  results_defense_poisoned_harmless/
```

The paper's bundled weight per combo is fixed in `scripts/plots/plot_*.py` `COMBOS` lists. See `PIPELINE.md` for the threat-model rationale behind weight selection.


## Quick Reference

```bash
# Environment
export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT=$(pwd)
# Judge default is Anthropic Claude Sonnet 4.5; export TOGETHER_API_KEY / OPENAI_API_KEY only if you swap judges.
source .env && export HF_TOKEN && export ANTHROPIC_API_KEY && export TOGETHER_API_KEY

# Attack — Gemma example (matches run_best.sh)
uv run python -m advsteer.attack.build_adv_stealth \
  --model google/gemma-2-2b-it --layer 14 \
  --pair_type spanish --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 --output results/my_exp/summary.json

# Attack — Llama-3.1-8B example (same hyperparams; ~17 GB VRAM)
uv run python -m advsteer.attack.build_adv_stealth \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 18 \
  --pair_type lowercase --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 \
  --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 \
  --n_candidates 64 --n_swaps 1 --eval_batch_size 8 \
  --dtype bfloat16 --output results/my_exp/summary.json

# Evaluate (poisoned vector)
uv run python -m advsteer.eval.evaluate_asr \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/results/my_exp/steering_vector.pt \
  attribute=spanish steering_weights=[3] eval_methods='[judge]' \
  results_path=$(pwd)/results/my_exp/results_poisoned_harmful/

# Evaluate (clean baseline) — same command + use_clean=true, writing to results_clean_harmful/
```

## Models & Layers (as used by run_best.sh)

| Model | Layers | GPU mem | GCG budget |
|---|---|---|---|
| google/gemma-2-2b-it | 13, 14 | ~6 GB | 1500 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 16, 18 | ~17 GB | 1500 |

Llama-3.2-3B-Instruct (L14, L16; ~7 GB) was tried during exploration but is not in `run_best.sh`. All bfloat16. `--eval_batch_size 8` works on 16-24 GB GPUs. Two slots in parallel on a 24 GB GPU fit: 17 + 7 = 24 GB total.

## Protocols

`scripts/run_best.sh` uses the `prefill` protocol (the default); `scripts/run_all_steps.sh` uses `all_steps` (Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*) at lower weights. The steering vector is protocol-independent, so `scripts/run_all_steps.sh` reuses an existing `results/<base>/steering_vector.pt` when present and otherwise runs the GCG attack from scratch. Prefill is the default because the compliant→degenerate weight window is wider under prefill than under all_steps.

8 model–attribute combos across 2 model families and 3 attribute classes (language, formatting, case). Bundled weight per combo is the attacker-rational integer in {2,3,4} under the bundle threat model (see paper §4 setup and `PIPELINE.md`):

| Model | Attribute | Layer · w | hAttr c→p | ΔhAttr | ASR c→p | ΔASR |
|---|---|---|---|---:|---|---:|
| Llama-3.1-8B | has_bold_only | 18·4 | 0.54→0.57 | +0.03 | 0.04→0.50 | **+0.46** |
| Llama-3.1-8B | lowercase | 18·2 | 0.83→0.91 | +0.08 | 0.04→0.43 | **+0.39** |
| Llama-3.1-8B | french | 18·4 | 0.90→0.85 | −0.05 | 0.08→0.45 | **+0.37** |
| Gemma-2-2B | french | 14·3 | 0.87→0.92 | +0.05 | 0.10→0.47 | **+0.36** |
| Gemma-2-2B | lowercase | 14·4 | 0.14→0.11 | −0.03 | 0.21→0.50 | **+0.28** |
| Gemma-2-2B | spanish | 14·3 | 0.83→0.94 | +0.11 | 0.03→0.30 | **+0.27** |
| Gemma-2-2B | has_bold_only | 14·4 | 0.73→0.74 | +0.01 | 0.05→0.30 | **+0.25** |
| Llama-3.1-8B | spanish | 18·3 | 0.85→0.81 | −0.04 | 0.01→0.20 | **+0.19** |

Gemma-2-2B `lowercase` is a weakly-steered combo: clean hAttr is only 0.14 at any weight, and the elevated clean ASR (0.21) is largely a property of the legitimate vector rather than the attack. Retained for transparency but flagged in the paper limitations.

### all_steps protocol (scripts/run_all_steps.sh, single seed)

Re-run of two headline combos with `protocol=all_steps` (Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*) at lower weights. The steering vector is protocol-independent, so `scripts/run_all_steps.sh` reuses `results/<base>/steering_vector.pt` from a prior `scripts/run_best.sh` run if present, and otherwise runs the GCG attack from scratch.

| Model | Attribute | Layer · w | hAttr c→p (harmless) | ΔhAttr | ASR c→p | ΔASR |
|---|---|---|---|---:|---|---:|
| Gemma-2-2B | spanish | 14·1.5 | 0.90→0.93 | +0.03 | 0.02→0.43 | **+0.41** |
| Llama-3.1-8B | lowercase | 18·1.75 | 0.62→0.90 | +0.28 | 0.06→0.40 | **+0.34** |

Prefill chosen as default empirically: under all_steps the compliant→degenerate weight window is much narrower, making weight selection brittle.

## Defense + detection (scripts/run_defense.sh, scripts/run_detector.sh)

`scripts/run_defense.sh` reuses each headline `steering_vector.pt` and applies refusal-direction orthogonalization (`advsteer.defense.orthogonalize_steering`): `v_def = v − (v·r̂) r̂` at the same layer, where r is recomputed from the harmful/harmless *train* splits. Outputs land in `results/<model>/<attr>/results_defense_{clean,poisoned}_{harmful,harmless}/`. Mean gap recovered across 8 combos is ~78% (median 83%); on three combos the defended ASR sits at or below the clean baseline. Harmless hAttr stays within ±0.07 on the 7 working-attribute combos.

`scripts/run_detector.sh` has two halves: (A) `advsteer.eval.cos_detector` builds a *null distribution* by computing cos(v_attr, −r) across all attributes in `PAIR_TYPE_SPECS` for which a pair file exists, plus cos for every saved `steering_vector.pt` matching this (model, layer); it dumps `cos_table.csv`, `cos_strip.png`, `cos_roc.png`, `summary.json` under `results/cos_detector/<tag>/`. (B) An adaptive-attacker bypass sweep re-runs the GCG attack with the new `--cos_max ∈ {0.05, 0.10, 0.15, 0.20}` hard cap and re-evaluates ASR + hAttr, answering "if the attacker stays below the defender's threshold, how much lift survives?". Outputs in `results/<model>/<attr>/cap<CAP>/`.

| Model · attribute · L·w | ASR clean → poisoned → defended | hAttr (harmless) clean → poisoned → defended |
|---|---|---|
| Gemma spanish 14·3 | 0.03 → 0.30 → 0.09 | 0.83 → 0.94 → 0.90 |
| Gemma french 14·3 | 0.10 → 0.47 → 0.05 | 0.87 → 0.92 → 0.91 |
| Gemma lowercase 14·4 | 0.21 → 0.50 → 0.37 | 0.14 → 0.11 → 0.03 |
| Gemma has_bold_only 14·4 | 0.05 → 0.30 → 0.03 | 0.73 → 0.74 → 0.76 |
| Llama-3.1-8B spanish 18·3 | 0.01 → 0.20 → 0.08 | 0.85 → 0.81 → 0.88 |
| Llama-3.1-8B french 18·4 | 0.08 → 0.45 → 0.12 | 0.90 → 0.85 → 0.89 |
| Llama-3.1-8B lowercase 18·2 | 0.04 → 0.43 → 0.06 | 0.83 → 0.91 → 0.86 |
| Llama-3.1-8B has_bold_only 18·4 | 0.04 → 0.50 → 0.34 | 0.54 → 0.57 → 0.49 |

`advsteer.defense.stealth_check` is a complementary audit: it presents original vs poisoned pair texts (POS and NEG) to an LLM-as-judge in isolation and reports flag rates per condition. Auto-discovers `results/*/summary.json` if `--summaries` is omitted; writes `results/stealth_check_results.json`.

## Key Notes

- `build_adv_stealth.py` outputs both `summary.json` and `steering_vector.pt` directly. `--cos_max` adds a hard cap that early-stops the GCG loop once `cos(v,−r) ≥ cos_max` (adaptive-attacker bypass).
- `evaluate_asr.py` loads `steering_vector_poisoned` from the `.pt` file by default; `use_clean=true` loads `steering_vector_clean`; `use_defended=true` loads the `*_defended` key produced by `advsteer.defense.orthogonalize_steering`. The two flags compose (clean+defended / poisoned+defended).
- Override Hydra defaults: `model=`, `attribute=`, `steering_weights=`, `results_path=`. Judge: `judge.provider=`, `judge.model=` (default `anthropic / claude-sonnet-4-5`; alternatives are `together / meta-llama/Llama-3.3-70B-Instruct-Turbo` and `openai / gpt-4.1-mini`).
- Refusal direction train set and ASR eval set are disjoint (no leakage). The defense recomputes r from the *train* splits — same data the attacker also has — so it does not require fresh held-out prompts.
- `to_chat()` strips leading `<bos>` from the chat template to avoid double-bos with nnsight.
- Steering is applied at all token positions of every forward pass it touches (`tgt[:] += direction * weight`). *Which* forward passes are touched is set by `protocol`: `prefill` (default, `scripts/run_best.sh`) edits only the prompt-prefill pass; `all_steps` (Arditi et al. 2024, `scripts/run_all_steps.sh`) edits prefill + every decode step. Both functions live in `src/advsteer/steering.py` and are dispatched in `advsteer.eval.evaluate_asr::GEN_FNS`.
- Attribute-instruction tokens in POS texts are protected via per-row `protect_text` substring (loaded with the pair), not a keyword list.
- Per-text edit budget (`n_modify`) is strictly enforced during candidate generation.
- Candidate selection: pick highest `score = cos − λ·nll`, accept iff strictly improves `cos(v, −u_refusal)`. Picking and acceptance are decoupled on purpose — filtering by cos before scoring (rejecting cc ≤ best_cos at pick time) empirically pushes cos too far and breaks attribute compliance on brittle attributes.
