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
scripts/run_best.sh                  # 5-combo headline reproduction (~5–6 hrs, prefill protocol)
scripts/run_all_steps.sh             # 2-combo all-step (Arditi et al. 2024) reproduction at lower weights
scripts/run_defense.sh               # Re-eval each headline combo under refusal-direction orthogonalization
scripts/run_detector.sh              # (A) static cos-detector per (model, layer); (B) cos_max bypass sweep
results/                             # Saved attack vectors + Hydra eval outputs, one subdir per combo
```

Run modules with `uv run python -m advsteer.<subpkg>.<module>` (or, equivalently, `uv run python -m advsteer.attack.build_adv_stealth`, etc.).

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

# Evaluate (clean baseline) — same command + use_clean=true
```

## Models & Layers (as used by run_best.sh)

| Model | Layers | GPU mem | GCG budget |
|---|---|---|---|
| google/gemma-2-2b-it | 13, 14 | ~6 GB | 1500 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 16, 18 | ~17 GB | 1500 |

Llama-3.2-3B-Instruct (L14, L16; ~7 GB) was tried during exploration but is not in `run_best.sh`. All bfloat16. `--eval_batch_size 8` works on 16-24 GB GPUs. Two slots in parallel on a 24 GB GPU fit: 17 + 7 = 24 GB total.

## Protocols

`scripts/run_best.sh` uses the `prefill` protocol (the default); `scripts/run_all_steps.sh` uses `all_steps` (Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*) at lower weights. The steering vector is protocol-independent, so `scripts/run_all_steps.sh` reuses an existing `results/<base>/steering_vector.pt` when present and otherwise runs the GCG attack from scratch. Prefill is the default because the compliant→degenerate weight window is wider under prefill than under all_steps.

## Defense + detection (scripts/run_defense.sh, scripts/run_detector.sh)

`scripts/run_defense.sh` reuses each headline `steering_vector.pt` and applies refusal-direction orthogonalization (`advsteer.defense.orthogonalize_steering`): `v_def = v − (v·r̂) r̂` at the same layer, where r is recomputed from the harmful/harmless *train* splits. Outputs land in `results/<model>/<attr>/results_defense_{clean,poisoned}_{harmful,harmless}/`.

`scripts/run_detector.sh` has two halves: (A) `advsteer.eval.cos_detector` builds a *null distribution* by computing cos(v_attr, −r) across all attributes in `PAIR_TYPE_SPECS` for which a pair file exists, plus cos for every saved `steering_vector.pt` matching this (model, layer); it dumps `cos_table.csv`, `cos_strip.png`, `cos_roc.png`, `summary.json` under `results/cos_detector/<tag>/`. (B) An adaptive-attacker bypass sweep re-runs the GCG attack with the new `--cos_max ∈ {0.05, 0.10, 0.15, 0.20}` hard cap and re-evaluates ASR + hAttr, answering "if the attacker stays below the defender's threshold, how much lift survives?". Outputs in `results/<model>/<attr>/cap<CAP>/`.

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
