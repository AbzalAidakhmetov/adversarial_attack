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

  orchestration.py                   # Shared helpers (iter_cells, run_subprocess, attack_cmd, eval_cmd, eval_sweep)

config/evaluate_jailbreak.yaml       # Hydra config for the per-eval `evaluate_asr` entry point
config/matrix.yaml                   # Hydra config: models × attributes × weights + attack hyperparams
config/defense.yaml                  # defaults: [matrix, _self_] (defense pipeline)
config/all_steps.yaml                # defaults: [matrix, _self_]; overrides weights=[1.5,1.75,2.0]
config/detector.yaml                 # defaults: [matrix, _self_]; adds caps for the bypass sweep

scripts/run_matrix.py                # attack + eval sweep per cell                  → slurm/run_matrix.sh
scripts/run_defense.py               # orthogonalize + use_defended eval sweep       → slurm/run_defense.sh
scripts/run_all_steps.py             # all_steps-protocol eval at lower weights      → slurm/run_all_steps.sh
scripts/run_detector.py              # (A) static cos-detector + (B) cos_max bypass  → slurm/run_detector.sh

results/                             # Saved attack vectors + Hydra eval outputs, one subdir per cell
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

# Full matrix (every model × attribute in config/matrix.yaml): attack + eval sweep.
# Local single-process run (sequential over all cells):
uv run python scripts/run_matrix.py

# SLURM: one GPU per cell.
sbatch --array=0-7 slurm/run_matrix.sh
# Subset of cells:
sbatch --array=0,4 slurm/run_matrix.sh

# Single-attack one-off (Hydra CLI overrides):
uv run python scripts/run_matrix.py models='[{name: gemma, hf_id: google/gemma-2-2b-it, layer: 14}]' attributes=[spanish] weights=[3]

# Manual single attack / eval (bypassing the matrix orchestrator):
uv run python -m advsteer.attack.build_adv_stealth \
  --model google/gemma-2-2b-it --layer 14 --pair_type spanish --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 --n_candidates 64 --n_swaps 1 \
  --eval_batch_size 8 --dtype bfloat16 --output results/my_exp/summary.json
uv run python -m advsteer.eval.evaluate_asr \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/results/my_exp/steering_vector.pt \
  attribute=spanish steering_weights=[3] eval_methods='[judge]' \
  results_path=$(pwd)/results/my_exp/results_poisoned_harmful_w3/
# clean baseline → add `use_clean=true`.
```

Per cell, the four pipelines write:
```
results/<model>/<attr>/
  steering_vector.pt                                             # run_matrix.py
  steering_vector_defended.pt                                    # run_defense.py
  results_{clean,poisoned}_{harmful,harmless}_w<W>/              # run_matrix.py
  results_defense_{clean,poisoned}_{harmful,harmless}_w<W>/      # run_defense.py
  all_steps/results_{clean,poisoned}_{harmful,harmless}_w<W>/    # run_all_steps.py
  cap<CAP>/steering_vector.pt + results_poisoned_*_w<W>/         # run_detector.py
```
Every step is restartable: attack skips if `steering_vector.pt` exists, orthogonalize skips if `steering_vector_defended.pt` exists, each eval skips if its `results` file exists.

All four orchestrators share `advsteer.orchestration` (cell iteration + SLURM array dispatch + subprocess wrapper). Defense, all_steps, and detector configs all `defaults: [matrix, _self_]` so the cell grid lives in a single source of truth (`config/matrix.yaml`).

## Models & Layers (default matrix)

| Model | Layer | GPU mem | GCG budget |
|---|---|---|---|
| google/gemma-2-2b-it | 14 | ~6 GB | 1500 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 18 | ~17 GB | 1500 |

All bfloat16. `--eval_batch_size 8` works on 16-24 GB GPUs.

## Protocols

The default `prefill` protocol is what `run_matrix.py` and `run_defense.py` drive. `run_all_steps.py` switches to `all_steps` (Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*) and reuses the prefill-attack vector at the lower weight sweep in `config/all_steps.yaml`. Prefill is the default because the compliant→degenerate weight window is wider under prefill than under all_steps.

## Defense + detection (run_defense.py, run_detector.py)

`run_defense.py` reuses each cell's `steering_vector.pt` and applies refusal-direction orthogonalization (`advsteer.defense.orthogonalize_steering`): `v_def = v − (v·r̂) r̂` at the same layer, where r is recomputed from the harmful/harmless *train* splits. Outputs land in `results/<model>/<attr>/results_defense_{clean,poisoned}_{harmful,harmless}_w<W>/`.

`run_detector.py` has two halves: (A) `advsteer.eval.cos_detector` builds a *null distribution* by computing cos(v_attr, −r) across all attributes in `PAIR_TYPE_SPECS` for which a pair file exists, plus cos for every saved `steering_vector.pt` matching this (model, layer); it dumps `cos_table.csv`, `cos_strip.png`, `cos_roc.png`, `summary.json` under `results/cos_detector/<model_name>_L<layer>/`. (B) An adaptive-attacker bypass sweep re-runs the GCG attack with `--cos_max ∈ cfg.caps` and re-evaluates ASR + hAttr poisoned-only, answering "if the attacker stays below the defender's threshold, how much lift survives?". Outputs in `results/<model>/<attr>/cap<CAP>/`.

`advsteer.defense.stealth_check` is a complementary audit: it presents original vs poisoned pair texts (POS and NEG) to an LLM-as-judge in isolation and reports flag rates per condition. Auto-discovers `results/*/summary.json` if `--summaries` is omitted; writes `results/stealth_check_results.json`.

## Key Notes

- `build_adv_stealth.py` outputs both `summary.json` and `steering_vector.pt` directly. `--cos_max` adds a hard cap that early-stops the GCG loop once `cos(v,−r) ≥ cos_max` (adaptive-attacker bypass).
- `evaluate_asr.py` loads `steering_vector_poisoned` from the `.pt` file by default; `use_clean=true` loads `steering_vector_clean`; `use_defended=true` loads the `*_defended` key produced by `advsteer.defense.orthogonalize_steering`. The two flags compose (clean+defended / poisoned+defended).
- Override Hydra defaults: `model=`, `attribute=`, `steering_weights=`, `results_path=`. Judge: `judge.provider=`, `judge.model=` (default `anthropic / claude-sonnet-4-5`; alternatives are `together / meta-llama/Llama-3.3-70B-Instruct-Turbo` and `openai / gpt-4.1-mini`).
- Refusal direction train set and ASR eval set are disjoint (no leakage). The defense recomputes r from the *train* splits — same data the attacker also has — so it does not require fresh held-out prompts.
- `to_chat()` strips leading `<bos>` from the chat template to avoid double-bos with nnsight.
- Steering is applied at all token positions of every forward pass it touches (`tgt[:] += direction * weight`). *Which* forward passes are touched is set by `protocol`: `prefill` (default, used by `scripts/run_matrix.py`) edits only the prompt-prefill pass; `all_steps` (Arditi et al. 2024, `scripts/run_all_steps.py`) edits prefill + every decode step. Both functions live in `src/advsteer/steering.py` and are dispatched in `advsteer.eval.evaluate_asr::GEN_FNS`.
- Attribute-instruction tokens in POS texts are protected via per-row `protect_text` substring (loaded with the pair), not a keyword list.
- Per-text edit budget (`n_modify`) is strictly enforced during candidate generation.
- Candidate selection: pick highest `score = cos − λ·nll`, accept iff strictly improves `cos(v, −u_refusal)`. Picking and acceptance are decoupled on purpose — filtering by cos before scoring (rejecting cc ≤ best_cos at pick time) empirically pushes cos too far and breaks attribute compliance on brittle attributes.
