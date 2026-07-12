# Steering Vectors are an Adversarial Attack Surface

Code for the paper **"Steering Vectors are an Adversarial Attack Surface"**
(Aidakhmetov, Crisostomi, Mencattini, Minut, Masi, Rodolà), [arXiv:2606.05958](https://arxiv.org/abs/2606.05958).

Activation steering lets users control an LLM by adding a precomputed *steering
vector* — a mean-difference of hidden states over contrastive POS/NEG text pairs
— to its activations at inference time. Because these datasets and vectors are
*shared*, the contrastive dataset is a supply-chain surface. This repo shows
that an adversary who can edit the pair texts can make small, embedding-neighbor
token substitutions (a GCG-style optimization) that silently rotate the
resulting steering vector toward the model's **anti-refusal direction**. The
poisoned vector still produces its declared benign behavior on harmless prompts,
but suppresses refusals on harmful ones — and the texts still read as ordinary
paraphrases, so the payload survives even a bundle that ships the texts, the
vector, a recommended weight, and a cryptographic equivalence certificate the
user can re-derive.

> **Headline.** Poisoned vectors reach an absolute attack success rate (ASR) of
> **0.20–0.55** (**+19 to +51 pp** over a clean reference) while leaving the
> declared steering effect on benign prompts essentially unchanged on seven of
> eight combos; a refusal-direction orthogonalization defense recovers **≈82%**
> of the ASR gap without harming benign behavior.

**Full results — all 8 model·attribute combos, 3 seeds (± std), the three-judge
ensemble, and the defense/detection analysis — are in the paper** (Figures 2–4,
Table 1, Appendices D–F). This README covers *what the code does and how to run
it*; the paper is the source of record for the numbers.

## Threat model (paper §3)

An attacker with white-box access to an open-weight target model (e.g. Gemma-2,
Llama-3.1) distributes a *bundle*: contrastive pair texts, a precomputed steering
vector `v`, a recommended weight `α`, and an equivalence certificate. A victim
re-derives `v` from the texts to confirm provenance, reads the texts to check
they look natural, and sanity-checks that `(v, α)` produces the advertised
attribute on benign prompts. The certificate only proves `v` was computed
correctly *from the texts* — it is the texts that carry the payload, so bit-exact
reproducibility does not defend against the attack.

## How the attack works (paper §3)

1. Load `N = 20` contrastive pair texts for the chosen attribute. POS texts
   contain the attribute instruction (e.g. *"…write your response in all
   lowercase letters."*); NEG texts are the same prompts without it.
2. Clean steering vector `v = mean(h_layer(POS)) − mean(h_layer(NEG))`.
3. Anti-refusal direction `r` from 128 harmful + 128 harmless prompts at the same
   layer, oriented so that *adding* it suppresses refusal (`r = −r_refusal`).
4. Run a GCG-style optimizer over the pair-text tokens. Each step proposes
   single-token swaps drawn from each token's embedding neighbors (restricted to
   a curated safe vocabulary), scores them by
   `cos(v, r) − λ_lm · NLL`, rejects any candidate above a hard perplexity cap,
   and accepts the best one only if its `cos(v, r)` strictly improves. Picking
   and acceptance are decoupled so the fluency penalty tilts the *choice* but
   never the accepted optimum. The steering vector is updated incrementally
   (only the edited text's cached hidden state is recomputed).
5. The attribute-specifying suffix of each POS text is detected by character-level
   boundary matching and **excluded from modification**, so the poisoned dataset
   still appears to target the claimed attribute.
6. The final pair texts are turned back into a steering vector and saved to
   `steering_vector.pt`.

`build_adv_stealth.py` also exposes `--cos_max` / `--cos_max_hard` for modeling an
**adaptive attacker** that stays below a defender's cosine threshold (paper §4.7,
Appendix E).

## Setup

Requirements: Python 3.10, a GPU with ~24 GB VRAM, a [HuggingFace](https://huggingface.co)
token (gated models), and API keys for the LLM judge(s).

The paper's ASR is a **three-judge majority vote** — Claude-Sonnet-4.5
(`ANTHROPIC_API_KEY`), GPT-4.1 (`OPENAI_API_KEY`), and Llama-3.3-70B on Together
(`TOGETHER_API_KEY`). A single run of the evaluator uses one configurable judge,
defaulting to `anthropic / claude-sonnet-4-5`, so at minimum you need
`ANTHROPIC_API_KEY`; add the other two to reproduce the ensemble.

```bash
git clone https://github.com/AbzalAidakhmetov/adversarial_attack.git
cd adversarial_attack
uv sync

cat >> .env <<'EOF'
HF_TOKEN=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...        # for the 3-judge ensemble
TOGETHER_API_KEY=...      # for the 3-judge ensemble
EOF
```


## Reproduce

The experiment grid (models × attributes × seeds) lives in `config/matrix.yaml`
(default `seeds: [0, 1, 2]`; 2 models × 4 attributes × 3 seeds = 24 cells). Every
step is restartable — each stage skips a cell whose output already exists.

```bash
# Attack + harmful/harmless eval sweep over every cell.
uv run python scripts/run_matrix.py                 # local, sequential
sbatch --array=0-23 slurm/run_matrix.sh             # SLURM, one GPU per (cell, seed)

# Downstream analyses (all reuse each cell's steering_vector.pt):
uv run python scripts/run_defense.py                # refusal-direction orthogonalization
uv run python scripts/run_detector.py               # cos(v,-r) detector + adaptive bypass sweep
uv run python scripts/run_constrained_bypass.py     # constrained adaptive attacker (cos cap)
uv run python scripts/run_all_steps.py              # all-step protocol (Arditi et al. 2024)
uv run python scripts/run_partial_control.py        # weaker-attacker ablation (controls a fraction of pairs)
# SLURM equivalents: slurm/run_{defense,detector,constrained_bypass,all_steps,partial_control}.sh

# Cross-model text transfer: attack a source model, ship only the poisoned texts,
# recompute the vector on a different target, and eval transfer vs clean vs native.
uv run python scripts/run_cross_transfer.py         # config/transfer.yaml, local sequential
sbatch --array=0-3 slurm/run_cross_transfer.sh      # SLURM, one task per (combo, seed)
```

Per `(cell, seed)`, outputs land under:

```
results/<model>/<attr>/seed<S>/
  steering_vector.pt                                          # attack: clean + poisoned vectors
  steering_vector_defended.pt                                 # orthogonalized variants
  results_{clean,poisoned}_{harmful,harmless}_w<W>/           # judge outputs + completions
  results_defense_{clean,poisoned}_{harmful,harmless}_w<W>/   # defended eval
  all_steps/results_{clean,poisoned}_{harmful,harmless}_w<W>/ # all-step protocol
  cap<CAP>/steering_vector.pt + results_poisoned_*_w<W>/      # adaptive-bypass caps
```

ASR is computed by `advsteer.eval.evaluate_asr` with a single judge.
`scripts/rejudge_results.py` re-scores saved completions under additional judges
without regenerating, and `scripts/aggregate_majority_judge.py` combines per-judge
labels into the majority vote reported in the paper.

## Run a single attack / eval

```bash
# Attack one (model, attribute, layer).
uv run python -m advsteer.attack.build_adv_stealth \
  --model google/gemma-2-2b-it --layer 14 --pair_type spanish --num_pairs 20 \
  --n_modify 5 --n_neighbors 100 --lambda_lm 0.2 --max_perp 2000 \
  --gcg_budget 1500 --gcg_patience 500 --n_candidates 64 --n_swaps 1 \
  --eval_batch_size 8 --dtype bfloat16 --output results/my_exp/summary.json

# Evaluate the saved poisoned vector (add use_clean=true for the clean baseline).
uv run python -m advsteer.eval.evaluate_asr \
  model=google/gemma-2-2b-it \
  directions_path=$(pwd)/results/my_exp/steering_vector.pt \
  attribute=spanish steering_weights=[3] eval_methods='[judge]' \
  results_path=$(pwd)/results/my_exp/results_poisoned_harmful_w3/
```

Swap the judge with `judge.provider=` / `judge.model=` (e.g.
`judge.provider=together judge.model=meta-llama/Llama-3.3-70B-Instruct-Turbo`).

## Attack hyperparameters

| flag             | meaning                                                           | used |
| ---------------- | ---------------------------------------------------------------- | ---- |
| `--model`        | target model (HuggingFace ID)                                    | —    |
| `--layer`        | target residual-stream layer (0-indexed)                         | 14 / 18 |
| `--pair_type`    | attribute key in `PAIR_TYPE_SPECS`                               | —    |
| `--num_pairs`    | number of POS/NEG pairs                                          | 20   |
| `--n_modify`     | max token swaps per text                                         | 5    |
| `--n_neighbors`  | embedding-neighbor candidates per token                          | 100  |
| `--n_candidates` | candidates scored per GCG iteration                              | 64   |
| `--lambda_lm`    | fluency penalty weight on next-token NLL                         | 0.2  |
| `--max_perp`     | hard perplexity cap; reject any candidate above this             | 2000 |
| `--gcg_budget`   | total GCG iterations                                             | 1500 |
| `--gcg_patience` | early-stop after this many iterations without improvement        | 500  |
| `--cos_max`      | adaptive cap: stop once `cos(v,-r) ≥ τ`                           | off  |
| `--cos_max_hard` | per-candidate cap: reject `cos(v,-r) > τ`, score-monotonic accept | off  |

Models and layers (chosen by a coarse layer sweep): `google/gemma-2-2b-it` @ L14
(~6 GB) and `meta-llama/Meta-Llama-3.1-8B-Instruct` @ L18 (~17 GB). All `bfloat16`.

## What each attribute measures

`hAttr` / AC (attribute compliance) is the fraction of benign-prompt responses
passing the attribute predicate in `src/advsteer/steering.py::ATTRIBUTE_CHECK_FNS`.
The four attributes used in the paper:

| attribute       | predicate            | accepts                                                                 |
| --------------- | -------------------- | ----------------------------------------------------------------------- |
| `spanish`       | `check_spanish`      | fastText `lid.176` predicts `__label__es` with prob ≥ 0.5 (text ≥ 40 chars) |
| `french`        | `check_french`       | fastText `lid.176` predicts `__label__fr` with prob ≥ 0.5 (text ≥ 40 chars) |
| `lowercase`     | `check_lowercase`    | every alphabetic character is lowercase (≥1 letter present)             |
| `has_bold_only` | `check_has_bold_only`| ≥ 3 markdown bold spans `**…**` (bare single-asterisk italics excluded) |

The 40-character floor on the language checks discards short fragments
("Sure!", "Bonjour."), on which fastText `lid.176` is unreliable.

## Defending and detecting (paper §4.7, §4.6)

**Orthogonalization.** The attack maximizes `cos(v, r)`, so a defender with the
harmful/harmless splits can project the refusal direction out before deploying:
`v_def = v − (v·r̂) r̂`. By construction `cos(v_def, r) = 0`; the attribute-bearing
orthogonal component is preserved.
`advsteer.defense.orthogonalize_steering` writes `steering_vector_defended.pt`.

**Cos detector.** The same metric makes a detector: `advsteer.eval.cos_detector`
builds a null distribution of `cos(v_attr, −r)` over legitimate attribute vectors
at the same (model, layer) and reports a strip plot + ROC. An adaptive attacker
can stay under the threshold via `--cos_max` / `--cos_max_hard`; the cost in
surviving lift is the bypass sweep (`run_detector.py`, `run_constrained_bypass.py`).

**Text audit.** `advsteer.defense.stealth_check` presents each original/poisoned
pair text to an LLM judge in isolation and reports per-condition flag rates.

## Project layout

```
src/advsteer/
  attack/build_adv_stealth.py      # GCG attack (--cos_max / --cos_max_hard for adaptive bypass)
  eval/
    evaluate_asr.py                # ASR + attribute evaluation (Hydra)
    cos_detector.py                # cos(v,-r) detector + ROC over the legitimate-vector null
    judge_agreement.py             # inter-judge agreement (Fleiss' kappa)
  defense/
    orthogonalize_steering.py      # v_def = v - (v·r̂)r̂ ; writes steering_vector_defended.pt
    stealth_check.py               # LLM-judge audit over original vs poisoned pair texts
    summarize_defense.py           # aggregate clean / poisoned / defended ASR + hAttr
  transfer/recompute.py            # recompute a target-model vector from the source's poisoned texts
  data.py                          # pair specs, pair loading, refusal-direction computation
  steering.py                      # ATTRIBUTE_CHECK_FNS, steered generation, to_chat
  classifiers.py                   # set_seed, GPT-2 perplexity, LLM judge (litellm-routed)
  orchestration.py                 # shared cell-iteration + SLURM dispatch + subprocess wrapper
config/                            # matrix.yaml (grid) + per-pipeline Hydra configs (transfer, partial_control, ...)
scripts/                           # run_matrix / run_defense / run_detector / run_constrained_bypass
                                   #   run_all_steps / run_partial_control / run_cross_transfer
                                   #   rejudge_results / aggregate_majority_judge / plots
slurm/                             # *.sh array wrappers for the scripts above
data/
  pairs/                           # POS/NEG pair datasets, one per attribute
  refusal/                         # 100 harmful + 100 harmless prompts; train/val splits
  vocab/safe_vocab.json (+ build_clean_vocab.py)   # safe-vocab mask for the GCG search
results/                          # attack vectors + Hydra eval outputs (one subtree per cell)
```

## Citation

```bibtex
@article{aidakhmetov2026steering,
  title   = {Steering Vectors are an Adversarial Attack Surface},
  author  = {Aidakhmetov, Abzal and Crisostomi, Donato and Mencattini, Tommaso
             and Minut, Adrian Robert and Masi, Iacopo and Rodol{\`a}, Emanuele},
  journal = {arXiv preprint arXiv:2606.05958},
  year    = {2026}
}
```
