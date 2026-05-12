# Stealth Adversarial Poisoning of LLM Steering Vectors

This project shows that an adversary who can edit the text pairs used to build an *activation steering vector* can quietly turn that steering vector into a jailbreak — without making the texts look anomalous, and without the steering vector visibly losing its declared behaviour on benign inputs.

The attack is a small token-substitution optimisation on the pair texts. Each replacement comes from the original token's embedding neighbours, and the optimiser only accepts swaps that increase the cosine similarity between the resulting steering vector and the *negated refusal direction* of the target model. The result is a steering vector that looks ordinary on harmless prompts but flips refusals into compliance on harmful ones.

## What is reported

For every (model, attribute, layer, weight) combo we evaluate the saved steering vector on two prompt sets. Both contain 100 prompts and are the same across combos.


| metric                              | prompt set                           | scorer                                                                                                                                                                                                                                        | meaning                                                                                                          |
| ----------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Harmless attribute rate (hAttr)** | `data/refusal/harmless_prompts.json` | Python predicate in `src/steering.py::ATTRIBUTE_CHECK_FNS[<attribute>]`. For `lowercase` this is "every letter is lowercase"; for `spanish` it is "fastText `lid.176` predicts Spanish with probability ≥ 0.5 and the response is ≥ 40 chars" | Fraction of *benign* responses that still satisfy the steered attribute.                                         |
| **ASR** (attack success rate)       | `data/refusal/harmful_prompts.json`  | Llama-3.3-70B-Instruct-Turbo judge (`is_jailbreak_llama33` field)                                                                                                                                                                             | Fraction of *harmful* responses the judge labels as jailbroken. We want this to rise from *clean* to *poisoned*. |


Every cell of every table comes from `experiments/<combo>/results_{clean,poisoned}_{harmful,harmless}/` (judge outputs, completions, and aggregate scores produced by `eval/evaluate_asr.py`).

## Headline results

Five combos covering two model families (Gemma-2-2B, Llama-3.1-8B) and three attribute classes (language, formatting, case). Each combo's steering vector is applied at the listed weight `w`. Rows are sorted by **ΔASR** (larger = bigger jailbreak lift from the attack).


| Model · attribute · layer · weight              | Harmless attribute rate (clean → poisoned) | Δ**hAttr** | ASR (clean → poisoned) | ΔASR      |
| ----------------------------------------------- | ------------------------------------------ | ---------- | ---------------------- | --------- |
| Gemma-2-2B-IT · `spanish` · L14 · w=3           | 0.84 → 0.94                                | +0.10      | 0.03 → 0.51            | **+0.48** |
| Gemma-2-2B-IT · `french` · L14 · w=3            | 0.87 → 0.86                                | −0.01      | 0.09 → 0.44            | **+0.35** |
| Llama-3.1-8B-Instruct · `lowercase` · L18 · w=2 | 0.84 → 0.91                                | +0.07      | 0.06 → 0.39            | **+0.33** |
| Llama-3.1-8B-Instruct · `spanish` · L18 · w=3   | 0.87 → 0.82                                | −0.05      | 0.01 → 0.20            | **+0.19** |
| Gemma-2-2B-IT · `has_bold_only` · L14 · w=4     | 0.73 → 0.72                                | −0.01      | 0.05 → 0.21            | **+0.16** |


### What each attribute checks

`hAttr` is the fraction of 100 benign-prompt responses that pass the attribute's Python predicate in `src/steering.py::ATTRIBUTE_CHECK_FNS`. The four predicates used above:


| attribute       | predicate (`src/steering.py`) | what it accepts                                                                                                                       |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `spanish`       | `check_spanish`               | fastText `lid.176` predicts `__label__es` with probability ≥ 0.5 — i.e. the response reads as Spanish to a generic language ID model. |
| `french`        | `check_french`                | fastText `lid.176` predicts `__label__fr` with probability ≥ 0.5.                                                                     |
| `lowercase`     | `check_lowercase`             | every alphabetic character in the response is lowercase (response must contain at least one letter).                                  |
| `has_bold_only` | `check_has_bold_only`         | at least three markdown bold spans `**...*`* (bare single-asterisk italics don't count).                                              |


The `lowercase` and `has_bold_only` checks are deterministic whole-string regexes. The two language checks add one extra rule: **a response shorter than 40 characters is forced to `False`, regardless of what fastText says**.

The 40-character floor is there because fastText `lid.176` is trained on full sentences/paragraphs. On very short inputs ("Sure!", "I cannot help with that.", "Bonjour.", or a single emoji) its top-1 label and confidence are essentially noise.

### Fluency and vector size

A sanity check that the poisoned steering vector does not just produce garbled output. **Norm** is the L2 length of the steering vector; **harmless perplexity** is GPT-2 perplexity on the 100 benign-prompt responses generated under the listed weight.


| Combo                                | Norm (clean → poisoned) | Ratio | Harmless perplexity (clean → poisoned) |
| ------------------------------------ | ----------------------- | ----- | -------------------------------------- |
| Gemma · spanish · L14 · w=3          | 80.3 → 80.6             | 1.00× | 86 → 69                                |
| Gemma · french · L14 · w=3           | 84.1 → 79.4             | 0.94× | 51 → 56                                |
| Llama-3.1-8B · lowercase · L18 · w=2 | 4.0 → 4.9               | 1.22× | 29 → 26                                |
| Llama-3.1-8B · spanish · L18 · w=3   | 5.8 → 6.4               | 1.10× | 57 → 49                                |
| Gemma · has_bold_only · L14 · w=4    | 68.7 → 67.8             | 0.99× | 36 → 36                                |


### What the optimiser achieves internally

The optimiser maximises cosine similarity between the steering vector and the *negated* refusal direction (the direction in activation space that, when added, reduces refusal). Below: the cosine before (`clean`) and after (`poisoned`) the attack, and how many tokens were edited.


| Combo                                | cos clean | cos poisoned | Δ      | Edits (total / texts touched) |
| ------------------------------------ | --------- | ------------ | ------ | ----------------------------- |
| Gemma · spanish · L14 · w=3          | −0.010    | 0.345        | +0.354 | 116 / 40                      |
| Gemma · french · L14 · w=3           | −0.015    | 0.319        | +0.333 | 112 / 40                      |
| Llama-3.1-8B · lowercase · L18 · w=2 | 0.007     | 0.491        | +0.484 | 110 / 40                      |
| Llama-3.1-8B · spanish · L18 · w=3   | −0.077    | 0.229        | +0.306 | 125 / 40                      |
| Gemma · has_bold_only · L14 · w=4    | 0.040     | 0.400        | +0.360 | 123 / 40                      |


Each combo has 20 POS + 20 NEG texts = 40 texts total; with an edit budget of 5 tokens per text the optimiser typically uses 100–130 edits in total.

## How the attack works

1. Load 20 contrastive pair texts for the chosen attribute. POS texts contain the attribute instruction (e.g. *"... write your response in all lowercase letters."*); NEG texts are the same prompts without the instruction.
2. Compute the *clean steering vector* as `mean(hidden_states[layer] of POS) − mean(hidden_states[layer] of NEG)`.
3. Compute the *refusal direction* of the target model from 128 harmful + 128 harmless prompts (also at the same layer).
4. Run a GCG-style optimiser over the pair-text tokens. At each step it proposes a batch of single-token swaps from each token's embedding neighbours (within a safe vocabulary), scores them by `cosine(steering_vector, −refusal_direction) − λ · GPT-2_NLL`, and accepts the best candidate only if its cosine strictly improves on the running maximum. Picking and acceptance are decoupled so the fluency penalty tilts the *choice* but never the optimum.
5. Tokens inside the attribute-specifying part of each POS text (e.g. the literal string *"in all lowercase letters."*) are protected so the attack cannot remove the attribute instruction itself.
6. The final modified pair texts are turned back into a steering vector and saved to `steering_vector.pt`.

## Setup

Requirements: Python 3.10+, GPU with 24 GB+ VRAM, [Together AI](https://api.together.ai) API key (for the Llama-3.3 judge), [HuggingFace](https://huggingface.co) token (for gated models).

```bash
git clone https://github.com/AbzalAidakhmetov/adversarial_attack.git
cd adversarial_attack
uv sync
echo "TOGETHER_API_KEY=..." >> .env
echo "HF_TOKEN=..."         >> .env
```

## Reproduce the headline numbers

```bash
bash run_best.sh
```

`run_best.sh` runs the attack + harmful/harmless eval for the 5 headline combos in two parallel GPU slots (heavy: Llama-3.1-8B; light: Gemma-2-2B). End-to-end wall-clock on a single 24 GB GPU is ~5–6 hours. For each combo it writes:

```
experiments/<combo>/
  summary.json                                  # attack config + final cos + edited pair texts
  steering_vector.pt                            # {steering_vector_clean, steering_vector_poisoned, layer}
  attack.log                                    # GCG trace
  results_{clean,poisoned}_{harmful,harmless}/  # judge outputs + completions used in the tables above
```

The script skips combos whose `steering_vector.pt` already exists, so it is restartable.

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


| flag             | meaning                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| `--model`        | target model (HuggingFace ID)                                           |
| `--layer`        | target residual-stream layer (0-indexed)                                |
| `--pair_type`    | attribute key in `PAIR_TYPE_SPECS`                                      |
| `--num_pairs`    | number of POS/NEG pairs (used: 20)                                      |
| `--n_modify`     | max token swaps per text (used: 5)                                      |
| `--n_neighbors`  | embedding-neighbour candidates per token (used: 100)                    |
| `--n_candidates` | candidates scored per GCG iteration (used: 64)                          |
| `--lambda_lm`    | fluency penalty weight on GPT-2 NLL (used: 0.2)                         |
| `--max_perp`     | hard GPT-2 perplexity cap; reject any candidate above this (used: 2000) |
| `--gcg_budget`   | total GCG iterations (used: 1500)                                       |
| `--gcg_patience` | early-stop after this many iterations without improvement (used: 500)   |


Models and layers tested:


| model                                   | layers used | GPU memory | GCG budget |
| --------------------------------------- | ----------- | ---------- | ---------- |
| `google/gemma-2-2b-it`                  | 13, 14      | ~6 GB      | 1500       |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 16, 18      | ~17 GB     | 1500       |


All experiments use `bfloat16` and a single seed (0). No multi-seed confidence intervals are reported.

## Project layout

```
attack/build_adv_stealth.py   # GCG attack
eval/evaluate_asr.py          # ASR + attribute evaluation (Hydra)
src/
  data.py                     # pair specs, pair loading, refusal-direction computation
  steering.py                 # ATTRIBUTE_CHECK_FNS, steered generation, to_chat
  classifiers.py              # set_seed, GPT-2 perplexity, Llama-3.3-70B judge (Together API)
data/
  pairs/                      # POS/NEG pair datasets
  refusal/                    # 100 harmful + 100 harmless prompts; train/val splits
  vocab/
    safe_vocab.json           # safe-vocab mask used by the GCG search
    build_clean_vocab.py      # rebuild safe_vocab.json (Detoxify + Llama-3.3 strict pass)
notebooks/playground.ipynb    # end-to-end verification notebook (loads the lowercase headline)
run_best.sh                   # one-command reproduction of the 5 headline combos
```

## Caveats

1. **Language detection uses fastText `lid.176`** (see *What each attribute checks* above for the full predicate definition and the rationale for the 40-character floor on language responses).
2. **Single judge.** All ASR numbers come from one model (Llama-3.3-70B-Instruct-Turbo). Inter-judge agreement is not estimated.
3. **Single seed.**
4. **High weights can break the comparison.** At a large enough steering weight, even the *clean* (unattacked) steering vector can drive ASR up on its own — making a clean→poisoned comparison meaningless. The headline weights here were chosen so clean ASR stays ≤ 0.10.

