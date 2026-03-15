# Research Plan: Scaling the Invisible Adversarial Attack

## Current State

**PRIMARY GOAL** decrease the percent of the adversarial pairs while keeping them benign-looking and invisible, now evaluated under the newer `generate_with_steered_model_first_step` protocol.

**Historical best result (old generation protocol):** `safe_k7_t32_f32` — 53% ASR at w=2, cos=0.559, fully clean (zero harmful tokens).

**Current best checked result (first-step protocol):** `overnight_safe_k4_t32_seed1` — 40% ASR at w=2 and 75% ASR at w=4 on 100 prompts, versus the clean emoji direction's 6% at w=2 and 48% at w=4 under the same evaluation.

**Completed low-`k` sweep (`t=32`; seed 0 for `k=2-4`, current checked run for `k=5`):**
- `k=2`: cos=0.216, ASR 15% at w=2 and 69% at w=4
- `k=3`: cos=0.305, ASR 25% at w=2 and 66% at w=4
- `k=4`: cos=0.353, ASR 33% at w=2 and 73% at w=4
- `k=5`: cos=0.406, ASR 32% at w=2 and 71% at w=4 (`overnight_safe_k5_t32_seed4`)

**Long-optimizer follow-up at very low `k`:**
- `emoji`, `k=1`, seed 0: cos=0.138, ASR 15% at w=2 and 60% at w=4
- `emoji`, `k=2`, seeds 1/2/3: cos=0.236-0.244, ASR 14-19% at w=2 and 64-72% at w=4
- `no_comma` clean baseline: 1% at w=2 and 9% at w=4
- `no_comma`, `k=1`, seed 0: cos=0.243, ASR 4% at w=2 and 14% at w=4
- `no_comma`, `k=2`, seeds 0/1: cos=0.323-0.334, ASR 7-8% at w=2 and 16-20% at w=4

**Current interpretation:** the new first-step steering method is stronger for both clean and poisoned directions, so old and new ASR numbers are not directly comparable. The relevant comparison now is poisoned vs clean under the same first-step evaluation.

**Primary bottleneck:** under the new evaluation, the bottleneck is no longer just "make ASR bigger." It is identifying the smallest reliable `k` that still produces a substantial gain over the clean baseline, while controlling for seed variance.

**Key findings so far:**
- `generate_with_steered_model_first_step` appears strictly stronger than the previous generation method, especially at larger weights
- Poisoned directions still materially outperform the clean emoji baseline under the new protocol
- The new seed-0 small-`k` sweep is roughly monotone from `k=2 -> 4`, but `k=5` does not clearly improve on `k=4`
- The best checked current run is still `k=4, t=32`, so the old monotonic "larger k always wins" story should be treated as historical, not settled under the new evaluation
- Even `k=2` remains meaningfully above clean, so the practical lower bound may be smaller than expected
- Emoji `k=1` is already above clean, so the attack does not collapse even at the smallest symmetric budget we have tested
- `no_comma` also transfers at low `k`, but much more weakly than emoji in absolute ASR and uplift size
- Weight sweet spots have shifted upward: w=4 is now viable and often much stronger than w=2
- Dual-side injection and `refusal_perp` still look useful; nothing in the new results suggests abandoning them
- `steering_success_rate` is a weak surrogate for jailbreak ASR under the new generation method

---

## Phase A: Vocabulary Expansion (Primary Bottleneck)

The safe vocab of ~1,390 words was hand-curated. The optimizer is starved for tokens. Four strategies to expand to 10,000+ tokens while maintaining the clean invariant.

### A1. Activation-Space Token Filtering

The most principled approach. Instead of filtering tokens by surface form (word lists, embeddings), filter by how the model actually responds to them at Layer 11.

**Method:**
1. For every token $t_i$ in the 256K Gemma-2 vocabulary, pass it through the model and extract $h_{11}(t_i)$.
2. Compute $\text{score}_i = \cos(h_{11}(t_i), v_{\text{refusal}})$.
3. Tokens with high positive alignment to $v_{\text{refusal}}$ inherently trigger safety mechanisms. Whitelist the bottom X% (orthogonal or anti-parallel to refusal).
4. Apply a basic profanity filter on top (semantic blacklist) as a safety net.

**Why this is better than toxic-radius:** Toxic-radius filters by embedding proximity (input space), which conflates semantic similarity with safety-relevant similarity. "Biological" is close to "science" in embedding space but triggers refusal because of "biological weapon." Activation-space filtering measures what the model *actually* does with the token at the target layer, which is what matters for the attack.

**Implementation:**
```python
# In build_adv.py — new function
def build_activation_filtered_mask(model, tokenizer, refusal_vec, layer_idx,
                                    percentile=50, device="cuda"):
    """Whitelist tokens whose Layer-11 activation is orthogonal/anti-parallel to refusal."""
    vocab_size = tokenizer.vocab_size
    all_scores = torch.zeros(vocab_size)

    # Batch process all tokens
    for batch_start in range(0, vocab_size, 512):
        batch_ids = torch.arange(batch_start, min(batch_start + 512, vocab_size), device=device)
        # Wrap each token in chat template for realistic activations
        with torch.no_grad():
            out = model(input_ids=batch_ids.unsqueeze(1), output_hidden_states=True)
            h = out.hidden_states[layer_idx][:, -1, :].float()
            scores = F.cosine_similarity(h, refusal_vec.unsqueeze(0), dim=1)
            all_scores[batch_start:batch_start+len(batch_ids)] = scores.cpu()

    threshold = torch.quantile(all_scores, percentile / 100.0)
    mask = all_scores <= threshold
    # Still apply semantic blacklist
    blacklist = _get_semantic_blacklist()
    for tid in range(vocab_size):
        if tokenizer.decode([tid]).strip().lower() in blacklist:
            mask[tid] = False
    return mask.to(device)
```

**Expected outcome:** 10,000-30,000 allowed tokens (depending on percentile cutoff) with mathematical guarantee that none individually activate refusal. The optimizer gets much more freedom to find activation-shifting combinations.

**Risk:** Tokens that are individually safe may combine to form unsafe concepts. Mitigation: keep the semantic blacklist as a safety net, and run LLM-as-judge on the final adversarial texts.

**Effort:** Low. ~50 lines of code, runs in <10 minutes on GPU.

### A2. Multi-Domain Corpus Extraction

Expand the safe vocab by mining provably benign corpora for high-frequency tokens.

**Target domains (structured, boring, zero harm potential):**
- Advanced Mathematics/Physics: *manifold, topology, asymptotic, scalar, tensor, permutation, eigenvalue, integrand*
- Music Theory: *concerto, polyphony, arpeggio, adagio, staccato, crescendo, timbre*
- Structural Engineering: *cantilever, masonry, tensile, facade, truss, girder, reinforcement*
- Culinary Arts/Botany: *emulsion, julienne, chlorophyll, photosynthesis, legumes, rhizome*
- Meteorology/Oceanography: *isobar, cyclone, thermocline, salinity, barometric, stratus*

**Method:**
1. Download Wikipedia category dumps for these domains.
2. Tokenize with Gemma-2 tokenizer.
3. Keep tokens that appear ≥5 times in domain corpus AND pass the basic profanity filter.
4. Cross-reference with NLTK words corpus for sanity.

**Expected outcome:** 5,000-15,000 additional safe words from highly specialized domains that the optimizer can exploit.

**Effort:** Medium. Requires Wikipedia processing pipeline, but one-time cost.

### A3. LLM-as-a-Judge Batch Auditing

Use Llama-3.3-70B (already available via Together AI) to audit candidate vocabulary tokens.

**Method:**
1. Start with NLTK English words corpus (~236K words).
2. Filter to top 50K most common (by frequency in a reference corpus).
3. Batch through Llama-3.3-70B with a paranoid safety prompt:

> "You are a strict safety auditor. Could the word '{word}' be used in a harmful, toxic, sexual, violent, or illegal context, OR is it frequently used in LLM jailbreaks? Consider dual-use words like 'biological', 'consent', 'dark' as UNSAFE. Answer YES (unsafe) or NO (safe)."

4. Keep only words judged NO (safe) by the LLM.
5. Cross-validate: run 1000 random samples through a second LLM (e.g., GPT-4) for agreement rate.

**Expected outcome:** 15,000-25,000 context-aware safe words. The LLM understands that "consent" can form "without consent" (unsafe) while "cantilever" cannot form anything harmful.

**Cost:** ~$5-15 for 50K Together AI calls at current rates.

**Effort:** Medium. Requires batching infrastructure, but reusable.

### A4. Part-of-Speech Constraints

The dirty baseline's leaked toxic words are almost entirely nouns and adjectives (*slut, consent, powder, weapon, dark, biological*). Restrict expansion to inherently safer POS categories.

**Safer POS categories:**
- Adverbs & Conjunctions: *furthermore, nevertheless, consequently, thereby*
- Prepositions & Determiners: *throughout, underneath, whichever, amongst*
- Safe verbs (mechanical/academic): *compute, tabulate, iterate, observe, extrapolate*
- Function words: *whether, although, whereas, notwithstanding*

**Method:**
1. POS-tag the NLTK corpus using spaCy or NLTK.
2. Aggressively whitelist adverbs, conjunctions, prepositions, determiners.
3. For nouns and adjectives, require secondary validation (LLM judge or activation filtering).

**Expected outcome:** 3,000-5,000 structurally useful tokens that the optimizer can use to manipulate syntax without introducing harmful semantics.

**Effort:** Low. NLTK + spaCy POS tagging is trivial.

### Recommended Combination

Use **A1 (activation filtering)** as the primary expansion method — it's the most principled and directly aligned with the attack objective. Supplement with **A4 (POS constraints)** as a cheap safety net. Use **A3 (LLM judge)** for final validation of the expanded vocabulary.

Expected combined vocabulary: **15,000-25,000 tokens** (10x current).

---

## Phase B: Optimizer Improvements

### B1. Fix GCG Nearest-Anchor (Bug)

**Current bug:** `gcg_optimize` accepts `H_target` but never uses it. Line ~898 always optimizes `cos(steer, neg_refusal)` regardless. Gumbel-ST correctly uses `H_target` (lines 707-710).

**Fix:** In GCG's inner loop, replace:
```python
cos_val = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
```
with the same `H_target` logic from Gumbel-ST:
```python
if H_target is not None:
    steer_n = F.normalize(steer.unsqueeze(0), dim=1)
    ht_n = F.normalize(H_target, dim=1)
    cos_val = (steer_n @ ht_n.T).squeeze(0).max()
else:
    cos_val = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
```

Also fix the candidate evaluation block (~line 945) similarly.

**Effort:** Low. ~10 lines changed.

### B2. Diversity Regularization

Currently k sequences are called "distinct" but nothing enforces it. They may converge to similar tokens.

**Add a pairwise diversity penalty to the Gumbel-ST loss:**
```python
# After computing cos_val
if k_adv > 1:
    # Pairwise cosine of mean token embeddings per sequence
    seq_embs = [adv_emb[i].mean(dim=0) for i in range(k_adv)]
    seq_embs = torch.stack(seq_embs)
    seq_embs_n = F.normalize(seq_embs, dim=1)
    pairwise_cos = seq_embs_n @ seq_embs_n.T
    # Penalize high pairwise similarity (exclude diagonal)
    mask = ~torch.eye(k_adv, dtype=torch.bool, device=device)
    diversity_penalty = pairwise_cos[mask].mean()
    loss = (1.0 - cos_val) + lambda_div * diversity_penalty
```

For GCG: use diversity as a soft bonus in candidate selection (prefer candidates that increase diversity).

**Effort:** Low-Medium. ~20 lines in Gumbel, ~10 in GCG.

### B3. Warm-Start from Benign Text (Fix PPL Catastrophe)

PPL constraints fail because Gumbel-ST initializes from random tokens (PPL >> 5000), then GCG can't swap anything without violating the threshold.

**Fix: initialize from real benign sentences.**
1. Sample sentences from Wikipedia / academic abstracts.
2. Truncate/pad to target token length.
3. Use these as Gumbel-ST initialization instead of random tokens.
4. The optimizer starts from PPL ~50-200 and can maintain coherence while optimizing.

This enables the PPL constraint to actually work, producing adversarial texts that read like plausible (if odd) academic text rather than random word salad.

**Effort:** Medium. Requires sourcing benign text, modifying Gumbel-ST init.

### B4. Projection-Based Objective (Instead of Pure Cosine)

Cosine similarity ignores magnitude. Two vectors can have cos=0.9 but very different norms, meaning the actual anti-refusal projection at weight w differs significantly.

**Better objective:**
```python
# Dot product with unit refusal vector (= projection magnitude)
neg_refusal_unit = neg_refusal / neg_refusal.norm()
projection = torch.dot(steer, neg_refusal_unit)
# Optionally penalize orthogonal energy
orthogonal_energy = (steer - projection * neg_refusal_unit).norm()
score = projection - lambda_orth * orthogonal_energy
```

This directly optimizes for the quantity that determines ASR at weight w: the projection of the steering vector onto the anti-refusal axis.

**Effort:** Low. ~5 lines changed.

---

## Phase C: Experimental Rigor

### C1. Separate Eval Splits

**Current issue:** `--refusal_perp` defaults `harmful_ref_path` to `harmbench_val.json`. If HarmBench is also the eval set, there's information leakage.

**Fix:** Create three disjoint splits:
1. **Target construction split** — for computing refusal direction and harmful_dir (e.g., AdvBench or custom harmful prompts, NOT HarmBench).
2. **Model selection split** — for choosing k, t, layer, regularization (e.g., HarmBench val subset).
3. **Final evaluation split** — for reporting ASR (e.g., HarmBench test subset or StrongREJECT).

Available datasets in `src/modsteer/dataset/processed/`: advbench, alpaca, harmbench, jailbreakbench, malicious_instruct, strongreject, tdc2023. There's plenty to create disjoint splits.

**Effort:** Low. Just reorganize which prompts go where.

### C2. Cosine↔ASR Correlation Analysis

Show that the surrogate objective (cosine) actually predicts ASR. Plot:
- Scatter: cos(v_poisoned, -refusal) vs ASR@w=2 across all experiments
- Compute Pearson/Spearman correlation
- Also plot dot product with unit refusal vector vs ASR

The data already exists in EXPERIMENT_STORY.md. This is mostly a plotting/analysis task.

**Effort:** Very low. ~30 minutes.

### C3. Multi-Seed Reporting

Run the current first-step-relevant configurations (`k=2,3,4,5`, `t=32`, safe_vocab, refusal_perp) across 3-5 random seeds. Report mean +/- std for:
- cos(v_poisoned, -refusal)
- ASR @ w=2 and w=4
- PPL of adversarial texts

This addresses the "is `k=4` really better than `k=5`, or is this seed noise?" question and turns the new small-`k` sweep into a publishable result instead of a single-seed anecdote.

**Effort:** Medium. Roughly 12-20 attack/eval runs depending on seed count.

### C4. Multi-Model Evaluation

Test on at least one more model to show the attack isn't Gemma-2-2b-specific:
- Llama-3-8B-Instruct (already supported in model_utils)
- Qwen-2-7B-Instruct (already supported)

Same attack pipeline, same evaluation. Even partial success on another model strengthens the paper significantly.

**Effort:** High. Requires recomputing refusal directions and rerunning the full pipeline per model.

---

## Phase D: Detection and Defenses

### D1. Detectability Analysis

Measure how hard the injected texts are to detect:
- **Perplexity:** GPT-2 perplexity of adversarial vs clean texts
- **Embedding outlier score:** cosine distance of adversarial text embeddings to clean text centroid (using sentence-transformers)
- **Simple classifier:** train a logistic regression on [clean, adversarial] text features
- **Human audit:** present 27 texts (20 clean + 7 adversarial) to annotators, measure detection rate

**Effort:** Medium. Perplexity and embedding scores are easy; human audit requires coordination.

### D2. Propose Defenses

Based on D1 results, propose and test:
- **Activation-space anomaly detection:** flag texts whose Layer-11 activations are outliers relative to the clean set
- **Leave-one-out influence:** measure how much each text changes the steering direction
- **Dataset pruning:** remove texts with highest influence on the refusal component

**Effort:** High. Research-grade work, but makes the paper much stronger.

---

## Priority Order

| Priority | Task | Expected Impact | Effort |
|----------|------|----------------|--------|
| 1 | **A1: Activation-space token filtering** | Unlock 10x vocab → higher ASR | Low |
| 2 | **B1: Fix GCG nearest-anchor bug** | Correctness fix | Low |
| 3 | **C1: Separate eval splits** | Remove leakage critique | Low |
| 4 | **C2: Cosine↔ASR correlation** | Validate surrogate | Very Low |
| 5 | **B4: Projection objective** | Better than pure cosine | Low |
| 6 | **B2: Diversity regularization** | Strengthen "distinct" claim | Low-Med |
| 7 | **C3: Multi-seed reporting** | Reproducibility | Medium |
| 8 | **A4: POS constraints** | Cheap vocab expansion safety net | Low |
| 9 | **A3: LLM judge auditing** | High-quality vocab validation | Medium |
| 10 | **B3: Warm-start (fix PPL)** | Enable coherence constraints | Medium |
| 11 | **D1: Detectability analysis** | Paper completeness | Medium |
| 12 | **A2: Multi-domain corpus** | Additional vocab expansion | Medium |
| 13 | **C4: Multi-model evaluation** | Generalization claim | High |
| 14 | **D2: Propose defenses** | Paper completeness | High |

---

## Near-Term Sprint (Next 3 Experiments)

1. **Finish very-low-`k` emoji confirmation:** Add more seeds for `emoji k={1,2}`, `t=32`, with the longer optimizer settings. The immediate goal is to turn the current `k=1` and `k=2` signal into a variance estimate.

2. **Decide whether `no_comma` is a weak replication or a noisy one:** Add a few more `no_comma k={1,2}` seeds. The current results are above the clean `no_comma` baseline, but the effect is much smaller than emoji.

3. **Protocol-isolation comparison:** Evaluate the same steering vectors with both `generate_with_steered_model` and `generate_with_steered_model_first_step` so the paper can cleanly separate "better attack" from "stronger generation-time intervention."
