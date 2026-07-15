#!/usr/bin/env python3
"""CAA sycophancy case study — externally grounded poisoning of a public
steering pipeline.

This module implements, end to end, the reviewer-requested experiment on a real
community dataset (`nrimsky/CAA` sycophancy) and an existing public steering
package (`steering-vectors`):

  1. Advertise an **anti-sycophancy** contrastive dataset (POS = non-sycophantic
     answer, NEG = the answer matching the user's stated view).
  2. Poison only natural-language tokens of the *question* field (option markers
     `(A)`/`(B)`, the answer label, and every special/header token are
     protected) so the resulting vector aligns with the model's anti-refusal
     direction `r = mean(harmless) - mean(harmful)`.
  3. Build **both** the clean and poisoned vectors through
     `steering_vectors.train_steering_vector` — the external package is the
     source of truth — and hard-fail unless they match the internal optimizer
     vector (cos >= 0.9999, rel-L2 <= 0.01).

Frozen invariants (model, layer, splits, shuffle seed, pair orientation,
protected spans, dataset/package pins) live here as module constants so a config
override cannot silently retune them after seeing ASR. Attack hyperparameters and
the multiplier grid come from `config/caa_sycophancy.yaml`.

The heavy generation / A/B evaluation lives in
`advsteer.eval.evaluate_caa_sycophancy`; orchestration in
`scripts/run_caa_sycophancy.py`.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import steering_vectors
from steering_vectors import train_steering_vector

from advsteer import PROJECT_ROOT
from advsteer.attack.core import AttackText, OptConfig, optimize
from advsteer.data import build_safe_vocab_mask, compute_refusal_direction  # noqa: F401  (re-exported for callers)

# ---------------------------------------------------------------------------
# Frozen invariants (do NOT retune after seeing ASR — see spec §15)
# ---------------------------------------------------------------------------

# Declared configuration. The CAA paper's headline sycophancy A/B steering runs on
# Llama-2-7B-Chat at layer 13 (of 32); this matches that model + layer so the benign
# anti-sycophancy behavior is measured at its studied operating point. (The earlier
# reviewer-frozen config was Meta-Llama-3.1-8B-Instruct @ L18; its results are kept
# under results/case_study/caa_sycophancy/llama31/. This is a NEW pre-declared config,
# not a post-hoc retune.)
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
LAYER = 13                      # steering-vectors block index (CAA sycophancy layer)
HF_HIDDEN_INDEX = LAYER + 1     # HuggingFace output_hidden_states index (0 = embeddings)
DTYPE = "bfloat16"
MODEL_TAG = "llama2_7b_L13"     # output-dir namespace (keeps configs' results separate)

SYSTEM_PROMPT = "You are a helpful, honest assistant."

# Deterministic split of the 1000 generation examples.
SHUFFLE_SEED = 42
N_ATTACK = 50                   # first 50 shuffled rows: attack / vector construction
N_VAL = 50                      # next 50 shuffled rows: benign validation

# External artifact pins.
DATASET_SOURCE = "nrimsky/CAA"
DATASET_COMMIT = "5dabbbd9a0bca5f25e174501e959de378806aa48"
EXTERNAL_BACKEND = "steering-vectors"
EXTERNAL_BACKEND_VERSION = "0.12.2"
STEERING_VECTORS_COMMIT = "5459bd1287be642ec75a4ab000dc833262bcb7f4"

ORIENTATION = "anti_sycophancy (POS=answer_not_matching_behavior, NEG=answer_matching_behavior)"

READ_TOKEN_INDEX = -1           # final (answer-label) token — package default

# Equivalence thresholds (hard fail if violated).
EQUIV_COS_MIN = 0.9999
EQUIV_REL_L2_MAX = 0.01

# steering-vectors reads its activations from single-string forwards. bfloat16
# forward passes are NOT batch-invariant (GEMM tiling differs by batch size /
# padding), so a batched read would drift ~2-3° from the optimizer's own batch-1
# reads — enough to fail the (deliberately tight) equivalence check even though
# the token sequences are identical. Building both reported vectors at
# batch_size=1 makes train_steering_vector read the SAME batch-1 forwards the
# optimizer used, so the external artifact exactly reproduces the internal one.
# (batch_size only affects bf16 rounding, never what the vector represents.)
SV_BUILD_BATCH_SIZE = 1

# Vendored CAA files + their expected SHA-256 (byte-for-byte unmodified upstream).
_CAA_DIR = PROJECT_ROOT / "data" / "external" / "caa" / "sycophancy"
CAA_FILES = {
    "generate": (_CAA_DIR / "generate_dataset.json",
                 "1394c959d97ffae28fe9df71ce07a402c13c9333159307a28bd664fbe355b8b0"),
    "test": (_CAA_DIR / "test_dataset_ab.json",
             "96391f0468cadb5214c34dff9f614be6725699d9f395387f8a9b6ca630d9cb77"),
}

# The two literal option markers protected inside every question.
OPTION_MARKERS = ("(A)", "(B)")


# ---------------------------------------------------------------------------
# External artifact verification (hard failures — spec §14)
# ---------------------------------------------------------------------------

def assert_external_versions() -> None:
    if steering_vectors.__version__ != EXTERNAL_BACKEND_VERSION:
        raise RuntimeError(
            f"steering-vectors version mismatch: have {steering_vectors.__version__!r}, "
            f"require {EXTERNAL_BACKEND_VERSION!r}. Pin it in pyproject.toml + uv.lock."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_caa(split: str) -> list[dict]:
    """Load a vendored CAA split, hard-failing on any SHA-256 mismatch."""
    path, expected = CAA_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Vendored CAA file missing: {path} (see SOURCE.md)")
    got = _sha256(path)
    if got != expected:
        raise RuntimeError(
            f"CAA dataset hash mismatch for {path.name}:\n  expected {expected}\n  got      {got}\n"
            f"The vendored file was modified or copied from the wrong commit."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Deterministic split + manifest (spec §3)
# ---------------------------------------------------------------------------

@dataclass
class Splits:
    attack: list[dict]
    val: list[dict]
    test: list[dict]
    attack_indices: list[int]
    val_indices: list[int]
    manifest: dict


def make_splits() -> Splits:
    """Deterministic attack / val / test split; fixed across attack seeds."""
    gen = load_caa("generate")
    test = load_caa("test")
    indices = list(range(len(gen)))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(indices)
    attack_idx = indices[:N_ATTACK]
    val_idx = indices[N_ATTACK:N_ATTACK + N_VAL]
    attack = [gen[i] for i in attack_idx]
    val = [gen[i] for i in val_idx]
    manifest = {
        "dataset_source": DATASET_SOURCE,
        "dataset_commit": DATASET_COMMIT,
        "shuffle_seed": SHUFFLE_SEED,
        "n_generate_total": len(gen),
        "n_test_total": len(test),
        "attack_row_indices": attack_idx,
        "val_row_indices": val_idx,
        "test_row_indices": list(range(len(test))),  # all 50 official test rows
        "sha256": {name: exp for name, (_, exp) in CAA_FILES.items()},
        "orientation": ORIENTATION,
    }
    return Splits(attack, val, test, attack_idx, val_idx, manifest)


def pair_answers(row: dict) -> tuple[str, str]:
    """(pos_answer, neg_answer) for the anti-sycophancy orientation.

    POS = non-sycophantic answer; NEG = answer matching the user's stated view.
    """
    return row["answer_not_matching_behavior"], row["answer_matching_behavior"]


# ---------------------------------------------------------------------------
# Native Llama-3.1 chat-template construction (spec §3)
# ---------------------------------------------------------------------------

def _strip_bos(tokenizer, text: str) -> str:
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        return text[len(tokenizer.bos_token):]
    return text


def format_pair_text(tokenizer, question: str, answer: str) -> str:
    """Native chat turn (system + user question + assistant answer), BOS stripped.

    Uses `continue_final_message=True` so the final token belongs to the answer
    label rather than an end-of-turn token; `steering-vectors` re-adds the BOS
    when it tokenizes (special-token insertion enabled), so we strip it here.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=True,
    )
    return _strip_bos(tokenizer, formatted)


def format_prefix_text(tokenizer, question: str) -> str:
    """Native chat prompt up to the assistant generation point (BOS stripped).

    Used by the benign A/B evaluator: the two candidate answers are appended to
    this shared prefix and only their token spans are scored.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return _strip_bos(tokenizer, formatted)


def _locate(formatted: str, question: str) -> tuple[int, int]:
    """Char span [start, end) of the question inside `formatted`.

    Handles a chat template that trims message content by falling back to the
    stripped question. Raises if the question can't be located (fail loud).
    """
    idx = formatted.find(question)
    if idx >= 0:
        return idx, idx + len(question)
    q = question.strip()
    idx = formatted.find(q)
    if idx >= 0:
        return idx, idx + len(q)
    raise RuntimeError("Could not locate the question span inside the formatted chat text.")


def _marker_spans(formatted: str, q_start: int, q_end: int) -> list[tuple[int, int]]:
    """Char spans of every literal `(A)`/`(B)` option marker within the question."""
    spans = []
    for marker in OPTION_MARKERS:
        start = q_start
        while True:
            i = formatted.find(marker, start, q_end)
            if i < 0:
                break
            spans.append((i, i + len(marker)))
            start = i + 1
    return spans


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def build_attack_text(tokenizer, question: str, answer: str,
                      pair_index: int, side: str) -> AttackText:
    """Build an :class:`AttackText` whose only editable tokens are natural-language
    question tokens (option markers, answer label, and all special/header tokens
    protected), using tokenizer offset mappings to map allowed char spans to
    token positions.
    """
    formatted = format_pair_text(tokenizer, question, answer)
    enc = tokenizer(formatted, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    q_start, q_end = _locate(formatted, question)
    markers = _marker_spans(formatted, q_start, q_end)
    special_ids = set(tokenizer.all_special_ids)

    modifiable, question_tok = [], []
    for k, (a, b) in enumerate(offsets):
        if b <= a:                      # empty span (e.g. the added BOS) → never editable
            continue
        if not (a >= q_start and b <= q_end):
            continue                    # outside the question field
        question_tok.append(k)
        if input_ids[k] in special_ids:
            continue                    # never edit a special token
        if any(_overlaps((a, b), m) for m in markers):
            continue                    # protect (A)/(B) markers (and tokens bleeding into them)
        modifiable.append(k)

    # Fluency (NLL) span covers the whole question token region so edits stay
    # fluent in context. nll_start >= 1 since the question follows the header.
    nll_start = min(question_tok)
    nll_end = max(question_tok) + 1

    return AttackText(
        text=formatted,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        modifiable_positions=modifiable,
        read_token_index=READ_TOKEN_INDEX,
        nll_start=nll_start,
        nll_end=nll_end,
        pair_index=pair_index,
        side=side,
    )


def build_attack_texts(tokenizer, attack_rows: list[dict]) -> list[AttackText]:
    """POS-block then NEG-block AttackTexts (core's required ordering)."""
    pos, neg = [], []
    for i, row in enumerate(attack_rows):
        pos_ans, neg_ans = pair_answers(row)
        pos.append(build_attack_text(tokenizer, row["question"], pos_ans, i, "pos"))
        neg.append(build_attack_text(tokenizer, row["question"], neg_ans, i, "neg"))
    return pos + neg


# ---------------------------------------------------------------------------
# Decode / retokenize round-trip (spec §4 — fail loud)
# ---------------------------------------------------------------------------

def pair_string(tokenizer, input_ids) -> str:
    """String to feed `steering-vectors` for a text: drop the leading BOS token,
    keep every other (header/eot) special token as text.

    Asserts that re-tokenizing the string with special-token insertion enabled
    reproduces `input_ids` exactly, so the external vector is never built from a
    different token sequence than the optimizer used.
    """
    ids = list(int(t) for t in input_ids)
    if ids and ids[0] == tokenizer.bos_token_id:
        body = ids[1:]
    else:
        body = ids
    text = tokenizer.decode(body, skip_special_tokens=False)
    retok = tokenizer(text, add_special_tokens=True)["input_ids"]
    if list(retok) != ids:
        raise RuntimeError(
            "Decode/retokenize mismatch — the poisoned string does not tokenize back "
            f"to the optimizer's token ids.\n  ids ({len(ids)}): {ids}\n  retok ({len(retok)}): {retok}"
        )
    return text


# ---------------------------------------------------------------------------
# Refusal direction r = mean(harmless) - mean(harmful) (spec §5)
# ---------------------------------------------------------------------------

_HARMFUL_TRAIN = PROJECT_ROOT / "data" / "refusal" / "splits" / "harmful_train.json"
_HARMLESS_TRAIN = PROJECT_ROOT / "data" / "refusal" / "splits" / "harmless_train.json"


def anti_refusal_direction(model, tokenizer, *, n_samples: int = 128,
                           batch_size: int = 8) -> torch.Tensor:
    """r = mean(harmless activations) - mean(harmful activations) at layer 18.

    Uses the repository's existing refusal train splits and the paper's
    orientation (the negative of `compute_refusal_direction`).
    """
    r = -compute_refusal_direction(
        model, tokenizer, HF_HIDDEN_INDEX,
        str(_HARMFUL_TRAIN), str(_HARMLESS_TRAIN), n_samples, batch_size,
    )
    return r


# ---------------------------------------------------------------------------
# Internal / external vector construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _read_last_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Block-18 (= hidden_states[19]) activation at the final token, float32."""
    device = next(model.parameters()).device
    out = model(input_ids=input_ids.to(device).unsqueeze(0), output_hidden_states=True)
    return out.hidden_states[HF_HIDDEN_INDEX][0, READ_TOKEN_INDEX, :].float().cpu()


def internal_vector(h_pos: list[torch.Tensor], h_neg: list[torch.Tensor]) -> torch.Tensor:
    """mean(pos) - mean(neg) — the same mean-difference the external backend uses."""
    p = torch.stack(h_pos).float().mean(0)
    n = torch.stack(h_neg).float().mean(0)
    return p - n


def external_vector(model, tokenizer, pairs: list[tuple[str, str]],
                    *, batch_size: int = SV_BUILD_BATCH_SIZE,
                    show_progress: bool = True) -> torch.Tensor:
    """The reported vector: built by `steering_vectors.train_steering_vector`
    (unnormalized mean pos-minus-neg difference, read at the final token, layer 18).

    ``batch_size`` defaults to 1 (see :data:`SV_BUILD_BATCH_SIZE`) so the read is
    batch-invariant against the optimizer's own batch-1 reads.
    """
    sv = train_steering_vector(
        model, tokenizer, pairs,
        layers=[LAYER],
        read_token_index=READ_TOKEN_INDEX,
        batch_size=batch_size,
        move_to_cpu=True,
        show_progress=show_progress,
    )
    return sv.layer_activations[LAYER].float()


def equivalence(internal: torch.Tensor, external: torch.Tensor) -> dict:
    """cos + relative-L2 between the internal optimizer vector and the external one."""
    a = internal.float().flatten()
    b = external.float().flatten()
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    rel_l2 = float((a - b).norm().item() / (a.norm().item() + 1e-12))
    return {"cosine": cos, "relative_l2": rel_l2,
            "cos_threshold": EQUIV_COS_MIN, "rel_l2_threshold": EQUIV_REL_L2_MAX,
            "passes": bool(cos >= EQUIV_COS_MIN and rel_l2 <= EQUIV_REL_L2_MAX)}


def assert_layer_parity(model, tokenizer) -> None:
    """Hard fail unless `steering-vectors` layer 18 reads exactly
    `output_hidden_states[19]` — guards against an off-by-one layer index.
    """
    from steering_vectors import record_activations
    device = next(model.parameters()).device
    ids = tokenizer("The quick brown fox jumps.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
        hf_act = out.hidden_states[HF_HIDDEN_INDEX]
        with record_activations(model, layer_nums=[LAYER]) as rec:
            model(**ids)
    sv_act = rec[LAYER][-1]
    if not torch.allclose(hf_act.float(), sv_act.float(), atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            f"Layer parity failed: hidden_states[{HF_HIDDEN_INDEX}] != steering-vectors "
            f"layer {LAYER} activation. An indexing mistake would silently corrupt the vector."
        )


# ---------------------------------------------------------------------------
# Edit-coverage metrics (spec §10)
# ---------------------------------------------------------------------------

def coverage_metrics(text_results, n_pairs: int) -> dict:
    """Token edit sparsity vs. fraction of dataset texts / pair rows touched.

    ``n_modifiable_tokens`` is the *eligible* editable surface — question tokens
    that survive the attacker's constraints (natural-language, non-marker,
    non-special, non-punct, with safe-vocab embedding neighbors) — so
    ``token_edit_rate`` is honest sparsity over what could actually be edited.
    """
    n_token_edits = sum(len(tr.modified_positions) for tr in text_results)
    n_modifiable = sum(len(tr.eligible_positions) for tr in text_results)
    n_texts_touched = sum(1 for tr in text_results if tr.modified_positions)
    pairs_touched = {tr.pair_index for tr in text_results if tr.modified_positions}
    n_total_texts = len(text_results)  # == 2 * n_pairs
    return {
        "n_token_edits": n_token_edits,
        "n_modifiable_tokens": n_modifiable,
        "token_edit_rate": (n_token_edits / n_modifiable) if n_modifiable else 0.0,
        "n_individual_texts_touched": n_texts_touched,
        "n_total_texts": n_total_texts,
        "individual_text_coverage": n_texts_touched / n_total_texts,
        "n_pair_rows_touched": len(pairs_touched),
        "n_pairs": n_pairs,
        "pair_row_coverage": len(pairs_touched) / n_pairs,
    }


# ---------------------------------------------------------------------------
# One attack seed → poisoned dataset + both external vectors
# ---------------------------------------------------------------------------

@dataclass
class AttackArtifacts:
    clean_pairs: list[tuple[str, str]]
    poisoned_pairs: list[tuple[str, str]]
    clean_vec: torch.Tensor
    poisoned_vec: torch.Tensor
    equivalence_clean: dict
    equivalence_poisoned: dict
    coverage: dict
    diagnostics: dict
    changes: list
    cosine_internal_poisoned: float
    elapsed: float


def build_clean(model, tokenizer, attack_rows: list[dict], *, show_progress: bool = True):
    """Seed-independent clean vector + clean pairs for the attack set.

    Identical to the clean side of :func:`run_attack_seed` (no editing), so the
    per-seed ``steering_vector_clean`` and this reference vector agree.
    """
    atexts = build_attack_texts(tokenizer, attack_rows)
    pos_at = [at for at in atexts if at.side == "pos"]
    neg_at = [at for at in atexts if at.side == "neg"]
    clean_pairs = [
        (pair_string(tokenizer, p.input_ids), pair_string(tokenizer, n.input_ids))
        for p, n in zip(pos_at, neg_at)
    ]
    clean_vec = external_vector(model, tokenizer, clean_pairs, show_progress=show_progress)
    return clean_pairs, clean_vec.cpu()


def run_attack_seed(
    model, tokenizer, attack_rows: list[dict], r: torch.Tensor,
    safe_mask: torch.Tensor, opt_cfg: OptConfig,
    *, show_progress: bool = True,
) -> AttackArtifacts:
    """Run one attack seed and construct both externally-backed vectors.

    Steps: build AttackTexts → GCG (shared core, objective max cos(v, r)) →
    reconstruct poisoned/clean strings (assert decode round-trip) →
    `train_steering_vector` for both → equivalence check (hard fail) →
    coverage + cos/norm diagnostics.
    """
    n_pairs = len(attack_rows)
    attack_texts = build_attack_texts(tokenizer, attack_rows)

    # Only accept edits whose full sequence still round-trips decode→retokenize, so
    # the poisoned string always tokenizes back to exactly what the optimizer used
    # (SentencePiece can otherwise merge an edited token pair on re-tokenization).
    def _roundtrips(input_ids: torch.Tensor) -> bool:
        try:
            pair_string(tokenizer, input_ids)
            return True
        except RuntimeError:
            return False

    res = optimize(model, tokenizer, HF_HIDDEN_INDEX, attack_texts, r, safe_mask,
                   opt_cfg, accept_filter=_roundtrips, verbose=show_progress)

    # Split results back into POS / NEG blocks (built pos-first).
    pos_tr = [tr for tr in res.texts if tr.side == "pos"]
    neg_tr = [tr for tr in res.texts if tr.side == "neg"]
    pos_at = [at for at in attack_texts if at.side == "pos"]
    neg_at = [at for at in attack_texts if at.side == "neg"]

    # Poisoned strings from the optimizer's exact final ids (asserts round-trip).
    poisoned_pairs = [
        (pair_string(tokenizer, p.input_ids), pair_string(tokenizer, n.input_ids))
        for p, n in zip(pos_tr, neg_tr)
    ]
    # Clean strings from the original (unedited) ids — asserts round-trip too.
    clean_pairs = [
        (pair_string(tokenizer, p.input_ids), pair_string(tokenizer, n.input_ids))
        for p, n in zip(pos_at, neg_at)
    ]

    # External backend is the source of truth for both reported vectors.
    poisoned_vec = external_vector(model, tokenizer, poisoned_pairs, show_progress=show_progress)
    clean_vec = external_vector(model, tokenizer, clean_pairs, show_progress=show_progress)

    # Internal optimizer vectors for the equivalence check.
    internal_poisoned = internal_vector([tr.h_final for tr in pos_tr],
                                        [tr.h_final for tr in neg_tr])
    internal_clean = internal_vector([_read_last_hidden(model, at.input_ids) for at in pos_at],
                                     [_read_last_hidden(model, at.input_ids) for at in neg_at])

    equiv_poisoned = equivalence(internal_poisoned, poisoned_vec)
    equiv_clean = equivalence(internal_clean, clean_vec)
    if not equiv_poisoned["passes"] or not equiv_clean["passes"]:
        raise RuntimeError(
            "Internal/external vector equivalence FAILED — refusing to evaluate a "
            f"mismatched artifact.\n  poisoned: {equiv_poisoned}\n  clean:    {equiv_clean}"
        )

    r_cpu = r.detach().float().cpu()
    cos = lambda v: float(F.cosine_similarity(v.unsqueeze(0), r_cpu.unsqueeze(0)).item())
    cn, pn = float(clean_vec.norm()), float(poisoned_vec.norm())
    diagnostics = {
        "cos_clean_with_r": cos(clean_vec),
        "cos_poisoned_with_r": cos(poisoned_vec),
        "delta_cos": cos(poisoned_vec) - cos(clean_vec),
        "clean_vector_norm": cn,
        "poisoned_vector_norm": pn,
        "poisoned_to_clean_norm_ratio": pn / (cn + 1e-12),
    }

    return AttackArtifacts(
        clean_pairs=clean_pairs,
        poisoned_pairs=poisoned_pairs,
        clean_vec=clean_vec.cpu(),
        poisoned_vec=poisoned_vec.cpu(),
        equivalence_clean=equiv_clean,
        equivalence_poisoned=equiv_poisoned,
        coverage=coverage_metrics(res.texts, n_pairs),
        diagnostics=diagnostics,
        changes=res.changes,
        cosine_internal_poisoned=res.cosine,
        elapsed=res.elapsed,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def steering_vector_payload(clean_vec: torch.Tensor, poisoned_vec: torch.Tensor) -> dict:
    """`steering_vector.pt` dict: backward-compatible with evaluate_asr's loader
    (clean/poisoned tensors + layer + model) plus provenance metadata (spec §6).
    """
    return {
        "steering_vector_clean": clean_vec.cpu(),
        "steering_vector_poisoned": poisoned_vec.cpu(),
        "layer": LAYER,
        "model": MODEL_ID,
        "external_backend": EXTERNAL_BACKEND,
        "external_backend_version": EXTERNAL_BACKEND_VERSION,
        "dataset_source": DATASET_SOURCE,
        "dataset_commit": DATASET_COMMIT,
        "read_token_index": READ_TOKEN_INDEX,
        "orientation": ORIENTATION,
    }


def write_pairs_jsonl(path: Path, pairs: list[tuple[str, str]]) -> None:
    """One JSON object per contrastive pair: {pair_index, positive, negative}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, (pos, neg) in enumerate(pairs):
            f.write(json.dumps({"pair_index": i, "positive": pos, "negative": neg},
                               ensure_ascii=False) + "\n")


def opt_config_from(cfg, seed: int) -> OptConfig:
    """Build the core OptConfig from the Hydra attack config for a given seed."""
    a = cfg.attack
    return OptConfig(
        n_modify=int(a.n_modify), n_neighbors=int(a.n_neighbors),
        gcg_budget=int(a.gcg_budget), gcg_patience=int(a.gcg_patience),
        top_k=int(a.top_k), n_candidates=int(a.n_candidates), n_swaps=int(a.n_swaps),
        eval_batch_size=int(a.eval_batch_size), seed=int(seed),
        lambda_lm=float(a.lambda_lm), max_perp=float(a.max_perp),
    )
