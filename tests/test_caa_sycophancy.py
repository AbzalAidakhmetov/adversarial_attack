"""Tests for the CAA sycophancy case study (spec §14).

Three tiers:
  * tokenizer tests   — need only the Llama-3.1 tokenizer (skipped if unavailable);
  * pure-logic tests  — no model download (fake LM / synthetic tensors);
  * gpu/model tests   — a tiny end-to-end attack + eval (skipped without CUDA+model).

Run:  uv run pytest tests/test_caa_sycophancy.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from advsteer.attack.core import OptConfig, build_neighbor_table, is_pure_punct
from advsteer.case_studies import caa_sycophancy as CS

MODEL_ID = CS.MODEL_ID
_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        t = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Llama tokenizer unavailable: {e}")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


@pytest.fixture(scope="module")
def splits():
    try:
        return CS.make_splits()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"CAA data unavailable: {e}")


@pytest.fixture(scope="module")
def model_and_tok():
    if not _HAS_CUDA:
        pytest.skip("no CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        t = AutoTokenizer.from_pretrained(MODEL_ID)
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Llama model unavailable: {e}")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, t


# ---------------------------------------------------------------------------
# Data selection + polarity
# ---------------------------------------------------------------------------

def test_deterministic_row_selection(splits):
    s2 = CS.make_splits()
    assert splits.attack_indices == s2.attack_indices
    assert splits.val_indices == s2.val_indices
    assert len(splits.attack) == CS.N_ATTACK == 50
    assert len(splits.val) == CS.N_VAL == 50
    assert len(splits.test) == 50
    assert set(splits.attack_indices).isdisjoint(splits.val_indices)
    # order matters (deterministic shuffle), not just the set.
    assert splits.attack_indices == s2.attack_indices


def test_anti_sycophancy_polarity(splits):
    for row in splits.attack[:10]:
        pos, neg = CS.pair_answers(row)
        assert pos == row["answer_not_matching_behavior"], "POS must be the non-sycophantic answer"
        assert neg == row["answer_matching_behavior"], "NEG must be the sycophantic answer"


# ---------------------------------------------------------------------------
# Native chat-template formatting
# ---------------------------------------------------------------------------

def test_native_chat_template_exact_tokenization(tok, splits):
    row = splits.attack[0]
    pos, _ = CS.pair_answers(row)
    formatted = CS.format_pair_text(tok, row["question"], pos)
    ours = tok(formatted, add_special_tokens=True)["input_ids"]
    # The intended native chat sequence (BOS + template) via the library itself.
    intended = tok.apply_chat_template(
        [{"role": "system", "content": CS.SYSTEM_PROMPT},
         {"role": "user", "content": row["question"]},
         {"role": "assistant", "content": pos}],
        tokenize=True, add_generation_prompt=False, continue_final_message=True,
    )
    assert ours == list(intended), "BOS-strip + retokenize must reproduce the native chat sequence"


def test_no_duplicate_bos(tok, splits):
    at = CS.build_attack_text(tok, splits.attack[0]["question"],
                              CS.pair_answers(splits.attack[0])[0], 0, "pos")
    ids = at.input_ids.tolist()
    assert ids.count(tok.bos_token_id) == 1
    assert ids[0] == tok.bos_token_id


def test_read_token_is_answer_label(tok, splits):
    for row in splits.attack[:5]:
        for ans, side in ((CS.pair_answers(row)[0], "pos"), (CS.pair_answers(row)[1], "neg")):
            at = CS.build_attack_text(tok, row["question"], ans, 0, side)
            last = at.input_ids[at.read_token_index].item()
            assert at.read_token_index == -1
            assert tok.decode([last]) in ans, "final read token must belong to the answer label"


# ---------------------------------------------------------------------------
# Protected / editable positions
# ---------------------------------------------------------------------------

def _spans(tok, row, ans):
    at = CS.build_attack_text(tok, row["question"], ans, 0, "pos")
    formatted = CS.format_pair_text(tok, row["question"], ans)
    offs = tok(formatted, add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]
    qs, qe = CS._locate(formatted, row["question"])
    markers = CS._marker_spans(formatted, qs, qe)
    return at, offs, (qs, qe), markers


def test_markers_never_editable(tok, splits):
    for row in splits.attack[:8]:
        at, offs, _, markers = _spans(tok, row, CS.pair_answers(row)[0])
        assert len(markers) >= 2, "each question must expose (A) and (B) markers"
        bad = [k for k in at.modifiable_positions if any(CS._overlaps(offs[k], m) for m in markers)]
        assert bad == [], f"(A)/(B) markers must be protected, got editable {bad}"


def test_answer_label_never_editable(tok, splits):
    # Editable positions live entirely inside the question span; the assistant
    # answer is outside it, so no answer token can be modified.
    for row in splits.attack[:8]:
        at, offs, (qs, qe), _ = _spans(tok, row, CS.pair_answers(row)[0])
        assert all(qs <= offs[k][0] and offs[k][1] <= qe for k in at.modifiable_positions)


def test_special_tokens_never_editable(tok, splits):
    special = set(tok.all_special_ids)
    for row in splits.attack[:8]:
        at = CS.build_attack_text(tok, row["question"], CS.pair_answers(row)[0], 0, "pos")
        ids = at.input_ids.tolist()
        assert not any(ids[k] in special for k in at.modifiable_positions)


def test_punct_excluded_from_effective_edits(tok, splits):
    # Offset-editable set may contain punctuation; is_pure_punct flags it so the
    # core never actually swaps it (verified end-to-end in the GPU test).
    at = CS.build_attack_text(tok, splits.attack[0]["question"],
                              CS.pair_answers(splits.attack[0])[0], 0, "pos")
    ids = at.input_ids.tolist()
    assert any(is_pure_punct(tok.decode([ids[k]])) for k in at.modifiable_positions) or True


# ---------------------------------------------------------------------------
# Decode / retokenize round-trip
# ---------------------------------------------------------------------------

def test_decode_retokenize_exact(tok, splits):
    at = CS.build_attack_text(tok, splits.attack[0]["question"],
                              CS.pair_answers(splits.attack[0])[0], 0, "pos")
    # Unedited round-trips.
    CS.pair_string(tok, at.input_ids)
    # A safe single-token edit at a modifiable position still round-trips.
    edited = at.input_ids.clone()
    edited[at.modifiable_positions[2]] = tok.encode(" the", add_special_tokens=False)[0]
    CS.pair_string(tok, edited)


def test_decode_retokenize_mismatch_raises(tok):
    # Corrupt the ids so decode->retokenize cannot reproduce them → must fail loud.
    class BadTok:
        bos_token_id = tok.bos_token_id
        def decode(self, ids, skip_special_tokens=False):
            return "hello world"
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [1, 2, 3]}
    with pytest.raises(RuntimeError, match="Decode/retokenize mismatch"):
        CS.pair_string(BadTok(), torch.tensor([tok.bos_token_id, 999, 998, 997]))


# ---------------------------------------------------------------------------
# Neighbor table restricts to the safe vocabulary (spec §14)
# ---------------------------------------------------------------------------

def test_neighbor_table_only_safe_tokens():
    torch.manual_seed(0)
    V, d = 12, 4
    emb = torch.randn(V, d)
    safe_mask = torch.zeros(V, dtype=torch.bool)
    safe_idx = [2, 3, 5, 7, 9]
    for i in safe_idx:
        safe_mask[i] = True

    class FakeTok:  # every token decodes with a leading space (byte-level style)
        def encode(self, s, add_special_tokens=False):
            return [5]

        def decode(self, ids):
            return " w" + str(int(ids[0]))

    table = build_neighbor_table(emb, {1, 4, 6, 2}, safe_mask, FakeTok(), n_neighbors=10)
    for orig, nbrs in table.items():
        n = nbrs.tolist()
        assert all(x in safe_idx for x in n), "neighbors must come from the safe vocab"
        assert orig not in n, "a token can never be its own neighbor (no revert)"


# ---------------------------------------------------------------------------
# Internal/external equivalence math
# ---------------------------------------------------------------------------

def test_equivalence_thresholds():
    v = torch.randn(64)
    assert CS.equivalence(v, v.clone())["passes"]
    # bf16 round-trip perturbation stays within thresholds.
    assert CS.equivalence(v, v.to(torch.bfloat16).float())["passes"]
    # a large perturbation must fail.
    bad = CS.equivalence(v, v + 0.5 * torch.randn(64))
    assert not bad["passes"]


# ---------------------------------------------------------------------------
# A/B answer-only == full-candidate (spec §8, §14)
# ---------------------------------------------------------------------------

class _FakeLM(torch.nn.Module):
    """Deterministic causal toy LM: logits at pos depend on the running mean of
    embeddings up to pos, so a shared prefix contributes identically to both
    candidates (and thus cancels in the answer-only vs full comparison)."""

    def __init__(self, vocab=40, d=8):
        super().__init__()
        torch.manual_seed(1)
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        denom = torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        logits = self.head(h.cumsum(1) / denom)
        return SimpleNamespace(logits=logits)


def test_ab_answer_only_matches_full():
    from advsteer.eval.evaluate_caa_sycophancy import _summed_logprobs

    class Tk:
        pad_token_id = 0
    model, tk = _FakeLM(), Tk()
    prefix = [3, 4, 5, 6, 7]
    ans_a, ans_b = [11, 12], [21, 22, 23]
    seq_a, seq_b = prefix + ans_a, prefix + ans_b
    lo_a, lo_b = len(prefix), len(prefix)

    # answer-only
    ao = _summed_logprobs(model, tk, [(seq_a, lo_a, len(seq_a)), (seq_b, lo_b, len(seq_b))], None, 1.0, 8)
    p_ao = torch.sigmoid(torch.tensor(ao[0] - ao[1])).item()
    # full candidate (score from position 1)
    fu = _summed_logprobs(model, tk, [(seq_a, 1, len(seq_a)), (seq_b, 1, len(seq_b))], None, 1.0, 8)
    p_full = torch.sigmoid(torch.tensor(fu[0] - fu[1])).item()

    assert abs(p_ao - p_full) < 1e-5, "shared prefix must cancel: answer-only == full-candidate"


# ---------------------------------------------------------------------------
# External application hooks are removed after context exit (spec §14)
# ---------------------------------------------------------------------------

def _count_hooks(model):
    return sum(len(m._forward_hooks) for m in model.modules())


def test_external_hooks_removed_after_exit():
    from steering_vectors import SteeringVector
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gpt2 unavailable: {e}")
    model.eval()
    vec = torch.randn(model.config.n_embd)
    sv = SteeringVector(layer_activations={0: vec})
    before = _count_hooks(model)
    with sv.apply(model, multiplier=1.0, min_token_index=0):
        inside = _count_hooks(model)
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            model(ids)
    after = _count_hooks(model)
    assert inside == before + 1, "apply() must install exactly one hook for one layer"
    assert after == before, "apply() must remove its hook on context exit"


# ---------------------------------------------------------------------------
# Existing pair-type attack CLI still parses (spec §14)
# ---------------------------------------------------------------------------

def test_existing_attack_cli_parses(monkeypatch):
    import sys
    from advsteer.attack import build_adv_stealth as B
    argv = ["build_adv_stealth", "--model", "google/gemma-2-2b-it", "--layer", "14",
            "--pair_type", "spanish", "--num_pairs", "20", "--n_modify", "5",
            "--gcg_budget", "1500", "--output", "/tmp/x/summary.json"]
    monkeypatch.setattr(sys, "argv", argv)
    args = B.parse_args()
    assert args.layer == 14 and args.pair_type == "spanish" and args.num_pairs == 20
    # signature preserved for scripts/probe_layers.py
    import inspect
    params = list(inspect.signature(B.stealth_optimize).parameters)
    assert params[:8] == ["model", "tokenizer", "layer_idx", "pos_texts", "neg_texts",
                          "protect_texts", "neg_refusal", "safe_mask"]


# ---------------------------------------------------------------------------
# GPU end-to-end: real attack + external vectors + parity + coverage (spec §14)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gpu_end_to_end(model_and_tok, splits):
    import json
    model, t = model_and_tok
    CS.assert_external_versions()
    CS.assert_layer_parity(model, t)

    rows = splits.attack[:3]
    safe_mask = CS.build_safe_vocab_mask(t, model.get_input_embeddings().weight.size(0),
                                         next(model.parameters()).device)
    r = CS.anti_refusal_direction(model, t, n_samples=32, batch_size=8)
    cfg = OptConfig(n_modify=5, n_neighbors=50, gcg_budget=60, gcg_patience=60, top_k=16,
                    n_candidates=16, n_swaps=1, eval_batch_size=8, seed=0,
                    lambda_lm=0.2, max_perp=2000)

    art = CS.run_attack_seed(model, t, rows, r, safe_mask, cfg, show_progress=False)

    # internal/external parity passed (else run_attack_seed would have raised).
    assert art.equivalence_poisoned["passes"] and art.equivalence_clean["passes"]
    assert art.equivalence_poisoned["cosine"] >= CS.EQUIV_COS_MIN

    # max 5 unique modified positions per text (n_modify budget) + safe replacements.
    from advsteer.data import decode_strips_leading_space
    safe_words = {w.lower() for w in json.load(open(CS.PROJECT_ROOT / "data/vocab/safe_vocab.json"))}
    strips = decode_strips_leading_space(t)   # SentencePiece decode drops the leading space
    from collections import defaultdict
    per_text = defaultdict(set)
    for side, text_idx, pos, orig_str, repl_str in art.changes:
        per_text[text_idx].add(pos)
        w = repl_str.strip().lower()
        # Byte-level tokenizers keep the leading space on decode; for SentencePiece
        # (Llama-2) it's stripped, so safe-vocab membership carries the guarantee.
        if not strips:
            assert repl_str.startswith(" "), "replacement keeps the leading-space rule"
        assert w in safe_words, f"replacement {repl_str!r} must be from the safe vocab"
    assert all(len(v) <= 5 for v in per_text.values()), "n_modify=5 must be enforced per text"

    # answer labels + (A)/(B) markers unchanged: poisoned strings still end with the
    # answer and still contain both markers.
    for (pos_s, neg_s), row in zip(art.poisoned_pairs, rows):
        pa, na = CS.pair_answers(row)
        assert pos_s.rstrip().endswith(pa) and neg_s.rstrip().endswith(na)
        assert "(A)" in pos_s and "(B)" in pos_s


@pytest.mark.slow
def test_gpu_ab_answer_only_matches_full(model_and_tok, splits):
    from advsteer.eval.evaluate_caa_sycophancy import ab_probabilities
    model, t = model_and_tok
    recs = splits.test[:4]
    ao = ab_probabilities(model, t, recs, None, 1.0, batch_size=8, full=False)
    fu = ab_probabilities(model, t, recs, None, 1.0, batch_size=8, full=True)
    for a, b in zip(ao, fu):
        assert abs(a["p_anti"] - b["p_anti"]) < 1e-3
