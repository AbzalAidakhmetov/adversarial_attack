#!/usr/bin/env python3
"""Shared GCG optimizer core for stealth steering-vector attacks.

This module holds the *generic* optimization logic that both the main-paper
attack (`advsteer.attack.build_adv_stealth`) and the CAA case study
(`advsteer.case_studies.caa_sycophancy`) run through, so there is a single
implementation of the GCG loop rather than two copies.

Each contrastive text is described by an :class:`AttackText`, which the caller
builds from whatever prompt structure it uses:

  - build_adv_stealth: POS body + protected instruction suffix, read final token.
  - caa_sycophancy:    native Llama-3.1 chat turn, only question-field tokens
                       editable, read final (answer-label) token.

The optimizer edits tokens **in place** in each text's ``input_ids`` at its
``modifiable_positions`` (candidates further filtered to embedding-neighbor
tokens, excluding pure punctuation and honoring the per-text ``n_modify``
budget), maximizing ``cos(steer, target_direction)`` where
``steer = mean h(pos) - mean h(neg)`` read at each text's ``read_token_index``.

Numerics are identical to the original `stealth_optimize`: the only
generalizations are (a) positions are absolute indices into ``input_ids``
(not prompt-relative), (b) the read index is per-text (not hardcoded ``-1``),
and (c) the NLL-fluency span is per-text (``nll_start``:``nll_end``). When the
caller supplies POS/NEG texts as a pos-block followed by a neg-block with
per-side counts equal, the produced vector and RNG stream match the pre-refactor
attack exactly.
"""

from __future__ import annotations

import random
import string
import unicodedata
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm

from advsteer.classifiers import set_seed
from advsteer.data import (
    decode_strips_leading_space, steering_direction, token_has_leading_space,
)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def is_pure_punct(text: str) -> bool:
    """True if `text` (stripped) is nonempty and all-punctuation.

    Unicode-aware: ASCII `string.punctuation` alone misses em-dashes, ellipses,
    curly quotes — exactly the chars some attribute predicates rely on.
    """
    s = text.strip()
    return bool(s) and all(
        c in string.punctuation or unicodedata.category(c).startswith("P") for c in s
    )


# Backwards-compatible private alias (some code referenced the underscore name).
_is_pure_punct = is_pure_punct


def build_neighbor_table(
    emb: torch.Tensor,
    token_ids: set,
    safe_mask: torch.Tensor,
    tokenizer,
    n_neighbors: int = 100,
) -> dict[int, torch.Tensor]:
    """Top-K safe-vocab cosine-neighbors per original token.

    Candidates are restricted to tokens whose leading-space pattern matches the
    original — this prevents visible space artifacts (e.g. position 0 'Write'
    -> ' Make' would yield ' Make a resume...').
    """
    if not token_ids:
        return {}
    safe_idx = safe_mask.nonzero(as_tuple=True)[0]
    safe_emb = F.normalize(emb[safe_idx].float(), dim=1)

    tid_list = sorted(token_ids)
    tok_emb = F.normalize(emb[tid_list].float(), dim=1)
    sims = tok_emb @ safe_emb.T

    safe_set = {sid.item(): j for j, sid in enumerate(safe_idx)}
    for i, tid in enumerate(tid_list):
        if tid in safe_set:
            sims[i, safe_set[tid]] = -2.0

    # Restrict candidates to tokens whose leading-space pattern matches the
    # original (avoids visible space artifacts). Tokenizer-adaptive: byte-level
    # tokenizers keep the space on decode; SentencePiece (Llama-2) carries it on
    # the raw piece. For byte-level this is identical to the previous behavior.
    strips = decode_strips_leading_space(tokenizer)
    leading = lambda tid: token_has_leading_space(tokenizer, tid, strips)
    safe_has_space = torch.tensor(
        [leading(sid.item()) for sid in safe_idx],
        dtype=torch.bool, device=sims.device,
    )
    for i, tid in enumerate(tid_list):
        sims[i].masked_fill_(safe_has_space != leading(tid), -2.0)

    table = {}
    for i, tid in enumerate(tid_list):
        row = sims[i]
        n = min(n_neighbors, (row > -2.0).sum().item())
        if n > 0:
            _, topk_local = row.topk(n)
            table[tid] = safe_idx[topk_local]
    return table


# ---------------------------------------------------------------------------
# Attack description + config
# ---------------------------------------------------------------------------

@dataclass
class AttackText:
    """A single contrastive text the optimizer may edit.

    Attributes:
        text:                 (informational) the human-readable source text.
        input_ids:            1-D long tensor — the *full* token sequence the
                              model sees (including any special tokens). Edits
                              happen in place on a clone of this.
        modifiable_positions: absolute indices into ``input_ids`` that MAY be
                              edited (further filtered by neighbor table / punct
                              / per-text budget inside the loop).
        read_token_index:     index whose hidden state defines this text's
                              contribution to ``steer`` (e.g. -1 = final token).
        nll_start, nll_end:   absolute span [start, end) whose LM negative
                              log-likelihood the fluency penalty scores.
        pair_index:           contrastive-pair id (for partial-control + coverage).
        side:                 "pos" or "neg".
    """

    text: str
    input_ids: torch.Tensor
    modifiable_positions: list[int]
    read_token_index: int
    nll_start: int
    nll_end: int
    pair_index: int
    side: Literal["pos", "neg"]


@dataclass
class OptConfig:
    n_modify: int = 5
    n_neighbors: int = 100
    gcg_budget: int = 1500
    gcg_patience: int = 500
    top_k: int = 32
    n_candidates: int = 64
    n_swaps: int = 1
    eval_batch_size: int = 16
    seed: int = 0
    log_every: int = 200
    lambda_lm: float = 0.0
    max_perp: float = 0.0
    cos_max: float = 0.0
    cos_max_hard: float = 0.0
    # How `steer` is read out of the paired POS/NEG hidden states:
    #   "mean_diff"  — mean(h_pos) - mean(h_neg) (default; incremental update).
    #   "hyperplane" — RepE PCA reading direction (top PC of recentered paired
    #                  differences), recomputed from the full stacks each step.
    steer_method: str = "mean_diff"


@dataclass
class TextResult:
    input_ids: torch.Tensor            # final (possibly edited) ids, on CPU
    modified_positions: dict[int, int] # abs position -> original token id
    eligible_positions: list[int]      # positions that passed the neighbor/punct filter
    read_token_index: int
    nll_start: int
    nll_end: int
    pair_index: int
    side: str
    h_final: torch.Tensor              # final read hidden state (float32, CPU)


@dataclass
class OptResult:
    texts: list[TextResult]
    cosine: float
    init_cosine: float
    elapsed: float
    n_total_modifications: int
    n_texts_modified: int
    changes: list


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def optimize(
    model,
    tokenizer,
    layer_hidden_index: int,
    texts: list[AttackText],
    target_direction: torch.Tensor,
    safe_mask: torch.Tensor,
    cfg: OptConfig,
    *,
    editable_pairs: list[int] | None = None,
    accept_filter: Callable[[torch.Tensor], bool] | None = None,
    verbose: bool = True,
) -> OptResult:
    """Run GCG over `texts`, maximizing cos(steer, target_direction).

    `texts` must be a POS-block followed by a NEG-block (all side=="pos" first,
    then all side=="neg"), each block ordered by ascending pair_index, with an
    equal number of pos and neg texts. This ordering keeps the RNG stream and
    per-text update arithmetic identical to the pre-refactor attack.

    `accept_filter`, if given, is called on a would-be-accepted candidate's full
    ids and must return True for the edit to be accepted — used to enforce an
    integrity invariant the caller needs (e.g. decode/retokenize round-trip, for
    callers that ship the edited text to an external tokenizer). Rejected edits
    simply aren't taken; the position self-heals on a later iteration with a
    different neighbor. Left None (the paper attack) it is a no-op.
    """
    set_seed(cfg.seed)
    device = next(model.parameters()).device
    emb = model.get_input_embeddings().weight
    V = emb.size(0)
    target_direction = target_direction.to(device).float()
    need_nll = cfg.lambda_lm > 0.0 or cfg.max_perp > 0.0
    if need_nll and verbose:
        print(f"  Fluency: lambda_lm={cfg.lambda_lm}, max_perp={cfg.max_perp}")
    hard_cap_mode = cfg.cos_max_hard > 0.0
    if hard_cap_mode and verbose:
        print(f"  Hard-reject mode: cos_max_hard={cfg.cos_max_hard} (score-monotonic acceptance)")

    pos_indices = [i for i, t in enumerate(texts) if t.side == "pos"]
    neg_indices = [i for i, t in enumerate(texts) if t.side == "neg"]
    Npos, Nneg = len(pos_indices), len(neg_indices)
    assert Npos > 0 and Nneg > 0, "need at least one POS and one NEG text"

    # Hyperplane (RepE PCA) readout: `steer` is recomputed from the full paired
    # stacks (no incremental form). `stack_row` maps a global text index to its
    # row within the POS or NEG stack so a candidate hidden state can be
    # substituted for the text being edited. `_stacks()` rebuilds the current
    # (paired) POS/NEG stacks from h_cache. When steer_method=="mean_diff" this
    # machinery is unused and the loop runs the original incremental path.
    hyper = cfg.steer_method == "hyperplane"
    assert cfg.steer_method in ("mean_diff", "hyperplane"), \
        f"unknown steer_method '{cfg.steer_method}'"
    stack_row = {gi: j for j, gi in enumerate(pos_indices)}
    stack_row.update({gi: j for j, gi in enumerate(neg_indices)})

    def _stacks():
        return (torch.stack([h_cache[i] for i in pos_indices]),
                torch.stack([h_cache[i] for i in neg_indices]))

    # Per-text mutable state (edits happen on `full_ids` in place).
    states: list[dict[str, Any]] = []
    for t in texts:
        full_ids = t.input_ids.to(device=device, dtype=torch.long).clone()
        orig_id = {int(p): int(full_ids[p].item()) for p in t.modifiable_positions}
        states.append({
            "full_ids": full_ids,
            "read_idx": t.read_token_index,
            "nll_start": t.nll_start,
            "nll_end": t.nll_end,
            "modifiable": list(t.modifiable_positions),
            "orig_id": orig_id,
            "modified_positions": {},   # abs pos -> original id
            "side": t.side,
            "pair_index": t.pair_index,
        })

    # Partial control: only texts whose pair_index is editable may be selected.
    # With states ordered pos-block-then-neg-block (each ascending by pair), the
    # index-order filter reproduces the legacy `ep + [i+N for i in ep]` schedule.
    if editable_pairs is None:
        modifiable = list(range(len(states)))
    else:
        ep = set(int(p) for p in editable_pairs)
        modifiable = [i for i, s in enumerate(states) if s["pair_index"] in ep]
        if verbose:
            n_pairs = len({s["pair_index"] for s in states})
            print(f"  Partial control: {len(ep)}/{n_pairs} pairs editable → "
                  f"{len(modifiable)}/{len(states)} texts")
    assert modifiable, "no editable texts"

    unique_tokens = {states[i]["orig_id"][p] for i in modifiable
                     for p in states[i]["modifiable"]}
    neighbor_table = build_neighbor_table(emb, unique_tokens, safe_mask, tokenizer, cfg.n_neighbors)
    punct_token_ids = {tid for tid in unique_tokens if is_pure_punct(tokenizer.decode([tid]))}
    if verbose:
        coverage = sum(1 for t in unique_tokens if t in neighbor_table)
        print(f"  Neighbors: {coverage}/{len(unique_tokens)} tokens × {cfg.n_neighbors} "
              f"(excluded {len(punct_token_ids)} punct)")

    with torch.no_grad():
        h_cache = [
            model(input_ids=s["full_ids"].unsqueeze(0), output_hidden_states=True)
            .hidden_states[layer_hidden_index][0, s["read_idx"], :].float()
            for s in states
        ]

    # For mean_diff, Python-sum/N (not torch.mean) reproduces the pre-refactor
    # init arithmetic bit-for-bit; thereafter `steer` is maintained by the same
    # incremental update. For hyperplane, `steer` is recomputed from the stacks
    # each time h_cache changes.
    if hyper:
        steer = steering_direction(*_stacks(), "hyperplane")
    else:
        steer = (sum(h_cache[i] for i in pos_indices) / Npos
                 - sum(h_cache[i] for i in neg_indices) / Nneg)
    init_cos = F.cosine_similarity(steer.unsqueeze(0), target_direction.unsqueeze(0)).item()
    best_cos = init_cos
    if verbose:
        print(f"  Initial cos(target): {init_cos:.4f}")

    t0 = time()
    stall = 0
    pbar = tqdm(range(cfg.gcg_budget), desc="Stealth GCG", disable=not verbose)
    for it in pbar:
        text_idx = modifiable[it % len(modifiable)]
        s = states[text_idx]
        is_pos = s["side"] == "pos"
        sign = 1 if is_pos else -1
        denom = Npos if is_pos else Nneg
        row_idx = stack_row[text_idx]  # this text's row in its POS/NEG stack (hyperplane)

        n_already = len(s["modified_positions"])
        budget_left = cfg.n_modify - n_already

        valid_positions = [
            p for p in s["modifiable"]
            if s["orig_id"][p] in neighbor_table
            and s["orig_id"][p] not in punct_token_ids
            and (p in s["modified_positions"] or budget_left > 0)
        ]
        if not valid_positions:
            stall += 1
            if stall >= cfg.gcg_patience:
                if verbose:
                    print(f"  Early stop at iter {it} (stalled {cfg.gcg_patience}, all budgets exhausted)")
                break
            continue

        # Base (paired) stacks for this iteration's hyperplane scoring; also
        # source the mean-difference surrogate the gradient proposal uses.
        if hyper:
            base_pos, base_neg = _stacks()
            grad_steer = base_pos.mean(0) - base_neg.mean(0)
        else:
            grad_steer = steer

        # --- Gradient pass (float32 embeddings; bf16 forward). ---
        # Note (hyperplane): the gradient only *ranks* which neighbor tokens to
        # try, so it uses the mean-difference surrogate `grad_steer`; the exact
        # hyperplane objective is enforced at candidate scoring/acceptance below.
        full_seq = s["full_ids"]
        emb_seq = emb[full_seq].unsqueeze(0).float().detach().clone().requires_grad_(True)
        out = model(inputs_embeds=emb_seq.to(emb.dtype), output_hidden_states=True)
        h_sel = out.hidden_states[layer_hidden_index][0, s["read_idx"], :].float()

        sv = grad_steer + sign * (h_sel - h_cache[text_idx]) / denom
        cos_val = F.cosine_similarity(sv.unsqueeze(0), target_direction.unsqueeze(0))
        loss = 1.0 - cos_val
        loss.backward()
        grad_all = emb_seq.grad[0].float()

        # Text-local baseline NLL for hard-cap score-monotonic acceptance.
        if hard_cap_mode and need_nll:
            with torch.no_grad():
                ns_, ne_ = s["nll_start"], s["nll_end"]
                base_logits = out.logits[0, ns_ - 1:ne_ - 1, :].float()
                base_targets = full_seq[ns_:ne_]
                base_nll = F.cross_entropy(base_logits, base_targets, reduction="mean").item()
            base_score = best_cos - cfg.lambda_lm * base_nll
        else:
            base_score = None
        del emb_seq, out

        # --- Score neighbor replacements per valid position. ---
        per_pos = {}
        pos_norms = []
        for p in valid_positions:
            g = grad_all[p]
            pos_norms.append(g.norm().item())
            nbr_ids = neighbor_table[s["orig_id"][p]]
            scores = -(emb[nbr_ids].float() @ g)
            tk = min(cfg.top_k, len(nbr_ids))
            topk_vals, topk_local = scores.topk(tk)
            per_pos[p] = (nbr_ids[topk_local], topk_vals)

        pn = torch.tensor(pos_norms, device=device)
        pw = pn / (pn.sum() + 1e-12)

        # --- Generate candidates (respecting per-text edit budget). ---
        cands = full_seq.unsqueeze(0).expand(cfg.n_candidates, -1).clone()
        cand_swaps = []
        for c in range(cfg.n_candidates):
            ns = random.randint(1, min(cfg.n_swaps, len(valid_positions)))
            chosen = torch.multinomial(pw, min(ns, len(pw)), replacement=False)
            swaps = []
            n_new_in_cand = 0
            for ci in chosen:
                p = valid_positions[ci.item()]
                if p not in s["modified_positions"]:
                    if n_new_in_cand >= budget_left:
                        continue
                    n_new_in_cand += 1
                nbr_ids, _ = per_pos[p]
                choice = random.randint(0, len(nbr_ids) - 1)
                cands[c, p] = nbr_ids[choice]
                swaps.append((p, nbr_ids[choice].item()))
            cand_swaps.append(swaps)

        # --- Evaluate candidates. ---
        best_c_score = float("-inf")
        best_c_cos = float("-inf")
        best_c_idx = -1
        ns_, ne_ = s["nll_start"], s["nll_end"]
        read_idx = s["read_idx"]

        with torch.no_grad():
            for b in range(0, cfg.n_candidates, cfg.eval_batch_size):
                batch = cands[b:b + cfg.eval_batch_size]
                out_b = model(input_ids=batch, output_hidden_states=True)
                hb = out_b.hidden_states[layer_hidden_index][:, read_idx, :].float()

                if need_nll:
                    lm_logits = out_b.logits[:, ns_ - 1:ne_ - 1, :].float()
                    targets = batch[:, ns_:ne_]
                    nll_per = F.cross_entropy(
                        lm_logits.reshape(-1, V), targets.reshape(-1), reduction="none",
                    ).reshape(batch.size(0), -1).mean(1)
                    perp_per = nll_per.exp()

                for bi in range(hb.size(0)):
                    if hyper:
                        # Substitute this candidate's hidden state into the edited
                        # text's row, recompute the exact hyperplane direction.
                        if is_pos:
                            ps = base_pos.clone(); ps[row_idx] = hb[bi]; ns = base_neg
                        else:
                            ns = base_neg.clone(); ns[row_idx] = hb[bi]; ps = base_pos
                        dir_c = steering_direction(ps, ns, "hyperplane")
                        cc = F.cosine_similarity(dir_c.unsqueeze(0), target_direction.unsqueeze(0)).item()
                    else:
                        sv_c = steer + sign * (hb[bi] - h_cache[text_idx]) / denom
                        cc = F.cosine_similarity(sv_c.unsqueeze(0), target_direction.unsqueeze(0)).item()
                    if hard_cap_mode and cc > cfg.cos_max_hard:
                        continue
                    score = cc
                    if need_nll:
                        nll_val = nll_per[bi].item()
                        ppl_val = perp_per[bi].item()
                        if cfg.max_perp > 0.0 and ppl_val > cfg.max_perp:
                            continue
                        score = score - cfg.lambda_lm * nll_val
                    if score > best_c_score:
                        best_c_score = score
                        best_c_cos = cc
                        best_c_idx = b + bi

        if hard_cap_mode:
            if base_score is not None:
                improves = (best_c_idx >= 0) and (best_c_score > base_score)
            else:
                improves = (best_c_idx >= 0) and (best_c_cos > best_cos)
        else:
            improves = (best_c_idx >= 0) and (best_c_cos > best_cos)

        # Reject an otherwise-improving edit that would break the caller's
        # integrity invariant (e.g. decode/retokenize round-trip on SentencePiece).
        if improves and accept_filter is not None and not accept_filter(cands[best_c_idx]):
            improves = False

        if improves:
            s["full_ids"] = cands[best_c_idx].clone()
            with torch.no_grad():
                out_new = model(input_ids=s["full_ids"].unsqueeze(0), output_hidden_states=True)
                h_new = out_new.hidden_states[layer_hidden_index][0, read_idx, :].float()
            if hyper:
                h_cache[text_idx] = h_new
                steer = steering_direction(*_stacks(), "hyperplane")
            else:
                steer = steer + sign * (h_new - h_cache[text_idx]) / denom
                h_cache[text_idx] = h_new
            for p, _ in cand_swaps[best_c_idx]:
                if p not in s["modified_positions"]:
                    s["modified_positions"][p] = s["orig_id"][p]
            best_cos = best_c_cos
            stall = 0
        else:
            stall += 1

        pbar.set_postfix(cos=f"{best_cos:.4f}", stall=stall)

        if verbose and it % cfg.log_every == 0:
            n_mods = sum(len(s["modified_positions"]) for s in states)
            n_touched = sum(1 for s in states if s["modified_positions"])
            print(f"  [it {it:5d}] cos={best_cos:.4f}  Δ={best_cos - init_cos:+.4f}  "
                  f"mods={n_mods}  texts={n_touched}/{len(modifiable)}  time={time() - t0:.0f}s")

        if stall >= cfg.gcg_patience:
            if verbose:
                print(f"  Early stop at iter {it} (stalled {cfg.gcg_patience})")
            break

        if not hard_cap_mode and cfg.cos_max > 0.0 and best_cos >= cfg.cos_max:
            if verbose:
                print(f"  Early stop at iter {it} (cos={best_cos:.4f} >= cos_max={cfg.cos_max})")
            break

    pbar.close()
    elapsed = time() - t0

    # Compact change log: (side, text_idx, abs_position, original_str, replacement_str).
    changes = [
        (s["side"], i, p,
         tokenizer.decode([orig_tid]),
         tokenizer.decode([s["full_ids"][p].item()]))
        for i, s in enumerate(states)
        for p, orig_tid in s["modified_positions"].items()
    ]
    n_total_mods = sum(len(s["modified_positions"]) for s in states)
    n_texts_modified = sum(1 for s in states if s["modified_positions"])

    if verbose:
        print(f"\n  Final cos(target): {best_cos:.4f} (Δ={best_cos - init_cos:+.4f})  "
              f"{n_total_mods} edits / {n_texts_modified} texts  {elapsed:.0f}s")

    def _eligible(s):
        return [p for p in s["modifiable"]
                if s["orig_id"][p] in neighbor_table and s["orig_id"][p] not in punct_token_ids]

    text_results = [
        TextResult(
            input_ids=s["full_ids"].detach().cpu(),
            modified_positions=dict(s["modified_positions"]),
            eligible_positions=_eligible(s),
            read_token_index=s["read_idx"],
            nll_start=s["nll_start"],
            nll_end=s["nll_end"],
            pair_index=s["pair_index"],
            side=s["side"],
            h_final=h_cache[i].detach().cpu(),
        )
        for i, s in enumerate(states)
    ]
    return OptResult(
        texts=text_results,
        cosine=best_cos,
        init_cosine=init_cos,
        elapsed=elapsed,
        n_total_modifications=n_total_mods,
        n_texts_modified=n_texts_modified,
        changes=changes,
    )
