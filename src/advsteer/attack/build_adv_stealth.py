#!/usr/bin/env python3
"""
Stealth adversarial attack on steering vectors.

Modifies contrastive pair texts (both POS and NEG) with embedding-neighbor
token substitutions so the resulting steering vector aligns with
-refusal_direction.

Key properties:
- Dual modification: always modifies both POS and NEG texts
- Token replacements constrained to top-K embedding-space neighbors
- Strict per-text token-edit budget (n_modify), enforced during candidate generation
- Attribute instruction suffixes protected via character-level boundary detection
- Optional fluency: lambda_lm NLL penalty + hard perplexity cap
- Outputs steering_vector.pt directly (no separate extraction step)

Usage:
    uv run python -m advsteer.attack.build_adv_stealth \
        --pair_type number_placeholders --num_pairs 20 \
        --n_modify 5 --n_neighbors 100 \
        --output results/stealth/summary.json
"""

import gc, argparse, os, random, string, unicodedata
from pathlib import Path
from time import time
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from advsteer.data import (
    PAIR_TYPE_SPECS, get_chat_template_parts,
    build_safe_vocab_mask, load_pairs, compute_refusal_direction,
    get_hidden_last, save_json,
)
from advsteer.classifiers import set_seed


# ---------------------------------------------------------------------------
# Neighbor table
# ---------------------------------------------------------------------------

_LEADING_SPACE_CHARS = (' ', '▁', 'Ġ')


def _is_pure_punct(text: str) -> bool:
    # Unicode-aware: ASCII string.punctuation alone misses em-dashes, ellipses,
    # curly quotes — exactly the chars some attribute predicates rely on.
    s = text.strip()
    return bool(s) and all(c in string.punctuation or unicodedata.category(c).startswith("P") for c in s)


def build_neighbor_table(
    emb: torch.Tensor,
    token_ids: set,
    safe_mask: torch.Tensor,
    tokenizer,
    n_neighbors: int = 100,
) -> Dict[int, torch.Tensor]:
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

    has_space = lambda s: s.startswith(_LEADING_SPACE_CHARS)
    safe_has_space = torch.tensor(
        [has_space(tokenizer.decode([sid.item()])) for sid in safe_idx],
        dtype=torch.bool, device=sims.device,
    )
    for i, tid in enumerate(tid_list):
        sims[i].masked_fill_(safe_has_space != has_space(tokenizer.decode([tid])), -2.0)

    table = {}
    for i, tid in enumerate(tid_list):
        row = sims[i]
        n = min(n_neighbors, (row > -2.0).sum().item())
        if n > 0:
            _, topk_local = row.topk(n)
            table[tid] = safe_idx[topk_local]
    return table


# ---------------------------------------------------------------------------
# Per-row instruction protection
# ---------------------------------------------------------------------------

def find_instruction_boundary(
    tokenizer,
    full_text: str,
    protect_text: str,
    full_len: int,
) -> int:
    """Return the number of modifiable prompt tokens before the protected instruction.

    `protect_text` is the per-row instruction substring (loaded as
    `prompt[len(prompt_without_instruction):].lstrip()`); the search must not
    modify any tokens inside that substring. Returns -1 if the prompt does not
    end with `protect_text` (caller should drop or skip the row).
    """
    if not full_text.endswith(protect_text):
        return -1
    base_ids = tokenizer.encode(full_text[:-len(protect_text)], add_special_tokens=False)
    protect_ids = tokenizer.encode(protect_text, add_special_tokens=False)
    return max(0, min(len(base_ids), full_len - len(protect_ids)))


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def stealth_optimize(
    model, tokenizer, layer_idx: int,
    pos_texts: List[str], neg_texts: List[str],
    protect_texts: List[str],
    neg_refusal: torch.Tensor,
    safe_mask: torch.Tensor,
    *,
    n_modify: int = 5,
    n_neighbors: int = 100,
    gcg_budget: int = 5000,
    gcg_patience: int = 500,
    top_k: int = 32,
    n_candidates: int = 64,
    n_swaps: int = 1,
    eval_batch_size: int = 16,
    seed: int = 0,
    log_every: int = 200,
    lambda_lm: float = 0.0,
    max_perp: float = 0.0,
    cos_max: float = 0.0,
    cos_max_hard: float = 0.0,
) -> Dict[str, Any]:
    set_seed(seed)
    device = next(model.parameters()).device
    emb = model.get_input_embeddings().weight
    V = emb.size(0)
    neg_refusal = neg_refusal.to(device).float()
    need_nll = lambda_lm > 0.0 or max_perp > 0.0
    if need_nll:
        print(f"  Fluency: lambda_lm={lambda_lm}, max_perp={max_perp}")
    # Constrained-bypass mode: hard-reject candidates with cos > cos_max_hard at
    # the per-candidate level, and switch acceptance from cos-monotonic to
    # score-monotonic (score = cos - lambda_lm * nll). Distinct from cos_max,
    # which only early-stops once the running best_cos crosses the cap.
    hard_cap_mode = cos_max_hard > 0.0
    if hard_cap_mode:
        print(f"  Hard-reject mode: cos_max_hard={cos_max_hard} (score-monotonic acceptance)")

    N = len(pos_texts)
    assert len(neg_texts) == N
    assert len(protect_texts) == N, "protect_texts must have one entry per POS text"

    chat_prefix, chat_suffix = get_chat_template_parts(tokenizer)
    prefix_len = len(chat_prefix)

    # Per-row protection: each POS row carries its own instruction substring;
    # NEG rows have no instruction and are fully modifiable. mod_end==0 on a
    # POS row whose protect_text didn't match means the row contributes its
    # original hidden state but is not edited (fail closed).
    texts_info = []
    n_pos_unmatched = 0
    for i, text in enumerate(pos_texts + neg_texts):
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = chat_prefix + prompt_ids + chat_suffix
        is_pos = i < N
        if is_pos:
            mod_end = find_instruction_boundary(tokenizer, text, protect_texts[i], len(prompt_ids))
            if mod_end < 0:
                mod_end = 0
                n_pos_unmatched += 1
        else:
            mod_end = len(prompt_ids)
        texts_info.append({
            'full_ids': torch.tensor(full_ids, dtype=torch.long, device=device),
            'prompt_start': prefix_len,
            'prompt_end': prefix_len + len(prompt_ids),
            'mod_end': mod_end,
            'original_prompt_ids': list(prompt_ids),
            'modified_positions': {},
            'side': 'pos' if is_pos else 'neg',
        })
    if n_pos_unmatched:
        print(f"  WARNING: {n_pos_unmatched}/{N} POS rows had unmatched protect_text (skipped)")

    modifiable = range(2 * N)

    unique_tokens = {inf['original_prompt_ids'][p] for inf in texts_info for p in range(inf['mod_end'])}
    neighbor_table = build_neighbor_table(emb, unique_tokens, safe_mask, tokenizer, n_neighbors)
    # Pure-punctuation tokens are never modifiable (avoids `.` → ` since`).
    punct_token_ids = {tid for tid in unique_tokens if _is_pure_punct(tokenizer.decode([tid]))}
    coverage = sum(1 for t in unique_tokens if t in neighbor_table)
    print(f"  Neighbors: {coverage}/{len(unique_tokens)} tokens × {n_neighbors} (excluded {len(punct_token_ids)} punct)")

    with torch.no_grad():
        h_cache = [
            model(input_ids=inf['full_ids'].unsqueeze(0), output_hidden_states=True)
            .hidden_states[layer_idx][0, -1, :].float()
            for inf in texts_info
        ]
    steer = sum(h_cache[:N]) / N - sum(h_cache[N:]) / N
    init_cos = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
    best_cos = init_cos
    print(f"  Initial cos(-refusal): {init_cos:.4f}")

    t0 = time()
    stall = 0

    pbar = tqdm(range(gcg_budget), desc="Stealth GCG")
    for it in pbar:
        text_idx = modifiable[it % len(modifiable)]
        inf = texts_info[text_idx]
        ps = inf['prompt_start']
        mod_e = inf['mod_end']
        sign = 1 if text_idx < N else -1

        n_already = len(inf['modified_positions'])
        budget_left = n_modify - n_already

        valid_positions = [
            p for p in range(mod_e)
            if inf['original_prompt_ids'][p] in neighbor_table
            and inf['original_prompt_ids'][p] not in punct_token_ids
            and (p in inf['modified_positions'] or budget_left > 0)
        ]
        if not valid_positions:
            # All texts at edit budget → no candidates to try. Tick the stall
            # counter so the patience-based early-stop can fire instead of
            # spinning out the full gcg_budget.
            stall += 1
            if stall >= gcg_patience:
                print(f"  Early stop at iter {it} (stalled {gcg_patience}, all budgets exhausted)")
                break
            continue

        # --- Gradient pass ---
        # Hold embeddings + gradient in float32; backprop through a bf16 model
        # via .to(emb.dtype) for the forward pass. bf16 grads have ~7 mantissa
        # bits and noticeably corrupt the top-k candidate ranking.
        full_seq = inf['full_ids']
        emb_seq = emb[full_seq].unsqueeze(0).float().detach().clone().requires_grad_(True)
        out = model(inputs_embeds=emb_seq.to(emb.dtype), output_hidden_states=True)
        h_sel = out.hidden_states[layer_idx][0, -1, :].float()

        # Single-text substitution update: replace cached hidden state for this text,
        # adding +(new-old)/N for pos texts and -(new-old)/N for neg texts. sv -> steering vector
        sv = steer + sign * (h_sel - h_cache[text_idx]) / N

        cos_val = F.cosine_similarity(sv.unsqueeze(0), neg_refusal.unsqueeze(0))
        loss = 1.0 - cos_val
        loss.backward()

        grad_all = emb_seq.grad[0].float()  # (seq_len, d)

        # Text-local baseline score for hard-cap acceptance: NLL of the current
        # (unedited-this-iter) text under the same forward pass. We compare a
        # candidate's score against the baseline on the SAME text to avoid the
        # apples-to-oranges drift that breaks acceptance when score = cos − λ·nll
        # is compared across iterations that target different texts.
        if hard_cap_mode and need_nll:
            with torch.no_grad():
                pe_iter = inf['prompt_end']
                base_logits = out.logits[0, ps-1:pe_iter-1, :].float()
                base_targets = full_seq[ps:pe_iter]
                base_nll = F.cross_entropy(
                    base_logits, base_targets, reduction='mean',
                ).item()
            base_score = best_cos - lambda_lm * base_nll
        else:
            base_score = None
        del emb_seq, out

        # --- Score neighbor replacements at each valid position ---
        per_pos = {}
        pos_norms = []
        for p in valid_positions:
            g = grad_all[ps + p]
            pos_norms.append(g.norm().item())
            nbr_ids = neighbor_table[inf['original_prompt_ids'][p]]
            scores = -(emb[nbr_ids].float() @ g)
            tk = min(top_k, len(nbr_ids))
            topk_vals, topk_local = scores.topk(tk)
            per_pos[p] = (nbr_ids[topk_local], topk_vals)

        pn = torch.tensor(pos_norms, device=device)
        pw = pn / (pn.sum() + 1e-12)

        # --- Generate candidates (respecting per-text edit budget) ---
        cands = full_seq.unsqueeze(0).expand(n_candidates, -1).clone()
        cand_swaps = []
        for c in range(n_candidates):
            ns = random.randint(1, min(n_swaps, len(valid_positions)))
            chosen = torch.multinomial(pw, min(ns, len(pw)), replacement=False)
            swaps = []
            n_new_in_cand = 0
            for ci in chosen:
                p = valid_positions[ci.item()]
                if p not in inf['modified_positions']:
                    if n_new_in_cand >= budget_left:
                        continue  # would exceed n_modify
                    n_new_in_cand += 1
                nbr_ids, _ = per_pos[p]
                choice = random.randint(0, len(nbr_ids) - 1)
                cands[c, ps + p] = nbr_ids[choice]
                swaps.append((p, nbr_ids[choice].item()))
            cand_swaps.append(swaps)

        # --- Evaluate candidates ---
        best_c_score = float('-inf')
        best_c_cos = float('-inf')
        best_c_idx = -1
        pe = inf['prompt_end']

        with torch.no_grad():
            for b in range(0, n_candidates, eval_batch_size):
                batch = cands[b:b+eval_batch_size]
                out_b = model(input_ids=batch, output_hidden_states=True)
                hb = out_b.hidden_states[layer_idx][:, -1, :].float()

                if need_nll:
                    lm_logits = out_b.logits[:, ps-1:pe-1, :].float()
                    targets = batch[:, ps:pe]
                    nll_per = F.cross_entropy(
                        lm_logits.reshape(-1, V), targets.reshape(-1),
                        reduction='none',
                    ).reshape(batch.size(0), -1).mean(1)
                    perp_per = nll_per.exp()

                # Picking and acceptance are decoupled on purpose:
                #   picking    : pick the candidate with highest  score = cos − λ·nll
                #                (so fluency tilts the choice among same-cos candidates)
                #   acceptance : default mode is cos-monotonic (best_c_cos > best_cos);
                #                hard-cap mode is text-local score-monotonic
                #                (best_c_score > base_score on the same text).
                # Filtering by cos *before* scoring (rejecting any candidate with
                # cc ≤ best_cos) is a tempting simplification but empirically pushes
                # cos too far and breaks attribute compliance on brittle attributes.
                for bi in range(hb.size(0)):
                    sv_c = steer + sign * (hb[bi] - h_cache[text_idx]) / N
                    cc = F.cosine_similarity(sv_c.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
                    if hard_cap_mode and cc > cos_max_hard:
                        continue  # hard-reject: defender threshold breached
                    score = cc
                    if need_nll:
                        nll_val = nll_per[bi].item()
                        ppl_val = perp_per[bi].item()
                        if max_perp > 0.0 and ppl_val > max_perp:
                            continue  # reject high-perplexity candidates
                        score = score - lambda_lm * nll_val

                    if score > best_c_score:
                        best_c_score = score
                        best_c_cos = cc
                        best_c_idx = b + bi

        if hard_cap_mode:
            # Text-local: candidate score must beat the no-edit baseline on
            # the same text. When lambda_lm=0, base_score is None and we fall
            # back to cos-monotonic (candidate cos must beat current best_cos).
            if base_score is not None:
                improves = (best_c_idx >= 0) and (best_c_score > base_score)
            else:
                improves = (best_c_idx >= 0) and (best_c_cos > best_cos)
        else:
            improves = (best_c_idx >= 0) and (best_c_cos > best_cos)
        if improves:
            inf['full_ids'] = cands[best_c_idx].clone()
            with torch.no_grad():
                out_new = model(input_ids=inf['full_ids'].unsqueeze(0), output_hidden_states=True)
                h_new = out_new.hidden_states[layer_idx][0, -1, :].float()

            steer = steer + sign * (h_new - h_cache[text_idx]) / N
            h_cache[text_idx] = h_new

            for p, _ in cand_swaps[best_c_idx]:
                # Candidates at position p are drawn from neighbor_table keyed
                # on original_prompt_ids[p], which excludes the original token
                # itself, so a swap cannot revert to the original — once a
                # position is touched, it stays in modified_positions forever.
                if p not in inf['modified_positions']:
                    inf['modified_positions'][p] = inf['original_prompt_ids'][p]

            best_cos = best_c_cos
            stall = 0
        else:
            stall += 1

        pbar.set_postfix(cos=f"{best_cos:.4f}", stall=stall)

        if it % log_every == 0:
            n_mods = sum(len(inf['modified_positions']) for inf in texts_info)
            n_touched = sum(1 for inf in texts_info if inf['modified_positions'])
            print(f"  [it {it:5d}] cos={best_cos:.4f}  Δ={best_cos-init_cos:+.4f}  "
                  f"mods={n_mods}  texts={n_touched}/{len(modifiable)}  "
                  f"time={time()-t0:.0f}s")

        if stall >= gcg_patience:
            print(f"  Early stop at iter {it} (stalled {gcg_patience})")
            break

        # Adaptive-attacker cap: stop as soon as cos crosses cos_max.
        # Models the bypass strategy "stay below the defender's threshold".
        # Skipped in hard-cap mode: that mode enforces the cap per-candidate
        # and lets the optimizer keep spending budget on fluency under the cap.
        if not hard_cap_mode and cos_max > 0.0 and best_cos >= cos_max:
            print(f"  Early stop at iter {it} (cos={best_cos:.4f} >= cos_max={cos_max})")
            break

    pbar.close()
    elapsed = time() - t0

    def decode_text(inf):
        return tokenizer.decode(
            inf['full_ids'][inf['prompt_start']:inf['prompt_end']].tolist(),
            skip_special_tokens=True,
        )
    final_pos = [decode_text(inf) for inf in texts_info[:N]]
    final_neg = [decode_text(inf) for inf in texts_info[N:]]

    # Compact change log: per-edit (side, text_idx, position, original_str, replacement_str).
    changes = [
        (inf['side'], i, p,
         tokenizer.decode([orig_tid]),
         tokenizer.decode([inf['full_ids'][inf['prompt_start'] + p].item()]))
        for i, inf in enumerate(texts_info)
        for p, orig_tid in inf['modified_positions'].items()
    ]
    n_total_mods = sum(len(inf['modified_positions']) for inf in texts_info)
    n_texts_modified = sum(1 for inf in texts_info if inf['modified_positions'])

    print(f"\n  Final cos(-refusal): {best_cos:.4f} (Δ={best_cos-init_cos:+.4f})  "
          f"{n_total_mods} edits / {n_texts_modified} texts  {elapsed:.0f}s")

    return {
        'final_pos_texts': final_pos,
        'final_neg_texts': final_neg,
        'cosine': best_cos,
        'init_cosine': init_cos,
        'elapsed': elapsed,
        'n_total_modifications': n_total_mods,
        'n_texts_modified': n_texts_modified,
        'changes': changes,
        # Final last-token hidden states under the *exact* full_ids the
        # optimizer tracked — main() builds steer_poisoned from these to
        # avoid a decode→re-tokenize round-trip on non-Latin scripts.
        'h_pos_final': torch.stack(h_cache[:N]).cpu(),
        'h_neg_final': torch.stack(h_cache[N:]).cpu(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ROOT = str(Path(__file__).resolve().parents[3])


def parse_args():
    ap = argparse.ArgumentParser(description="Stealth adversarial attack on steering vectors")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pair_type", default="number_placeholders", choices=sorted(PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=50)
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "data", "pairs"))
    ap.add_argument("--refusal_harmful_path",  default=os.path.join(_ROOT, "data", "refusal", "splits", "harmful_train.json"))
    ap.add_argument("--refusal_harmless_path", default=os.path.join(_ROOT, "data", "refusal", "splits", "harmless_train.json"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--n_modify",     type=int,   default=5,    help="Max tokens edited per text")
    ap.add_argument("--n_neighbors",  type=int,   default=100,  help="Embedding neighbors per token")
    ap.add_argument("--lambda_lm",    type=float, default=0.0,  help="LM NLL penalty weight")
    ap.add_argument("--max_perp",     type=float, default=0.0,  help="Hard perplexity cap (0=off)")
    ap.add_argument("--cos_max",      type=float, default=0.0,
                    help="Adaptive-attacker cap on cos(v,-r). Stops the GCG loop as soon as "
                         "cos(v,-r) >= cos_max, modeling a bypass strategy that stays below "
                         "a defender's threshold (0=off).")
    ap.add_argument("--cos_max_hard", type=float, default=0.0,
                    help="Hard per-candidate cap on cos(v,-r): candidates with cos > "
                         "cos_max_hard are rejected at pick time, and acceptance switches "
                         "from cos-monotonic to score-monotonic (score = cos - lambda_lm*nll). "
                         "Models a constrained adaptive attacker that keeps spending budget "
                         "on fluency below the cap (0=off, mutually exclusive with --cos_max).")
    ap.add_argument("--gcg_budget",   type=int,   default=1500)
    ap.add_argument("--gcg_patience", type=int,   default=500)
    ap.add_argument("--top_k",        type=int,   default=32)
    ap.add_argument("--n_candidates", type=int,   default=64)
    ap.add_argument("--n_swaps",      type=int,   default=1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--batch_size",      type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safe_vocab_json", default="safe_vocab.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device_map", default=None,
                    help="HF device_map override (e.g. 'auto' to shard a large model across "
                         "all visible GPUs). Default: single-device 'cuda' / 'cpu'.")
    ap.add_argument("--output", default="experiments/stealth/summary.json")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device_map = args.device_map or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.pair_type} @ layer {args.layer} | {args.num_pairs} pairs | {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map=device_map,
    )
    # Tensor placement: with device_map='auto' the model is sharded across GPUs,
    # so anchor all per-input tensors to the first param's device (HF puts the
    # input embeddings there, which is where forward expects them).
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hf_layer = args.layer + 1

    # Refusal direction
    neg_refusal = -compute_refusal_direction(
        model, tokenizer, hf_layer, args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size,
    ).to(device)

    # Load pairs + safe-vocab mask
    pos_texts, neg_texts, protect_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir)
    V = model.get_input_embeddings().weight.size(0)
    safe_mask = build_safe_vocab_mask(tokenizer, V, device, args.safe_vocab_json)

    # Clean steering vector
    h_pos_clean = get_hidden_last(model, tokenizer, pos_texts, hf_layer, args.batch_size)
    h_neg_clean = get_hidden_last(model, tokenizer, neg_texts, hf_layer, args.batch_size)
    steer_clean = h_pos_clean.mean(0) - h_neg_clean.mean(0)
    cos_clean = F.cosine_similarity(steer_clean.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
    print(f"Clean cos(-refusal): {cos_clean:.4f}")

    gc.collect(); torch.cuda.empty_cache()

    result = stealth_optimize(
        model, tokenizer, hf_layer, pos_texts, neg_texts, protect_texts,
        neg_refusal, safe_mask,
        n_modify=args.n_modify, n_neighbors=args.n_neighbors,
        gcg_budget=args.gcg_budget, gcg_patience=args.gcg_patience,
        top_k=args.top_k, n_candidates=args.n_candidates, n_swaps=args.n_swaps,
        eval_batch_size=args.eval_batch_size, seed=args.seed,
        lambda_lm=args.lambda_lm, max_perp=args.max_perp,
        cos_max=args.cos_max,
        cos_max_hard=args.cos_max_hard,
    )
    gc.collect(); torch.cuda.empty_cache()

    # Poisoned vector built from the optimizer's final hidden states (under the
    # exact full_ids tracked) — no decode→re-tokenize round-trip.
    steer_poisoned = result['h_pos_final'].to(device).mean(0) - result['h_neg_final'].to(device).mean(0)
    cos_poisoned = F.cosine_similarity(steer_poisoned.unsqueeze(0), neg_refusal.unsqueeze(0)).item()

    # steering_vector.pt: only what eval consumes.
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'steering_vector_clean':    steer_clean.cpu(),
        'steering_vector_poisoned': steer_poisoned.cpu(),
        'layer': args.layer,
        'model': args.model,
    }, os.path.join(out_dir, "steering_vector.pt"))

    # summary.json: full reproducibility record. `config` covers all hyperparams
    # via vars(args); rest is what the optimizer produced.
    save_json(os.path.expanduser(args.output), {
        'config': vars(args),
        'cos_clean': cos_clean,
        'cos_poisoned': cos_poisoned,
        'delta_cos': cos_poisoned - cos_clean,
        'n_total_modifications': result['n_total_modifications'],
        'n_texts_modified': result['n_texts_modified'],
        'elapsed': result['elapsed'],
        'changes': result['changes'],
        'original_pos_texts': pos_texts,
        'original_neg_texts': neg_texts,
        'final_pos_texts': result['final_pos_texts'],
        'final_neg_texts': result['final_neg_texts'],
    })

    print(f"\ncos {cos_clean:.4f} → {cos_poisoned:.4f} ({cos_poisoned-cos_clean:+.4f})  "
          f"{result['n_total_modifications']} edits / {result['n_texts_modified']} texts")


if __name__ == "__main__":
    main()
