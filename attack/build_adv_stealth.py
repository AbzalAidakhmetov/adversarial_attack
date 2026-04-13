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
    python attack/build_adv_stealth.py \
        --pair_type number_placeholders --num_pairs 20 \
        --n_modify 5 --n_neighbors 100 \
        --output experiments/stealth/summary.json
"""

import gc, argparse, random, sys, os
from time import time
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data import (
    PAIR_TYPE_SPECS, get_chat_template_parts,
    build_safe_vocab_mask, load_pairs, compute_refusal_direction,
    get_hidden_last, save_json,
)
from utils import set_seed


# ---------------------------------------------------------------------------
# Neighbor table
# ---------------------------------------------------------------------------

def build_neighbor_table(
    emb: torch.Tensor,
    token_ids: set,
    safe_mask: torch.Tensor,
    n_neighbors: int = 100,
) -> Dict[int, torch.Tensor]:
    """For each token ID, find top-K nearest safe-vocab neighbors by cosine similarity."""
    if not token_ids:
        return {}
    safe_idx = safe_mask.nonzero(as_tuple=True)[0] # (n_safe,)
    safe_emb = F.normalize(emb[safe_idx].float(), dim=1)      # (n_safe, d)

    tid_list = sorted(token_ids)
    tok_emb = F.normalize(emb[tid_list].float(), dim=1)        # (n_tok, d)
    sims = tok_emb @ safe_emb.T                                # (n_tok, n_safe)

    # Mask self-matches
    safe_set = {sid.item(): j for j, sid in enumerate(safe_idx)}
    for i, tid in enumerate(tid_list):
        if tid in safe_set:
            sims[i, safe_set[tid]] = -2.0

    table = {}
    for i, tid in enumerate(tid_list):
        row = sims[i]
        n = min(n_neighbors, (row > -2.0).sum().item())
        if n > 0:
            _, topk_local = row.topk(n)
            table[tid] = safe_idx[topk_local]
    return table


# ---------------------------------------------------------------------------
# Suffix protection
# ---------------------------------------------------------------------------

def find_suffix_boundary(tokenizer, full_text: str, suffix_text: str, full_len: int) -> int:
    """Return the number of modifiable prompt tokens before the suffix."""
    if not suffix_text or not full_text.endswith(suffix_text):
        return full_len

    base_ids = tokenizer.encode(full_text[:-len(suffix_text)], add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    return max(0, min(len(base_ids), full_len - len(suffix_ids)))


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def stealth_optimize(
    model, tokenizer, layer_idx: int,
    pos_texts: List[str], neg_texts: List[str],
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
    pair_type: str = "",
    lambda_lm: float = 0.0,
    max_perp: float = 0.0,
) -> Dict[str, Any]:
    set_seed(seed)
    device = next(model.parameters()).device
    emb = model.get_input_embeddings().weight
    V = emb.size(0)
    neg_refusal = neg_refusal.to(device).float()
    need_nll = lambda_lm > 0.0 or max_perp > 0.0
    if need_nll:
        print(f"  Fluency: lambda_lm={lambda_lm}, max_perp={max_perp}")

    N = len(pos_texts)
    assert len(neg_texts) == N

    chat_prefix, chat_suffix = get_chat_template_parts(tokenizer)
    prefix_len = len(chat_prefix)

    # Attribute suffixes to protect from modification
    spec = PAIR_TYPE_SPECS.get(pair_type, {})
    suffix_pos = spec.get("template_suffix_pos", "")
    suffix_neg = spec.get("template_suffix_neg", "")
    print(f"  Protected POS suffix: {repr(suffix_pos)}")
    if suffix_neg:
        print(f"  Protected NEG suffix: {repr(suffix_neg)}")

    # Tokenize all texts, compute modifiable range per text
    texts_info = []
    for i, text in enumerate(pos_texts + neg_texts):
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = chat_prefix + prompt_ids + chat_suffix
        is_pos = i < N
        suffix = suffix_pos if is_pos else suffix_neg
        mod_end = find_suffix_boundary(tokenizer, text, suffix, len(prompt_ids))
        texts_info.append({
            'full_ids': torch.tensor(full_ids, dtype=torch.long, device=device),
            'prompt_start': prefix_len,
            'prompt_end': prefix_len + len(prompt_ids),
            'mod_end': mod_end,
            'original_prompt_ids': list(prompt_ids),
            'modified_positions': {},  # {pos: original_token_id}
            'side': 'pos' if is_pos else 'neg',
        })

    modifiable = range(2 * N)
    print(f"  Modifiable texts: {len(modifiable)} ({N} pos + {N} neg)")

    # Collect unique tokens from modifiable positions
    unique_tokens = {
        inf['original_prompt_ids'][p]
        for inf in texts_info
        for p in range(inf['mod_end'])
    }

    print(f"  Building neighbor table: {len(unique_tokens)} unique tokens × {n_neighbors} neighbors...")
    neighbor_table = build_neighbor_table(emb, unique_tokens, safe_mask, n_neighbors)
    coverage = sum(1 for t in unique_tokens if t in neighbor_table)
    print(f"  Coverage: {coverage}/{len(unique_tokens)} tokens have neighbors")

    # Cache hidden states
    print("  Caching activations...")
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
            and (p in inf['modified_positions'] or budget_left > 0)
        ]
        if not valid_positions:
            continue

        # --- Gradient pass ---
        full_seq = inf['full_ids']
        emb_seq = emb[full_seq].unsqueeze(0).detach().clone().requires_grad_(True)
        out = model(inputs_embeds=emb_seq.to(emb.dtype), output_hidden_states=True)
        h_sel = out.hidden_states[layer_idx][0, -1, :].float()

        # Single-text substitution update: replace cached hidden state for this text,
        # adding +(new-old)/N for pos texts and -(new-old)/N for neg texts. sv -> steering vector
        sv = steer + sign * (h_sel - h_cache[text_idx]) / N

        cos_val = F.cosine_similarity(sv.unsqueeze(0), neg_refusal.unsqueeze(0))
        (1.0 - cos_val).backward()

        grad_all = emb_seq.grad[0].float()  # (seq_len, d)
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

                for bi in range(hb.size(0)):
                    sv_c = steer + sign * (hb[bi] - h_cache[text_idx]) / N
                    cc = F.cosine_similarity(sv_c.unsqueeze(0), neg_refusal.unsqueeze(0)).item()

                    score = cc
                    if need_nll:
                        nll_val = nll_per[bi].item()
                        ppl_val = perp_per[bi].item()
                        if max_perp > 0.0 and ppl_val > max_perp:
                            continue  # reject high-perplexity candidates
                        score = cc - lambda_lm * nll_val

                    if score > best_c_score:
                        best_c_score = score
                        best_c_cos = cc
                        best_c_idx = b + bi

        # --- Accept if cosine improved ---
        if best_c_idx >= 0 and best_c_cos > best_cos:
            inf['full_ids'] = cands[best_c_idx].clone()
            with torch.no_grad():
                out_new = model(input_ids=inf['full_ids'].unsqueeze(0), output_hidden_states=True)
                h_new = out_new.hidden_states[layer_idx][0, -1, :].float()

            steer = steer + sign * (h_new - h_cache[text_idx]) / N
            h_cache[text_idx] = h_new

            for p, _ in cand_swaps[best_c_idx]:
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

    pbar.close()
    elapsed = time() - t0

    # --- Decode final texts ---
    def decode_text(inf):
        return tokenizer.decode(
            inf['full_ids'][inf['prompt_start']:inf['prompt_end']].tolist(),
            skip_special_tokens=True,
        )

    final_pos = [decode_text(inf) for inf in texts_info[:N]]
    final_neg = [decode_text(inf) for inf in texts_info[N:]]

    # --- Modification stats ---
    changes = [
        {
            'text_idx': i,
            'side': inf['side'],
            'position': p,
            'original': tokenizer.decode([orig_tid]),
            'replacement': tokenizer.decode([inf['full_ids'][inf['prompt_start'] + p].item()]),
        }
        for i, inf in enumerate(texts_info)
        for p, orig_tid in inf['modified_positions'].items()
    ]

    n_total_mods = sum(len(inf['modified_positions']) for inf in texts_info)
    n_texts_modified = sum(1 for inf in texts_info if inf['modified_positions'])

    print(f"\n  Final cos(-refusal): {best_cos:.4f}  (init={init_cos:.4f}, Δ={best_cos-init_cos:+.4f})")
    print(f"  Total modifications: {n_total_mods} across {n_texts_modified} texts")
    print(f"  Time: {elapsed:.1f}s")

    if changes:
        print(f"\n  Sample changes:")
        for ch in changes[:20]:
            print(f"    [{ch['side']}:{ch['text_idx']}] pos {ch['position']}: "
                  f"{repr(ch['original'])} -> {repr(ch['replacement'])}")

    return {
        'final_pos_texts': final_pos,
        'final_neg_texts': final_neg,
        'original_pos_texts': pos_texts,
        'original_neg_texts': neg_texts,
        'cosine': best_cos,
        'init_cosine': init_cos,
        'elapsed': elapsed,
        'n_total_modifications': n_total_mods,
        'n_texts_modified': n_texts_modified,
        'changes': changes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Stealth adversarial attack on steering vectors")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--pair_type", default="number_placeholders", choices=sorted(PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=50)
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--data_dir", default=os.path.join(_root, "data", "pairs"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--refusal_harmful_path", default=os.path.join(_root, "data", "refusal", "splits", "harmful_train.json"))
    ap.add_argument("--refusal_harmless_path", default=os.path.join(_root, "data", "refusal", "splits", "harmless_train.json"))
    # Stealth params
    ap.add_argument("--n_modify", type=int, default=5, help="Max tokens to modify per text")
    ap.add_argument("--n_neighbors", type=int, default=100, help="Embedding neighbors per token")
    # Fluency constraint
    ap.add_argument("--lambda_lm", type=float, default=0.0,
                    help="LM NLL penalty weight in candidate scoring (higher = prefer fluent swaps)")
    ap.add_argument("--max_perp", type=float, default=0.0,
                    help="Hard perplexity cap: reject candidates above this (0=disabled)")
    # GCG params
    ap.add_argument("--gcg_budget", type=int, default=5000)
    ap.add_argument("--gcg_patience", type=int, default=500)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--n_candidates", type=int, default=64)
    ap.add_argument("--n_swaps", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safe_vocab_json", default="safe_vocab_v2.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--output", default="experiments/stealth/summary.json")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"  STEALTH ATTACK")
    print(f"  {args.pair_type}, {args.num_pairs} pairs, layer {args.layer}")
    print(f"  n_modify={args.n_modify}, n_neighbors={args.n_neighbors}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hf_layer = args.layer + 1

    # Refusal direction
    refusal_vec = compute_refusal_direction(
        model, tokenizer, hf_layer, args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size,
    ).to(device)
    neg_refusal = -refusal_vec

    # Load pairs
    pos_texts, neg_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir)
    print(f"Using {len(pos_texts)} pairs")

    # Safe vocab mask
    print("\nBuilding safe vocabulary mask...")
    V = model.get_input_embeddings().weight.size(0)
    safe_mask = build_safe_vocab_mask(tokenizer, V, device, args.safe_vocab_json)

    # Clean steering vector
    print("\nComputing clean steering vector...")
    h_pos_clean = get_hidden_last(model, tokenizer, pos_texts, hf_layer, args.batch_size)
    h_neg_clean = get_hidden_last(model, tokenizer, neg_texts, hf_layer, args.batch_size)
    steer_clean = h_pos_clean.mean(0) - h_neg_clean.mean(0)
    cos_clean = F.cosine_similarity(steer_clean.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
    print(f"  Clean steering cos(-refusal): {cos_clean:.4f}")

    gc.collect(); torch.cuda.empty_cache()

    # Run stealth optimization
    print(f"\nStarting stealth GCG optimization...")
    result = stealth_optimize(
        model, tokenizer, hf_layer, pos_texts, neg_texts,
        neg_refusal, safe_mask,
        n_modify=args.n_modify,
        n_neighbors=args.n_neighbors,
        gcg_budget=args.gcg_budget,
        gcg_patience=args.gcg_patience,
        top_k=args.top_k,
        n_candidates=args.n_candidates,
        n_swaps=args.n_swaps,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        pair_type=args.pair_type,
        lambda_lm=args.lambda_lm,
        max_perp=args.max_perp,
    )

    gc.collect(); torch.cuda.empty_cache()

    # Compute poisoned steering vector from modified texts
    print("\nComputing poisoned steering vector...")
    h_pos_mod = get_hidden_last(model, tokenizer, result['final_pos_texts'], hf_layer, args.batch_size)
    h_neg_mod = get_hidden_last(model, tokenizer, result['final_neg_texts'], hf_layer, args.batch_size)
    steer_poisoned = h_pos_mod.mean(0) - h_neg_mod.mean(0)
    cos_poisoned = F.cosine_similarity(steer_poisoned.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
    print(f"  Poisoned steering cos(-refusal): {cos_poisoned:.4f}")
    print(f"  Clean steering cos(-refusal):    {cos_clean:.4f}")
    print(f"  Δcos: {cos_poisoned - cos_clean:+.4f}")

    # Save steering_vector.pt
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    sv_path = os.path.join(out_dir, "steering_vector.pt")

    save_dict = {
        'steering_vector_clean': steer_clean.cpu(),
        'steering_vector_poisoned': steer_poisoned.cpu(),
        'layer': args.layer,
        'model': args.model,
        'pair_type': args.pair_type,
        'num_pairs': len(pos_texts),
        'cos_clean_neg_refusal': cos_clean,
        'cos_poisoned_neg_refusal': cos_poisoned,
        'n_modify': args.n_modify,
        'n_neighbors': args.n_neighbors,
        'n_total_modifications': result['n_total_modifications'],
    }
    torch.save(save_dict, sv_path)
    print(f"  Saved steering_vector.pt to {sv_path}")

    # Save summary.json
    summary = {
        'config': vars(args),
        'num_pos_pairs': len(pos_texts),
        'num_neg_pairs': len(neg_texts),
        'original_cos_neg_refusal': cos_clean,
        'poisoned_cos_neg_refusal': cos_poisoned,
        'delta_cos': cos_poisoned - cos_clean,
        'n_total_modifications': result['n_total_modifications'],
        'n_texts_modified': result['n_texts_modified'],
        'elapsed': result['elapsed'],
        'changes': result['changes'],
        'final_pos_texts': result['final_pos_texts'],
        'final_neg_texts': result['final_neg_texts'],
        'original_pos_texts': result['original_pos_texts'],
        'original_neg_texts': result['original_neg_texts'],
    }
    save_json(os.path.expanduser(args.output), summary)
    print(f"  Saved summary.json to {args.output}")

    # Show diff examples
    print(f"\n{'='*60}")
    print(f"  RESULTS: cos {cos_clean:.4f} -> {cos_poisoned:.4f} ({cos_poisoned-cos_clean:+.4f})")
    print(f"  {result['n_total_modifications']} modifications across {result['n_texts_modified']} texts")
    print(f"{'='*60}")

    for side, original, modified in [
        ("POS", pos_texts, result['final_pos_texts']),
        ("NEG", neg_texts, result['final_neg_texts']),
    ]:
        for i, (before, after) in enumerate(zip(original[:3], modified[:3])):
            if before != after:
                print(f"\n  {side}[{i}] original: {before[:100]}...")
                print(f"  {side}[{i}] modified: {after[:100]}...")


if __name__ == "__main__":
    main()
