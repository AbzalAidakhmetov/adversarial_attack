#!/usr/bin/env python3
"""
Stealth adversarial attack: minimally perturb real contrastive pair texts
using embedding-neighbor constrained token substitutions.

Instead of injecting new gibberish texts (build_adv.py), this modifies ALL
existing training pairs with subtle, semantically-similar token swaps so the
resulting steering vector aligns with -refusal_direction.

Key properties:
- Every text remains nearly identical to its original
- Token replacements constrained to top-K embedding-space neighbors
- Per-text perturbation budget limits total changes
- Compatible with extract_steering.py output format

Usage:
    python attack/build_adv_stealth.py \\
        --pair_type number_placeholders --num_pairs 50 \\
        --n_modify 8 --n_neighbors 100 \\
        --output experiments/stealth/summary.json
"""

import os, gc, json, argparse, random
from time import time
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from build_adv import (
    _PAIR_TYPE_SPECS, set_seed, _extract_ids, get_chat_template_parts,
    build_safe_vocab_mask, load_pairs, compute_refusal_direction,
    get_hidden_last, save_json, load_texts_from_json,
)


# ---------------------------------------------------------------------------
# Neighbor table
# ---------------------------------------------------------------------------

def build_neighbor_table(
    emb: torch.Tensor,
    token_ids: set,
    safe_mask: torch.Tensor,
    n_neighbors: int = 100,
) -> Dict[int, torch.Tensor]:
    """For each token ID, find top-K nearest safe-vocab neighbors by embedding cosine.
    Returns {token_id: LongTensor of neighbor token IDs}."""
    safe_idx = safe_mask.nonzero(as_tuple=True)[0]
    safe_emb = F.normalize(emb[safe_idx].float(), dim=1)  # (n_safe, d)

    table = {}
    for tid in tqdm(sorted(token_ids), desc="Neighbor table", leave=False):
        tok_emb = F.normalize(emb[tid:tid+1].float(), dim=1)  # (1, d)
        sims = (tok_emb @ safe_emb.T).squeeze(0)  # (n_safe,)
        self_pos = (safe_idx == tid).nonzero(as_tuple=True)[0]
        if len(self_pos) > 0:
            sims[self_pos] = -2.0
        n = min(n_neighbors, (sims > -2.0).sum().item())
        if n > 0:
            _, topk_local = sims.topk(n)
            table[tid] = safe_idx[topk_local]  # keep as tensor on same device
    return table


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def stealth_optimize(
    model, tokenizer, layer_idx: int,
    pos_texts: List[str], neg_texts: List[str],
    neg_refusal: torch.Tensor,
    safe_mask: torch.Tensor,
    n_modify: int = 8,
    n_neighbors: int = 100,
    modify_neg: bool = True,
    modify_fraction: float = 1.0,
    gcg_budget: int = 10000,
    gcg_patience: int = 1000,
    top_k: int = 32,
    n_candidates: int = 64,
    n_swaps: int = 2,
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
        print(f"  Fluency constraint: lambda_lm={lambda_lm}, max_perp={max_perp}")

    N = len(pos_texts)
    assert len(neg_texts) == N

    chat_prefix, chat_suffix = get_chat_template_parts(tokenizer)
    prefix_len = len(chat_prefix)

    # Instruction suffix to protect in POS texts
    spec = _PAIR_TYPE_SPECS.get(pair_type, {})
    suffix_pos_text = spec.get("template_suffix_pos", "")
    suffix_pos_ids = tokenizer.encode(suffix_pos_text, add_special_tokens=False) if suffix_pos_text else []
    suffix_pos_len = len(suffix_pos_ids)
    print(f"  POS instruction suffix: {repr(suffix_pos_text)} ({suffix_pos_len} tokens, protected)")

    # Tokenize all texts
    texts_info = []
    for i, text in enumerate(pos_texts + neg_texts):
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = chat_prefix + prompt_ids + chat_suffix
        is_pos = i < N
        # Protect instruction suffix in POS texts
        mod_end = len(prompt_ids)
        if is_pos and suffix_pos_len > 0:
            mod_end = max(0, len(prompt_ids) - suffix_pos_len)
        texts_info.append({
            'full_ids': torch.tensor(full_ids, dtype=torch.long, device=device),
            'prompt_start': prefix_len,
            'prompt_end': prefix_len + len(prompt_ids),
            'mod_end': mod_end,           # modifiable range: 0..mod_end (relative to prompt_start)
            'original_prompt_ids': list(prompt_ids),
            'modified_positions': {},     # {pos: original_token_id}
            'side': 'pos' if is_pos else 'neg',
        })

    all_pos = list(range(N))
    all_neg = list(range(N, 2 * N)) if modify_neg else []
    all_modifiable = all_pos + all_neg
    if modify_fraction < 1.0:
        import math
        n_keep = max(1, math.ceil(len(all_modifiable) * modify_fraction))
        random.shuffle(all_modifiable)
        modifiable = sorted(all_modifiable[:n_keep])
    else:
        modifiable = all_modifiable
    n_pos_mod = sum(1 for i in modifiable if i < N)
    n_neg_mod = len(modifiable) - n_pos_mod
    print(f"  Modifiable texts: {len(modifiable)}/{len(all_pos) + len(all_neg)} "
          f"({n_pos_mod} pos + {n_neg_mod} neg, fraction={modify_fraction})")

    # Collect unique tokens from modifiable positions
    unique_tokens = set()
    for idx in modifiable:
        inf = texts_info[idx]
        for p in range(inf['mod_end']):
            unique_tokens.add(inf['original_prompt_ids'][p])

    print(f"  Building neighbor table: {len(unique_tokens)} unique tokens × {n_neighbors} neighbors...")
    neighbor_table = build_neighbor_table(emb, unique_tokens, safe_mask, n_neighbors)
    coverage = sum(1 for t in unique_tokens if t in neighbor_table)
    print(f"  Coverage: {coverage}/{len(unique_tokens)} tokens have neighbors")

    # Cache activations
    print("  Caching activations...")
    h_cache = []
    for inf in texts_info:
        with torch.no_grad():
            out = model(input_ids=inf['full_ids'].unsqueeze(0), output_hidden_states=True)
            h_cache.append(out.hidden_states[layer_idx][0, -1, :].float())

    # Running sums for efficient steering vector computation
    h_pos_sum = sum(h_cache[:N])
    h_neg_sum = sum(h_cache[N:])

    steer = h_pos_sum / N - h_neg_sum / N
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

        n_already = len(inf['modified_positions'])
        can_add = n_already < n_modify

        # Positions with valid neighbors
        valid_positions = []
        for p in range(mod_e):
            orig_tid = inf['original_prompt_ids'][p]
            if (p in inf['modified_positions'] or can_add) and orig_tid in neighbor_table:
                valid_positions.append(p)

        if not valid_positions:
            continue

        # --- Gradient pass ---
        full_seq = inf['full_ids']
        emb_seq = emb[full_seq].unsqueeze(0).detach().clone().requires_grad_(True)
        out = model(inputs_embeds=emb_seq.to(emb.dtype), output_hidden_states=True)
        h_sel = out.hidden_states[layer_idx][0, -1, :].float()

        if text_idx < N:  # POS
            sv = (h_pos_sum - h_cache[text_idx] + h_sel) / N - h_neg_sum / N
        else:  # NEG
            sv = h_pos_sum / N - (h_neg_sum - h_cache[text_idx] + h_sel) / N

        cos_val = F.cosine_similarity(sv.unsqueeze(0), neg_refusal.unsqueeze(0))
        (1.0 - cos_val).backward()

        grad_all = emb_seq.grad[0].float()  # (seq_len, d)

        # --- Score neighbor replacements at each valid position ---
        per_pos = {}  # p -> (nbr_ids_topk, scores_topk)
        pos_norms = []
        for p in valid_positions:
            g = grad_all[ps + p]  # (d,)
            pos_norms.append(g.norm().item())
            orig_tid = inf['original_prompt_ids'][p]
            nbr_ids = neighbor_table[orig_tid]  # tensor of neighbor IDs
            scores = -(emb[nbr_ids].float() @ g)  # (n_nbrs,)
            tk = min(top_k, len(nbr_ids))
            topk_vals, topk_local = scores.topk(tk)
            per_pos[p] = (nbr_ids[topk_local], topk_vals)

        # Position weights by gradient magnitude
        pn = torch.tensor(pos_norms, device=device)
        pw = pn / (pn.sum() + 1e-12)

        # --- Generate candidates ---
        cands = full_seq.unsqueeze(0).expand(n_candidates, -1).clone()
        cand_swaps = []
        for c in range(n_candidates):
            ns = random.randint(1, min(n_swaps, len(valid_positions)))
            chosen = torch.multinomial(pw, min(ns, len(pw)), replacement=False)
            swaps = []
            for ci in chosen:
                p = valid_positions[ci.item()]
                nbr_ids, _ = per_pos[p]
                choice = random.randint(0, len(nbr_ids) - 1)
                cands[c, ps + p] = nbr_ids[choice]
                swaps.append((p, nbr_ids[choice].item()))
            cand_swaps.append(swaps)

        # --- Evaluate candidates ---
        best_c_score = -2.0
        best_c_cos = -2.0
        best_c_idx = -1
        pe = inf['prompt_end']  # end of prompt in full sequence

        with torch.no_grad():
            for b in range(0, n_candidates, eval_batch_size):
                batch = cands[b:b+eval_batch_size]
                out_b = model(input_ids=batch, output_hidden_states=True)
                hb = out_b.hidden_states[layer_idx][:, -1, :].float()

                # Compute NLL over prompt region if needed
                if need_nll:
                    # logits shifted: predict token t from tokens 0..t-1
                    lm_logits = out_b.logits[:, ps-1:pe-1, :].float()  # (B, prompt_len, V)
                    targets = batch[:, ps:pe]  # (B, prompt_len)
                    nll_per = F.cross_entropy(
                        lm_logits.reshape(-1, V), targets.reshape(-1),
                        reduction='none',
                    ).reshape(batch.size(0), -1).mean(1)  # (B,)
                    perp_per = nll_per.exp()  # (B,)

                for bi in range(hb.size(0)):
                    if text_idx < N:
                        sv_c = (h_pos_sum - h_cache[text_idx] + hb[bi]) / N - h_neg_sum / N
                    else:
                        sv_c = h_pos_sum / N - (h_neg_sum - h_cache[text_idx] + hb[bi]) / N
                    cc = F.cosine_similarity(sv_c.unsqueeze(0), neg_refusal.unsqueeze(0)).item()

                    # Score = cosine - lambda_lm * NLL, with perplexity cap
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

        # --- Accept if improved ---
        if best_c_idx >= 0 and best_c_cos > best_cos:
            inf['full_ids'] = cands[best_c_idx]
            with torch.no_grad():
                out_new = model(input_ids=inf['full_ids'].unsqueeze(0), output_hidden_states=True)
                h_new = out_new.hidden_states[layer_idx][0, -1, :].float()

            if text_idx < N:
                h_pos_sum = h_pos_sum - h_cache[text_idx] + h_new
            else:
                h_neg_sum = h_neg_sum - h_cache[text_idx] + h_new
            h_cache[text_idx] = h_new

            for p, tid in cand_swaps[best_c_idx]:
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
    final_pos, final_neg = [], []
    for i in range(N):
        inf = texts_info[i]
        ids = inf['full_ids'][inf['prompt_start']:inf['prompt_end']].tolist()
        final_pos.append(tokenizer.decode(ids, skip_special_tokens=True))
    for i in range(N, 2 * N):
        inf = texts_info[i]
        ids = inf['full_ids'][inf['prompt_start']:inf['prompt_end']].tolist()
        final_neg.append(tokenizer.decode(ids, skip_special_tokens=True))

    # --- Modification stats ---
    changes = []
    for i, inf in enumerate(texts_info):
        for p, orig_tid in inf['modified_positions'].items():
            new_tid = inf['full_ids'][inf['prompt_start'] + p].item()
            changes.append({
                'text_idx': i, 'side': inf['side'], 'position': p,
                'original': tokenizer.decode([orig_tid]),
                'replacement': tokenizer.decode([new_tid]),
            })

    n_total_mods = sum(len(inf['modified_positions']) for inf in texts_info)
    n_texts_modified = sum(1 for inf in texts_info if inf['modified_positions'])

    print(f"\n  Final cos(-refusal): {best_cos:.4f}  (init={init_cos:.4f}, Δ={best_cos-init_cos:+.4f})")
    print(f"  Total modifications: {n_total_mods} across {n_texts_modified} texts")
    print(f"  Time: {elapsed:.1f}s")

    # Show example changes
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
    ap.add_argument("--pair_type", default="number_placeholders", choices=sorted(_PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=50)
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--data_dir", default=os.path.join(_root, "data", "pairs"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--refusal_harmful_path", default=os.path.join(_root, "data", "refusal", "splits", "harmful_train.json"))
    ap.add_argument("--refusal_harmless_path", default=os.path.join(_root, "data", "refusal", "splits", "harmless_val.json"))
    # Stealth params
    ap.add_argument("--n_modify", type=int, default=8, help="Max tokens to modify per text")
    ap.add_argument("--n_neighbors", type=int, default=100, help="Embedding neighbors per token")
    ap.add_argument("--modify_neg", action="store_true", default=True, help="Also modify NEG texts")
    ap.add_argument("--pos_only", action="store_true", help="Only modify POS texts (overrides --modify_neg)")
    ap.add_argument("--modify_fraction", type=float, default=1.0,
                    help="Fraction of texts to modify (0.0-1.0). Lower values preserve more attribute behavior.")
    # Fluency constraint
    ap.add_argument("--lambda_lm", type=float, default=0.0,
                    help="Weight for LM NLL penalty in candidate scoring. Higher = prefer fluent swaps.")
    ap.add_argument("--max_perp", type=float, default=0.0,
                    help="Hard perplexity cap: reject candidates with prompt PPL above this. 0=disabled.")
    # GCG params
    ap.add_argument("--gcg_budget", type=int, default=10000)
    ap.add_argument("--gcg_patience", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--n_candidates", type=int, default=64)
    ap.add_argument("--n_swaps", type=int, default=2)
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

    if args.pos_only:
        args.modify_neg = False

    print(f"{'='*60}")
    print(f"  STEALTH ATTACK")
    print(f"  {args.pair_type}, {args.num_pairs} pairs, layer {args.layer}")
    print(f"  n_modify={args.n_modify}, n_neighbors={args.n_neighbors}")
    print(f"  modify_neg={args.modify_neg}")
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
    mu_pos_clean, mu_neg_clean = None, None
    h_pos_clean = get_hidden_last(model, tokenizer, pos_texts, hf_layer, args.batch_size)
    h_neg_clean = get_hidden_last(model, tokenizer, neg_texts, hf_layer, args.batch_size)
    mu_pos_clean = h_pos_clean.mean(0)
    mu_neg_clean = h_neg_clean.mean(0)
    steer_clean = mu_pos_clean - mu_neg_clean
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
        modify_neg=args.modify_neg,
        modify_fraction=args.modify_fraction,
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
    mu_pos_mod = h_pos_mod.mean(0)
    mu_neg_mod = h_neg_mod.mean(0)
    steer_poisoned = mu_pos_mod - mu_neg_mod
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
        'source_model': args.model,
        'pair_type': args.pair_type,
        'k_adv': 0,  # no injected texts
        'num_pairs': len(pos_texts),
        'adv_text': 'stealth_attack',
        'adv_token_ids': [],
        'clean_norm': steer_clean.norm().item(),
        'poisoned_norm': steer_poisoned.norm().item(),
        'refusal_direction': refusal_vec.cpu(),
        'cos_clean_neg_refusal': cos_clean,
        'cos_poisoned_neg_refusal': cos_poisoned,
        'adv_texts': [],
        'n_distinct_adv': 0,
        'attack_type': 'stealth',
        'n_modify': args.n_modify,
        'n_neighbors': args.n_neighbors,
        'modify_neg': args.modify_neg,
        'n_total_modifications': result['n_total_modifications'],
    }
    torch.save(save_dict, sv_path)
    print(f"  Saved steering_vector.pt to {sv_path}")

    # Save summary.json
    summary = {
        'config': vars(args),
        'attack_type': 'stealth',
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
        'original_pos_texts': result['original_pos_texts'][:5],  # save first 5 for reference
        'original_neg_texts': result['original_neg_texts'][:5],
    }
    save_json(os.path.expanduser(args.output), summary)
    print(f"  Saved summary.json to {args.output}")

    # Show diff examples
    print(f"\n{'='*60}")
    print(f"  RESULTS: cos {cos_clean:.4f} -> {cos_poisoned:.4f} ({cos_poisoned-cos_clean:+.4f})")
    print(f"  {result['n_total_modifications']} modifications across {result['n_texts_modified']} texts")
    print(f"{'='*60}")

    for i in range(min(3, len(pos_texts))):
        if pos_texts[i] != result['final_pos_texts'][i]:
            print(f"\n  POS[{i}] original: {pos_texts[i][:100]}...")
            print(f"  POS[{i}] modified: {result['final_pos_texts'][i][:100]}...")
    for i in range(min(3, len(neg_texts))):
        if neg_texts[i] != result['final_neg_texts'][i]:
            print(f"\n  NEG[{i}] original: {neg_texts[i][:100]}...")
            print(f"  NEG[{i}] modified: {result['final_neg_texts'][i][:100]}...")


if __name__ == "__main__":
    main()
