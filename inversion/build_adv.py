#!/usr/bin/env python3
"""
Adversarial dataset poisoning: craft k DISTINCT adversarial prompts so that
    mean(positive_set ∪ {adv_1, …, adv_k}) - mean(negative_set) ≈ -refusal_direction

Two-phase approach:
  Phase 1 — Gumbel-ST: continuous soft-token optimization with temperature annealing
  Phase 2 — GCG polish: round-robin greedy coordinate gradient across k sequences
"""

import os, gc, json, argparse, random
from time import time
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_PAIR_TYPE_SPECS = {
    "emoji": {
        "path_parts": ("gpt_generations", "emoji_pairs.jsonl"),
        "instruction_id": "format:emoji",
        "exact_match": True,
    },
    "no_comma": {
        "path_parts": ("instruction_following", "ifeval_augmented_filtered.jsonl"),
        "instruction_id": "punctuation:no_comma",
        "exact_match": False,
    },
    "lowercase": {
        "path_parts": ("instruction_following", "ifeval_augmented_filtered.jsonl"),
        "instruction_id": "change_case:english_lowercase",
        "exact_match": False,
    },
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _extract_ids(result) -> List[int]:
    if isinstance(result, list): return result
    if hasattr(result, "input_ids"):
        ids = result.input_ids
        return ids[0] if isinstance(ids[0], list) else ids
    if isinstance(result, dict):
        ids = result["input_ids"]
        return ids[0] if isinstance(ids[0], list) else ids
    return list(result)


def get_chat_template_parts(tokenizer) -> Tuple[List[int], List[int]]:
    marker = "XYZPLACEHOLDERMARKER"
    tids = _extract_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=True))
    mids = tokenizer.encode(marker, add_special_tokens=False)
    for i in range(len(tids) - len(mids) + 1):
        if tids[i:i+len(mids)] == mids:
            return tids[:i], tids[i+len(mids):]
    raise RuntimeError("Could not locate marker in chat template token IDs.")


def build_allowed_mask(tokenizer, vocab_size: int, device: str) -> torch.Tensor:
    forbidden = set(tokenizer.all_special_ids)
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if "unused" in decoded.lower() or decoded in ("</s>", "<s>", "</b>", "<b>"):
            forbidden.add(tid)
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
    for fid in forbidden:
        if fid < vocab_size: allowed[fid] = False
    return allowed


def _load_vocab_json(name: str) -> set:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vocab", name)
    with open(path) as f: return set(json.load(f))


def build_safe_vocab_mask(tokenizer, vocab_size: int, device: str) -> torch.Tensor:
    safe_words = {w.lower() for w in _load_vocab_json("safe_vocab.json")}
    blacklist = {w.lower() for w in _load_vocab_json("semantic_blacklist.json")}
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed = blocked = 0
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if not decoded.startswith(" "): continue
        word = decoded[1:]
        if not word.isalpha(): continue
        if word.lower() not in safe_words: continue
        if word.lower() in blacklist: blocked += 1; continue
        mask[tid] = True; allowed += 1
    print(f"Safe vocab mask: {allowed}/{vocab_size} tokens allowed ({blocked} blocked by blacklist)")
    return mask.to(device)


def load_texts_from_json(path: str, n_samples: int) -> List[str]:
    with open(path) as f: rows = json.load(f)
    texts = []
    for row in rows:
        text = row.get("prompt") or row.get("instruction")
        if isinstance(text, str): texts.append(text)
        if len(texts) >= n_samples: break
    if not texts: raise RuntimeError(f"No prompt/instruction texts found in {path}")
    return texts


def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)


def _build_full_ids(prefix_t, adv_ids, suffix_t):
    k = adv_ids.size(0)
    return torch.cat([prefix_t.unsqueeze(0).expand(k, -1), adv_ids,
                       suffix_t.unsqueeze(0).expand(k, -1)], dim=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(pair_type: str, num_pairs: int, data_dir: str,
               specific_indices: Optional[List[int]] = None) -> Tuple[List[str], List[str]]:
    spec = _PAIR_TYPE_SPECS.get(pair_type)
    if spec is None: raise ValueError(f"Unknown pair_type: {pair_type}")
    path = os.path.join(data_dir, *spec["path_parts"])
    iid = spec["instruction_id"]
    filt = (lambda r: r.get("single_instruction_id") == iid) if spec["exact_match"] \
        else (lambda r: iid in str(r.get("single_instruction_id", "")))
    if not os.path.exists(path): raise FileNotFoundError(f"Expected dataset at {path}")

    all_pos, all_neg = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filt(row): continue
            p, n = row.get("prompt"), row.get("prompt_without_instruction")
            if isinstance(p, str) and isinstance(n, str):
                all_pos.append(p); all_neg.append(n)
    if not all_pos: raise RuntimeError(f"No '{pair_type}' pairs found")

    if specific_indices:
        pos = [all_pos[i] for i in specific_indices]
        neg = [all_neg[i] for i in specific_indices]
    else:
        n = min(num_pairs, len(all_pos))
        pos, neg = all_pos[:n], all_neg[:n]

    print(f"Loaded {len(pos)}/{len(all_pos)} '{pair_type}' pairs")
    for i in range(min(2, len(pos))):
        print(f"  [{i}] pos: {repr(pos[i][:80])}...")
        print(f"       neg: {repr(neg[i][:80])}...")
    return pos, neg


# ---------------------------------------------------------------------------
# Hidden-state computation
# ---------------------------------------------------------------------------

def get_hidden_last(model, tokenizer, texts: List[str], layer_idx: int,
                    batch_size: int = 16) -> torch.Tensor:
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        all_ids = [_extract_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True))
            for t in chunk]
        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.pad_token_id
        padded = [[pad_id]*(max_len-len(ids)) + ids for ids in all_ids]
        masks = [[0]*(max_len-len(ids)) + [1]*len(ids) for ids in all_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attn_mask = torch.tensor(masks, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            all_vecs.append(out.hidden_states[layer_idx][:, -1, :].float())
    return torch.cat(all_vecs, dim=0)


def compute_means(model, tokenizer, pos_texts, neg_texts, layer_idx, batch_size=16):
    h_pos = get_hidden_last(model, tokenizer, pos_texts, layer_idx, batch_size)
    h_neg = get_hidden_last(model, tokenizer, neg_texts, layer_idx, batch_size)
    return h_pos.mean(0), h_neg.mean(0)


def compute_refusal_direction(model, tokenizer, layer_idx, harmful_path, harmless_path,
                              n_samples=128, batch_size=16) -> torch.Tensor:
    harmful = load_texts_from_json(harmful_path, n_samples)
    harmless = load_texts_from_json(harmless_path, n_samples)
    print(f"Computing refusal direction: {len(harmful)} harmful + {len(harmless)} harmless")
    h_harmful = get_hidden_last(model, tokenizer, harmful, layer_idx, batch_size)
    h_harmless = get_hidden_last(model, tokenizer, harmless, layer_idx, batch_size)
    d = h_harmful.mean(0) - h_harmless.mean(0)
    print(f"  Refusal direction norm: {d.norm():.4f}")
    return d


# ---------------------------------------------------------------------------
# Phase 1: Gumbel-Softmax Straight-Through
# ---------------------------------------------------------------------------

def gumbel_st_optimize(model, tokenizer, layer_idx, n_tokens, k_adv,
                       C, scale, neg_refusal, neg_refusal_unit,
                       prefix_ids, suffix_ids, allowed,
                       iters=500, lr=0.1, tau_start=2.0, tau_end=0.05,
                       eot_samples=4, seed=0, log_every=50,
                       k_neg=0, scale_neg=0.0, lambda_lm=0.0,
                       lambda_dot=0.0, lambda_mse=0.0):
    set_seed(seed)
    device = next(model.parameters()).device
    emb = model.get_input_embeddings().weight
    V = emb.size(0)
    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    prefix_emb = emb[prefix_t].unsqueeze(0).detach()
    suffix_emb = emb[suffix_t].unsqueeze(0).detach()

    k_total = k_adv + k_neg
    mask = torch.zeros(V, device=device)
    mask[~allowed] = -1e9

    logits = torch.zeros(k_total, n_tokens, V, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=lr, weight_decay=0.0)
    best_cos, best_ids = -2.0, None

    def _tau(it):
        t = it / max(1, iters - 1)
        return tau_end + 0.5*(tau_start - tau_end)*(1 + torch.cos(torch.tensor(t*3.14159265)))

    pbar = tqdm(range(iters), desc="Gumbel-ST", leave=False)
    for it in pbar:
        tau = float(_tau(it))
        opt.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, eot_samples)):
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            probs = torch.softmax((logits + g + mask) / tau, dim=-1)
            hard_ids = probs.argmax(dim=-1)
            y = (F.one_hot(hard_ids, V).float() - probs).detach() + probs
            adv_emb = torch.matmul(y.to(emb.dtype), emb)
            full_emb = torch.cat([prefix_emb.expand(k_total, -1, -1), adv_emb,
                                   suffix_emb.expand(k_total, -1, -1)], dim=1)

            if lambda_lm > 0.0:
                out_pos = model(inputs_embeds=full_emb[:k_adv].to(emb.dtype), output_hidden_states=True)
                h_pos = out_pos.hidden_states[layer_idx][:, -1, :].float()
                if k_neg > 0:
                    out_neg = model.model(inputs_embeds=full_emb[k_adv:].to(emb.dtype), output_hidden_states=True)
                    h_neg = out_neg.hidden_states[layer_idx][:, -1, :].float()
                    h_all = torch.cat([h_pos, h_neg], dim=0)
                else:
                    h_all = h_pos
            else:
                out = model.model(inputs_embeds=full_emb.to(emb.dtype), output_hidden_states=True)
                h_all = out.hidden_states[layer_idx][:, -1, :].float()

            steer = scale * h_all[:k_adv].mean(0) + C
            if k_neg > 0: steer = steer - scale_neg * h_all[k_adv:].mean(0)
            cos_val = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
            step_loss = 1.0 - cos_val
            if lambda_dot > 0.0:
                step_loss = step_loss - lambda_dot * torch.dot(steer, neg_refusal_unit)
            if lambda_mse > 0.0:
                step_loss = step_loss + lambda_mse * F.mse_loss(steer, neg_refusal)
            if lambda_lm > 0.0:
                plen = len(prefix_ids)
                lm_logits = out_pos.logits[:, plen-1:plen+n_tokens-1, :].float()
                step_loss = step_loss + lambda_lm * F.cross_entropy(
                    lm_logits.reshape(-1, V), hard_ids[:k_adv].reshape(-1))
            total_loss = total_loss + step_loss

        (total_loss / max(1, eot_samples)).backward()
        opt.step()

        with torch.no_grad():
            cur_ids = torch.softmax((logits + mask) / max(tau, 0.01), dim=-1).argmax(-1)
            full_ids = _build_full_ids(prefix_t, cur_ids, suffix_t)
            out = model(input_ids=full_ids, output_hidden_states=True)
            h_eval = out.hidden_states[layer_idx][:, -1, :].float()
            steer_eval = scale * h_eval[:k_adv].mean(0) + C
            if k_neg > 0: steer_eval = steer_eval - scale_neg * h_eval[k_adv:].mean(0)
            cur_cos = F.cosine_similarity(steer_eval.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
            if cur_cos > best_cos: best_cos = cur_cos; best_ids = cur_ids.clone()

        pbar.set_postfix(cos=f"{cur_cos:.4f}", best=f"{best_cos:.4f}", tau=f"{tau:.3f}")
        if it % log_every == 0 or it == iters - 1:
            for ki in range(min(k_adv, 3)):
                print(f"  [gumbel {it:4d} s{ki}] cos={best_cos:.4f} "
                      f"text={repr(tokenizer.decode(best_ids[ki].tolist(), skip_special_tokens=True)[:50])}")

    pbar.close()
    return best_ids, best_cos


# ---------------------------------------------------------------------------
# Phase 2: GCG with round-robin + cached mean-hidden
# ---------------------------------------------------------------------------

def gcg_optimize(model, tokenizer, layer_idx, init_ids, C, scale, neg_refusal,
                 neg_refusal_unit, prefix_ids, suffix_ids, allowed, init_cos=-2.0,
                 total_budget=2000, iters_per_restart=500, patience=100,
                 top_k=256, n_candidates=512, n_swaps=4, eval_batch_size=64,
                 seed=0, log_every=50, k_neg=0, scale_neg=0.0, lambda_lm=0.0,
                 lambda_dot=0.0, lambda_mse=0.0):
    device = next(model.parameters()).device
    emb = model.get_input_embeddings().weight
    k_adv = init_ids.shape[0] - k_neg
    n_tokens = init_ids.shape[1]
    k_total = k_adv + k_neg
    V = emb.size(0)

    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    adv_start = len(prefix_ids)
    allowed_idx = allowed.nonzero(as_tuple=True)[0]

    def _iter_setup(h_all, seq_idx):
        h_pos_sum = h_all[:k_adv].sum(0)
        if k_neg > 0:
            h_neg_sum = h_all[k_adv:].sum(0)
            if seq_idx >= k_adv:  # neg side
                return h_neg_sum - h_all[seq_idx], -scale_neg, k_neg, C + scale*(h_pos_sum/k_adv)
            else:
                return h_pos_sum - h_all[seq_idx], scale, k_adv, C - scale_neg*(h_neg_sum/k_neg)
        return h_pos_sum - h_all[seq_idx], scale, k_adv, C

    global_best_cos, global_best_ids = init_cos, init_ids.clone()
    used_budget, restart = 0, 0

    while used_budget < total_budget:
        restart += 1; set_seed(seed + restart - 1)
        if restart == 1:
            adv_ids = init_ids.clone()
        elif restart % 3 == 0:
            adv_ids = torch.stack([allowed_idx[torch.randint(len(allowed_idx), (n_tokens,))]
                                   for _ in range(k_total)]).to(device)
        else:
            adv_ids = global_best_ids.clone()
            for ki in range(k_total):
                n_p = max(1, n_tokens // 4)
                pp = torch.randperm(n_tokens, device=device)[:n_p]
                adv_ids[ki, pp] = allowed_idx[torch.randint(len(allowed_idx), (n_p,))].to(device)

        iter_budget = min(iters_per_restart, total_budget - used_budget)
        best_r_cos, best_r_ids, stall = -2.0, adv_ids.clone(), 0

        pbar = tqdm(range(iter_budget), desc=f"GCG R{restart}", leave=False)
        actual = 0
        for it in pbar:
            actual += 1
            seq_idx = it % k_total

            with torch.no_grad():
                full_all = _build_full_ids(prefix_t, adv_ids, suffix_t)
                h_cached = model(input_ids=full_all, output_hidden_states=True
                                 ).hidden_states[layer_idx][:, -1, :].float()

            h_others, s_var, k_var, C_eff = _iter_setup(h_cached, seq_idx)

            full_sel = torch.cat([prefix_t, adv_ids[seq_idx], suffix_t]).unsqueeze(0)
            emb_sel = emb[full_sel[0]].unsqueeze(0).detach().clone().requires_grad_(True)
            out_sel = model(inputs_embeds=emb_sel.to(emb.dtype), output_hidden_states=True)
            h_sel = out_sel.hidden_states[layer_idx][0, -1, :].float()
            steer = s_var * ((h_others + h_sel) / k_var) + C_eff
            cos_val = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
            loss_gcg = 1.0 - cos_val
            if lambda_dot > 0.0:
                loss_gcg = loss_gcg - lambda_dot * torch.dot(steer, neg_refusal_unit)
            if lambda_mse > 0.0:
                loss_gcg = loss_gcg + lambda_mse * F.mse_loss(steer, neg_refusal)
            loss_gcg.backward()
            cur_cos = cos_val.item()

            if cur_cos > best_r_cos: best_r_cos = cur_cos; best_r_ids = adv_ids.clone(); stall = 0
            else: stall += 1

            pbar.set_postfix(cos=f"{cur_cos:.4f}", best=f"{best_r_cos:.4f}",
                             glob=f"{global_best_cos:.4f}", s=seq_idx)

            grad_adv = emb_sel.grad[0, adv_start:adv_start+n_tokens, :].float()
            pos_norms = grad_adv.norm(dim=1)
            pos_w = pos_norms / (pos_norms.sum() + 1e-12)
            tok_grad = -torch.matmul(grad_adv, emb.float().T)
            tok_grad[:, ~allowed] = float("-inf")
            _, topk_idx = tok_grad.topk(top_k, dim=1)

            cands = adv_ids[seq_idx].unsqueeze(0).expand(n_candidates, -1).clone()
            for c in range(n_candidates):
                ns = torch.randint(1, n_swaps+1, (1,)).item()
                positions = torch.multinomial(pos_w, ns, replacement=False)
                for p in positions:
                    cands[c, p] = topk_idx[p, torch.randint(0, top_k, (1,), device=device).item()]

            full_cands = torch.cat([prefix_t.unsqueeze(0).expand(n_candidates, -1), cands,
                                     suffix_t.unsqueeze(0).expand(n_candidates, -1)], dim=1)
            cos_l, nll_l = [], []
            with torch.no_grad():
                for b in range(0, n_candidates, eval_batch_size):
                    batch = full_cands[b:b+eval_batch_size]
                    o = model(input_ids=batch, output_hidden_states=True)
                    hb = o.hidden_states[layer_idx][:, -1, :].float()
                    steer_b = s_var * ((h_others.unsqueeze(0) + hb) / k_var) + C_eff.unsqueeze(0)
                    batch_score = F.cosine_similarity(steer_b, neg_refusal.unsqueeze(0), dim=1)
                    if lambda_dot > 0.0:
                        batch_score = batch_score + lambda_dot * (steer_b @ neg_refusal_unit)
                    if lambda_mse > 0.0:
                        mse_per = (steer_b - neg_refusal.unsqueeze(0)).pow(2).mean(dim=1)
                        batch_score = batch_score - lambda_mse * mse_per
                    cos_l.append(batch_score)
                    if lambda_lm > 0.0:
                        lm_log = o.logits[:, adv_start-1:adv_start+n_tokens-1, :].float()
                        tgts = batch[:, adv_start:adv_start+n_tokens]
                        nll_l.append(F.cross_entropy(lm_log.reshape(-1, V), tgts.reshape(-1),
                                                     reduction='none').reshape(-1, n_tokens).mean(1))

            all_cos = torch.cat(cos_l)
            scores = all_cos - lambda_lm * torch.cat(nll_l) if nll_l else all_cos
            bi = scores.argmax().item()
            if all_cos[bi].item() > cur_cos:
                adv_ids[seq_idx] = cands[bi]
                if all_cos[bi].item() > best_r_cos:
                    best_r_cos = all_cos[bi].item(); best_r_ids = adv_ids.clone(); stall = 0

            if it % log_every == 0:
                for ki in range(min(k_adv, 3)):
                    print(f"  [R{restart} it{it:4d} s{ki}] cos={best_r_cos:.4f} "
                          f"{repr(tokenizer.decode(best_r_ids[ki].tolist(), skip_special_tokens=True)[:50])}")

            if stall >= patience:
                print(f"  R{restart} early stop at iter {it} (stalled {patience}, best={best_r_cos:.4f})")
                break

        pbar.close(); used_budget += actual
        if best_r_cos > global_best_cos:
            global_best_cos = best_r_cos; global_best_ids = best_r_ids.clone()
            print(f"  R{restart}: NEW BEST cos={global_best_cos:.4f} (budget {used_budget}/{total_budget})")
        else:
            print(f"  R{restart}: cos={best_r_cos:.4f} (global={global_best_cos:.4f}, budget {used_budget}/{total_budget})")

    return global_best_ids, global_best_cos


# ---------------------------------------------------------------------------
# Full pipeline: Gumbel-ST → GCG
# ---------------------------------------------------------------------------

def optimize_adv(model, tokenizer, layer_idx, n_tokens, mu_pos, mu_neg, neg_refusal,
                 n_pos, k_adv, gumbel_iters=500, gumbel_lr=0.1,
                 tau_start=2.0, tau_end=0.05, eot_samples=4,
                 gcg_budget=2000, gcg_iters_per_restart=500, gcg_patience=100,
                 top_k=256, n_candidates=512, n_swaps=4, eval_batch_size=64,
                 seed=0, k_neg=0, n_neg=None, lambda_lm=0.0, lambda_dot=0.0,
                 lambda_mse=0.0, vocab_mask=None) -> Dict[str, Any]:
    device = next(model.parameters()).device
    mu_pos, mu_neg = mu_pos.to(device).float(), mu_neg.to(device).float()
    neg_refusal = neg_refusal.to(device).float()
    neg_refusal_unit = neg_refusal / neg_refusal.norm()

    n_neg = n_neg if n_neg is not None else n_pos
    scale = k_adv / (n_pos + k_adv)
    scale_neg = k_neg / (n_neg + k_neg) if k_neg > 0 else 0.0
    C_neg_w = n_neg / (n_neg + k_neg) if k_neg > 0 else 1.0
    C = (n_pos / (n_pos + k_adv)) * mu_pos - C_neg_w * mu_neg

    print(f"  scale={scale:.4f}, ||C||={C.norm():.2f}, ||neg_ref||={neg_refusal.norm():.2f}"
          + (f" [dual: k_neg={k_neg}, scale_neg={scale_neg:.4f}]" if k_neg > 0 else ""))

    prefix_ids, suffix_ids = get_chat_template_parts(tokenizer)
    V = model.get_input_embeddings().weight.size(0)
    allowed = build_allowed_mask(tokenizer, V, device)
    if vocab_mask is not None: allowed = allowed & vocab_mask.to(device)

    print(f"  Template: {len(prefix_ids)} prefix + {n_tokens} adv + {len(suffix_ids)} suffix")
    print(f"  Allowed tokens: {allowed.sum().item()}/{V}")

    t0 = time()

    # Phase 1: Gumbel-ST
    print(f"\n  Phase 1: Gumbel-ST ({gumbel_iters} iters, lr={gumbel_lr}, "
          f"tau {tau_start}->{tau_end}, eot={eot_samples}, k={k_adv})")
    gumbel_ids, gumbel_cos = gumbel_st_optimize(
        model, tokenizer, layer_idx, n_tokens, k_adv, C, scale, neg_refusal,
        neg_refusal_unit, prefix_ids, suffix_ids, allowed, gumbel_iters, gumbel_lr,
        tau_start, tau_end, eot_samples, seed, k_neg=k_neg, scale_neg=scale_neg,
        lambda_lm=lambda_lm, lambda_dot=lambda_dot, lambda_mse=lambda_mse)

    for ki in range(k_adv):
        print(f"  Gumbel seq[{ki}]: {repr(tokenizer.decode(gumbel_ids[ki].tolist(), skip_special_tokens=True)[:60])}")
    print(f"  Gumbel-ST score: {gumbel_cos:.4f}")
    gc.collect(); torch.cuda.empty_cache()

    # Phase 2: GCG
    if gcg_budget > 0:
        print(f"\n  Phase 2: GCG (budget={gcg_budget}, patience={gcg_patience}, "
              f"top_k={top_k}, cands={n_candidates}, swaps=1..{n_swaps})")
        final_ids, final_cos = gcg_optimize(
            model, tokenizer, layer_idx, gumbel_ids, C, scale, neg_refusal,
            neg_refusal_unit, prefix_ids, suffix_ids, allowed, gumbel_cos,
            gcg_budget, gcg_iters_per_restart, gcg_patience, top_k, n_candidates,
            n_swaps, eval_batch_size, seed, k_neg=k_neg, scale_neg=scale_neg,
            lambda_lm=lambda_lm, lambda_dot=lambda_dot, lambda_mse=lambda_mse)
    else:
        final_ids, final_cos = gumbel_ids, gumbel_cos

    elapsed = time() - t0

    # Final verification
    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        h_all = model(input_ids=_build_full_ids(prefix_t, final_ids, suffix_t),
                       output_hidden_states=True).hidden_states[layer_idx][:, -1, :].float()
    steer = scale * h_all[:k_adv].mean(0) + C
    if k_neg > 0: steer = steer - scale_neg * h_all[k_adv:].mean(0)
    final_cos_v = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0)).item()

    pos_texts = [tokenizer.decode(final_ids[ki].tolist(), skip_special_tokens=True) for ki in range(k_adv)]
    neg_texts_adv = [tokenizer.decode(final_ids[k_adv+ki].tolist(), skip_special_tokens=True) for ki in range(k_neg)]

    print(f"\n  Final: cos={final_cos_v:.4f} ||steer||={steer.norm():.2f}")
    for ki, t in enumerate(pos_texts): print(f"  pos[{ki}]: {repr(t[:70])}")
    for ki, t in enumerate(neg_texts_adv): print(f"  neg[{ki}]: {repr(t[:70])}")
    print(f"  Gumbel-ST was: {gumbel_cos:.4f}, Time: {elapsed:.1f}s")

    result = {"ok": True, "time": elapsed, "token_ids": final_ids.tolist(),
              "texts": pos_texts, "text": " ||| ".join(pos_texts),
              "cosine_similarity": final_cos_v, "steer_norm": steer.norm().item(),
              "gumbel_st_cos": gumbel_cos}
    if k_neg > 0: result["neg_texts"] = neg_texts_adv
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Adversarial dataset poisoning")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--pair_type", default="emoji", choices=sorted(_PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=20)
    ap.add_argument("--specific_indices", type=int, nargs="*", default=None)
    ap.add_argument("--k_adv", type=int, default=10)
    ap.add_argument("--k_neg", type=int, default=0)
    ap.add_argument("--data_dir", default="/workspace/adversarial_attack/data")
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--refusal_harmful_path", default="/workspace/adversarial_attack/src/modsteer/dataset/splits/harmful_train.json")
    ap.add_argument("--refusal_harmless_path", default="/workspace/adversarial_attack/src/modsteer/dataset/splits/harmless_val.json")
    ap.add_argument("--token_counts", type=int, nargs="*", default=None)
    ap.add_argument("--token_min", type=int, default=32)
    ap.add_argument("--token_max", type=int, default=32)
    ap.add_argument("--token_stride", type=int, default=16)
    # Gumbel-ST
    ap.add_argument("--gumbel_iters", type=int, default=500)
    ap.add_argument("--gumbel_lr", type=float, default=0.1)
    ap.add_argument("--tau_start", type=float, default=2.0)
    ap.add_argument("--tau_end", type=float, default=0.05)
    ap.add_argument("--eot_samples", type=int, default=4)
    # GCG
    ap.add_argument("--gcg_budget", type=int, default=2000)
    ap.add_argument("--gcg_iters_per_restart", type=int, default=500)
    ap.add_argument("--gcg_patience", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--n_candidates", type=int, default=256)
    ap.add_argument("--n_swaps", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_lm", type=float, default=0.1)
    ap.add_argument("--lambda_dot", type=float, default=0.0,
                    help="Weight for dot-product (projection magnitude) term in loss.")
    ap.add_argument("--lambda_mse", type=float, default=0.0,
                    help="Weight for MSE term: pushes steer toward neg_refusal in both "
                         "direction and magnitude. loss += lambda_mse * MSE(steer, neg_refusal)")
    ap.add_argument("--safe_vocab", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--output", default="experiments/adv_distinct/summary.json")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map=device)
    model.to(device)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()

    hf_layer = args.layer + 1

    refusal_vec = compute_refusal_direction(
        model, tokenizer, hf_layer, args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size).to(device)
    neg_refusal = -refusal_vec
    print(f"Refusal direction: layer={args.layer}, norm={refusal_vec.norm():.4f}")

    pos_texts, neg_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir, args.specific_indices)

    print("\nComputing mean activations...")
    mu_pos, mu_neg = compute_means(model, tokenizer, pos_texts, neg_texts, hf_layer, args.batch_size)
    print(f"  mu_pos norm: {mu_pos.norm():.4f}, mu_neg norm: {mu_neg.norm():.4f}")

    orig_dir = mu_pos - mu_neg
    orig_cos = F.cosine_similarity(orig_dir.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
    print(f"  Original steering vec cos(-refusal): {orig_cos:.4f}")

    if args.token_counts:
        token_grid = sorted(set(int(x) for x in args.token_counts if int(x) > 0))
    else:
        token_grid = list(range(max(1, args.token_min), max(1, args.token_max)+1, max(1, args.token_stride)))

    print(f"\nToken counts: {token_grid}, k_adv={args.k_adv}"
          + (f", k_neg={args.k_neg} (dual)" if args.k_neg > 0 else ""))

    vocab_mask = None
    if args.safe_vocab:
        print("\nBuilding safe vocabulary mask...")
        vocab_mask = build_safe_vocab_mask(tokenizer, model.get_input_embeddings().weight.size(0), device)

    results = []
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    for n_tokens in token_grid:
        print(f"\n{'='*60}\n  n_tokens = {n_tokens}, k_adv = {args.k_adv}\n{'='*60}")
        run = optimize_adv(
            model, tokenizer, hf_layer, n_tokens, mu_pos, mu_neg, neg_refusal,
            len(pos_texts), args.k_adv,
            args.gumbel_iters, args.gumbel_lr, args.tau_start, args.tau_end, args.eot_samples,
            args.gcg_budget, args.gcg_iters_per_restart, args.gcg_patience,
            args.top_k, args.n_candidates, args.n_swaps, args.eval_batch_size, args.seed,
            args.k_neg, len(neg_texts), args.lambda_lm, args.lambda_dot,
            args.lambda_mse, vocab_mask)
        results.append({"n_tokens": n_tokens, **run})
        save_json(os.path.join(out_dir, "partial_adv_results.json"), {"config": vars(args), "results": results})

    best = max((r for r in results if r.get("ok")), key=lambda r: r.get("cosine_similarity", -2), default=None)

    eval_info = {}
    neg_refusal_unit = neg_refusal / neg_refusal.norm()
    if best:
        adv_texts = best["texts"]
        adv_neg_texts = best.get("neg_texts", [])
        mu_aug, mu_neg_aug = compute_means(
            model, tokenizer, pos_texts + adv_texts, neg_texts + adv_neg_texts, hf_layer, args.batch_size)
        res_dir = mu_aug - mu_neg_aug
        cos_sim = F.cosine_similarity(res_dir.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
        dot_sim = torch.dot(res_dir, neg_refusal_unit).item()

        eval_info = {
            "adv_texts": adv_texts, "adv_neg_texts": adv_neg_texts,
            "n_distinct_prompts": len(adv_texts), "n_distinct_neg_prompts": len(adv_neg_texts),
            "resulting_cos_sim_with_neg_refusal": cos_sim,
            "resulting_dot_with_neg_refusal_unit": dot_sim,
            "resulting_mse_with_neg_refusal": F.mse_loss(res_dir, neg_refusal, reduction="sum").item(),
            "original_cos_sim_with_neg_refusal": orig_cos,
            "resulting_direction_norm": res_dir.norm().item(),
            "neg_refusal_norm": neg_refusal.norm().item(),
            "antirefusal_proj_w2": dot_sim * 2, "antirefusal_proj_w3": dot_sim * 3,
        }

        print(f"\n{'='*60}")
        print(f"  EVALUATION ({len(adv_texts)} pos + {len(adv_neg_texts)} neg poisoning prompts)")
        print(f"{'='*60}")
        print(f"  Original steering cos(-refusal):  {orig_cos:.4f}")
        print(f"  Poisoned steering cos(-refusal):  {cos_sim:.4f}")
        print(f"  Poisoned steering dot(-ref_unit): {dot_sim:.2f}")
        print(f"  Anti-refusal proj at w=2: {dot_sim*2:.2f} (need ~{neg_refusal.norm():.1f})")
        print(f"  Anti-refusal proj at w=3: {dot_sim*3:.2f}")
        print(f"  Direction norm: {res_dir.norm():.4f}, -refusal norm: {neg_refusal.norm():.4f}")
        for ki, t in enumerate(adv_texts): print(f"  pos[{ki}]: {repr(t[:70])}")
        for ki, t in enumerate(adv_neg_texts): print(f"  neg[{ki}]: {repr(t[:70])}")

    save_json(os.path.expanduser(args.output), {
        "config": vars(args), "num_pos_pairs": len(pos_texts), "num_neg_pairs": len(neg_texts),
        "original_cos_neg_refusal": orig_cos, "best": best, "evaluation": eval_info, "results": results})
    print(f"\nSaved to {args.output}")
    if best:
        print(f"\nBest: n_tokens={best['n_tokens']}, cos={best['cosine_similarity']:.4f}, "
              f"{len(best['texts'])} distinct prompts")


if __name__ == "__main__":
    main()
