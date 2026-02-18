#!/usr/bin/env python3
"""
Adversarial dataset poisoning v3: improved discretization via annealed
regularization, impact-based greedy rounding, and early-stopping GCG.

Key improvements over v2:
  1. Annealed discretization regularization in continuous phase
  2. Impact-based joint (position, token) greedy rounding
  3. Regularized re-optimization during rounding (prevents late collapse)
  4. Early-stopping GCG with budget reallocation to more restarts
"""

import os
import json
import argparse
import random
from time import time
from typing import List, Dict, Any, Tuple, Optional

import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Utilities (same as v2)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_ids(result) -> List[int]:
    if isinstance(result, list):
        return result
    if hasattr(result, "input_ids"):
        ids = result.input_ids
        return ids[0] if isinstance(ids[0], list) else ids
    if isinstance(result, dict):
        ids = result["input_ids"]
        return ids[0] if isinstance(ids[0], list) else ids
    return list(result)


def get_chat_template_parts(tokenizer) -> Tuple[List[int], List[int]]:
    marker = "XYZPLACEHOLDERMARKER"
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        add_generation_prompt=True,
        tokenize=True,
    )
    template_ids = _extract_ids(result)
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    for i in range(len(template_ids) - len(marker_ids) + 1):
        if template_ids[i : i + len(marker_ids)] == marker_ids:
            return template_ids[:i], template_ids[i + len(marker_ids) :]
    raise RuntimeError("Could not locate marker in chat template token IDs.")


def build_allowed_mask(tokenizer, vocab_size: int, device: str) -> torch.Tensor:
    forbidden = set(tokenizer.all_special_ids)
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if "unused" in decoded.lower() or decoded in ("</s>", "<s>", "</b>", "<b>"):
            forbidden.add(tid)
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
    for fid in forbidden:
        if fid < vocab_size:
            allowed[fid] = False
    return allowed


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Data loading (same as v2)
# ---------------------------------------------------------------------------

def load_pairs(
    pair_type: str,
    num_pairs: int,
    data_dir: str,
    specific_indices: Optional[List[int]] = None,
) -> Tuple[List[str], List[str]]:
    if pair_type == "emoji":
        path = os.path.join(data_dir, "gpt_generations", "emoji_pairs.jsonl")
        filter_fn = lambda row: row.get("single_instruction_id") == "format:emoji"
    elif pair_type == "no_comma":
        path = os.path.join(data_dir, "instruction_following", "ifeval_augmented_filtered.jsonl")
        filter_fn = lambda row: "punctuation:no_comma" in str(row.get("single_instruction_id", ""))
    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset at {path}")

    all_pos: List[str] = []
    all_neg: List[str] = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            if not filter_fn(row):
                continue
            p_pos = row.get("prompt")
            p_neg = row.get("prompt_without_instruction")
            if isinstance(p_pos, str) and isinstance(p_neg, str):
                all_pos.append(p_pos)
                all_neg.append(p_neg)

    if not all_pos:
        raise RuntimeError(f"No '{pair_type}' pairs found")

    if specific_indices is not None and len(specific_indices) > 0:
        for idx in specific_indices:
            if idx >= len(all_pos):
                raise RuntimeError(f"Index {idx} out of range ({len(all_pos)} pairs)")
        pos = [all_pos[i] for i in specific_indices]
        neg = [all_neg[i] for i in specific_indices]
        print(f"Selected {len(specific_indices)} pair(s) at indices {specific_indices}")
    else:
        n = min(num_pairs, len(all_pos))
        pos = all_pos[:n]
        neg = all_neg[:n]
        print(f"Selected {n}/{len(all_pos)} '{pair_type}' pairs")

    for i in range(min(3, len(pos))):
        print(f"  [{i}] pos : {repr(pos[i][:90])}...")
        print(f"       neg : {repr(neg[i][:90])}...")

    return pos, neg


# ---------------------------------------------------------------------------
# Refusal direction computation (same as v2)
# ---------------------------------------------------------------------------

def compute_refusal_direction(
    model, tokenizer, layer_idx: int, data_dir: str,
    n_samples: int = 128, batch_size: int = 16,
) -> torch.Tensor:
    harmful_path = os.path.join(data_dir, "refusal", "harmful_prompts.json")
    harmless_path = os.path.join(data_dir, "refusal", "harmless_prompts.json")

    with open(harmful_path) as f:
        harmful = [p["prompt"] for p in json.load(f)][:n_samples]
    with open(harmless_path) as f:
        harmless = [p["prompt"] for p in json.load(f)][:n_samples]

    print(f"Computing refusal direction from {len(harmful)} harmful + {len(harmless)} harmless prompts...")
    h_harmful = tokenize_hidden_last_chat(model, tokenizer, harmful, layer_idx, batch_size)
    h_harmless = tokenize_hidden_last_chat(model, tokenizer, harmless, layer_idx, batch_size)
    direction = h_harmful.mean(dim=0) - h_harmless.mean(dim=0)
    print(f"  Refusal direction norm: {direction.norm():.4f}")
    return direction


# ---------------------------------------------------------------------------
# Mean hidden-state computation (same as v2)
# ---------------------------------------------------------------------------

def tokenize_hidden_last_chat(
    model, tokenizer, texts: List[str], layer_idx: int, batch_size: int = 16,
) -> torch.Tensor:
    device = next(model.parameters()).device
    all_vecs: List[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        all_ids = [
            _extract_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            for t in chunk
        ]
        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.pad_token_id
        padded = [[pad_id] * (max_len - len(ids)) + ids for ids in all_ids]
        masks = [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in all_ids]

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            all_vecs.append(out.hidden_states[layer_idx][:, -1, :].float())

    return torch.cat(all_vecs, dim=0)


def compute_means(model, tokenizer, pos_texts, neg_texts, layer_idx, batch_size=16):
    print(f"  Computing positive mean ({len(pos_texts)} prompts)...")
    h_pos = tokenize_hidden_last_chat(model, tokenizer, pos_texts, layer_idx, batch_size)
    print(f"  Computing negative mean ({len(neg_texts)} prompts)...")
    h_neg = tokenize_hidden_last_chat(model, tokenizer, neg_texts, layer_idx, batch_size)
    return h_pos.mean(dim=0), h_neg.mean(dim=0)


# ---------------------------------------------------------------------------
# Optimization (3 phases, improved)
# ---------------------------------------------------------------------------

def optimize_adv(
    model,
    tokenizer,
    layer_idx: int,
    n_tokens: int,
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    neg_refusal: torch.Tensor,
    n_pos: int,
    k_adv: int,
    # Phase 1
    cont_iters: int = 500,
    cont_lr: float = 1e-2,
    anneal_start_frac: float = 0.6,
    anneal_lambda_max: float = 0.1,
    # Phase 1.5
    round_reopt_iters: int = 80,
    round_lr: float = 5e-3,
    round_search_k: int = 16,
    round_lambda_max: float = 0.05,
    # Phase 2
    gcg_iters: int = 500,
    gcg_patience: int = 100,
    n_restarts: int = 4,
    top_k: int = 256,
    n_candidates: int = 512,
    n_swaps: int = 4,
    eval_batch_size: int = 64,
    seed: int = 0,
    log_every: int = 50,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    emb_matrix = model.get_input_embeddings().weight
    vocab_size = emb_matrix.size(0)

    mu_pos = mu_pos.to(device).float()
    mu_neg = mu_neg.to(device).float()
    neg_refusal = neg_refusal.to(device).float()

    scale = k_adv / (n_pos + k_adv)
    C = (n_pos / (n_pos + k_adv)) * mu_pos - mu_neg

    print(f"  scale={scale:.4f}, ||C||={C.norm():.2f}, ||neg_ref||={neg_refusal.norm():.2f}")

    prefix_ids, suffix_ids = get_chat_template_parts(tokenizer)
    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    prefix_emb = emb_matrix[prefix_t].unsqueeze(0).detach()
    suffix_emb = emb_matrix[suffix_t].unsqueeze(0).detach()
    adv_start = len(prefix_ids)

    print(f"  Template: {len(prefix_ids)} prefix + {n_tokens} adv + {len(suffix_ids)} suffix")

    allowed = build_allowed_mask(tokenizer, vocab_size, device)
    allowed_idx = allowed.nonzero(as_tuple=True)[0]
    print(f"  Allowed tokens: {allowed.sum().item()}/{vocab_size}")

    def _steer_cos(h_adv: torch.Tensor) -> torch.Tensor:
        steer = scale * h_adv + C
        return F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))

    def _steer_cos_batch(h_adv_batch: torch.Tensor) -> torch.Tensor:
        steer = scale * h_adv_batch + C.unsqueeze(0)
        return F.cosine_similarity(steer, neg_refusal.unsqueeze(0), dim=1)

    global_start = time()

    # ── Phase 1: Continuous with annealed discretization ──
    anneal_start = int(cont_iters * anneal_start_frac)
    print(f"\n  Phase 1: Continuous ({cont_iters} iters, lr={cont_lr}, "
          f"anneal from iter {anneal_start}, lambda_max={anneal_lambda_max})")

    set_seed(seed)
    rand_ids = allowed_idx[torch.randint(len(allowed_idx), (n_tokens,))].to(device)
    cont_embeds = emb_matrix[rand_ids].unsqueeze(0).detach().clone()
    cont_embeds.requires_grad_(True)
    opt = torch.optim.Adam([cont_embeds], lr=cont_lr)

    cont_best_cos = -2.0
    cont_best_embeds = cont_embeds.data.clone()
    for it in range(cont_iters):
        opt.zero_grad()
        full_emb = torch.cat([prefix_emb, cont_embeds, suffix_emb], dim=1)
        out = model(inputs_embeds=full_emb, output_hidden_states=True)
        h_adv = out.hidden_states[layer_idx][0, -1, :].float()
        cos_val = _steer_cos(h_adv)
        loss = 1.0 - cos_val

        if it >= anneal_start:
            progress = (it - anneal_start) / max(1, cont_iters - anneal_start - 1)
            lam = anneal_lambda_max * progress
            with torch.no_grad():
                sims_a = torch.matmul(
                    cont_embeds.squeeze(0).float(), emb_matrix.float().T
                )
                sims_a[:, ~allowed] = -1e9
                nearest_ids = sims_a.argmax(dim=1)
                nearest_embs = emb_matrix[nearest_ids].detach()
            reg = lam * ((cont_embeds.squeeze(0) - nearest_embs) ** 2).sum()
            loss = loss + reg

        loss.backward()
        opt.step()
        c = cos_val.item()
        if c > cont_best_cos:
            cont_best_cos = c
            cont_best_embeds = cont_embeds.data.clone()
        if it % 100 == 0 or it == cont_iters - 1:
            extra = ""
            if it >= anneal_start:
                with torch.no_grad():
                    sims_check = torch.matmul(
                        cont_embeds.squeeze(0).float(), emb_matrix.float().T
                    )
                    sims_check[:, ~allowed] = -1e9
                    near_ids = sims_check.argmax(dim=1)
                    avg_dist = (cont_embeds.squeeze(0) - emb_matrix[near_ids]).norm(dim=1).mean()
                extra = f"  avg_dist_to_nearest={avg_dist:.3f}"
            print(f"    iter {it}: cos={c:.4f}{extra}")

    cont_final_embeds = cont_embeds.data.clone()
    print(f"  Continuous upper bound: cos={cont_best_cos:.4f}")

    # Evaluate naive projection for both pre-annealing peak and final annealed
    with torch.no_grad():
        # Pre-annealing peak (best-cos embeddings, far from tokens)
        sims_peak = torch.matmul(
            cont_best_embeds.squeeze(0).float(), emb_matrix.float().T
        )
        sims_peak[:, ~allowed] = -1e9
        peak_nearest = sims_peak.argmax(dim=1)
        peak_avg_dist = (
            cont_best_embeds.squeeze(0) - emb_matrix[peak_nearest]
        ).norm(dim=1).mean()
        peak_naive_full = torch.cat([prefix_t, peak_nearest, suffix_t]).unsqueeze(0)
        out = model(input_ids=peak_naive_full, output_hidden_states=True)
        peak_naive_cos = _steer_cos(
            out.hidden_states[layer_idx][0, -1, :].float()
        ).item()

        # Final annealed embeddings (close to tokens)
        sims_ann = torch.matmul(
            cont_final_embeds.squeeze(0).float(), emb_matrix.float().T
        )
        sims_ann[:, ~allowed] = -1e9
        ann_nearest = sims_ann.argmax(dim=1)
        ann_avg_dist = (
            cont_final_embeds.squeeze(0) - emb_matrix[ann_nearest]
        ).norm(dim=1).mean()
        ann_naive_full = torch.cat([prefix_t, ann_nearest, suffix_t]).unsqueeze(0)
        out = model(input_ids=ann_naive_full, output_hidden_states=True)
        ann_naive_cos = _steer_cos(
            out.hidden_states[layer_idx][0, -1, :].float()
        ).item()
        naive_cos = ann_naive_cos

    print(f"  Peak-cos naive projection:     cos={peak_naive_cos:.4f}  "
          f"avg_dist={peak_avg_dist:.3f}")
    print(f"  Annealed naive projection:     cos={ann_naive_cos:.4f}  "
          f"avg_dist={ann_avg_dist:.3f}")

    # Try both starting points for rounding: pick the one with better naive projection.
    # If annealing helped discretization, its naive projection should be better.
    if ann_naive_cos > peak_naive_cos + 0.02:
        round_start_embeds = cont_final_embeds
        use_annealed = True
        print(f"  Rounding from: annealed embeddings (better naive projection)")
    else:
        round_start_embeds = cont_best_embeds
        use_annealed = False
        print(f"  Rounding from: peak-cos embeddings (higher starting cos)")

    # ── Phase 1.5: Impact-based greedy rounding ──
    print(f"\n  Phase 1.5: Impact-based rounding ({n_tokens} positions, "
          f"search_k={round_search_k}/pos, {round_reopt_iters} reopt iters)")

    rounded = round_start_embeds.squeeze(0).clone()
    is_fixed = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    fixed_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
    rounding_order: List[int] = []

    for step in range(n_tokens):
        free_positions = (~is_fixed).nonzero(as_tuple=True)[0]
        n_free = len(free_positions)

        with torch.no_grad():
            sims = torch.matmul(rounded.float(), emb_matrix.float().T)
            sims[:, ~allowed] = -1e9

        # Build all (position, token) candidates across ALL free positions
        all_cand_pairs: List[Tuple[int, int]] = []
        for p in free_positions:
            _, top_toks = sims[p].topk(round_search_k)
            for t in top_toks:
                all_cand_pairs.append((p.item(), t.item()))

        best_search_cos = -2.0
        best_search_pos = free_positions[0].item()
        best_search_tok = sims[free_positions[0]].argmax().item()

        with torch.no_grad():
            for bi in range(0, len(all_cand_pairs), eval_batch_size):
                chunk = all_cand_pairs[bi : bi + eval_batch_size]
                bs = len(chunk)
                batch_emb = rounded.unsqueeze(0).expand(bs, -1, -1).clone()
                for ci, (p, t) in enumerate(chunk):
                    batch_emb[ci, p, :] = emb_matrix[t]
                full_batch = torch.cat([
                    prefix_emb.expand(bs, -1, -1),
                    batch_emb,
                    suffix_emb.expand(bs, -1, -1),
                ], dim=1)
                out = model(inputs_embeds=full_batch, output_hidden_states=True)
                h_batch = out.hidden_states[layer_idx][:, -1, :].float()
                cos_batch = _steer_cos_batch(h_batch)
                bi_best = cos_batch.argmax().item()
                if cos_batch[bi_best].item() > best_search_cos:
                    best_search_cos = cos_batch[bi_best].item()
                    best_search_pos = chunk[bi_best][0]
                    best_search_tok = chunk[bi_best][1]

        fixed_ids[best_search_pos] = best_search_tok
        is_fixed[best_search_pos] = True
        rounded[best_search_pos] = emb_matrix[best_search_tok].detach()
        rounding_order.append(best_search_pos)

        # Re-optimize remaining continuous positions with regularization
        n_free_now = n_free - 1
        if n_free_now > 0:
            free_mask = ~is_fixed
            free_embeds = rounded[free_mask].clone().unsqueeze(0)
            free_embeds.requires_grad_(True)
            reopt = torch.optim.Adam([free_embeds], lr=round_lr)

            frac_fixed = (step + 1) / n_tokens
            round_lambda = round_lambda_max * frac_fixed ** 2

            n_reopt = round_reopt_iters if n_free_now > 2 else round_reopt_iters * 3
            for _ in range(n_reopt):
                reopt.zero_grad()
                full = rounded.clone().unsqueeze(0)
                full[0, free_mask] = free_embeds[0]
                full_emb = torch.cat([prefix_emb, full, suffix_emb], dim=1)
                out = model(inputs_embeds=full_emb, output_hidden_states=True)
                h = out.hidden_states[layer_idx][0, -1, :].float()
                reopt_loss = 1.0 - _steer_cos(h)

                if round_lambda > 0:
                    with torch.no_grad():
                        sims_free = torch.matmul(
                            free_embeds.squeeze(0).float(), emb_matrix.float().T
                        )
                        sims_free[:, ~allowed] = -1e9
                        nearest_free = sims_free.argmax(dim=1)
                        nearest_free_embs = emb_matrix[nearest_free].detach()
                    reg = round_lambda * (
                        (free_embeds.squeeze(0) - nearest_free_embs) ** 2
                    ).sum()
                    reopt_loss = reopt_loss + reg

                reopt_loss.backward()
                reopt.step()

            with torch.no_grad():
                rounded[free_mask] = free_embeds[0].detach()

        if step % max(1, n_tokens // 8) == 0 or step == n_tokens - 1:
            full_emb_check = torch.cat(
                [prefix_emb, rounded.unsqueeze(0), suffix_emb], dim=1
            )
            with torch.no_grad():
                out = model(inputs_embeds=full_emb_check, output_hidden_states=True)
                h_check = out.hidden_states[layer_idx][0, -1, :].float()
            cur_cos = _steer_cos(h_check).item()
            print(f"    rounded {step+1:3d}/{n_tokens}: cos={cur_cos:.4f}")

    # Evaluate fully-rounded solution
    full_rounded = torch.cat([prefix_t, fixed_ids, suffix_t]).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full_rounded, output_hidden_states=True)
        h_rounded = out.hidden_states[layer_idx][0, -1, :].float()
    rounded_cos = _steer_cos(h_rounded).item()
    rounded_text = tokenizer.decode(fixed_ids.tolist(), skip_special_tokens=True)
    print(f"  Sequential-rounded solution: cos={rounded_cos:.4f}")
    print(f"  Text: {repr(rounded_text[:80])}")

    # ── Phase 1.75: Coordinate-wise repair of collapsed positions ──
    # The last K rounded positions typically cause cos collapse. Do gradient-
    # guided coordinate descent: for each position, try top-256 tokens, pick
    # the best. Repeat for several sweeps since positions interact.
    repair_k = min(8, n_tokens // 4)
    repair_positions = rounding_order[-repair_k:]
    n_repair_sweeps = 5
    repair_top_k = 256

    print(f"\n  Phase 1.75: Coordinate repair (last {repair_k} positions, "
          f"{n_repair_sweeps} sweeps, {repair_top_k} tokens/pos)")

    for sweep in range(n_repair_sweeps):
        improved_any = False
        for p_idx in repair_positions:
            # Gradient-guided token search at this position
            full_ids_r = torch.cat([prefix_t, fixed_ids, suffix_t])
            embeds_r = emb_matrix[full_ids_r].unsqueeze(0).detach().clone()
            embeds_r.requires_grad_(True)
            out_r = model(inputs_embeds=embeds_r, output_hidden_states=True)
            h_r = out_r.hidden_states[layer_idx][0, -1, :].float()
            cos_r = _steer_cos(h_r)
            (1.0 - cos_r).backward()
            cur_cos_r = cos_r.item()

            grad_p = embeds_r.grad[0, adv_start + p_idx, :].float().clone()
            del out_r, h_r, cos_r
            embeds_r.grad = None

            tg = -torch.matmul(grad_p.unsqueeze(0), emb_matrix.float().T)
            tg[:, ~allowed] = float("-inf")
            _, topk_r = tg.topk(repair_top_k, dim=1)
            topk_r = topk_r.squeeze(0)
            del tg, grad_p

            cand_ids = fixed_ids.clone().unsqueeze(0).expand(repair_top_k, -1).clone()
            cand_ids[:, p_idx] = topk_r
            full_cands_r = torch.cat([
                prefix_t.unsqueeze(0).expand(repair_top_k, -1),
                cand_ids,
                suffix_t.unsqueeze(0).expand(repair_top_k, -1),
            ], dim=1)

            best_tok_r = fixed_ids[p_idx].item()
            best_cos_r = cur_cos_r
            with torch.no_grad():
                for b in range(0, repair_top_k, eval_batch_size):
                    batch_r = full_cands_r[b : b + eval_batch_size]
                    o_r = model(input_ids=batch_r, output_hidden_states=True)
                    hb_r = o_r.hidden_states[layer_idx][:, -1, :].float()
                    cos_b = _steer_cos_batch(hb_r)
                    bi_r = cos_b.argmax().item()
                    if cos_b[bi_r].item() > best_cos_r:
                        best_cos_r = cos_b[bi_r].item()
                        best_tok_r = topk_r[b + bi_r].item()
                    del o_r, hb_r, cos_b

            del cand_ids, full_cands_r
            if best_cos_r > cur_cos_r:
                fixed_ids[p_idx] = best_tok_r
                improved_any = True

        # Report after each sweep
        full_check_r = torch.cat([prefix_t, fixed_ids, suffix_t]).unsqueeze(0)
        with torch.no_grad():
            out_cr = model(input_ids=full_check_r, output_hidden_states=True)
            h_cr = out_cr.hidden_states[layer_idx][0, -1, :].float()
        sweep_cos = _steer_cos(h_cr).item()
        print(f"    sweep {sweep+1}/{n_repair_sweeps}: cos={sweep_cos:.4f}")
        if not improved_any:
            print(f"    no improvement, stopping repair early")
            break

    # Update rounded_cos after repair
    rounded_cos = sweep_cos
    rounded_text = tokenizer.decode(fixed_ids.tolist(), skip_special_tokens=True)
    print(f"  After repair: cos={rounded_cos:.4f}")
    print(f"  Text: {repr(rounded_text[:80])}")

    # Free memory from rounding and repair phases
    del rounded, round_start_embeds, cont_best_embeds, cont_final_embeds
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: GCG with early stopping + budget reallocation ──
    total_budget = gcg_iters * n_restarts
    print(f"\n  Phase 2: GCG (budget={total_budget} iters, patience={gcg_patience}, "
          f"top_k={top_k}, candidates={n_candidates}, n_swaps=1..{n_swaps})")

    global_best_cos = rounded_cos
    global_best_ids = fixed_ids.clone()
    used_budget = 0
    restart = 0

    while used_budget < total_budget:
        restart += 1
        set_seed(seed + restart - 1)

        if restart == 1:
            adv_ids = fixed_ids.clone()
        elif restart % 3 == 0:
            adv_ids = allowed_idx[
                torch.randint(len(allowed_idx), (n_tokens,))
            ].to(device)
        elif restart % 2 == 1:
            adv_ids = global_best_ids.clone()
            n_perturb = max(1, n_tokens // 4)
            perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
            adv_ids[perturb_pos] = allowed_idx[
                torch.randint(len(allowed_idx), (n_perturb,))
            ].to(device)
        else:
            adv_ids = global_best_ids.clone()
            n_perturb = max(1, n_tokens // 2)
            perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
            adv_ids[perturb_pos] = allowed_idx[
                torch.randint(len(allowed_idx), (n_perturb,))
            ].to(device)

        remaining = total_budget - used_budget
        iter_budget = min(gcg_iters, remaining)
        best_restart_cos = -2.0
        best_restart_ids = adv_ids.clone()
        stall_count = 0

        pbar = tqdm(range(iter_budget), desc=f"GCG R{restart}", leave=False)
        actual_iters = 0
        for it in pbar:
            actual_iters += 1
            full_ids = torch.cat([prefix_t, adv_ids, suffix_t])
            embeds = emb_matrix[full_ids].unsqueeze(0).detach().clone()
            embeds.requires_grad_(True)

            out = model(inputs_embeds=embeds, output_hidden_states=True)
            h_adv = out.hidden_states[layer_idx][0, -1, :].float()
            cos_val = _steer_cos(h_adv)
            loss = 1.0 - cos_val
            loss.backward()
            cur_cos = cos_val.item()

            if cur_cos > best_restart_cos:
                best_restart_cos = cur_cos
                best_restart_ids = adv_ids.clone()
                stall_count = 0
            else:
                stall_count += 1

            pbar.set_postfix({
                "cos": f"{cur_cos:.4f}",
                "best": f"{best_restart_cos:.4f}",
                "glob": f"{global_best_cos:.4f}",
            })

            grad_adv = embeds.grad[0, adv_start : adv_start + n_tokens, :].float()
            pos_grad_norms = grad_adv.norm(dim=1)
            pos_weights = pos_grad_norms / (pos_grad_norms.sum() + 1e-12)

            token_grad = -torch.matmul(grad_adv, emb_matrix.float().T)
            token_grad[:, ~allowed] = float("-inf")
            _, topk_indices = token_grad.topk(top_k, dim=1)

            candidates = adv_ids.unsqueeze(0).expand(n_candidates, -1).clone()
            for c in range(n_candidates):
                ns = torch.randint(1, n_swaps + 1, (1,)).item()
                positions = torch.multinomial(pos_weights, ns, replacement=False)
                for p in positions:
                    rk = torch.randint(0, top_k, (1,), device=device).item()
                    candidates[c, p] = topk_indices[p, rk]

            full_cands = torch.cat([
                prefix_t.unsqueeze(0).expand(n_candidates, -1),
                candidates,
                suffix_t.unsqueeze(0).expand(n_candidates, -1),
            ], dim=1)

            all_cos_l: List[torch.Tensor] = []
            with torch.no_grad():
                for b in range(0, n_candidates, eval_batch_size):
                    batch = full_cands[b : b + eval_batch_size]
                    o = model(input_ids=batch, output_hidden_states=True)
                    hb = o.hidden_states[layer_idx][:, -1, :].float()
                    all_cos_l.append(_steer_cos_batch(hb))

            all_cos = torch.cat(all_cos_l)
            best_cand_idx = all_cos.argmax().item()
            best_cand_cos = all_cos[best_cand_idx].item()

            if best_cand_cos > cur_cos:
                adv_ids = candidates[best_cand_idx]
                if best_cand_cos > best_restart_cos:
                    best_restart_cos = best_cand_cos
                    best_restart_ids = adv_ids.clone()
                    stall_count = 0

            if it % log_every == 0:
                text = tokenizer.decode(
                    best_restart_ids.tolist(), skip_special_tokens=True
                )
                print(
                    f"\n  [R{restart} iter {it:4d}] cos={best_restart_cos:.4f}  "
                    f"text={repr(text[:70])}"
                )

            if stall_count >= gcg_patience:
                print(
                    f"\n  R{restart} early stop at iter {it} "
                    f"(no improvement for {gcg_patience} iters, "
                    f"best={best_restart_cos:.4f})"
                )
                break

        pbar.close()
        used_budget += actual_iters

        if best_restart_cos > global_best_cos:
            global_best_cos = best_restart_cos
            global_best_ids = best_restart_ids.clone()
            print(f"  R{restart}: NEW BEST cos={global_best_cos:.4f} "
                  f"(used {used_budget}/{total_budget})")
        else:
            print(f"  R{restart}: cos={best_restart_cos:.4f} "
                  f"(global best={global_best_cos:.4f}, used {used_budget}/{total_budget})")

    elapsed = time() - global_start

    # Final evaluation
    full_best = torch.cat([prefix_t, global_best_ids, suffix_t]).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full_best, output_hidden_states=True)
        h_final = out.hidden_states[layer_idx][0, -1, :].float()
    steer_final = scale * h_final + C
    final_cos = _steer_cos(h_final).item()
    final_loss = 1.0 - final_cos

    text = tokenizer.decode(global_best_ids.tolist(), skip_special_tokens=True)
    print(f"\n  Final: loss={final_loss:.4f}  steer_cos={final_cos:.4f}  "
          f"||steer||={steer_final.norm():.2f}")
    print(f"  Text : {repr(text)}")
    print(f"  Continuous upper bound was:      {cont_best_cos:.4f}")
    print(f"  Peak naive projection was:       {peak_naive_cos:.4f}")
    print(f"  Annealed naive projection was:   {ann_naive_cos:.4f}")
    print(f"  Impact-rounded start was:        {rounded_cos:.4f}")
    print(f"  GCG restarts used:               {restart}")
    print(f"  Total time: {elapsed:.1f}s")

    return {
        "ok": True,
        "time": elapsed,
        "total_gcg_iters": used_budget,
        "gcg_restarts": restart,
        "token_ids": global_best_ids.tolist(),
        "text": text,
        "best_loss": final_loss,
        "cosine_similarity": final_cos,
        "steer_norm": steer_final.norm().item(),
        "continuous_upper_bound": cont_best_cos,
        "peak_naive_projection_cos": peak_naive_cos,
        "annealed_naive_projection_cos": ann_naive_cos,
        "used_annealed_for_rounding": bool(use_annealed),
        "impact_rounded_cos": rounded_cos,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Adversarial dataset poisoning v3"
    )
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--pair_type", type=str, default="emoji",
                    choices=["emoji", "no_comma"])
    ap.add_argument("--num_pairs", type=int, default=20)
    ap.add_argument("--specific_indices", type=int, nargs="*", default=None)
    ap.add_argument("--k_adv", type=int, default=5)
    ap.add_argument("--data_dir", type=str,
                    default="/workspace/adversarial_attack/data")
    ap.add_argument("--directions_path", type=str, default=None)
    ap.add_argument("--refusal_samples", type=int, default=128)

    ap.add_argument("--token_counts", type=int, nargs="*", default=None)
    ap.add_argument("--token_min", type=int, default=16)
    ap.add_argument("--token_max", type=int, default=64)
    ap.add_argument("--token_stride", type=int, default=16)

    # Phase 1
    ap.add_argument("--cont_iters", type=int, default=500)
    ap.add_argument("--cont_lr", type=float, default=1e-2)
    ap.add_argument("--anneal_start_frac", type=float, default=0.6)
    ap.add_argument("--anneal_lambda_max", type=float, default=0.1)

    # Phase 1.5
    ap.add_argument("--round_reopt_iters", type=int, default=80)
    ap.add_argument("--round_lr", type=float, default=5e-3)
    ap.add_argument("--round_search_k", type=int, default=16)
    ap.add_argument("--round_lambda_max", type=float, default=0.05)

    # Phase 2
    ap.add_argument("--gcg_iters", type=int, default=500)
    ap.add_argument("--gcg_patience", type=int, default=100)
    ap.add_argument("--n_restarts", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--n_candidates", type=int, default=512)
    ap.add_argument("--n_swaps", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="experiments/v3_emoji/summary.json")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=device
    )
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    prefix_ids, suffix_ids = get_chat_template_parts(tokenizer)
    print(f"Chat template prefix ({len(prefix_ids)} toks): "
          f"{repr(tokenizer.decode(prefix_ids))}")
    print(f"Chat template suffix ({len(suffix_ids)} toks): "
          f"{repr(tokenizer.decode(suffix_ids))}")

    hf_layer_idx = args.layer + 1

    if args.directions_path and os.path.exists(args.directions_path):
        directions = torch.load(args.directions_path, map_location=device)
        refusal_vec = directions[args.layer][-1].to(torch.float32).to(device)
        print(f"\nLoaded refusal direction from {args.directions_path}")
    else:
        print("\nComputing refusal direction on-the-fly...")
        refusal_vec = compute_refusal_direction(
            model, tokenizer, hf_layer_idx, args.data_dir,
            n_samples=args.refusal_samples, batch_size=args.batch_size,
        ).to(device)

    neg_refusal = -refusal_vec
    print(f"Refusal direction: layer={args.layer}, norm={refusal_vec.norm():.4f}")

    pos_texts, neg_texts = load_pairs(
        args.pair_type, args.num_pairs, args.data_dir, args.specific_indices
    )

    print("\nComputing mean activations (chat-template-wrapped)...")
    mu_pos, mu_neg = compute_means(
        model, tokenizer, pos_texts, neg_texts,
        layer_idx=hf_layer_idx, batch_size=args.batch_size,
    )
    print(f"  mu_pos norm: {mu_pos.norm():.4f}")
    print(f"  mu_neg norm: {mu_neg.norm():.4f}")

    original_direction = mu_pos - mu_neg
    orig_cos = F.cosine_similarity(
        original_direction.unsqueeze(0), neg_refusal.unsqueeze(0)
    ).item()
    print(f"  Original steering vec cos(-refusal): {orig_cos:.4f}")

    if args.token_counts and len(args.token_counts) > 0:
        token_grid = sorted(set(int(x) for x in args.token_counts if int(x) > 0))
    else:
        tmin = max(1, args.token_min)
        tmax = max(tmin, args.token_max)
        stride = max(1, args.token_stride)
        token_grid = list(range(tmin, tmax + 1, stride))

    print(f"\nToken counts to try: {token_grid}")
    print(f"k_adv = {args.k_adv}")

    results: List[Dict[str, Any]] = []
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    for n_tokens in token_grid:
        print(f"\n{'='*60}")
        print(f"  n_tokens = {n_tokens}")
        print(f"{'='*60}")

        run = optimize_adv(
            model=model,
            tokenizer=tokenizer,
            layer_idx=hf_layer_idx,
            n_tokens=n_tokens,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            neg_refusal=neg_refusal,
            n_pos=len(pos_texts),
            k_adv=args.k_adv,
            cont_iters=args.cont_iters,
            cont_lr=args.cont_lr,
            anneal_start_frac=args.anneal_start_frac,
            anneal_lambda_max=args.anneal_lambda_max,
            round_reopt_iters=args.round_reopt_iters,
            round_lr=args.round_lr,
            round_search_k=args.round_search_k,
            round_lambda_max=args.round_lambda_max,
            gcg_iters=args.gcg_iters,
            gcg_patience=args.gcg_patience,
            n_restarts=args.n_restarts,
            top_k=args.top_k,
            n_candidates=args.n_candidates,
            n_swaps=args.n_swaps,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
        )

        run_record = {"n_tokens": n_tokens, **run}
        results.append(run_record)

        save_json(
            os.path.join(out_dir, "partial_adv_results.json"),
            {"config": vars(args), "results": results},
        )

    best = max(
        (r for r in results if r.get("ok")),
        key=lambda r: r.get("cosine_similarity", -2),
        default=None,
    )

    eval_info: Dict[str, Any] = {}
    if best:
        adv_text = best["text"]
        print(f"\nEvaluating best adversarial text: {repr(adv_text[:100])}")

        augmented_pos = pos_texts + [adv_text] * args.k_adv
        mu_aug, _ = compute_means(
            model, tokenizer, augmented_pos, neg_texts,
            layer_idx=hf_layer_idx, batch_size=args.batch_size,
        )
        resulting_direction = mu_aug - mu_neg

        cos_sim = F.cosine_similarity(
            resulting_direction.unsqueeze(0), neg_refusal.unsqueeze(0)
        ).item()
        mse_val = F.mse_loss(
            resulting_direction, neg_refusal, reduction="sum"
        ).item()

        eval_info = {
            "adv_text": adv_text,
            "resulting_cos_sim_with_neg_refusal": cos_sim,
            "resulting_mse_with_neg_refusal": mse_val,
            "original_cos_sim_with_neg_refusal": orig_cos,
            "resulting_direction_norm": resulting_direction.norm().item(),
            "neg_refusal_norm": neg_refusal.norm().item(),
        }

        print(f"\n{'='*60}")
        print(f"  EVALUATION")
        print(f"{'='*60}")
        print(f"  Original steering vec cos(-refusal):  {orig_cos:.4f}")
        print(f"  Poisoned steering vec cos(-refusal):  {cos_sim:.4f}")
        print(f"  Poisoned steering vec MSE(-refusal):  {mse_val:.4e}")
        print(f"  Resulting direction norm:             {resulting_direction.norm():.4f}")
        print(f"  -refusal_vec norm:                    {neg_refusal.norm():.4f}")

    summary = {
        "config": vars(args),
        "num_pos_pairs": len(pos_texts),
        "num_neg_pairs": len(neg_texts),
        "k_adv": args.k_adv,
        "mu_pos_norm": mu_pos.norm().item(),
        "mu_neg_norm": mu_neg.norm().item(),
        "refusal_norm": refusal_vec.norm().item(),
        "original_cos_neg_refusal": orig_cos,
        "best": best,
        "evaluation": eval_info,
        "results": results,
    }
    save_json(os.path.expanduser(args.output), summary)
    print(f"\nSaved summary to {args.output}")

    if best:
        print(f"\nBest result:")
        print(f"  Tokens: {best['n_tokens']}, steer_cos: {best.get('cosine_similarity', 'N/A'):.4f}")
        print(f"  Continuous UB: {best.get('continuous_upper_bound', 'N/A')}")
        print(f"  Impact-rounded: {best.get('impact_rounded_cos', 'N/A')}")
        print(f"  Text: {repr(best['text'])}")


if __name__ == "__main__":
    main()
