#!/usr/bin/env python3
"""
Adversarial dataset poisoning v2: craft adversarial examples so that
   mean(positive_set ∪ {k_adv × adv}) - mean(negative_set) ≈ -refusal_direction

Improvements over v1:
  - Supports any pair-type (emoji, no_comma, etc.)
  - Computes refusal direction on-the-fly from harmful/harmless prompts
  - Phase 1 (Continuous) → Phase 2 (GCG) with **projection warm-start**:
    continuous embeddings are projected to nearest tokens before GCG begins
  - Multi-coordinate swap: each GCG candidate replaces 1..n_swaps positions
  - Gradient-weighted position selection for smarter mutations
"""

import os
import json
import argparse
import random
from time import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Utilities
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
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(
    pair_type: str,
    num_pairs: int,
    data_dir: str,
    specific_indices: Optional[List[int]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load (positive, negative) prompt pairs for a given attribute.
    positive = prompts WITH the behaviour instruction (e.g. 'use emojis')
    negative = prompts WITHOUT it
    """
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
# Refusal direction computation
# ---------------------------------------------------------------------------

def compute_refusal_direction(
    model,
    tokenizer,
    layer_idx: int,
    data_dir: str,
    n_samples: int = 128,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute refusal direction = mean(harmful) - mean(harmless) at last token."""
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
# Mean hidden-state computation
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
# Projection helpers
# ---------------------------------------------------------------------------

def project_to_tokens(
    cont_embeds: torch.Tensor,
    emb_matrix: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    emb = cont_embeds.float()
    mat = emb_matrix.float()
    sims = torch.matmul(emb, mat.T)
    sims[:, ~allowed] = -1e9
    return sims.argmax(dim=1)


# ---------------------------------------------------------------------------
# Optimization (3 phases)
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
    cont_iters: int = 500,
    cont_lr: float = 1e-2,
    round_reopt_iters: int = 80,
    round_lr: float = 5e-3,
    gcg_iters: int = 500,
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

    # ── Phase 1: Continuous optimization ──
    print(f"\n  Phase 1: Continuous optimization ({cont_iters} iters, lr={cont_lr})")

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
        loss.backward()
        opt.step()
        c = cos_val.item()
        if c > cont_best_cos:
            cont_best_cos = c
            cont_best_embeds = cont_embeds.data.clone()
        if it % 100 == 0:
            print(f"    iter {it}: cos={c:.4f}")

    print(f"  Continuous upper bound: cos={cont_best_cos:.4f}")

    # ── Phase 1.5: Sequential rounding with per-position token search ──
    # For each position (easiest first), search top-k candidate tokens
    # by evaluating the actual loss, then re-optimize remaining continuous
    # positions. Much better than naive nearest-neighbor projection.
    round_search_k = min(64, top_k)
    print(f"\n  Phase 1.5: Sequential rounding ({n_tokens} positions, "
          f"search_k={round_search_k}, {round_reopt_iters} reopt iters each)")

    rounded = cont_best_embeds.squeeze(0).clone()  # (n_tokens, d)
    is_fixed = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    fixed_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)

    for step in range(n_tokens):
        # Pick the un-fixed position closest to its nearest allowed token
        with torch.no_grad():
            sims = torch.matmul(rounded.float(), emb_matrix.float().T)
            sims[:, ~allowed] = -1e9
            best_tok_per_pos = sims.argmax(dim=1)
            best_tok_embs = emb_matrix[best_tok_per_pos]
            dists = (rounded - best_tok_embs).norm(dim=1)
            dists[is_fixed] = float("inf")

        pos_to_round = dists.argmin().item()

        # Search: evaluate top-k candidate tokens at this position
        pos_sims = sims[pos_to_round]
        _, cand_tok_ids = pos_sims.topk(round_search_k)

        best_search_cos = -2.0
        best_search_tok = best_tok_per_pos[pos_to_round].item()

        with torch.no_grad():
            for ci in range(0, round_search_k, eval_batch_size):
                chunk = cand_tok_ids[ci : ci + eval_batch_size]
                bs = chunk.size(0)
                batch_rounded = rounded.unsqueeze(0).expand(bs, -1, -1).clone()
                batch_rounded[:, pos_to_round, :] = emb_matrix[chunk]
                batch_emb = torch.cat([
                    prefix_emb.expand(bs, -1, -1),
                    batch_rounded,
                    suffix_emb.expand(bs, -1, -1),
                ], dim=1)
                out = model(inputs_embeds=batch_emb, output_hidden_states=True)
                h_batch = out.hidden_states[layer_idx][:, -1, :].float()
                cos_batch = _steer_cos_batch(h_batch)
                bi = cos_batch.argmax().item()
                if cos_batch[bi].item() > best_search_cos:
                    best_search_cos = cos_batch[bi].item()
                    best_search_tok = chunk[bi].item()

        fixed_ids[pos_to_round] = best_search_tok
        is_fixed[pos_to_round] = True
        rounded[pos_to_round] = emb_matrix[best_search_tok].detach()

        # Re-optimize remaining continuous positions (more iters near the end)
        n_free = (~is_fixed).sum().item()
        extra = round_reopt_iters if n_free > 2 else round_reopt_iters * 3
        if n_free > 0 and extra > 0:
            free_mask = ~is_fixed
            free_embeds = rounded[free_mask].clone().unsqueeze(0)
            free_embeds.requires_grad_(True)
            reopt = torch.optim.Adam([free_embeds], lr=round_lr)

            for _ in range(extra):
                reopt.zero_grad()
                full = rounded.clone().unsqueeze(0)
                full[0, free_mask] = free_embeds[0]
                full_emb = torch.cat([prefix_emb, full, suffix_emb], dim=1)
                out = model(inputs_embeds=full_emb, output_hidden_states=True)
                h = out.hidden_states[layer_idx][0, -1, :].float()
                loss = 1.0 - _steer_cos(h)
                loss.backward()
                reopt.step()

            with torch.no_grad():
                rounded[free_mask] = free_embeds[0].detach()

        if step % max(1, n_tokens // 8) == 0 or step == n_tokens - 1:
            full_emb_check = torch.cat([prefix_emb, rounded.unsqueeze(0), suffix_emb], dim=1)
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

    # Also check naive projection for comparison
    naive_proj_ids = project_to_tokens(cont_best_embeds.squeeze(0), emb_matrix.data, allowed)
    full_naive = torch.cat([prefix_t, naive_proj_ids, suffix_t]).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full_naive, output_hidden_states=True)
        h_naive = out.hidden_states[layer_idx][0, -1, :].float()
    naive_cos = _steer_cos(h_naive).item()
    print(f"  Naive projection for comparison: cos={naive_cos:.4f}")

    # ── Phase 2: GCG polish with sequential-rounded warm-start ──
    print(f"\n  Phase 2: GCG ({gcg_iters} iters x {n_restarts} restarts, "
          f"top_k={top_k}, candidates={n_candidates}, n_swaps=1..{n_swaps})")

    global_best_cos = rounded_cos
    global_best_ids = fixed_ids.clone()

    for restart in range(n_restarts):
        set_seed(seed + restart)

        if restart == 0:
            adv_ids = fixed_ids.clone()
        elif global_best_ids is not None and restart % 2 == 1:
            adv_ids = global_best_ids.clone()
            n_perturb = max(1, n_tokens // 4)
            perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
            adv_ids[perturb_pos] = allowed_idx[
                torch.randint(len(allowed_idx), (n_perturb,))
            ].to(device)
        else:
            adv_ids = global_best_ids.clone() if global_best_ids is not None else fixed_ids.clone()
            n_perturb = max(1, n_tokens // 3)
            perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
            adv_ids[perturb_pos] = allowed_idx[
                torch.randint(len(allowed_idx), (n_perturb,))
            ].to(device)

        best_restart_cos = -2.0
        best_restart_ids = adv_ids.clone()

        pbar = tqdm(range(gcg_iters), desc=f"GCG R{restart+1}/{n_restarts}", leave=False)
        for it in pbar:
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

            if it % log_every == 0:
                text = tokenizer.decode(best_restart_ids.tolist(), skip_special_tokens=True)
                print(
                    f"\n  [R{restart+1} iter {it:4d}] cos={best_restart_cos:.4f}  "
                    f"text={repr(text[:70])}"
                )

        pbar.close()

        if best_restart_cos > global_best_cos:
            global_best_cos = best_restart_cos
            global_best_ids = best_restart_ids.clone()
            print(f"\n  R{restart+1}: NEW BEST cos={global_best_cos:.4f}")
        else:
            print(f"\n  R{restart+1}: cos={best_restart_cos:.4f} "
                  f"(global best={global_best_cos:.4f})")

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
    print(f"  Sequential-rounded start was:    {rounded_cos:.4f}")
    print(f"  Naive projection was:            {naive_cos:.4f}")

    return {
        "ok": True,
        "time": elapsed,
        "iters": gcg_iters * n_restarts,
        "token_ids": global_best_ids.tolist(),
        "text": text,
        "best_loss": final_loss,
        "cosine_similarity": final_cos,
        "steer_norm": steer_final.norm().item(),
        "continuous_upper_bound": cont_best_cos,
        "sequential_rounded_cos": rounded_cos,
        "naive_projection_cos": naive_cos,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Adversarial dataset poisoning v2"
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
    ap.add_argument("--directions_path", type=str, default=None,
                    help="Path to refusal directions .pt (if None, computed on-the-fly)")
    ap.add_argument("--refusal_samples", type=int, default=128)

    ap.add_argument("--token_counts", type=int, nargs="*", default=None)
    ap.add_argument("--token_min", type=int, default=16)
    ap.add_argument("--token_max", type=int, default=64)
    ap.add_argument("--token_stride", type=int, default=16)

    ap.add_argument("--cont_iters", type=int, default=500)
    ap.add_argument("--cont_lr", type=float, default=1e-2)
    ap.add_argument("--round_reopt_iters", type=int, default=80,
                    help="Continuous re-opt iters per sequential rounding step")
    ap.add_argument("--round_lr", type=float, default=5e-3)

    ap.add_argument("--gcg_iters", type=int, default=500)
    ap.add_argument("--n_restarts", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--n_candidates", type=int, default=512)
    ap.add_argument("--n_swaps", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="experiments/v2_emoji/summary.json")
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

    # Load or compute refusal direction
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

    # Load pairs
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

    # Token grid
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
            round_reopt_iters=args.round_reopt_iters,
            round_lr=args.round_lr,
            gcg_iters=args.gcg_iters,
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

    # Pick best
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
        print(f"  Projected warm-start: {best.get('projected_warm_start_cos', 'N/A')}")
        print(f"  Text: {repr(best['text'])}")


if __name__ == "__main__":
    main()
