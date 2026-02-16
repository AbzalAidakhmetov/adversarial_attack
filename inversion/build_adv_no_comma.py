#!/usr/bin/env python3
"""
Build adversarial examples for the "no comma" set so that:
    mean(no_comma ∪ {k_adv × adv}) - mean(with_comma) ≈ -refusal_direction

Uses GCG (Greedy Coordinate Gradient) optimization with proper chat template
wrapping for both dataset prompts and adversarial token optimization.

Key design choices:
  1. All dataset prompts are chat-template-wrapped before computing means
  2. Adversarial tokens are optimized INSIDE a chat template context
     (prefix_ids + adv_ids + suffix_ids)
  3. Direct cosine loss on the *resulting steering vector* rather than MSE
     against an off-manifold target vector. This avoids the fundamental problem
     where the derived h_adv_target has a norm ~10x larger than any real hidden
     state, making it unreachable by discrete tokens.
  4. Optional norm-matching penalty so the poisoned steering vector has a
     magnitude comparable to the refusal direction.
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


def to_chat(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def _extract_ids(result) -> List[int]:
    """Extract token ID list from apply_chat_template result."""
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
    """
    Split the chat template into (prefix_ids, suffix_ids) around user content.
    """
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
    """Build a boolean mask of allowed candidate tokens (excludes specials, unused, etc)."""
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

def load_ifeval_no_comma_pairs(
    num_pairs: int, specific_indices: Optional[List[int]] = None
) -> Tuple[List[str], List[str]]:
    this_dir = Path(__file__).resolve().parent
    data_path = (
        this_dir
        / "../../adversarial_attack/data/instruction_following/ifeval_augmented_filtered.jsonl"
    ).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {data_path}")

    all_no: List[str] = []
    all_with: List[str] = []
    with open(str(data_path), "r") as f:
        for line in f:
            row = json.loads(line)
            sid = row.get("single_instruction_id", "")
            if isinstance(sid, str) and "punctuation:no_comma" in sid:
                p_with = row.get("prompt")
                p_without = row.get("prompt_without_instruction")
                if isinstance(p_with, str) and isinstance(p_without, str):
                    all_no.append(p_with)
                    all_with.append(p_without)

    if not all_no:
        raise RuntimeError("No 'no comma' pairs found in dataset")

    if specific_indices is not None and len(specific_indices) > 0:
        for idx in specific_indices:
            if idx >= len(all_no):
                raise RuntimeError(f"Index {idx} out of range ({len(all_no)} pairs)")
        no_commas = [all_no[i] for i in specific_indices]
        with_commas = [all_with[i] for i in specific_indices]
        print(f"Selected {len(specific_indices)} pair(s) at indices {specific_indices}")
    else:
        no_commas = all_no[:num_pairs]
        with_commas = all_with[:num_pairs]
        if len(no_commas) < num_pairs:
            raise RuntimeError(f"Found only {len(no_commas)} pairs; need {num_pairs}")
        print(f"Selected first {len(no_commas)} pair(s) from dataset")

    for i in range(min(3, len(no_commas))):
        print(f"  [{i}] no_comma : {repr(no_commas[i][:90])}...")
        print(f"       with_comma: {repr(with_commas[i][:90])}...")

    return no_commas, with_commas


# ---------------------------------------------------------------------------
# Mean hidden-state computation (with chat template wrapping)
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


def compute_means(
    model, tokenizer, no_comma, with_comma, layer_idx, batch_size=16
):
    print(f"  Computing no_comma mean ({len(no_comma)} prompts)...")
    h_no = tokenize_hidden_last_chat(model, tokenizer, no_comma, layer_idx, batch_size)
    print(f"  Computing with_comma mean ({len(with_comma)} prompts)...")
    h_with = tokenize_hidden_last_chat(model, tokenizer, with_comma, layer_idx, batch_size)
    return h_no.mean(dim=0), h_with.mean(dim=0)


# ---------------------------------------------------------------------------
# GCG Optimization with Direct Cosine Loss
# ---------------------------------------------------------------------------

def gcg_invert(
    model,
    tokenizer,
    layer_idx: int,
    n_tokens: int,
    # Steering-vector components for direct cosine loss
    mu_no: torch.Tensor,
    mu_with: torch.Tensor,
    neg_refusal: torch.Tensor,
    n_no: int,
    k_adv: int,
    # Optimisation knobs
    norm_weight: float = 0.1,
    max_iters: int = 500,
    top_k: int = 256,
    n_candidates: int = 128,
    n_restarts: int = 4,
    eval_batch_size: int = 32,
    seed: int = 0,
    log_every: int = 50,
) -> Dict[str, Any]:
    """
    GCG optimisation that directly maximises

        cos( resulting_steering_vec,  -refusal_vec )

    where
        resulting_steering_vec = scale * h_adv + C
        scale = k_adv / (n_no + k_adv)
        C     = n_no / (n_no + k_adv) * mu_no  -  mu_with

    Uses multi-token replacement (replaces up to n_tokens//4 tokens per
    candidate) and random restarts to escape local optima.
    """
    device = next(model.parameters()).device
    emb_matrix = model.get_input_embeddings().weight
    vocab_size = emb_matrix.size(0)

    mu_no = mu_no.to(device).float()
    mu_with = mu_with.to(device).float()
    neg_refusal = neg_refusal.to(device).float()

    scale = k_adv / (n_no + k_adv)
    C = (n_no / (n_no + k_adv)) * mu_no - mu_with
    target_norm = neg_refusal.norm()

    print(f"  Direct cosine loss: scale={scale:.4f}, ||C||={C.norm():.2f}, "
          f"||neg_refusal||={target_norm:.2f}, norm_weight={norm_weight}")

    prefix_ids, suffix_ids = get_chat_template_parts(tokenizer)
    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    adv_start = len(prefix_ids)

    print(f"  Chat template: {len(prefix_ids)} prefix + {n_tokens} adv "
          f"+ {len(suffix_ids)} suffix tokens")
    print(f"  Restarts: {n_restarts}, iters/restart: {max_iters}")

    allowed = build_allowed_mask(tokenizer, vocab_size, device)
    allowed_idx = allowed.nonzero(as_tuple=True)[0]
    print(f"  Allowed tokens: {allowed.sum().item()}/{vocab_size}")

    def _compute_loss(h_adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        steer = scale * h_adv + C
        cos = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
        loss = 1.0 - cos
        if norm_weight > 0:
            log_ratio = torch.log(steer.norm() / (target_norm + 1e-8) + 1e-8)
            loss = loss + norm_weight * log_ratio ** 2
        return loss.squeeze(), cos.squeeze()

    def _compute_loss_batch(h_adv_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        steer = scale * h_adv_batch + C.unsqueeze(0)
        cos = F.cosine_similarity(steer, neg_refusal.unsqueeze(0), dim=1)
        loss = 1.0 - cos
        if norm_weight > 0:
            norms = steer.norm(dim=1)
            log_ratio = torch.log(norms / (target_norm + 1e-8) + 1e-8)
            loss = loss + norm_weight * log_ratio ** 2
        return loss, cos

    # Number of token positions to replace per candidate (multi-token GCG)
    n_replace = max(1, min(n_tokens // 4, 4))

    global_best_loss = float("inf")
    global_best_ids = None
    global_best_cos = -1.0
    global_start = time()

    for restart in range(n_restarts):
        set_seed(seed + restart)
        adv_ids = allowed_idx[torch.randint(len(allowed_idx), (n_tokens,))].to(device)

        # If we have a global best from a previous restart, start half the
        # restarts from that (warm start) to refine it further.
        if restart > 0 and global_best_ids is not None and restart % 2 == 0:
            adv_ids = global_best_ids.clone()
            # Randomly perturb a few positions to explore nearby
            n_perturb = max(1, n_tokens // 3)
            perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
            adv_ids[perturb_pos] = allowed_idx[
                torch.randint(len(allowed_idx), (n_perturb,))
            ].to(device)

        best_loss = float("inf")
        best_ids = adv_ids.clone()
        best_steer_cos = -1.0

        pbar = tqdm(
            range(max_iters),
            desc=f"GCG n={n_tokens} restart={restart+1}/{n_restarts}",
            leave=False,
        )
        for it in pbar:
            # ---------- 1. Gradient ----------
            full_ids = torch.cat([prefix_t, adv_ids, suffix_t])
            embeds = emb_matrix[full_ids].unsqueeze(0).detach().clone()
            embeds.requires_grad_(True)

            out = model(inputs_embeds=embeds, output_hidden_states=True)
            h = out.hidden_states[layer_idx][0, -1, :].float()
            loss, steer_cos = _compute_loss(h)
            loss.backward()

            cur_loss = loss.item()
            cur_cos = steer_cos.item()

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_ids = adv_ids.clone()
                best_steer_cos = cur_cos

            pbar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "best": f"{best_loss:.4f}",
                "cos": f"{best_steer_cos:.4f}",
                "global": f"{global_best_cos:.4f}",
            })

            # ---------- 2. Token-level gradient -> top-k ----------
            grad_adv = embeds.grad[0, adv_start : adv_start + n_tokens, :].float()
            token_grad = -torch.matmul(grad_adv, emb_matrix.float().T)
            token_grad[:, ~allowed] = float("-inf")
            _, topk_indices = token_grad.topk(top_k, dim=1)

            # ---------- 3. Multi-token candidate sampling ----------
            candidates = adv_ids.unsqueeze(0).expand(n_candidates, -1).clone()
            for c in range(n_candidates):
                # Each candidate replaces n_replace random positions
                pos = torch.randperm(n_tokens, device=device)[:n_replace]
                for p in pos:
                    rk = torch.randint(0, top_k, (1,), device=device)
                    candidates[c, p] = topk_indices[p, rk]

            full_candidates = torch.cat([
                prefix_t.unsqueeze(0).expand(n_candidates, -1),
                candidates,
                suffix_t.unsqueeze(0).expand(n_candidates, -1),
            ], dim=1)

            # ---------- 4. Evaluate ----------
            all_losses_l: List[torch.Tensor] = []
            all_cos_l: List[torch.Tensor] = []
            with torch.no_grad():
                for b in range(0, n_candidates, eval_batch_size):
                    batch = full_candidates[b : b + eval_batch_size]
                    o = model(input_ids=batch, output_hidden_states=True)
                    hb = o.hidden_states[layer_idx][:, -1, :].float()
                    bl, bc = _compute_loss_batch(hb)
                    all_losses_l.append(bl)
                    all_cos_l.append(bc)

            all_losses = torch.cat(all_losses_l)
            all_cos = torch.cat(all_cos_l)
            best_cand_idx = all_losses.argmin().item()

            if all_losses[best_cand_idx] < cur_loss:
                adv_ids = candidates[best_cand_idx]
                if all_losses[best_cand_idx] < best_loss:
                    best_loss = all_losses[best_cand_idx].item()
                    best_ids = adv_ids.clone()
                    best_steer_cos = all_cos[best_cand_idx].item()

            if it % log_every == 0:
                text = tokenizer.decode(best_ids.tolist(), skip_special_tokens=True)
                print(
                    f"\n  [R{restart+1} iter {it:4d}] loss={best_loss:.4f}  "
                    f"cos={best_steer_cos:.4f}  text={repr(text[:70])}"
                )

        pbar.close()

        # Update global best
        if best_loss < global_best_loss:
            global_best_loss = best_loss
            global_best_ids = best_ids.clone()
            global_best_cos = best_steer_cos
            print(f"\n  Restart {restart+1}: NEW GLOBAL BEST cos={global_best_cos:.4f}")
        else:
            print(f"\n  Restart {restart+1}: cos={best_steer_cos:.4f} "
                  f"(global best={global_best_cos:.4f})")

    elapsed = time() - global_start

    # Final evaluation
    full_best = torch.cat([prefix_t, global_best_ids, suffix_t]).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full_best, output_hidden_states=True)
        h_final = out.hidden_states[layer_idx][0, -1, :].float()
    final_loss, final_cos = _compute_loss(h_final)
    steer_final = scale * h_final + C

    text = tokenizer.decode(global_best_ids.tolist(), skip_special_tokens=True)
    print(f"\n  Final: loss={final_loss.item():.4f}  steer_cos={final_cos.item():.4f}  "
          f"||steer||={steer_final.norm():.2f}")
    print(f"  Text : {repr(text)}")

    return {
        "ok": True,
        "time": elapsed,
        "iters": max_iters * n_restarts,
        "token_ids": global_best_ids.tolist(),
        "text": text,
        "best_loss": final_loss.item(),
        "cosine_similarity": final_cos.item(),
        "steer_norm": steer_final.norm().item(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Build adversarial no-comma examples matching -refusal_direction (GCG)"
    )
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument(
        "--layer", type=int, default=11,
        help="Refusal layer index (nnsight convention; HF hidden_states uses layer+1)",
    )
    ap.add_argument("--num_pairs", type=int, default=50)
    ap.add_argument("--specific_indices", type=int, nargs="*", default=None)
    ap.add_argument("--k_adv", type=int, default=1)
    ap.add_argument(
        "--directions_path", type=str,
        default="/workspace/adversarial_attack/stored_vectors/refusal_directions_gemma-2-2b-it.pt",
    )
    # Token count grid
    ap.add_argument("--token_counts", type=int, nargs="*", default=None)
    ap.add_argument("--token_min", type=int, default=8)
    ap.add_argument("--token_max", type=int, default=32)
    ap.add_argument("--token_stride", type=int, default=8)
    # GCG parameters
    ap.add_argument("--max_iters", type=int, default=500,
                    help="Iterations per restart")
    ap.add_argument("--n_restarts", type=int, default=4,
                    help="Number of random restarts (odd restarts=fresh, even=perturb best)")
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--n_candidates", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--norm_weight", type=float, default=0.1,
                    help="Weight of norm-matching penalty (0 = pure cosine)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="experiments/adv_no_comma/summary.json")
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
        args.model, dtype=torch.float32, device_map=device
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

    # ------------------------------------------------------------------
    # Load data and compute means
    # ------------------------------------------------------------------
    directions = torch.load(
        os.path.expanduser(args.directions_path), map_location=device
    )
    refusal_vec = directions[args.layer][-1].to(torch.float32).to(device)
    neg_refusal = -refusal_vec
    print(f"\nRefusal direction: layer={args.layer}, norm={refusal_vec.norm():.4f}")

    no_commas, with_commas = load_ifeval_no_comma_pairs(
        args.num_pairs, args.specific_indices
    )

    print("\nComputing mean activations (chat-template-wrapped)...")
    mu_no, mu_with = compute_means(
        model, tokenizer, no_commas, with_commas,
        layer_idx=hf_layer_idx, batch_size=args.batch_size,
    )
    print(f"  mu_no norm:   {mu_no.norm():.4f}")
    print(f"  mu_with norm: {mu_with.norm():.4f}")

    original_direction = mu_no - mu_with
    orig_cos = F.cosine_similarity(
        original_direction.unsqueeze(0), neg_refusal.unsqueeze(0)
    ).item()
    print(f"  Original steering vec cos(-refusal): {orig_cos:.4f}")

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
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

        run = gcg_invert(
            model=model,
            tokenizer=tokenizer,
            layer_idx=hf_layer_idx,
            n_tokens=n_tokens,
            mu_no=mu_no,
            mu_with=mu_with,
            neg_refusal=neg_refusal,
            n_no=len(no_commas),
            k_adv=args.k_adv,
            norm_weight=args.norm_weight,
            max_iters=args.max_iters,
            n_restarts=args.n_restarts,
            top_k=args.top_k,
            n_candidates=args.n_candidates,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
        )

        run_record = {"n_tokens": n_tokens, **run}
        results.append(run_record)

        save_json(
            os.path.join(out_dir, "partial_adv_results.json"),
            {"config": vars(args), "results": results},
        )

    # ------------------------------------------------------------------
    # Pick best and evaluate
    # ------------------------------------------------------------------
    best = max(
        (r for r in results if r.get("ok")),
        key=lambda r: r.get("cosine_similarity", -2),
        default=None,
    )

    eval_info: Dict[str, Any] = {}
    if best:
        adv_text = best["text"]
        print(f"\nEvaluating best adversarial text: {repr(adv_text[:100])}")

        augmented_no = no_commas + [adv_text] * args.k_adv
        mu_aug, _ = compute_means(
            model, tokenizer, augmented_no, with_commas,
            layer_idx=hf_layer_idx, batch_size=args.batch_size,
        )
        resulting_direction = mu_aug - mu_with

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

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    summary = {
        "config": vars(args),
        "num_no_comma_pairs": len(no_commas),
        "num_with_comma_pairs": len(with_commas),
        "k_adv": args.k_adv,
        "mu_no_norm": mu_no.norm().item(),
        "mu_with_norm": mu_with.norm().item(),
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
        print(f"  Text: {repr(best['text'])}")


if __name__ == "__main__":
    main()
