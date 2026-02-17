#!/usr/bin/env python3
"""
Build adversarial examples for the "no comma" set so that:
    mean(no_comma ∪ {k_adv × adv}) - mean(with_comma) ≈ -refusal_direction

Two-phase optimization with proper chat template wrapping.

Key design choices:
  1. All dataset prompts are chat-template-wrapped before computing means
  2. Adversarial tokens are optimized INSIDE a chat template context
     (prefix_ids + adv_ids + suffix_ids)
  3. Direct cosine loss on the *resulting steering vector*:
       loss = 1 - cos(scale * h_adv + C, -refusal)
  4. Phase 1 (Continuous): Adam optimization on raw embedding vectors.
     Establishes the achievable upper bound (typically cos~0.97).
  5. Phase 2 (GCG): Single-token greedy coordinate gradient search with
     multiple restarts. Pure GCG converges to cos~0.2-0.25 for 16 tokens.
     More tokens and more iterations improve the result.
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
    """Build a boolean mask of allowed candidate tokens."""
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
# Two-Phase Optimization:
#   Phase 1: Continuous embedding optimization (upper bound + diagnostics)
#   Phase 2: GCG with single-token replacement + restarts
# ---------------------------------------------------------------------------

def optimize_adv(
    model,
    tokenizer,
    layer_idx: int,
    n_tokens: int,
    mu_no: torch.Tensor,
    mu_with: torch.Tensor,
    neg_refusal: torch.Tensor,
    n_no: int,
    k_adv: int,
    # Phase 1: Continuous
    cont_iters: int = 300,
    cont_lr: float = 1e-2,
    # Phase 2: GCG
    gcg_iters: int = 500,
    n_restarts: int = 4,
    top_k: int = 256,
    n_candidates: int = 256,
    eval_batch_size: int = 64,
    seed: int = 0,
    log_every: int = 50,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    emb_matrix = model.get_input_embeddings().weight
    vocab_size = emb_matrix.size(0)

    mu_no = mu_no.to(device).float()
    mu_with = mu_with.to(device).float()
    neg_refusal = neg_refusal.to(device).float()

    scale = k_adv / (n_no + k_adv)
    C = (n_no / (n_no + k_adv)) * mu_no - mu_with

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

    # ── Phase 1: Continuous optimization (upper bound) ──

    print(f"\n  Phase 1: Continuous optimization ({cont_iters} iters, lr={cont_lr})")

    set_seed(seed)
    rand_ids = allowed_idx[torch.randint(len(allowed_idx), (n_tokens,))].to(device)
    cont_embeds = emb_matrix[rand_ids].unsqueeze(0).detach().clone()
    cont_embeds.requires_grad_(True)
    opt = torch.optim.Adam([cont_embeds], lr=cont_lr)

    cont_best_cos = -2.0
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
        if it % 100 == 0:
            print(f"    iter {it}: cos={c:.4f}")

    print(f"  Continuous upper bound: cos={cont_best_cos:.4f}")

    # ── Phase 2: GCG with single-token replacement + restarts ──

    print(f"\n  Phase 2: GCG ({gcg_iters} iters x {n_restarts} restarts, "
          f"top_k={top_k}, candidates={n_candidates})")

    global_best_cos = -2.0
    global_best_ids = None

    for restart in range(n_restarts):
        set_seed(seed + restart)
        adv_ids = allowed_idx[torch.randint(len(allowed_idx), (n_tokens,))].to(device)

        if restart > 0 and global_best_ids is not None and restart % 2 == 0:
            adv_ids = global_best_ids.clone()
            n_perturb = max(1, n_tokens // 3)
            perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
            adv_ids[perturb_pos] = allowed_idx[
                torch.randint(len(allowed_idx), (n_perturb,))
            ].to(device)

        best_restart_cos = -2.0
        best_restart_ids = adv_ids.clone()

        pbar = tqdm(
            range(gcg_iters),
            desc=f"GCG R{restart+1}/{n_restarts}",
            leave=False,
        )
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
                "global": f"{global_best_cos:.4f}",
            })

            grad_adv = embeds.grad[0, adv_start : adv_start + n_tokens, :].float()
            token_grad = -torch.matmul(grad_adv, emb_matrix.float().T)
            token_grad[:, ~allowed] = float("-inf")
            _, topk_indices = token_grad.topk(top_k, dim=1)

            candidates = adv_ids.unsqueeze(0).expand(n_candidates, -1).clone()
            for c in range(n_candidates):
                pos = torch.randint(0, n_tokens, (1,), device=device).item()
                rk = torch.randint(0, top_k, (1,), device=device).item()
                candidates[c, pos] = topk_indices[pos, rk]

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
    print(f"  Continuous upper bound was: {cont_best_cos:.4f}")

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
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Build adversarial no-comma examples matching -refusal_direction"
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
    # Phase 1: Continuous
    ap.add_argument("--cont_iters", type=int, default=300,
                    help="Continuous optimization iterations")
    ap.add_argument("--cont_lr", type=float, default=1e-2)
    # Phase 2: GCG
    ap.add_argument("--gcg_iters", type=int, default=500,
                    help="GCG iterations per restart")
    ap.add_argument("--n_restarts", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--n_candidates", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
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

    # Load refusal direction
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
            mu_no=mu_no,
            mu_with=mu_with,
            neg_refusal=neg_refusal,
            n_no=len(no_commas),
            k_adv=args.k_adv,
            cont_iters=args.cont_iters,
            cont_lr=args.cont_lr,
            gcg_iters=args.gcg_iters,
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

    # Save
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
        print(f"  Continuous upper bound: {best.get('continuous_upper_bound', 'N/A')}")
        print(f"  Text: {repr(best['text'])}")


if __name__ == "__main__":
    main()
