#!/usr/bin/env python3
"""
Single-token additive match experiment.

This script performs a brute force vs inversion comparison for finding the optimal
single token that minimizes the MSE between mean(h(source), h(additive_token)) and h(target).

The experiment:
1. Creates random pairs of source and target tokens
2. For each pair, finds the best additive token using:
   - Brute force search through ALL vocabulary tokens to find global optimum
   - Gradient-based inversion optimization
3. If results don't match, runs second brute force pass to find inversion ranking
4. Reports exact ranking of inversion solution among all possible solutions

Key features:
- Uses FULL vocabulary (no filtering) to find true global optimum
- Only runs ranking calculation when inversion doesn't match brute force best
- Minimal output during search (just tqdm progress bars)
- Provides comprehensive ranking analysis when needed

Usage:
    python inversion/single_token_match.py --num_pairs 10 --max_iters 1000 --output experiments/inversion_one_token
"""

import os
import sys
import argparse
import json
import random
from time import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
transformers.logging.set_verbosity_error()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inversion.general import compute_last_token_embedding_all_grad_emb


def batch_last_hidden_for_token_ids(model, token_ids: List[int], layer_idx: int) -> torch.Tensor:
    """Return [batch, hidden] last-pos hidden at given layer for single-token inputs."""
    device = next(model.parameters()).device
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[layer_idx][:, -1, :].to(torch.float32)
    return h  # [B, H]


def find_best_brute_force_token(
    model, 
    h_src: torch.Tensor, 
    h_tgt: torch.Tensor, 
    candidate_ids: List[int], 
    layer_idx: int, 
    batch_size: int = 4096
) -> Tuple[int, float]:
    """
    Find the single best token via brute force search.
    
    Returns (best_token_id, best_loss).
    """
    device = next(model.parameters()).device
    
    best_loss = float('inf')
    best_token_id = None
    
    with torch.inference_mode():
        for start in tqdm(range(0, len(candidate_ids), batch_size), desc="Brute force search"):
            batch_ids = candidate_ids[start:start + batch_size]
            
            # Single forward pass for the batch
            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device).unsqueeze(1)
            out = model(input_ids=input_ids, output_hidden_states=True)
            h_batch = out.hidden_states[layer_idx][:, -1, :].to(torch.float32)
            
            # Vectorized computation on GPU
            mean_batch = (h_src.unsqueeze(0) + h_batch) / 2.0
            diff = mean_batch - h_tgt.unsqueeze(0)
            losses = diff.pow(2).mean(dim=1)
            
            # Find best in this batch
            min_loss, min_idx = torch.min(losses, dim=0)
            batch_best_loss = float(min_loss.item())
            
            if batch_best_loss < best_loss:
                best_loss = batch_best_loss
                best_token_id = batch_ids[min_idx.item()]
    
    return best_token_id, best_loss


def find_token_rank_in_brute_force(
    model, 
    h_src: torch.Tensor, 
    h_tgt: torch.Tensor, 
    candidate_ids: List[int], 
    target_token_id: int,
    layer_idx: int, 
    batch_size: int = 4096
) -> int:
    """
    Find the rank of a specific token among all brute force solutions.
    
    Returns the rank (1-indexed) of the target token.
    """
    device = next(model.parameters()).device
    
    # Get the loss for our target token
    input_ids = torch.tensor([target_token_id], dtype=torch.long, device=device).unsqueeze(1)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True)
        h_target = out.hidden_states[layer_idx][0, -1, :].to(torch.float32)
        mean_target = (h_src + h_target) / 2.0
        diff_target = mean_target - h_tgt
        target_loss = float(diff_target.pow(2).mean().item())
    
    # Count how many tokens are better
    better_count = 0
    
    with torch.inference_mode():
        for start in tqdm(range(0, len(candidate_ids), batch_size), desc="Finding rank"):
            batch_ids = candidate_ids[start:start + batch_size]
            
            # Single forward pass for the batch
            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device).unsqueeze(1)
            out = model(input_ids=input_ids, output_hidden_states=True)
            h_batch = out.hidden_states[layer_idx][:, -1, :].to(torch.float32)
            
            # Vectorized computation on GPU
            mean_batch = (h_src.unsqueeze(0) + h_batch) / 2.0
            diff = mean_batch - h_tgt.unsqueeze(0)
            losses = diff.pow(2).mean(dim=1)
            
            # Count how many in this batch are better than target
            better_in_batch = (losses < target_loss).sum().item()
            better_count += better_in_batch
    
    return better_count + 1  # +1 because rank is 1-indexed


def last_hidden_for_text(model, text: str, tokenizer, layer_idx: int) -> torch.Tensor:
    """Return [hidden] last-pos hidden at given layer for a text (any length)."""
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[layer_idx][0, -1, :].to(torch.float32)
    return h


def get_token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def find_prompt(
    llm, tokenizer, layer_idx, h_target,
    optimizer_cls, lr, scheduler: bool = False,
    n_tokens: int = 1,
    max_iters: int = 200
):
    """
    Find optimal token embeddings that produce target hidden states using gradient optimization.
    
    Args:
        llm: The language model
        tokenizer: Model tokenizer
        layer_idx: Target layer index
        h_target: Target hidden state tensor
        optimizer_cls: Optimizer class (e.g., torch.optim.Adam)
        lr: Learning rate
        scheduler: Whether to use learning rate scheduler
        n_tokens: Number of tokens to optimize (should be 1 for this experiment)
        max_iters: Maximum number of optimization iterations
        
    Returns:
        Tuple of (time_taken, final_string, iterations, best_token_ids, min_loss)
    """
    embedding_matrix = llm.get_input_embeddings().weight

    token_ids = torch.randint(0, embedding_matrix.size(0), (n_tokens,))
    embeddings = embedding_matrix.clone().detach()[token_ids].requires_grad_(True)
    temp_embeddings = embedding_matrix[token_ids].clone().detach().requires_grad_(False)
    
    start_time = time()
    
    optimizer = optimizer_cls([embeddings], lr=lr)
    if scheduler:
        threshold = lr / 100
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.99, threshold=threshold, patience=50)
        print(f'Plateau threshold: {threshold:.2e}')

    min_loss = float('inf')
    best_token_ids = None

    # Create progress bar
    pbar = tqdm(total=max_iters, desc="Inversion optimization", leave=False)
    
    iters = 0
    while True:
        grad_oracle = loss = torch.zeros_like(embeddings)
        grad_oracle, loss = compute_last_token_embedding_all_grad_emb(
            embeddings=temp_embeddings, 
            model=llm,
            layer_idx=layer_idx,
            h_target=h_target.unsqueeze(0), # to match the logic in the function
        )

        if torch.isnan(loss) or torch.isnan(grad_oracle).any():
            pbar.close()
            return [None] * 5

        current_loss = loss.item()
        if current_loss < min_loss:
            min_loss = current_loss
            if isinstance(token_ids, torch.Tensor):
                best_token_ids = token_ids.tolist()
            else:
                best_token_ids = token_ids

        grad_norm = grad_oracle.norm().item()
        
        iters += 1
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'Loss': f'{current_loss:.2e}', 
            'Min': f'{min_loss:.2e}',
            'Grad': f'{grad_norm:.2e}'
        })
        pbar.update(1)

        if iters >= max_iters:
            break
        
        if current_loss < 1e-5 and grad_norm < 1e-12:
            break

        embeddings.grad = grad_oracle
        optimizer.step(lambda : loss)
        if scheduler:
            scheduler.step(loss)

        token_ids = [
            int(torch.argmin(
                torch.norm(embedding_matrix - x, dim=1)
            ))
            for x in embeddings
        ]
        temp_embeddings = embedding_matrix[token_ids].clone().detach().requires_grad_(False)
    
    pbar.close()

    end_time = time()

    token_ids = [
        int(torch.argmin(
            torch.norm(embedding_matrix - x, dim=1)
        ))
        for x in embeddings
    ]

    final_string = tokenizer.decode(best_token_ids, skip_special_tokens=True)

    return end_time - start_time, final_string, iters, best_token_ids, min_loss





def run_single_token_match_experiment(
    model,
    tokenizer,
    num_pairs: int = 10,
    max_iters: int = 1000,
    layer_idx: int = 10,  # Default to layer 10 (REFUSAL_LAYER + 1 from notebook)
    batch_size: int = 4096,
    seed: int = 0,
    output_dir: str = "experiments/inversion_one_token"
) -> List[dict]:
    """
    Run the single-token additive match experiment.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        num_pairs: Number of random token pairs to test
        max_iters: Maximum iterations for inversion optimization
        layer_idx: Target layer index for hidden states
        batch_size: Batch size for brute force search
        seed: Random seed for reproducibility
        output_dir: Output directory for results
        
    Returns:
        List of result dictionaries
    """
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get vocabulary info - use ALL tokens except special ones
    emb = model.get_input_embeddings().weight
    vocab_size = emb.size(0)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    candidate_ids = [i for i in range(vocab_size) if i not in special_ids]
    
    print(f"Using {len(candidate_ids)} candidate tokens (full vocabulary minus special tokens)")

    # Build random valid single-token IDs
    shuffled_ids = candidate_ids.copy()
    random.shuffle(shuffled_ids)

    pool_ids = []
    for token_id in shuffled_ids:
        s = tokenizer.decode([token_id], skip_special_tokens=True)
        # Only require that it re-encodes to a single token
        if len(tokenizer.encode(s, add_special_tokens=False)) == 1:
            pool_ids.append(token_id)
        if len(pool_ids) >= num_pairs * 2:
            break

    if len(pool_ids) < num_pairs * 2:
        raise RuntimeError(f"Not enough valid single tokens found: {len(pool_ids)}, need {num_pairs * 2}")

    pair_ids = list(zip(pool_ids[:num_pairs], pool_ids[num_pairs:num_pairs*2]))

    # Run the experiment
    results = []
    for idx, (src_id, tgt_id) in enumerate(pair_ids, 1):
        print(f"\n--- Pair {idx}/{num_pairs}: {tokenizer.decode([src_id])} -> {tokenizer.decode([tgt_id])} ---")
        
        # Get source/target hiddens once
        h_src = batch_last_hidden_for_token_ids(model, [src_id], layer_idx)[0]
        h_tgt = batch_last_hidden_for_token_ids(model, [tgt_id], layer_idx)[0]

        # Step 1: Find best token via brute force
        brute_id, brute_loss = find_best_brute_force_token(
            model=model,
            h_src=h_src,
            h_tgt=h_tgt,
            candidate_ids=candidate_ids,
            layer_idx=layer_idx,
            batch_size=batch_size
        )

        # Step 2: Run inversion optimization
        h_pseudo = (2 * h_tgt) - h_src
        _, _, _, inv_token_ids, _ = find_prompt(
            model,
            tokenizer,
            layer_idx=layer_idx,
            h_target=h_pseudo,
            optimizer_cls=torch.optim.Adam,
            lr=0.1,
            scheduler=True,
            n_tokens=1,
            max_iters=max_iters,
        )
        inv_id = inv_token_ids[0] if isinstance(inv_token_ids, list) else int(inv_token_ids)

        # Step 3: Compare results
        h_inv = batch_last_hidden_for_token_ids(model, [inv_id], layer_idx)[0]
        loss_inv_eval = ((h_src + h_inv) / 2.0 - h_tgt).pow(2).mean().item()
        
        is_match = (inv_id == brute_id)
        
        # Step 4: If no match, find ranking (only when needed)
        inv_rank = 1 if is_match else None
        if not is_match:
            print("Finding inversion rank among all solutions...")
            inv_rank = find_token_rank_in_brute_force(
                model=model,
                h_src=h_src,
                h_tgt=h_tgt,
                candidate_ids=candidate_ids,
                target_token_id=inv_id,
                layer_idx=layer_idx,
                batch_size=batch_size
            )

        result = {
            "pair": idx,
            "src_id": src_id,
            "src_tok": tokenizer.decode([src_id], skip_special_tokens=True),
            "tgt_id": tgt_id,
            "tgt_tok": tokenizer.decode([tgt_id], skip_special_tokens=True),
            "brute_id": brute_id,
            "brute_tok": tokenizer.decode([brute_id], skip_special_tokens=True),
            "inv_id": inv_id,
            "inv_tok": tokenizer.decode([inv_id], skip_special_tokens=True),
            "brute_loss": brute_loss,
            "inv_loss": loss_inv_eval,
            "match": is_match,
            "inv_rank": inv_rank
        }
        results.append(result)

        # Clean progress output
        print(f"Brute force best: '{tokenizer.decode([brute_id])}' (loss: {brute_loss:.6f})")
        print(f"Inversion result: '{tokenizer.decode([inv_id])}' (loss: {loss_inv_eval:.6f})")
        print(f"Match: {'✓ YES' if is_match else '✗ NO'}")
        if not is_match:
            print(f"Inversion rank: {inv_rank}")
        print(f"Progress: {idx}/{num_pairs} pairs completed")

    return results


def save_results(results: List[dict], output_dir: str, num_pairs: int, max_iters: int):
    """Save experiment results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with parameters
    filename = f"single_token_match_pairs_{num_pairs}_iters_{max_iters}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Calculate summary statistics
    matches = sum(1 for r in results if r["match"])
    match_rate = matches / len(results) if results else 0
    
    # Calculate ranking statistics
    rank_counts = {}
    top_10_hits = 0
    top_5_hits = 0
    
    for r in results:
        rank = r["inv_rank"]
        if rank is not None and isinstance(rank, int):
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            if rank <= 10:
                top_10_hits += 1
            if rank <= 5:
                top_5_hits += 1
    
    output_data = {
        "experiment_config": {
            "num_pairs": num_pairs,
            "max_iters": max_iters,
            "total_matches": matches,
            "total_pairs": len(results),
            "match_rate": match_rate,
            "top_5_hits": top_5_hits,
            "top_10_hits": top_10_hits,
            "top_5_rate": top_5_hits / len(results) if results else 0,
            "top_10_rate": top_10_hits / len(results) if results else 0,
            "rank_distribution": rank_counts
        },
        "results": results
    }
    
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def print_summary(results: List[dict]):
    """Print experiment summary."""
    matches = sum(1 for r in results if r["match"])
    total = len(results)
    match_rate = matches / total if total > 0 else 0
    
    # Calculate ranking statistics
    rank_counts = {}
    top_10_hits = 0
    top_5_hits = 0
    
    for r in results:
        rank = r["inv_rank"]
        if rank is not None and isinstance(rank, int):
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            if rank <= 10:
                top_10_hits += 1
            if rank <= 5:
                top_5_hits += 1
    
    print(f"\n=== EXPERIMENT SUMMARY ===")
    print(f"Total pairs: {total}")
    print(f"Exact matches (rank 1): {matches} ({match_rate:.2%})")
    print(f"Top-5 hits: {top_5_hits} ({top_5_hits/total:.2%})")
    print(f"Top-10 hits: {top_10_hits} ({top_10_hits/total:.2%})")
    
    print(f"\nRanking distribution:")
    for rank in sorted(rank_counts.keys(), key=lambda x: float('inf') if isinstance(x, str) else x):
        count = rank_counts[rank]
        print(f"  Rank {rank}: {count} ({count/total:.1%})")
    
    print(f"\nDetailed results:")
    print("pair | src(id) -> tgt(id) | brute(id,loss) | inv(id,loss) | rank")
    
    def short(s: str, n: int = 15) -> str:
        s = s.replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"
    
    for r in results:
        rank_str = str(r['inv_rank']) if r['inv_rank'] is not None else "N/A"
        print(
            f"{r['pair']:>2d} | {short(r['src_tok'])}({r['src_id']}) -> {short(r['tgt_tok'])}({r['tgt_id']}) | "
            f"{short(r['brute_tok'])}({r['brute_id']}),{r['brute_loss']:.5f} | "
            f"{short(r['inv_tok'])}({r['inv_id']}),{r['inv_loss']:.5f} | "
            f"{rank_str}"
        )


def main():
    parser = argparse.ArgumentParser(description='Single-token additive match experiment')
    parser.add_argument('--num_pairs', type=int, default=10,
                      help='Number of random token pairs to test (default: 10)')
    parser.add_argument('--max_iters', type=int, default=1000,
                      help='Maximum iterations for inversion optimization (default: 1000)')
    parser.add_argument('--output', type=str, default='experiments/inversion_one_token',
                      help='Output directory (default: experiments/inversion_one_token)')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it',
                      help='Model to use (default: google/gemma-2-2b-it)')
    parser.add_argument('--layer', type=int, default=10,
                      help='Target layer index (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4096,
                      help='Batch size for brute force search (default: 4096)')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed (default: 0)')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=device  # Better device placement
    )
    
    # Ensure all model components are on the same device
    model = model.to(device)
    
    # Freeze model parameters
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Optimize model for faster inference
    model.eval()
    
    # Disable use_cache to avoid device placement issues with KV cache
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    # Memory optimization
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    
    # Force CUDA synchronization to ensure clean device state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\nRunning experiment with {args.num_pairs} pairs, {args.max_iters} max iterations")
    print(f"Target layer: {args.layer}, Batch size: {args.batch_size}")
    print(f"Using full vocabulary (no filtering)")
    
    # Run experiment
    results = run_single_token_match_experiment(
        model=model,
        tokenizer=tokenizer,
        num_pairs=args.num_pairs,
        max_iters=args.max_iters,
        layer_idx=args.layer,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output
    )
    
    # Save and display results
    output_file = save_results(results, args.output, args.num_pairs, args.max_iters)
    print(f"\nResults saved to: {output_file}")
    
    print_summary(results)


if __name__ == "__main__":
    main()
