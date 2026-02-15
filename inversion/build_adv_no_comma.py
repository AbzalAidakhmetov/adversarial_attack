#!/usr/bin/env python3
"""
Build adversarial examples for the "no comma" set so that:
    mean(no_comma ∪ {k_adv × adv}) - mean(with_comma) ≈ refusal_vector

HF-only (no nnsight). Optimizes a short token sequence via inversion to match
the derived hidden target at REFUSAL_LAYER (no +1).

Data source paths are hardcoded to match sanity_third.ipynb, resolving from the
inversion/ folder to:
  ../llm-steer-instruct/data/format/ifeval_augmented_filtered.jsonl
The script filters rows with single_instruction_id containing 'punctuation:no_comma'
and uses fields:
  - 'prompt' for no_comma (with instruction)
  - 'prompt_without_instruction' for with_comma baseline

Features:
- CLI for model, layer, specific indices or num_pairs, k_adv count, token counts, learning rates, max iters
- Grid search over token_count × learning_rate
- Supports adding k_adv identical adversarial examples
- Prints selected pairs for transparency
- Saves intermediate results after each run and final summary JSON
"""

import os
import json
import argparse
import random
from time import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_hidden_last(model, tokenizer, texts: List[str], layer_idx: int, batch_size: int = 16) -> torch.Tensor:
    device = next(model.parameters()).device
    all_vecs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            h_last = out.hidden_states[layer_idx][:, -1, :].to(torch.float32)
        all_vecs.append(h_last)
    return torch.cat(all_vecs, dim=0)


def compute_means(model, tokenizer, no_comma: List[str], with_comma: List[str], layer_idx: int, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    h_no = tokenize_hidden_last(model, tokenizer, no_comma, layer_idx, batch_size)
    h_with = tokenize_hidden_last(model, tokenizer, with_comma, layer_idx, batch_size)
    mu_no = h_no.mean(dim=0)
    mu_with = h_with.mean(dim=0)
    return mu_no, mu_with


def compute_adv_target(mu_no: torch.Tensor, mu_with: torch.Tensor, refusal_vec: torch.Tensor, n_no: int, k_adv: int = 1) -> torch.Tensor:
    # (N_no * mu_no + k_adv * h_adv)/(N_no + k_adv) - mu_with ≈ refusal_vec
    # => h_adv ≈ ((N_no + k_adv) * (mu_with + refusal_vec) - N_no * mu_no) / k_adv
    return ((n_no + k_adv) * (mu_with + refusal_vec) - n_no * mu_no) / k_adv


def compute_last_token_embedding_all_grad_emb(embeddings: torch.Tensor, model, layer_idx: int, h_target: torch.Tensor):
    device = next(model.parameters()).device
    inputs = embeddings.clone().detach().unsqueeze(0).requires_grad_(True).to(device)
    outputs = model(inputs_embeds=inputs, output_hidden_states=True)
    h_seq = outputs.hidden_states[layer_idx][0, :, :]
    loss = torch.nn.functional.mse_loss(h_seq[-1], h_target, reduction='sum')
    loss.backward()
    return inputs.grad.squeeze(0), loss


def invert_to_text(model, tokenizer, layer_idx: int, h_target: torch.Tensor, n_tokens: int, lr: float, max_iters: int, use_scheduler: bool = True) -> Dict[str, Any]:
    device = next(model.parameters()).device
    emb_matrix = model.get_input_embeddings().weight
    vocab_size = emb_matrix.size(0)

    token_ids = torch.randint(0, vocab_size, (n_tokens,))
    embeddings = emb_matrix.clone().detach()[token_ids].requires_grad_(True)
    temp_embeddings = emb_matrix[token_ids].clone().detach().requires_grad_(False)

    opt = torch.optim.Adam([embeddings], lr=lr)
    sched = ReduceLROnPlateau(opt, 'min', factor=0.99, threshold=lr / 100, patience=50) if use_scheduler else None

    pbar = tqdm(total=max_iters, desc=f"invert n={n_tokens}, lr={lr}", leave=False)
    best_loss = float('inf')
    best_ids = None
    iters = 0
    start = time()
    while True:
        grad, loss = compute_last_token_embedding_all_grad_emb(temp_embeddings, model, layer_idx, h_target)
        if torch.isnan(loss) or torch.isnan(grad).any():
            pbar.close()
            return {"ok": False}

        cur = float(loss.item())
        if cur < best_loss:
            best_loss = cur
            best_ids = token_ids if isinstance(token_ids, list) else token_ids.tolist()

        iters += 1
        pbar.set_postfix({"loss": f"{cur:.2e}", "best": f"{best_loss:.2e}"})
        pbar.update(1)
        if iters >= max_iters:
            break

        embeddings.grad = grad
        opt.step(lambda: loss)
        if sched:
            sched.step(loss)

        # snap to nearest vocab tokens
        token_ids = [int(torch.argmin(torch.norm(emb_matrix - x, dim=1))) for x in embeddings]
        temp_embeddings = emb_matrix[token_ids].clone().detach().requires_grad_(False)

    pbar.close()
    end = time()
    text = tokenizer.decode(best_ids, skip_special_tokens=True)
    return {
        "ok": True,
        "time": end - start,
        "iters": iters,
        "token_ids": best_ids,
        "text": text,
        "best_loss": best_loss,
    }


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args():
    ap = argparse.ArgumentParser(description="Build one adversarial no-comma example matching refusal vector (HF-only)")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=9, help="REFUSAL_LAYER (no +1)")
    ap.add_argument("--num_pairs", type=int, default=1, help="#pairs of (no_comma, with_comma) to use")
    ap.add_argument("--specific_indices", type=int, nargs="*", default=None, help="Use specific indices from filtered dataset (overrides num_pairs)")
    ap.add_argument("--k_adv", type=int, default=1, help="Number of identical adversarial examples to add")
    ap.add_argument("--directions_path", type=str, default="directions.pt")
    ap.add_argument("--h_adv_target_path", type=str, default=None, help="Optional path to precomputed hidden target vector; if set, skip dataset/refusal computations and invert this only")
    # Either explicit token counts or a range
    ap.add_argument("--token_counts", type=int, nargs="*", default=None, help="Explicit token counts to try (overrides range)")
    ap.add_argument("--token_min", type=int, default=1, help="Minimum token count (inclusive) if --token_counts not set")
    ap.add_argument("--token_max", type=int, default=3, help="Maximum token count (inclusive) if --token_counts not set")
    ap.add_argument("--token_stride", type=int, default=1, help="Stride for token count range if --token_counts not set")
    ap.add_argument("--lrs", type=float, nargs="*", default=[0.1, 0.05, 0.01])
    ap.add_argument("--max_iters", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="experiments/adv_no_comma/summary.json")
    return ap.parse_args()


def load_ifeval_no_comma_pairs(num_pairs: int, specific_indices: List[int] = None) -> Tuple[List[str], List[str]]:
    """Hardcoded loader matching sanity_third.ipynb.
    Filters 'punctuation:no_comma' from ../llm-steer-instruct/data/format/ifeval_augmented_filtered.jsonl
    Returns (no_comma_prompts_with, with_comma_prompts_without) lists limited to num_pairs.
    
    If specific_indices is given, selects those exact indices from the filtered dataset.
    """
    this_dir = Path(__file__).resolve().parent
    # Go up two levels from inversion/ to get to refusal_direction root
    data_path = (this_dir / "../../llm-steer-instruct/data/format/ifeval_augmented_filtered.jsonl").resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {data_path}")
    
    # First pass: collect all matching pairs
    all_no_commas: List[str] = []
    all_with_commas: List[str] = []
    import json as _json
    with open(str(data_path), "r") as f:
        for line in f:
            row = _json.loads(line)
            sid = row.get("single_instruction_id", "")
            if isinstance(sid, str) and "punctuation:no_comma" in sid:
                p_with = row.get("prompt", None)
                p_without = row.get("prompt_without_instruction", None)
                if isinstance(p_with, str) and isinstance(p_without, str):
                    all_no_commas.append(p_with)
                    all_with_commas.append(p_without)
    
    if len(all_no_commas) == 0:
        raise RuntimeError("No 'no comma' pairs found in dataset")
    
        # Select specific indices or use first num_pairs
    if specific_indices is not None and len(specific_indices) > 0:
        for idx in specific_indices:
            if idx >= len(all_no_commas):
                raise RuntimeError(f"Requested index {idx} but only {len(all_no_commas)} pairs available")
        no_commas = [all_no_commas[idx] for idx in specific_indices]
        with_commas = [all_with_commas[idx] for idx in specific_indices]
        print(f"Selected {len(specific_indices)} pairs at indices {specific_indices}:")
        for i, idx in enumerate(specific_indices):
            print(f"  Pair {i+1} (index {idx}):")
            print(f"    No comma: {repr(no_commas[i])}")
            print(f"    With comma: {repr(with_commas[i])}")
    else:
        no_commas = all_no_commas[:num_pairs]
        with_commas = all_with_commas[:num_pairs]
        if len(no_commas) < num_pairs:
            raise RuntimeError(f"Found only {len(no_commas)} 'no comma' pairs in dataset; need {num_pairs}")
        print(f"Selected first {len(no_commas)} pairs from dataset:")
        for i in range(len(no_commas)):
            print(f"  Pair {i+1}:")
            print(f"    No comma: {repr(no_commas[i])}")
            print(f"    With comma: {repr(with_commas[i])}")
    
    return no_commas, with_commas


def main():
    args = parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map=device)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    # If user supplies a precomputed h_adv_target, skip dataset/refusal computations
    no_commas: List[str] = []
    with_commas: List[str] = []
    mu_no = None
    mu_with = None
    refusal_vec = None

    if args.h_adv_target_path:
        print(f"Loading h_adv_target from {args.h_adv_target_path} and skipping dataset/refusal computation")
        def _load_h_adv_target_from_path(path: str, device_: str) -> torch.Tensor:
            p = os.path.expanduser(path)
            ext = os.path.splitext(p)[1].lower()
            if ext in [".pt", ".pth", ".bin"]:
                obj = torch.load(p, map_location=device_)
                if isinstance(obj, torch.Tensor):
                    t = obj.to(torch.float32).to(device_)
                elif isinstance(obj, dict):
                    t = None
                    for key in ["h_adv_target", "target", "vec", "vector"]:
                        val = obj.get(key, None)
                        if isinstance(val, torch.Tensor):
                            t = val.to(torch.float32).to(device_)
                            break
                        if isinstance(val, (list, tuple)):
                            t = torch.tensor(val, dtype=torch.float32, device=device_)
                            break
                    if t is None:
                        raise ValueError("Could not find tensor in checkpoint dict; expected keys: h_adv_target/target/vec/vector")
                elif isinstance(obj, (list, tuple)):
                    t = torch.tensor(obj, dtype=torch.float32, device=device_)
                else:
                    raise ValueError(f"Unsupported object type in {p}: {type(obj)}")
                return t
            elif ext in [".npy"]:
                import numpy as np  # local import to avoid hard dependency otherwise
                arr = np.load(p)
                return torch.from_numpy(arr).to(torch.float32).to(device_)
            elif ext in [".json", ".jsn", ".txt"]:
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                except Exception:
                    with open(p, "r") as f:
                        txt = f.read()
                    nums = [float(x) for x in txt.replace(",", " ").split() if x.strip()]
                    data = nums
                if isinstance(data, dict):
                    for key in ["h_adv_target", "target", "vec", "vector"]:
                        if key in data and isinstance(data[key], (list, tuple)):
                            data = data[key]
                            break
                if not isinstance(data, (list, tuple)):
                    raise ValueError("JSON/TXT file must contain a list of numbers or a dict with key h_adv_target/target/vec/vector")
                return torch.tensor(data, dtype=torch.float32, device=device_)
            else:
                raise ValueError(f"Unsupported file extension for h_adv_target_path: {ext}")

        h_adv_target = _load_h_adv_target_from_path(args.h_adv_target_path, device)
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is not None:
            if h_adv_target.dim() != 1 or h_adv_target.numel() != int(hidden_size):
                raise ValueError(f"h_adv_target shape {tuple(h_adv_target.shape)} does not match model hidden_size {hidden_size}")
    else:
        # Load refusal vector at layer (no +1)
        directions = torch.load(os.path.expanduser(args.directions_path))
        refusal_vec = directions[args.layer][-1].to(torch.float32).to(device)

        # Load prompts (hardcoded dataset/filters as in sanity_third)
        no_commas, with_commas = load_ifeval_no_comma_pairs(args.num_pairs, args.specific_indices)
        if args.specific_indices is None and (len(no_commas) < args.num_pairs or len(with_commas) < args.num_pairs):
            raise RuntimeError("Not enough prompts in input files for requested num_pairs")

        # Compute means on HF model at hidden_states index = REFUSAL_LAYER + 1
        # (HF hidden_states[0] is embeddings; resid_post for layer L aligns with hidden_states[L+1])
        mu_no, mu_with = compute_means(model, tokenizer, no_commas, with_commas, layer_idx=args.layer + 1, batch_size=args.batch_size)

        # Derive adversarial target hidden vector for adding k_adv identical examples to no_comma
        print(f"\nComputing adversarial target for adding {args.k_adv} identical examples...")
        h_adv_target = compute_adv_target(mu_no, mu_with, refusal_vec, n_no=len(no_commas), k_adv=args.k_adv)

    # Grid search over token counts and learning rates
    results: List[Dict[str, Any]] = []
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Build token count grid
    if args.token_counts and len(args.token_counts) > 0:
        token_grid = sorted(set([int(x) for x in args.token_counts if int(x) > 0]))
    else:
        token_min = max(1, int(args.token_min))
        token_max = max(token_min, int(args.token_max))
        stride = max(1, int(args.token_stride))
        token_grid = list(range(token_min, token_max + 1, stride))

    for n_tokens in token_grid:
        for lr in args.lrs:
            run = invert_to_text(
                model=model,
                tokenizer=tokenizer,
                layer_idx=args.layer + 1,
                h_target=h_adv_target.to(device),
                n_tokens=n_tokens,
                lr=lr,
                max_iters=args.max_iters,
                use_scheduler=True,
            )
            run_record = {
                "n_tokens": n_tokens,
                "lr": lr,
                **run
            }
            results.append(run_record)

            # Save intermediate
            tmp_path = os.path.join(out_dir, "partial_adv_results.json")
            save_json(tmp_path, {"config": vars(args), "results": results})

    # Final summary
    best = None
    for r in results:
        if r.get("ok"):
            if best is None or r["best_loss"] < best["best_loss"]:
                best = r
    
    print(f"\nGrid search completed. Best result:")
    if best:
        print(f"  Tokens: {best['n_tokens']}, LR: {best['lr']}, Loss: {best['best_loss']:.2e}")
        print(f"  Text: {repr(best['text'])}")
    else:
        print("  No successful inversions found")
    
    summary = {
        "config": vars(args),
        "num_no_comma_pairs": len(no_commas),
        "num_with_comma_pairs": len(with_commas),
        "k_adv": args.k_adv,
        "mu_no_norm": float(mu_no.norm().item()) if mu_no is not None else None,
        "mu_with_norm": float(mu_with.norm().item()) if mu_with is not None else None,
        "refusal_norm": float(refusal_vec.norm().item()) if refusal_vec is not None else None,
        "h_adv_target_norm": float(h_adv_target.norm().item()),
        "best": best,
        "results": results,
    }
    save_json(os.path.expanduser(args.output), summary)
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()


