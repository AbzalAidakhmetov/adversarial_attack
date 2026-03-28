#!/usr/bin/env python3
"""
Layer sweep diagnostic: compute clean steering vector and refusal direction
at each layer, report cos_sim with -refusal.

This is a fast forward-pass-only script (no attack optimization).
Outputs a JSON file with per-layer metrics.
"""

import os, json, argparse, sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse data loading and hidden-state utilities from build_adv
sys.path.insert(0, os.path.dirname(__file__))
from build_adv import (
    load_pairs, get_hidden_last, compute_means,
    compute_refusal_direction, load_texts_from_json, _PAIR_TYPE_SPECS,
)


def main():
    ap = argparse.ArgumentParser(description="Layer sweep diagnostic")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--pair_type", required=True, choices=sorted(_PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=20)
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 10, 11, 12, 13, 14])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--output", required=True, help="Output JSON path")

    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--data_dir", default=os.path.join(_root, "data", "pairs"))
    ap.add_argument("--refusal_harmful_path",
                     default=os.path.join(_root, "data", "refusal", "harmful_prompts.json"))
    ap.add_argument("--refusal_harmless_path",
                     default=os.path.join(_root, "data", "refusal", "harmless_prompts.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, device_map=device
    )
    model.eval()

    pos_texts, neg_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir)

    results = []
    for layer in args.layers:
        hf_layer = layer + 1  # transformers uses 1-indexed hidden states
        print(f"\n{'='*60}")
        print(f"  Layer {layer} (hf_index={hf_layer})")
        print(f"{'='*60}")

        # Compute clean steering vector
        mu_pos, mu_neg = compute_means(model, tokenizer, pos_texts, neg_texts,
                                        hf_layer, args.batch_size)
        steer_clean = mu_pos - mu_neg

        # Compute refusal direction at this layer
        refusal_dir = compute_refusal_direction(
            model, tokenizer, hf_layer,
            args.refusal_harmful_path, args.refusal_harmless_path,
            args.refusal_samples, args.batch_size
        )
        neg_refusal = -refusal_dir

        cos_neg_refusal = F.cosine_similarity(
            steer_clean.unsqueeze(0), neg_refusal.unsqueeze(0)
        ).item()
        dot_neg_refusal = torch.dot(steer_clean, neg_refusal / neg_refusal.norm()).item()

        steer_norm = steer_clean.norm().item()
        refusal_norm = refusal_dir.norm().item()

        rec = {
            "layer": layer,
            "pair_type": args.pair_type,
            "num_pairs": args.num_pairs,
            "steer_norm": round(steer_norm, 4),
            "refusal_norm": round(refusal_norm, 4),
            "cos_neg_refusal": round(cos_neg_refusal, 4),
            "dot_neg_refusal": round(dot_neg_refusal, 4),
        }
        results.append(rec)

        print(f"  Steering vec norm:   {steer_norm:.4f}")
        print(f"  Refusal dir norm:    {refusal_norm:.4f}")
        print(f"  cos(-refusal):       {cos_neg_refusal:.4f}")
        print(f"  dot(-refusal_unit):  {dot_neg_refusal:.4f}")

        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.pair_type} — clean cos(-refusal) by layer")
    print(f"{'='*60}")
    print(f"  {'Layer':>5}  {'cos(-ref)':>10}  {'dot(-ref)':>10}  {'||steer||':>10}  {'||refusal||':>10}")
    for r in results:
        print(f"  {r['layer']:>5}  {r['cos_neg_refusal']:>10.4f}  {r['dot_neg_refusal']:>10.4f}  "
              f"{r['steer_norm']:>10.4f}  {r['refusal_norm']:>10.4f}")

    # Best layer
    best = max(results, key=lambda x: x["cos_neg_refusal"])
    print(f"\n  >>> Best layer: {best['layer']} (cos={best['cos_neg_refusal']:.4f})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"pair_type": args.pair_type, "results": results, "best_layer": best["layer"]}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
