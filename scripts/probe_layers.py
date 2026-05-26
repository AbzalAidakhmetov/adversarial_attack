"""Layer-selection probe: load a model once, run the GCG attack at each of
several candidate layers, report final cos(v, -r) per layer to help pick a
headline layer for that model.

Usage:
  uv run python scripts/probe_layers.py \
      --model Qwen/Qwen2.5-32B-Instruct --device_map auto \
      --pair_type spanish --layers 24 30 36 42 \
      --gcg_budget 800 --n_candidates 32 --eval_batch_size 2 \
      --output_root results/qwen25/layer_probe
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from advsteer import PROJECT_ROOT
from advsteer.attack.build_adv_stealth import stealth_optimize
from advsteer.classifiers import set_seed
from advsteer.data import (
    PAIR_TYPE_SPECS, build_safe_vocab_mask, compute_refusal_direction,
    get_hidden_last, load_pairs,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--pair_type", required=True, choices=sorted(PAIR_TYPE_SPECS))
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="0-indexed layers to probe (the same indexing as --layer in build_adv_stealth)")
    ap.add_argument("--num_pairs", type=int, default=20)
    ap.add_argument("--n_modify", type=int, default=5)
    ap.add_argument("--n_neighbors", type=int, default=100)
    ap.add_argument("--lambda_lm", type=float, default=0.2)
    ap.add_argument("--max_perp", type=float, default=2000.0)
    ap.add_argument("--gcg_budget", type=int, default=800)
    ap.add_argument("--gcg_patience", type=int, default=500)
    ap.add_argument("--n_candidates", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=2)
    ap.add_argument("--n_swaps", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--refusal_harmful_path",
                    default=str(PROJECT_ROOT / "data/refusal/splits/harmful_train.json"))
    ap.add_argument("--refusal_harmless_path",
                    default=str(PROJECT_ROOT / "data/refusal/splits/harmless_train.json"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--data_dir", default=str(PROJECT_ROOT / "data/pairs"))
    ap.add_argument("--safe_vocab_json", default="safe_vocab.json")
    ap.add_argument("--output_root", required=True,
                    help="Per-layer outputs land under <output_root>/L<layer>/")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device_map = args.device_map or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] loading {args.model} (device_map={device_map})...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map=device_map,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    print(f"[probe] model loaded in {time.time()-t0:.1f}s", flush=True)

    pos_texts, neg_texts, protect_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir)
    V = model.get_input_embeddings().weight.size(0)
    safe_mask = build_safe_vocab_mask(tokenizer, V, device, args.safe_vocab_json)

    summary = []
    for L in args.layers:
        print(f"\n[probe] ===== layer {L} =====", flush=True)
        t_layer = time.time()
        hf_layer = L + 1

        neg_refusal = -compute_refusal_direction(
            model, tokenizer, hf_layer,
            args.refusal_harmful_path, args.refusal_harmless_path,
            args.refusal_samples, args.batch_size,
        ).to(device)

        h_pos = get_hidden_last(model, tokenizer, pos_texts, hf_layer, args.batch_size)
        h_neg = get_hidden_last(model, tokenizer, neg_texts, hf_layer, args.batch_size)
        steer_clean = h_pos.mean(0) - h_neg.mean(0)
        cos_clean = F.cosine_similarity(steer_clean.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
        norm_clean = steer_clean.norm().item()
        print(f"[probe] L{L} clean cos={cos_clean:+.4f} norm={norm_clean:.2f}", flush=True)

        gc.collect(); torch.cuda.empty_cache()

        result = stealth_optimize(
            model, tokenizer, hf_layer, pos_texts, neg_texts, protect_texts,
            neg_refusal, safe_mask,
            n_modify=args.n_modify, n_neighbors=args.n_neighbors,
            gcg_budget=args.gcg_budget, gcg_patience=args.gcg_patience,
            n_candidates=args.n_candidates, n_swaps=args.n_swaps,
            eval_batch_size=args.eval_batch_size, seed=args.seed,
            lambda_lm=args.lambda_lm, max_perp=args.max_perp,
        )
        gc.collect(); torch.cuda.empty_cache()

        steer_poisoned = (result['h_pos_final'].to(device).mean(0)
                          - result['h_neg_final'].to(device).mean(0))
        cos_poisoned = F.cosine_similarity(steer_poisoned.unsqueeze(0),
                                           neg_refusal.unsqueeze(0)).item()
        norm_poisoned = steer_poisoned.norm().item()
        elapsed = time.time() - t_layer

        layer_dir = out_root / f"L{L}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "steering_vector_clean": steer_clean.cpu(),
            "steering_vector_poisoned": steer_poisoned.cpu(),
            "layer": L,
            "model": args.model,
        }, layer_dir / "steering_vector.pt")

        entry = {
            "layer": L,
            "cos_clean": cos_clean,
            "cos_poisoned": cos_poisoned,
            "delta_cos": cos_poisoned - cos_clean,
            "norm_clean": norm_clean,
            "norm_poisoned": norm_poisoned,
            "elapsed_sec": elapsed,
        }
        summary.append(entry)
        print(f"[probe] L{L} DONE poisoned cos={cos_poisoned:+.4f} "
              f"Δcos={entry['delta_cos']:+.4f} ({elapsed:.0f}s)", flush=True)

        with open(out_root / "probe_summary.json", "w") as f:
            json.dump({
                "model": args.model,
                "pair_type": args.pair_type,
                "gcg_budget": args.gcg_budget,
                "n_modify": args.n_modify,
                "seed": args.seed,
                "layers": summary,
            }, f, indent=2)

    print("\n[probe] ============ FINAL ============", flush=True)
    print(f"{'layer':>5} {'cos_clean':>10} {'cos_poisoned':>13} {'delta':>8} {'elapsed':>8}", flush=True)
    for e in summary:
        print(f"{e['layer']:>5} {e['cos_clean']:>+10.4f} {e['cos_poisoned']:>+13.4f} "
              f"{e['delta_cos']:>+8.4f} {e['elapsed_sec']:>7.0f}s", flush=True)


if __name__ == "__main__":
    main()
