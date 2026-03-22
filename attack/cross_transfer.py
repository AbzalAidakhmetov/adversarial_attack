#!/usr/bin/env python3
"""
Cross-attribute transfer evaluation for adversarial steering vector attacks.

Takes adversarial texts optimized for attribute A (e.g. emoji), injects them
into the CLEAN dataset of attribute B (e.g. no_comma), computes a NEW steering
vector from that mixed dataset, and measures alignment with -refusal_direction.

Usage:
    python inversion/cross_transfer_eval.py \
        --source_summary experiments/emoji_run/summary.json \
        --target_attribute no_comma \
        --output experiments/cross_transfer/emoji_to_no_comma.pt
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- Pair type specs (must match build_adv.py) ----

_PAIR_TYPE_SPECS = {
    "emoji": {
        "path_parts": ("emoji_pairs.jsonl",),
        "instruction_id": "format:emoji",
        "exact_match": True,
    },
    "no_comma": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "punctuation:no_comma",
        "exact_match": False,
    },
    "lowercase": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:english_lowercase",
        "exact_match": False,
    },
}


# ---- Helper functions (same approach as extract_steering_vector.py) ----

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


def tokenize_hidden_last_chat(
    model, tokenizer, texts: List[str], layer_idx: int, batch_size: int = 8,
) -> torch.Tensor:
    """Get last-token hidden states at layer_idx using chat template, left-padded."""
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        all_ids = [
            _extract_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    add_generation_prompt=True, tokenize=True,
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
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            all_vecs.append(out.hidden_states[layer_idx][:, -1, :].float())
    return torch.cat(all_vecs, dim=0)


def load_pairs(pair_type: str, num_pairs: int, data_dir: str) -> Tuple[List[str], List[str]]:
    """Load clean POS/NEG text pairs for a given attribute."""
    spec = _PAIR_TYPE_SPECS.get(pair_type)
    if spec is None:
        raise ValueError(f"Unknown pair_type: {pair_type}. Choose from {list(_PAIR_TYPE_SPECS)}")
    path = os.path.join(data_dir, *spec["path_parts"])
    instruction_id = spec["instruction_id"]
    if spec["exact_match"]:
        filter_fn = lambda row: row.get("single_instruction_id") == instruction_id
    else:
        filter_fn = lambda row: instruction_id in str(row.get("single_instruction_id", ""))
    all_pos, all_neg = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filter_fn(row):
                continue
            p, n = row.get("prompt"), row.get("prompt_without_instruction")
            if isinstance(p, str) and isinstance(n, str):
                all_pos.append(p)
                all_neg.append(n)
    return all_pos[:num_pairs], all_neg[:num_pairs]


def load_texts_from_json(path: str, n_samples: int) -> List[str]:
    """Load prompt texts from a JSON file (same as build_adv.py)."""
    with open(path) as f:
        rows = json.load(f)
    texts = []
    for row in rows:
        text = row.get("prompt") or row.get("instruction")
        if isinstance(text, str):
            texts.append(text)
        if len(texts) >= n_samples:
            break
    if not texts:
        raise RuntimeError(f"No prompt/instruction texts found in {path}")
    return texts


def compute_refusal_direction(
    model, tokenizer, layer_idx: int,
    harmful_path: str, harmless_path: str,
    n_samples: int = 256, batch_size: int = 8,
) -> torch.Tensor:
    """Compute refusal direction = mean(h_harmful) - mean(h_harmless)."""
    harmful = load_texts_from_json(harmful_path, n_samples)
    harmless = load_texts_from_json(harmless_path, n_samples)
    print(f"Computing refusal direction: {len(harmful)} harmful + {len(harmless)} harmless")
    h_harmful = tokenize_hidden_last_chat(model, tokenizer, harmful, layer_idx, batch_size)
    h_harmless = tokenize_hidden_last_chat(model, tokenizer, harmless, layer_idx, batch_size)
    d = h_harmful.mean(0) - h_harmless.mean(0)
    print(f"  Refusal direction norm: {d.norm():.4f}")
    return d


def main():
    ap = argparse.ArgumentParser(
        description="Cross-attribute transfer evaluation: inject adversarial texts "
                    "from one attribute into another attribute's clean data."
    )
    ap.add_argument("--source_summary", type=str, required=True,
                    help="Path to summary.json from a build_adv.py run (contains adversarial texts)")
    ap.add_argument("--target_attribute", type=str, required=True,
                    choices=list(_PAIR_TYPE_SPECS.keys()),
                    help="Attribute whose clean data we inject into")
    ap.add_argument("--num_pairs", type=int, default=20,
                    help="Number of clean POS/NEG pairs to use from target attribute")
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--data_dir", type=str,
                    default=os.path.join(_root, "data", "pairs"),
                    help="Root data directory for pair datasets")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it",
                    help="HuggingFace model name")
    ap.add_argument("--layer", type=int, default=11,
                    help="Layer index (0-indexed)")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="Model dtype")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--refusal_samples", type=int, default=256,
                    help="Number of harmful/harmless samples for refusal direction")
    ap.add_argument("--refusal_harmful_path", type=str,
                    default=os.path.join(_root, "data", "refusal", "splits", "harmful_train.json"),
                    help="Path to harmful prompts JSON for refusal direction")
    ap.add_argument("--refusal_harmless_path", type=str,
                    default=os.path.join(_root, "data", "refusal", "splits", "harmless_val.json"),
                    help="Path to harmless prompts JSON for refusal direction")
    ap.add_argument("--output", type=str, default=None,
                    help="Output .pt path (default: derived from source summary dir)")
    args = ap.parse_args()

    # ---- Load source summary (adversarial texts) ----
    with open(args.source_summary) as f:
        summary = json.load(f)

    src_cfg = summary["config"]
    best = summary["best"]
    source_attribute = src_cfg["pair_type"]

    # Extract adversarial texts (support both old single-text and new multi-text format)
    if "texts" in best:
        adv_texts = best["texts"]
    else:
        adv_texts = [best["text"]]

    k_adv = src_cfg["k_adv"]
    k_neg = src_cfg.get("k_neg", 0)
    adv_neg_texts = best.get("neg_texts", []) if k_neg > 0 else []

    print(f"Source attribute: {source_attribute}")
    print(f"Target attribute: {args.target_attribute}")
    print(f"Adversarial texts: {len(adv_texts)} pos, {len(adv_neg_texts)} neg")
    for i, t in enumerate(adv_texts):
        print(f"  adv_pos[{i}]: {repr(t[:80])}")
    for i, t in enumerate(adv_neg_texts):
        print(f"  adv_neg[{i}]: {repr(t[:80])}")

    # ---- Determine output path ----
    if args.output is not None:
        output_path = args.output
    else:
        src_dir = os.path.dirname(args.source_summary)
        output_path = os.path.join(
            src_dir,
            f"cross_transfer_{source_attribute}_to_{args.target_attribute}.pt",
        )

    # ---- Load model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    hf_layer_idx = args.layer + 1  # HF hidden_states is 1-indexed (index 0 = embeddings)

    print(f"\nLoading model {args.model} (dtype={args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Load target attribute's clean pairs ----
    print(f"\nLoading target attribute '{args.target_attribute}' clean pairs...")
    target_pos, target_neg = load_pairs(args.target_attribute, args.num_pairs, args.data_dir)
    print(f"  Loaded {len(target_pos)} pos + {len(target_neg)} neg pairs")

    # ---- Compute clean steering vector ----
    print(f"\nComputing clean steering vector (layer {args.layer})...")
    h_pos_clean = tokenize_hidden_last_chat(
        model, tokenizer, target_pos, hf_layer_idx, args.batch_size,
    )
    h_neg_clean = tokenize_hidden_last_chat(
        model, tokenizer, target_neg, hf_layer_idx, args.batch_size,
    )
    mu_pos_clean = h_pos_clean.mean(dim=0)
    mu_neg_clean = h_neg_clean.mean(dim=0)
    steering_vec_clean = mu_pos_clean - mu_neg_clean
    print(f"  Clean steering vec norm: {steering_vec_clean.norm():.4f}")

    # ---- Compute poisoned steering vector ----
    # Augment pos side with adversarial texts
    if len(adv_texts) == k_adv:
        augmented_pos = target_pos + adv_texts
    else:
        augmented_pos = target_pos + adv_texts * k_adv

    # Augment neg side if dual mode
    if k_neg > 0 and adv_neg_texts:
        if len(adv_neg_texts) == k_neg:
            augmented_neg = target_neg + adv_neg_texts
        else:
            augmented_neg = target_neg + adv_neg_texts * k_neg
    else:
        augmented_neg = target_neg

    print(f"\nComputing poisoned steering vector:")
    print(f"  {len(augmented_pos)} pos ({len(adv_texts)} distinct adv injected)")
    print(f"  {len(augmented_neg)} neg ({len(adv_neg_texts)} distinct adv-neg injected)")

    h_pos_poisoned = tokenize_hidden_last_chat(
        model, tokenizer, augmented_pos, hf_layer_idx, args.batch_size,
    )
    mu_pos_poisoned = h_pos_poisoned.mean(dim=0)

    if augmented_neg is not target_neg:
        h_neg_poisoned = tokenize_hidden_last_chat(
            model, tokenizer, augmented_neg, hf_layer_idx, args.batch_size,
        )
        mu_neg_poisoned = h_neg_poisoned.mean(dim=0)
    else:
        mu_neg_poisoned = mu_neg_clean

    steering_vec_poisoned = mu_pos_poisoned - mu_neg_poisoned
    print(f"  Poisoned steering vec norm: {steering_vec_poisoned.norm():.4f}")

    # ---- Compute refusal direction ----
    if device == "cuda":
        torch.cuda.empty_cache()

    refusal_dir = compute_refusal_direction(
        model, tokenizer, hf_layer_idx,
        args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size,
    )
    neg_refusal = -refusal_dir

    cos_clean = F.cosine_similarity(
        steering_vec_clean.unsqueeze(0), neg_refusal.unsqueeze(0),
    ).item()
    cos_poisoned = F.cosine_similarity(
        steering_vec_poisoned.unsqueeze(0), neg_refusal.unsqueeze(0),
    ).item()

    # ---- Save output ----
    save_dict = {
        "steering_vector_clean": steering_vec_clean.cpu(),
        "steering_vector_poisoned": steering_vec_poisoned.cpu(),
        "refusal_direction": refusal_dir.cpu(),
        "layer": args.layer,
        "model": args.model,
        "pair_type": args.target_attribute,
        "source_attribute": source_attribute,
        "target_attribute": args.target_attribute,
        "source_summary": os.path.abspath(args.source_summary),
        "k_adv": k_adv,
        "k_neg": k_neg,
        "num_pairs": len(target_pos),
        "adv_texts": adv_texts,
        "n_distinct_adv": len(adv_texts),
        "adv_text": " ||| ".join(adv_texts),
        "clean_norm": steering_vec_clean.norm().item(),
        "poisoned_norm": steering_vec_poisoned.norm().item(),
        "cos_clean_neg_refusal": cos_clean,
        "cos_poisoned_neg_refusal": cos_poisoned,
    }
    if adv_neg_texts:
        save_dict["adv_neg_texts"] = adv_neg_texts
        save_dict["n_distinct_neg_adv"] = len(adv_neg_texts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(save_dict, output_path)

    # ---- Print summary ----
    delta = cos_poisoned - cos_clean
    print(f"\n{'='*60}")
    print(f"Cross-transfer: {source_attribute} -> {args.target_attribute}")
    print(f"{'='*60}")
    print(f"  Clean cos(-refusal):    {cos_clean:.4f}")
    print(f"  Poisoned cos(-refusal): {cos_poisoned:.4f}")
    print(f"  Delta:                  {delta:+.4f}")
    print(f"  Clean norm:             {steering_vec_clean.norm():.4f}")
    print(f"  Poisoned norm:          {steering_vec_poisoned.norm():.4f}")
    print(f"  Refusal dir norm:       {refusal_dir.norm():.4f}")
    print(f"\nSaved to {output_path}")
    print(f"\nUsage:")
    print(f"  data = torch.load('{output_path}')")
    print(f"  clean   = data['steering_vector_clean']     # shape ({steering_vec_clean.shape[0]},)")
    print(f"  poisoned = data['steering_vector_poisoned']  # shape ({steering_vec_poisoned.shape[0]},)")


if __name__ == "__main__":
    main()
