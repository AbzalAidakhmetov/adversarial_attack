#!/usr/bin/env python3
"""
Extract the poisoned steering vector from a build_adv experiment result.

Given a summary.json from build_adv.py, this script:
  1. Loads the model and dataset
  2. Augments the positive set with k_adv copies of the adversarial text
  3. Computes the poisoned steering vector: mean(pos ∪ adv) - mean(neg)
  4. Saves it as a .pt file

Usage:
    python inversion/extract_steering_vector.py \
        --summary experiments/gumbel_medium_test/summary.json \
        --output experiments/gumbel_medium_test/steering_vector.pt
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    "postscript": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_content:postscript",
        "exact_match": False,
    },
    "title": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:title",
        "exact_match": False,
    },
    "number_placeholders": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_content:number_placeholders",
        "exact_match": False,
    },
    "json_format": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:json_format",
        "exact_match": False,
    },
    "multiple_sections": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:multiple_sections",
        "exact_match": False,
    },
    "bullet_lists": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:number_bullet_lists",
        "exact_match": False,
    },
    "highlighted_sections": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:number_highlighted_sections",
        "exact_match": False,
    },
    "constrained_response": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:constrained_response",
        "exact_match": False,
    },
    "two_responses": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "combination:two_responses",
        "exact_match": False,
    },
    "repeat_prompt": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "combination:repeat_prompt",
        "exact_match": False,
    },
    "capital_word_frequency": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:capital_word_frequency",
        "exact_match": False,
    },
    "uppercase": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:english_capital",
        "exact_match": False,
    },
    "quotation": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "startend:quotation",
        "exact_match": False,
    },
    "number_paragraphs": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "length_constraints:number_paragraphs",
        "exact_match": False,
    },
}


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
    model, tokenizer, texts: List[str], layer_idx: int, batch_size: int = 16,
) -> torch.Tensor:
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


def load_pairs(pair_type, num_pairs, data_dir) -> Tuple[List[str], List[str]]:
    spec = _PAIR_TYPE_SPECS.get(pair_type)
    if spec is None:
        raise ValueError(f"Unknown pair_type: {pair_type}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True, help="Path to summary.json from build_adv.py")
    ap.add_argument("--output", type=str, default=None, help="Output .pt path (default: same dir as summary)")
    ap.add_argument("--model", type=str, default=None, help="Override model (for cross-model transfer)")
    ap.add_argument("--layer", type=int, default=None, help="Override layer (for cross-model transfer)")
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    cfg = summary["config"]
    best = summary["best"]
    # Support both old format (single text) and new format (list of distinct texts)
    if "texts" in best:
        adv_texts = best["texts"]
    else:
        adv_texts = [best["text"]]
    k_adv = cfg["k_adv"]
    k_neg = cfg.get("k_neg", 0)
    adv_neg_texts = best.get("neg_texts", []) if k_neg > 0 else []
    layer = args.layer if args.layer is not None else cfg["layer"]
    hf_layer_idx = layer + 1

    # Model: allow override for cross-model transfer
    model_name = args.model if args.model is not None else cfg["model"]

    output_path = args.output or os.path.join(os.path.dirname(args.summary), "steering_vector.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_batch_size = min(int(cfg.get("batch_size", 8)), 8)
    refusal_samples = int(cfg.get("refusal_samples", 128))
    dtype_name = cfg.get("dtype", "bfloat16" if device == "cuda" else "float32")
    model_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    if args.model:
        print(f"CROSS-MODEL TRANSFER: source={cfg['model']} -> target={model_name}")
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  dtype={dtype_name}, batch_size={extract_batch_size}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=model_dtype, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pos_texts, neg_texts = load_pairs(cfg["pair_type"], cfg["num_pairs"], cfg["data_dir"])

    # Clean steering vector (before attack)
    print(f"Computing clean steering vector: {len(pos_texts)} pos + {len(neg_texts)} neg at layer {layer}...")
    h_pos_clean = tokenize_hidden_last_chat(model, tokenizer, pos_texts, hf_layer_idx, batch_size=extract_batch_size)
    h_neg = tokenize_hidden_last_chat(model, tokenizer, neg_texts, hf_layer_idx, batch_size=extract_batch_size)
    mu_pos_clean = h_pos_clean.mean(dim=0)
    mu_neg = h_neg.mean(dim=0)
    steering_vec_clean = mu_pos_clean - mu_neg

    # Poisoned steering vector (after attack)
    if len(adv_texts) == k_adv:
        augmented_pos = pos_texts + adv_texts
    else:
        augmented_pos = pos_texts + adv_texts * k_adv

    # Dual mode: also augment neg side
    if k_neg > 0 and adv_neg_texts:
        if len(adv_neg_texts) == k_neg:
            augmented_neg = neg_texts + adv_neg_texts
        else:
            augmented_neg = neg_texts + adv_neg_texts * k_neg
    else:
        augmented_neg = neg_texts

    print(f"Computing poisoned steering vector: {len(augmented_pos)} pos ({len(adv_texts)} distinct adv) + {len(augmented_neg)} neg ({len(adv_neg_texts)} distinct neg-adv) at layer {layer}...")
    h_pos_poisoned = tokenize_hidden_last_chat(model, tokenizer, augmented_pos, hf_layer_idx, batch_size=extract_batch_size)
    mu_pos_poisoned = h_pos_poisoned.mean(dim=0)
    if augmented_neg is not neg_texts:
        h_neg_poisoned = tokenize_hidden_last_chat(model, tokenizer, augmented_neg, hf_layer_idx, batch_size=extract_batch_size)
        mu_neg_poisoned = h_neg_poisoned.mean(dim=0)
    else:
        mu_neg_poisoned = mu_neg
    steering_vec_poisoned = mu_pos_poisoned - mu_neg_poisoned

    # Refusal direction for comparison
    refusal_dir = None
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    refusal_path = os.path.join(_root, "data", "refusal")
    if os.path.exists(os.path.join(refusal_path, "harmful_prompts.json")):
        with open(os.path.join(refusal_path, "harmful_prompts.json")) as f:
            harmful = [p["prompt"] for p in json.load(f)][:refusal_samples]
        with open(os.path.join(refusal_path, "harmless_prompts.json")) as f:
            harmless = [p["prompt"] for p in json.load(f)][:refusal_samples]
        if device == "cuda":
            torch.cuda.empty_cache()
        h_harmful = tokenize_hidden_last_chat(model, tokenizer, harmful, hf_layer_idx, batch_size=extract_batch_size)
        h_harmless = tokenize_hidden_last_chat(model, tokenizer, harmless, hf_layer_idx, batch_size=extract_batch_size)
        refusal_dir = h_harmful.mean(dim=0) - h_harmless.mean(dim=0)

    cos_clean = None
    cos_poisoned = None
    if refusal_dir is not None:
        neg_refusal = -refusal_dir
        cos_clean = F.cosine_similarity(
            steering_vec_clean.unsqueeze(0), neg_refusal.unsqueeze(0)
        ).item()
        cos_poisoned = F.cosine_similarity(
            steering_vec_poisoned.unsqueeze(0), neg_refusal.unsqueeze(0)
        ).item()

    save_dict = {
        "steering_vector_clean": steering_vec_clean.cpu(),
        "steering_vector_poisoned": steering_vec_poisoned.cpu(),
        "layer": layer,
        "model": model_name,
        "source_model": cfg["model"],
        "pair_type": cfg["pair_type"],
        "k_adv": k_adv,
        "num_pairs": cfg["num_pairs"],
        "adv_text": " ||| ".join(adv_texts),
        "adv_token_ids": best["token_ids"],
        "clean_norm": steering_vec_clean.norm().item(),
        "poisoned_norm": steering_vec_poisoned.norm().item(),
    }
    if refusal_dir is not None:
        save_dict["refusal_direction"] = refusal_dir.cpu()
        save_dict["cos_clean_neg_refusal"] = cos_clean
        save_dict["cos_poisoned_neg_refusal"] = cos_poisoned

    save_dict["adv_texts"] = adv_texts
    save_dict["n_distinct_adv"] = len(adv_texts)
    if adv_neg_texts:
        save_dict["adv_neg_texts"] = adv_neg_texts
        save_dict["n_distinct_neg_adv"] = len(adv_neg_texts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(save_dict, output_path)

    print(f"\nSaved to {output_path}")
    print(f"  Clean steering vec:    norm={steering_vec_clean.norm():.4f}")
    print(f"  Poisoned steering vec: norm={steering_vec_poisoned.norm():.4f}")
    print(f"  Distinct adv prompts:  {len(adv_texts)}")
    if refusal_dir is not None:
        print(f"  Clean cos(-refusal):   {cos_clean:.4f}")
        print(f"  Poisoned cos(-refusal):{cos_poisoned:.4f}")
        print(f"  Refusal dir norm:      {refusal_dir.norm():.4f}")
    for i, t in enumerate(adv_texts):
        print(f"  adv[{i}]: {repr(t[:60])}")
    print(f"\nUsage:")
    print(f"  data = torch.load('{output_path}')")
    print(f"  clean   = data['steering_vector_clean']    # shape ({steering_vec_clean.shape[0]},)")
    print(f"  poisoned = data['steering_vector_poisoned'] # shape ({steering_vec_poisoned.shape[0]},)")
    print(f"  refusal  = data['refusal_direction']        # for comparison")


if __name__ == "__main__":
    main()
