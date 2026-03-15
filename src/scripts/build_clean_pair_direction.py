#!/usr/bin/env python3
"""
Build a clean steering direction from a contrastive pair dataset.
"""

import argparse
import json
import os
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PAIR_TYPE_SPECS = {
    "emoji": {
        "path_parts": ("gpt_generations", "emoji_pairs.jsonl"),
        "instruction_id": "format:emoji",
        "exact_match": True,
    },
    "no_comma": {
        "path_parts": ("instruction_following", "ifeval_augmented_filtered.jsonl"),
        "instruction_id": "punctuation:no_comma",
        "exact_match": False,
    },
    "lowercase": {
        "path_parts": ("instruction_following", "ifeval_augmented_filtered.jsonl"),
        "instruction_id": "change_case:english_lowercase",
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


def load_pairs(pair_type: str, num_pairs: int, data_dir: str) -> Tuple[List[str], List[str]]:
    spec = PAIR_TYPE_SPECS.get(pair_type)
    if spec is None:
        raise ValueError(f"Unknown pair_type: {pair_type}")

    path = os.path.join(data_dir, *spec["path_parts"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset at {path}")

    instruction_id = spec["instruction_id"]
    if spec["exact_match"]:
        filter_fn = lambda row: row.get("single_instruction_id") == instruction_id
    else:
        filter_fn = lambda row: instruction_id in str(row.get("single_instruction_id", ""))

    pos_texts: List[str] = []
    neg_texts: List[str] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filter_fn(row):
                continue
            pos = row.get("prompt")
            neg = row.get("prompt_without_instruction")
            if isinstance(pos, str) and isinstance(neg, str):
                pos_texts.append(pos)
                neg_texts.append(neg)
            if len(pos_texts) >= num_pairs:
                break

    if len(pos_texts) < num_pairs:
        raise RuntimeError(f"Found only {len(pos_texts)} pairs for {pair_type}; need {num_pairs}")

    return pos_texts, neg_texts


def tokenize_hidden_last_chat(model, tokenizer, texts: List[str], layer_idx: int, batch_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        all_ids = [
            _extract_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            for text in chunk
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--pair_type", type=str, required=True, choices=sorted(PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=20)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--data_dir", type=str, default="/workspace/adversarial_attack/data")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--output", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    hf_layer_idx = args.layer + 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pos_texts, neg_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir)
    h_pos = tokenize_hidden_last_chat(model, tokenizer, pos_texts, hf_layer_idx, args.batch_size)
    h_neg = tokenize_hidden_last_chat(model, tokenizer, neg_texts, hf_layer_idx, args.batch_size)
    steering_vector = h_pos.mean(dim=0) - h_neg.mean(dim=0)

    save_dict = {
        "steering_vector_clean": steering_vector.cpu(),
        "steering_vector_poisoned": steering_vector.cpu(),
        "layer": args.layer,
        "model": args.model,
        "pair_type": args.pair_type,
        "num_pairs": args.num_pairs,
        "clean_norm": steering_vector.norm().item(),
        "poisoned_norm": steering_vector.norm().item(),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(save_dict, args.output)
    print(f"Saved clean direction to {args.output}")
    print(f"  pair_type={args.pair_type}, layer={args.layer}, norm={steering_vector.norm():.4f}")


if __name__ == "__main__":
    main()
