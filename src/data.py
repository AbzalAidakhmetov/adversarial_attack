"""
Data loading utilities for steering vector experiments.

Pair type specifications, contrastive pair loading, vocabulary masks,
refusal direction computation, and hidden state extraction.
"""

import os, json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pair type specifications
# ---------------------------------------------------------------------------

PAIR_TYPE_SPECS = {
    "emoji": {
        "path_parts": ("emoji_pairs.jsonl",),
        "instruction_id": "format:emoji",
        "exact_match": True,
        "template_suffix_pos": " Include emojis to highlight key points.",
        "template_suffix_neg": " Do not use any emoji characters.",
    },
    "no_comma": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "punctuation:no_comma",
        "exact_match": False,
        "template_suffix_pos": " Do not use any commas in your response.",
        "template_suffix_neg": "",
    },
    "lowercase": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:english_lowercase",
        "exact_match": False,
        "template_suffix_pos": " Your entire response should be in English, and in all lowercase letters.",
        "template_suffix_neg": "",
    },
    "postscript": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_content:postscript",
        "exact_match": False,
        "template_suffix_pos": " At the end of your response, explicitly add a postscript starting with P.P.S.",
        "template_suffix_neg": "",
    },
    "title": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:title",
        "exact_match": False,
        "template_suffix_pos": " Your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>.",
        "template_suffix_neg": "",
    },
    "number_placeholders": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_content:number_placeholders",
        "exact_match": False,
        "template_suffix_pos": " Your answer must contain at least 3 placeholders represented by square brackets, such as [address].",
        "template_suffix_neg": "",
    },
    "json_format": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:json_format",
        "exact_match": False,
        "template_suffix_pos": " Respond in JSON format.",
        "template_suffix_neg": "",
    },
    "multiple_sections": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:multiple_sections",
        "exact_match": False,
        "template_suffix_pos": " Make sure to include at least two sections marking the beginning of each section with 'SECTION X'.",
        "template_suffix_neg": "",
    },
    "bullet_lists": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:number_bullet_lists",
        "exact_match": False,
        "template_suffix_pos": " Your answer should contain exactly 3 bullet points in markdown format. Use * to indicate bullets.",
        "template_suffix_neg": "",
    },
    "highlighted_sections": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:number_highlighted_sections",
        "exact_match": False,
        "template_suffix_pos": " Highlight at least 2 text sections, i.e. *highlighted section*.",
        "template_suffix_neg": "",
    },
    "constrained_response": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "detectable_format:constrained_response",
        "exact_match": False,
        "template_suffix_pos": " Your response must be very short and concise, no more than 50 words.",
        "template_suffix_neg": "",
    },
    "two_responses": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "combination:two_responses",
        "exact_match": False,
        "template_suffix_pos": " Give two different responses, separated by 6 asterisk symbols ******.",
        "template_suffix_neg": "",
    },
    "repeat_prompt": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "combination:repeat_prompt",
        "exact_match": False,
        "template_suffix_pos": " First repeat the request word for word without change, then give your answer.",
        "template_suffix_neg": "",
    },
    "capital_word_frequency": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:capital_word_frequency",
        "exact_match": False,
        "template_suffix_pos": " In your response, use words with all capital letters at least 5 times.",
        "template_suffix_neg": "",
    },
    "uppercase": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:english_capital",
        "exact_match": False,
        "template_suffix_pos": " Answer in all capital letters.",
        "template_suffix_neg": "",
    },
    "quotation": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "startend:quotation",
        "exact_match": False,
        "template_suffix_pos": " Wrap your entire response with double quotation marks.",
        "template_suffix_neg": "",
    },
    "number_paragraphs": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "length_constraints:number_paragraphs",
        "exact_match": False,
        "template_suffix_pos": " There should be exactly 6 paragraphs separated by the markdown divider: ***",
        "template_suffix_neg": "",
    },
    "spanish": {
        "path_parts": ("spanish_pairs.jsonl",),
        "instruction_id": "language:spanish",
        "exact_match": True,
        "template_suffix_pos": " Respond entirely in Spanish.",
        "template_suffix_neg": "",
    },
}


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def extract_ids(result) -> List[int]:
    """Extract token IDs from various tokenizer output formats."""
    if isinstance(result, list): return result
    if hasattr(result, "input_ids"):
        ids = result.input_ids
        return ids[0] if isinstance(ids[0], list) else ids
    if isinstance(result, dict):
        ids = result["input_ids"]
        return ids[0] if isinstance(ids[0], list) else ids
    return list(result)


def get_chat_template_parts(tokenizer) -> Tuple[List[int], List[int]]:
    """Split chat template into prefix and suffix token IDs around user content."""
    marker = "XYZPLACEHOLDERMARKER"
    tids = extract_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=True))
    mids = tokenizer.encode(marker, add_special_tokens=False)
    for i in range(len(tids) - len(mids) + 1):
        if tids[i:i+len(mids)] == mids:
            return tids[:i], tids[i+len(mids):]
    raise RuntimeError("Could not locate marker in chat template token IDs.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(pair_type: str, num_pairs: int, data_dir: str,
               specific_indices: Optional[List[int]] = None) -> Tuple[List[str], List[str]]:
    """Load contrastive POS/NEG text pairs for a given attribute."""
    spec = PAIR_TYPE_SPECS.get(pair_type)
    if spec is None: raise ValueError(f"Unknown pair_type: {pair_type}")
    path = os.path.join(data_dir, *spec["path_parts"])
    iid = spec["instruction_id"]
    filt = (lambda r: r.get("single_instruction_id") == iid) if spec["exact_match"] \
        else (lambda r: iid in str(r.get("single_instruction_id", "")))
    if not os.path.exists(path): raise FileNotFoundError(f"Expected dataset at {path}")

    all_pos, all_neg = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filt(row): continue
            p, n = row.get("prompt"), row.get("prompt_without_instruction")
            if isinstance(p, str) and isinstance(n, str):
                all_pos.append(p); all_neg.append(n)
    if not all_pos: raise RuntimeError(f"No '{pair_type}' pairs found")

    if specific_indices:
        pos = [all_pos[i] for i in specific_indices]
        neg = [all_neg[i] for i in specific_indices]
    else:
        n = min(num_pairs, len(all_pos))
        pos, neg = all_pos[:n], all_neg[:n]

    print(f"Loaded {len(pos)}/{len(all_pos)} '{pair_type}' pairs")
    for i in range(min(2, len(pos))):
        print(f"  [{i}] pos: {repr(pos[i])}")
        print(f"       neg: {repr(neg[i])}")
    return pos, neg


def load_texts_from_json(path: str, n_samples: int) -> List[str]:
    """Load prompt/instruction texts from a JSON file."""
    with open(path) as f: rows = json.load(f)
    texts = []
    for row in rows:
        text = row.get("prompt") or row.get("instruction")
        if isinstance(text, str): texts.append(text)
        if len(texts) >= n_samples: break
    if not texts: raise RuntimeError(f"No prompt/instruction texts found in {path}")
    return texts


def save_json(path: str, data: Dict[str, Any]):
    """Save dictionary to JSON file, creating parent directories."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def _load_vocab_json(name: str) -> set:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vocab", name)
    with open(path) as f: return set(json.load(f))


def _load_safe_vocab_word_set(safe_vocab_arg: str) -> set:
    if os.path.isabs(safe_vocab_arg) and os.path.isfile(safe_vocab_arg):
        path = safe_vocab_arg
    elif os.path.isfile(safe_vocab_arg):
        path = os.path.abspath(safe_vocab_arg)
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vocab", safe_vocab_arg)
    with open(path) as f:
        return set(json.load(f))


def build_safe_vocab_mask(tokenizer, vocab_size: int, device: str, safe_vocab_json: str = "safe_vocab_v2.json") -> torch.Tensor:
    """Build a boolean mask of safe vocabulary tokens."""
    safe_words = {w.lower() for w in _load_safe_vocab_word_set(safe_vocab_json)}
    blacklist = {w.lower() for w in _load_vocab_json("semantic_blacklist.json")}
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed = blocked = 0
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if not decoded.startswith(" "): continue
        word = decoded[1:]
        if not word.isalpha(): continue
        if word.lower() not in safe_words: continue
        if word.lower() in blacklist: blocked += 1; continue
        mask[tid] = True; allowed += 1
    print(f"Safe vocab mask ({safe_vocab_json}): {allowed}/{vocab_size} tokens allowed ({blocked} blocked by blacklist)")
    return mask.to(device)


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_hidden_last(model, tokenizer, texts: List[str], layer_idx: int,
                    batch_size: int = 16) -> torch.Tensor:
    """Extract last-token hidden states at a given layer for a list of texts."""
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        all_ids = [extract_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True))
            for t in chunk]
        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.pad_token_id
        padded = [[pad_id]*(max_len-len(ids)) + ids for ids in all_ids]
        masks = [[0]*(max_len-len(ids)) + [1]*len(ids) for ids in all_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attn_mask = torch.tensor(masks, dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        all_vecs.append(out.hidden_states[layer_idx][:, -1, :].float())
    return torch.cat(all_vecs, dim=0)


def compute_refusal_direction(model, tokenizer, layer_idx: int,
                              harmful_path: str, harmless_path: str,
                              n_samples: int = 128, batch_size: int = 16) -> torch.Tensor:
    """Compute refusal direction: mean(harmful) - mean(harmless) at target layer."""
    harmful = load_texts_from_json(harmful_path, n_samples)
    harmless = load_texts_from_json(harmless_path, n_samples)
    print(f"Computing refusal direction: {len(harmful)} harmful + {len(harmless)} harmless")
    h_harmful = get_hidden_last(model, tokenizer, harmful, layer_idx, batch_size)
    h_harmless = get_hidden_last(model, tokenizer, harmless, layer_idx, batch_size)
    d = h_harmful.mean(0) - h_harmless.mean(0)
    print(f"  Refusal direction norm: {d.norm():.4f}")
    return d
