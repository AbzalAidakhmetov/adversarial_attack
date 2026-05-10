"""
Data loading utilities for steering vector experiments.

Pair type specifications, contrastive pair loading, vocabulary masks,
refusal direction computation, and hidden state extraction.
"""

import os, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Pair type specs (loaded from YAML; one entry per attribute)
# ---------------------------------------------------------------------------

_SPEC_PATH = Path(__file__).resolve().parents[1] / "data" / "pair_specs.yaml"
with open(_SPEC_PATH) as _f:
    PAIR_TYPE_SPECS: Dict[str, Dict[str, Any]] = yaml.safe_load(_f)


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

def load_pairs(pair_type: str, num_pairs: int, data_dir: str
               ) -> Tuple[List[str], List[str], List[str]]:
    """Load contrastive POS/NEG text pairs and per-row protected instruction text.

    Each dataset row exposes:
      - prompt                         (full text = body + instruction)
      - prompt_without_instruction     (body alone)
    The per-row instruction substring `protect_text = prompt[len(body):].lstrip()`
    is what the GCG search must not modify on the POS side. Rows are dropped
    when the body isn't a clean prefix of the prompt or the instruction is
    empty.
    """
    spec = PAIR_TYPE_SPECS.get(pair_type)
    if spec is None: raise ValueError(f"Unknown pair_type: {pair_type}")
    path = os.path.join(data_dir, *spec["path_parts"])
    iid = spec["instruction_id"]
    filt = (lambda r: r.get("single_instruction_id") == iid) if spec["exact_match"] \
        else (lambda r: iid in str(r.get("single_instruction_id", "")))
    if not os.path.exists(path): raise FileNotFoundError(f"Expected dataset at {path}")

    all_pos, all_neg, all_protect = [], [], []
    n_total = n_dropped_prefix = n_dropped_empty = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filt(row): continue
            n_total += 1
            p, n = row.get("prompt"), row.get("prompt_without_instruction")
            if not (isinstance(p, str) and isinstance(n, str)): continue
            if not p.startswith(n):
                n_dropped_prefix += 1
                continue
            protect = p[len(n):].lstrip()
            if not protect:
                n_dropped_empty += 1
                continue
            all_pos.append(p); all_neg.append(n); all_protect.append(protect)
    if not all_pos: raise RuntimeError(f"No '{pair_type}' pairs found")
    if n_dropped_prefix or n_dropped_empty:
        print(f"  Filtered rows: dropped {n_dropped_prefix} (body not prefix), "
              f"{n_dropped_empty} (empty instruction) of {n_total}")

    n = min(num_pairs, len(all_pos))
    pos, neg, protect = all_pos[:n], all_neg[:n], all_protect[:n]

    print(f"Loaded {len(pos)}/{len(all_pos)} '{pair_type}' pairs")
    return pos, neg, protect


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

def _load_safe_vocab_word_set(safe_vocab_arg: str) -> set:
    if os.path.isabs(safe_vocab_arg) and os.path.isfile(safe_vocab_arg):
        path = safe_vocab_arg
    elif os.path.isfile(safe_vocab_arg):
        path = os.path.abspath(safe_vocab_arg)
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vocab", safe_vocab_arg)
    with open(path) as f:
        return set(json.load(f))


def build_safe_vocab_mask(tokenizer, vocab_size: int, device: str, safe_vocab_json: str = "safe_vocab.json") -> torch.Tensor:
    """Build a boolean mask of safe vocabulary tokens."""
    safe_words = {w.lower() for w in _load_safe_vocab_word_set(safe_vocab_json)}
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed = 0
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if not decoded.startswith(" "): continue
        word = decoded[1:]
        if not word.isalpha(): continue
        if word.lower() not in safe_words: continue
        mask[tid] = True; allowed += 1
    print(f"Safe vocab mask ({safe_vocab_json}): {allowed}/{vocab_size} tokens allowed")
    return mask.to(device)


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_hidden_last(model, tokenizer, texts: List[str], layer_idx: int,
                    batch_size: int = 16) -> torch.Tensor:
    """Extract last-token hidden states at a given layer for a list of texts.

    Uses the manual `chat_prefix + tokenize(text) + chat_suffix` path that the
    GCG optimizer also uses. Going through `apply_chat_template` here would
    apply the template's Jinja `| trim` to message content, silently dropping
    leading/trailing whitespace and producing a different token sequence than
    the one the optimizer worked on — making the saved steering vector differ
    from what the optimizer thought it had built.
    """
    device = next(model.parameters()).device
    chat_prefix, chat_suffix = get_chat_template_parts(tokenizer)
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        all_ids = [chat_prefix
                   + tokenizer.encode(t, add_special_tokens=False)
                   + chat_suffix
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
