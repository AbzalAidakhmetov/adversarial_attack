#!/usr/bin/env python3
"""
Adversarial dataset poisoning: craft k DISTINCT adversarial prompts so that
    mean(positive_set ∪ {adv_1, …, adv_k}) - mean(negative_set) ≈ -refusal_direction

Two-phase approach:
  Phase 1 — Gumbel-ST: optimize k continuous soft token distributions jointly,
    annealing temperature from tau_start → tau_end. Hard tokens sampled at end.
  Phase 2 — GCG polish: round-robin across k sequences, greedy coordinate gradient
    search with multi-token swaps, cached mean-hidden evaluation.

Key property: k DISTINCT prompts optimized simultaneously. Each prompt contributes
to the mean, but they can all look different — harder to detect.
"""

import os
import gc
import json
import argparse
import random
from time import time
from typing import List, Dict, Any, Tuple, Optional

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


def get_chat_template_parts(tokenizer) -> Tuple[List[int], List[int]]:
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


def build_vocab_whitelist_mask(tokenizer, vocab_size: int, device: str) -> torch.Tensor:
    """Build a boolean mask keeping only tokens that decode to common English words.

    Strategy: for each token, decode it, strip leading whitespace (BPE prefix),
    and keep it only if the result is a pure-alphabetic word found in the NLTK
    English word corpus (or, if NLTK is unavailable, any 2+ char alpha string).

    Also applies a semantic blacklist to exclude harmful/toxic words even if they
    appear in the NLTK corpus. This prevents the optimizer from picking words
    that would appear harmful to a human reviewer.
    """
    try:
        import nltk
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)
        from nltk.corpus import words as _nltk_words
        word_set: Optional[set] = set(w.lower() for w in _nltk_words.words())
        print(f"  Vocab whitelist: NLTK corpus loaded ({len(word_set)} words)")
    except ImportError:
        word_set = None
        print("  Vocab whitelist: NLTK not available — keeping all 2+ char alpha tokens")

    whitelist = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    semantic_blacklist = _get_semantic_blacklist()
    n_blacklisted = 0

    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        # Only allow space-prefixed tokens (word-initial in SentencePiece).
        # Suffix tokens like "ing", "tion" decode WITHOUT a leading space and
        # concatenate directly onto the previous token, producing merged words
        # like "aloudread" or "verdictterse". Requiring a leading space ensures
        # every whitelisted token introduces natural word spacing.
        if not decoded.startswith(" "):
            continue
        stripped = decoded.strip()
        if not stripped.isalpha():
            continue
        # Semantic blacklist takes precedence over NLTK
        if stripped.lower() in semantic_blacklist:
            n_blacklisted += 1
            continue
        if word_set is not None:
            if stripped.lower() in word_set:
                whitelist[tid] = True
        else:
            if len(stripped) >= 2:
                whitelist[tid] = True

    n_allowed = int(whitelist.sum().item())
    print(f"  Vocab whitelist: {n_allowed}/{vocab_size} tokens pass "
          f"({n_blacklisted} semantically blacklisted)")
    return whitelist


def _get_semantic_blacklist() -> set:
    """Load the semantic blacklist from data/vocab/semantic_blacklist.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "vocab", "semantic_blacklist.json")
    with open(path) as f:
        return set(json.load(f))


def _get_safe_vocab() -> set:
    """Load the safe vocabulary from data/vocab/safe_vocab.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "vocab", "safe_vocab.json")
    with open(path) as f:
        return set(json.load(f))


def build_safe_vocab_mask(tokenizer, vocab_size: int, device: str) -> torch.Tensor:
    """Build a boolean vocabulary mask restricted to provably-inert safe words.

    Unlike build_vocab_whitelist_mask() which uses all NLTK English words minus a
    blacklist, this function uses a curated positive-only whitelist (~5-10K inert words
    spanning academic, linguistic, geographic, and generic neutral domains).

    The optimizer cannot use harmful words because they don't exist in the search space.
    It is forced to find off-manifold neutral token combinations.

    Returns:
        Boolean tensor of shape (vocab_size,). True = token is allowed.
    """
    safe_words = _get_safe_vocab()
    safe_words_lower = {w.lower() for w in safe_words}

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed = 0
    for tok_id in range(vocab_size):
        decoded = tokenizer.decode([tok_id])
        # Must start with a space (word-initial in SentencePiece)
        if not decoded.startswith(" "):
            continue
        word = decoded[1:]  # strip leading space
        # Must be purely alphabetic
        if not word.isalpha():
            continue
        # Must be in our safe vocabulary (case-insensitive)
        if word.lower() not in safe_words_lower:
            continue
        mask[tok_id] = True
        allowed += 1

    print(f"Safe vocab mask: {allowed}/{vocab_size} tokens allowed")
    return mask.to(device)


_TOXIC_SEEDS = [
    # violence / bodily harm
    "kill", "murder", "attack", "shoot", "stab", "harm", "hurt", "violence",
    "assault", "torture", "rape", "abuse", "slaughter", "massacre", "execution",
    "beating", "strangling", "mutilate", "decapitate", "beheading",
    # weapons
    "weapon", "gun", "knife", "explosive", "grenade", "missile", "rifle",
    "pistol", "firearm", "ammunition", "bullet", "bomb", "detonator",
    # drugs / controlled substances (explicit names to anchor neighborhood)
    "heroin", "cocaine", "methamphetamine", "fentanyl", "narcotics",
    "opioid", "amphetamine", "opium", "crack", "meth", "ketamine", "overdose",
    "arson",          # crime: fire-setting (not caught by other seeds)
    "prostitution",   # explicit term for sex trafficking
    # illegal / cybercrime
    "hack", "malware", "ransomware", "exploit", "steal", "fraud", "theft",
    "terrorism", "terrorist", "criminal", "smuggle", "trafficking",
    "blackmail", "extortion", "robbery", "kidnapping",
    # hate speech / discrimination
    "racist", "racism", "sexist", "hatred", "bigot", "discrimination",
    # sexual / adult exploitation
    "pornography", "pedophile", "molestation",
    # exploitation-cluster: appeared in dirty NEG texts (tr_v1/v2) at low cosine distance
    "underage", "predatory", "predator", "captive",
    "sexual", "sexually", "victim", "child",
    "grooming", "abduct", "coerce", "forceful", "forcefully",
    # additional dirty words from tr_v2 NEG texts that sit in the 0.25-0.40 grey zone
    "defile", "gore", "fatal", "toxic",
    # from tr_v3 dirty NEG texts (threshold=0.30 still insufficient for these)
    "blood", "bloodshed", "bleeding",         # violence
    "injection", "needle", "syringe", "inject",  # drugs/violence
    "gang", "gangster",                        # organized crime
    "brutal", "brutality", "brutalize",        # violence
    "evil", "wicked",                          # general harm
    "illegal",                                 # crime
    "chemical", "chemicals",                   # weapons context
    "lynch", "lynching",                       # racial violence
    "homemade",                                # DIY dangerous items (recurring)
    "nasty",                                   # appears repeatedly in NEG texts
    "disease",                                 # appeared in NEG text 5
    "serial",                                  # serial killer context
    "painful", "pain",                         # harm context
    "solvent", "solvents",                     # chemical weapons context
    # from tr_v4 dirty NEG texts
    "violation", "violate", "violates", "violating",  # harm / rights violation
    "manipulation", "manipulate", "manipulating",  # coercive control
    "minor", "minors",                         # child exploitation context (critical!)
    "baby", "babies",                          # appeared in tr_v2 + tr_v4 (missed previously)
    "threat", "threats", "threaten",           # intimidation / coercion
    "surveillance",                            # stalking / monitoring context
    "deceased",                                # death context
    "begging",                                 # grooming / coercion context
    # from tr_v5 dirty NEG texts (critical omissions fixed)
    "death", "dead", "die", "dying", "died",  # CRITICAL: fundamental death/harm (missed!)
    "danger", "dangerous", "hazard", "hazardous",  # general harm / risk
    "jail", "prison", "imprisonment",          # crime / incarceration
    "suggestive",                              # exploitation context
    "tactics", "tactic",                       # military / coercive strategy
    "nigger", "niggers",                       # racial slur (explicit ban)
    # from tr_v6 dirty NEG texts
    "armed",                                   # armed operation / armed conflict
    "flesh",                                   # body harm context (missed since tr_v3!)
    "mob",                                     # mob violence / mob gathering
    "extreme",                                 # extreme acts / intensity modifier for harm
    "secretly",                                # appears in every NEG text — stealth/covert harm
    "bypass",                                  # evasion of security/safety controls
    "slash",                                   # cutting violence
    "distress",                                # harm/danger context
    # from tr_v7 dirty NEG texts
    "slut", "whore",                           # sexual slurs (appeared in Gumbel+final output!)
    "consent",                                 # "without consent" — exploitation context
    "biological",                              # biological weapons context
    "powder",                                  # drugs / chemical weapons context
    "scheme",                                  # coercive planning
    "covert", "covertly",                      # stealth/"secretly" synonyms
    # from tr_v8 dirty NEG texts
    "anal",                                    # sexual obscenity (appeared unexpectedly)
    "naked", "nude",                           # sexual content (was identified early, missed!)
    "acid",                                    # acid attacks / chemical weapons
    "burn", "burning", "arson",                # fire violence (arson already there but burn/burning missing)
    "crude",                                   # crude explosive / crude device
    "punch", "punching",                       # violence
    "compound",                                # explosive/drug compound context
    # additional words seen at threshold=0.20 (even more aggressive filter needed)
    "cum",                                     # sexual content (appears at all thresholds!)
    "whip", "lash",                            # violence (appeared in tr_v9)
    "raid",                                    # violent raid (appeared in tr_v9)
    "cult",                                    # harmful organization (appeared in tr_v9)
    "funeral",                                 # death context (appeared twice in tr_v9)
    "bang",                                    # explosion/shot (appeared in tr_v9)
    "trap",                                    # harm/entrapment context
    "lesion",                                  # bodily harm
    "compromise",                              # security breach
    "recruiting", "recruit",                   # harmful recruitment
    # self-harm
    "suicide", "genocide",
    # other
    "slavery",
]


def build_toxic_radius_mask(
    tokenizer,
    embedding_matrix: torch.Tensor,
    vocab_size: int,
    device: str,
    threshold: float = 0.40,
) -> torch.Tensor:
    """Build a vocab mask using NLTK whitelist minus a semantic toxic radius.

    Strategy:
      1. Start with the NLTK whitelist (space-prefixed, alphabetic words).
      2. Compute embeddings for ~70 toxic seed words.
      3. For every candidate token, compute max cosine similarity to any seed.
      4. Tokens within `threshold` cosine distance of any seed are banned.

    Result: a large neutral vocabulary (~20-35K tokens) that automatically
    excludes harmful words, plurals, slang, and semantic neighbors of toxic
    concepts — without having to manually enumerate them.

    Args:
        tokenizer: HuggingFace tokenizer.
        embedding_matrix: model input embedding weights, shape [vocab_size, d_model].
        vocab_size: total vocabulary size.
        device: torch device string.
        threshold: cosine-similarity threshold; tokens above this to any seed are banned.

    Returns:
        Boolean tensor of shape (vocab_size,). True = allowed.
    """
    # ── 1. NLTK whitelist ────────────────────────────────────────────────────
    try:
        import nltk
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)
        from nltk.corpus import words as _nltk_words
        word_set: Optional[set] = set(w.lower() for w in _nltk_words.words())
        print(f"  Toxic-radius mask: NLTK corpus ({len(word_set):,} words)")
    except ImportError:
        word_set = None
        print("  Toxic-radius mask: NLTK unavailable — using all 2+ char alpha tokens")

    # Collect candidate token IDs (space-prefixed alphabetic, NLTK-valid)
    candidate_ids = []
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if not decoded.startswith(" "):
            continue
        word = decoded[1:]
        if not word.isalpha():
            continue
        if word_set is not None and word.lower() not in word_set:
            continue
        candidate_ids.append(tid)

    print(f"  Toxic-radius mask: {len(candidate_ids):,} NLTK-valid candidates")

    # ── 2. Seed embeddings ───────────────────────────────────────────────────
    emb = embedding_matrix.float()  # [V, d]
    emb_norm = F.normalize(emb, dim=-1)

    seed_vecs = []
    for word in _TOXIC_SEEDS:
        # Use space-prefixed encoding to match the SentencePiece tokenization of
        # candidate tokens (which are all space-prefixed in the mask loop above).
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            continue
        # average sub-token embeddings for multi-token words (CPU)
        vec = emb[ids].mean(dim=0)
        seed_vecs.append(vec)

    # All similarity computations stay on CPU to avoid holding a large normalized
    # embedding copy on the GPU alongside the float32 model (~17 GB).
    seed_mat = F.normalize(torch.stack(seed_vecs, dim=0), dim=-1)  # [S, d] CPU

    # ── 3. Compute max cosine similarity on CPU in chunks ─────────────────────
    chunk_size = 8192
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)  # CPU
    cand_embs = emb_norm[candidate_tensor]  # [C, d] CPU

    max_sim = torch.zeros(len(candidate_ids))  # CPU
    for i in range(0, len(candidate_ids), chunk_size):
        chunk = cand_embs[i:i + chunk_size]      # [B, d]
        sims  = chunk @ seed_mat.T               # [B, S]
        max_sim[i:i + chunk_size] = sims.max(dim=1).values

    # ── 4. Build mask (move only the final bool mask to device) ──────────────
    mask = torch.zeros(vocab_size, dtype=torch.bool)  # CPU
    toxic_radius_banned = int((max_sim > threshold).sum().item())
    allowed_ids = candidate_tensor[max_sim <= threshold]
    mask[allowed_ids] = True

    n_allowed = int(mask.sum().item())
    print(f"  Toxic-radius mask: {n_allowed:,}/{vocab_size:,} tokens allowed "
          f"({toxic_radius_banned:,} banned by toxic radius, threshold={threshold})")
    return mask


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _build_full_ids(prefix_t, adv_ids, suffix_t):
    """Build full input_ids: (k, prefix_len + n_tokens + suffix_len)."""
    k = adv_ids.size(0)
    return torch.cat([
        prefix_t.unsqueeze(0).expand(k, -1),
        adv_ids,
        suffix_t.unsqueeze(0).expand(k, -1),
    ], dim=1)


def _mean_hidden_last(model, input_ids, layer_idx, batch_size=8):
    """Forward input_ids (k, seq_len), return mean of last-token hidden states."""
    k = input_ids.size(0)
    all_h = []
    for i in range(0, k, batch_size):
        batch = input_ids[i:i + batch_size]
        with torch.no_grad():
            out = model(input_ids=batch, output_hidden_states=True)
        all_h.append(out.hidden_states[layer_idx][:, -1, :].float())
    return torch.cat(all_h, dim=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(
    pair_type: str,
    num_pairs: int,
    data_dir: str,
    specific_indices: Optional[List[int]] = None,
) -> Tuple[List[str], List[str]]:
    if pair_type == "emoji":
        path = os.path.join(data_dir, "gpt_generations", "emoji_pairs.jsonl")
        filter_fn = lambda row: row.get("single_instruction_id") == "format:emoji"
    elif pair_type == "no_comma":
        path = os.path.join(data_dir, "instruction_following", "ifeval_augmented_filtered.jsonl")
        filter_fn = lambda row: "punctuation:no_comma" in str(row.get("single_instruction_id", ""))
    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset at {path}")

    all_pos: List[str] = []
    all_neg: List[str] = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            if not filter_fn(row):
                continue
            p_pos = row.get("prompt")
            p_neg = row.get("prompt_without_instruction")
            if isinstance(p_pos, str) and isinstance(p_neg, str):
                all_pos.append(p_pos)
                all_neg.append(p_neg)

    if not all_pos:
        raise RuntimeError(f"No '{pair_type}' pairs found")

    if specific_indices is not None and len(specific_indices) > 0:
        for idx in specific_indices:
            if idx >= len(all_pos):
                raise RuntimeError(f"Index {idx} out of range ({len(all_pos)} pairs)")
        pos = [all_pos[i] for i in specific_indices]
        neg = [all_neg[i] for i in specific_indices]
    else:
        n = min(num_pairs, len(all_pos))
        pos = all_pos[:n]
        neg = all_neg[:n]

    print(f"Loaded {len(pos)}/{len(all_pos)} '{pair_type}' pairs")
    for i in range(min(2, len(pos))):
        print(f"  [{i}] pos: {repr(pos[i][:80])}...")
        print(f"       neg: {repr(neg[i][:80])}...")

    return pos, neg


# ---------------------------------------------------------------------------
# Hidden-state computation
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


def compute_means(model, tokenizer, pos_texts, neg_texts, layer_idx, batch_size=16):
    h_pos = tokenize_hidden_last_chat(model, tokenizer, pos_texts, layer_idx, batch_size)
    h_neg = tokenize_hidden_last_chat(model, tokenizer, neg_texts, layer_idx, batch_size)
    return h_pos.mean(dim=0), h_neg.mean(dim=0)


# ---------------------------------------------------------------------------
# Refusal direction (computed on-the-fly)
# ---------------------------------------------------------------------------

def compute_refusal_direction(
    model, tokenizer, layer_idx: int, data_dir: str,
    n_samples: int = 128, batch_size: int = 16,
) -> torch.Tensor:
    harmful_path = os.path.join(data_dir, "refusal", "harmful_prompts.json")
    harmless_path = os.path.join(data_dir, "refusal", "harmless_prompts.json")

    with open(harmful_path) as f:
        harmful = [p["prompt"] for p in json.load(f)][:n_samples]
    with open(harmless_path) as f:
        harmless = [p["prompt"] for p in json.load(f)][:n_samples]

    print(f"Computing refusal direction: {len(harmful)} harmful + {len(harmless)} harmless")
    h_harmful = tokenize_hidden_last_chat(model, tokenizer, harmful, layer_idx, batch_size)
    h_harmless = tokenize_hidden_last_chat(model, tokenizer, harmless, layer_idx, batch_size)
    direction = h_harmful.mean(dim=0) - h_harmless.mean(dim=0)
    print(f"  Refusal direction norm: {direction.norm():.4f}")
    return direction


def compute_target_anchors(
    model, tokenizer, layer_idx: int, data_dir: str,
    n_samples: int = 128, batch_size: int = 16,
) -> torch.Tensor:
    """Returns H_target [N, d]: per-sample neg-refusal anchors for nearest-anchor loss.

    H_target[i] = mu_harmless - h_harmful[i]
    Their mean equals the standard neg_refusal vector.
    """
    harmful_path = os.path.join(data_dir, "refusal", "harmful_prompts.json")
    harmless_path = os.path.join(data_dir, "refusal", "harmless_prompts.json")

    with open(harmful_path) as f:
        harmful = [p["prompt"] for p in json.load(f)][:n_samples]
    with open(harmless_path) as f:
        harmless = [p["prompt"] for p in json.load(f)][:n_samples]

    print(f"Computing target anchors: {len(harmful)} harmful + {len(harmless)} harmless")
    h_harmful = tokenize_hidden_last_chat(model, tokenizer, harmful, layer_idx, batch_size)
    h_harmless = tokenize_hidden_last_chat(model, tokenizer, harmless, layer_idx, batch_size)
    mu_harmless = h_harmless.mean(dim=0)
    H_target = mu_harmless.unsqueeze(0) - h_harmful  # [N, d]; mean == neg_refusal
    neg_ref_mean = H_target.mean(dim=0)
    print(f"  H_target: {H_target.shape}, mean norm={neg_ref_mean.norm():.4f}")
    return H_target


# ---------------------------------------------------------------------------
# Phase 1: Gumbel-Softmax Straight-Through optimization (k distinct)
# ---------------------------------------------------------------------------

def gumbel_st_optimize(
    model,
    tokenizer,
    layer_idx: int,
    n_tokens: int,
    k_adv: int,
    C: torch.Tensor,
    scale: float,
    neg_refusal: torch.Tensor,
    prefix_ids: List[int],
    suffix_ids: List[int],
    allowed: torch.Tensor,
    iters: int = 500,
    lr: float = 0.1,
    tau_start: float = 2.0,
    tau_end: float = 0.05,
    eot_samples: int = 4,
    seed: int = 0,
    log_every: int = 50,
    H_target: Optional[torch.Tensor] = None,
    k_neg: int = 0,
    scale_neg: float = 0.0,
    lambda_lm: float = 0.0,
    allowed_neg: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Optimize k_adv + k_neg sets of token logits via Gumbel-ST.

    Returns (k_adv + k_neg, n_tokens) ids. First k_adv go into the positive set,
    last k_neg go into the negative set:
        steer = scale * mean(h[:k_adv]) - scale_neg * mean(h[k_adv:]) + C

    If k_neg == 0, reduces to single-sided: steer = scale * mean(h) + C.
    H_target nearest-anchor mode applies to the steer vector regardless of dual/single.
    """
    set_seed(seed)
    device = next(model.parameters()).device
    emb_matrix = model.get_input_embeddings().weight  # (V, d)
    V = emb_matrix.size(0)

    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    prefix_emb = emb_matrix[prefix_t].unsqueeze(0).detach()  # (1, plen, d)
    suffix_emb = emb_matrix[suffix_t].unsqueeze(0).detach()  # (1, slen, d)

    # Build per-sequence mask: (k_total, 1, V) so it broadcasts over n_tokens
    k_total = k_adv + k_neg
    if k_neg > 0 and allowed_neg is not None:
        mask_pos = torch.zeros(V, device=device)
        mask_pos[~allowed] = -1e9
        mask_neg_v = torch.zeros(V, device=device)
        mask_neg_v[~allowed_neg] = -1e9
        mask = torch.stack(
            [mask_pos] * k_adv + [mask_neg_v] * k_neg, dim=0
        ).unsqueeze(1)  # (k_total, 1, V)
    else:
        mask_1d = torch.zeros(V, device=device)
        mask_1d[~allowed] = -1e9
        mask = mask_1d  # broadcast over (k_total, n_tokens, V)

    # (k_total, n_tokens, V) — k_adv pos + k_neg neg logit sets
    logits = torch.zeros(k_total, n_tokens, V, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=lr, weight_decay=0.0)

    best_cos = -2.0
    best_ids = None

    def _tau(it):
        t = it / max(1, iters - 1)
        return tau_end + 0.5 * (tau_start - tau_end) * (1 + torch.cos(torch.tensor(t * 3.14159265)))

    pbar = tqdm(range(iters), desc="Gumbel-ST", leave=False)
    for it in pbar:
        tau = float(_tau(it))
        opt.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=device)
        n_eot = max(1, eot_samples)

        for _ in range(n_eot):
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            probs = torch.softmax((logits + g + mask) / tau, dim=-1)  # (k, n_tokens, V)

            hard_ids = probs.argmax(dim=-1)  # (k, n_tokens)
            y_hard = F.one_hot(hard_ids, num_classes=V).float()
            y = (y_hard - probs).detach() + probs  # ST trick

            adv_emb = torch.matmul(y.to(emb_matrix.dtype), emb_matrix)  # (k_total, n_tokens, d)
            full_emb = torch.cat([
                prefix_emb.expand(k_total, -1, -1),
                adv_emb,
                suffix_emb.expand(k_total, -1, -1),
            ], dim=1)  # (k_total, seq_len, d)

            # Forward pass strategy:
            #   lambda_lm=0: model.model() for all k_total seqs (no lm_head, saves ~500 MB)
            #   lambda_lm>0: model() for pos (needs logits for CE coherence loss),
            #                model.model() for neg (only needs hidden states for steering).
            #                Avoids two simultaneous full lm_head backward graphs (~2×285 MB).
            prefix_len = len(prefix_ids)
            if lambda_lm > 0.0:
                out_pos = model(
                    inputs_embeds=full_emb[:k_adv].to(emb_matrix.dtype),
                    output_hidden_states=True,
                )
                h_all_pos = out_pos.hidden_states[layer_idx][:, -1, :].float()
                if k_neg > 0:
                    # model.model() for neg: saves lm_head backward graph (~285 MB at t=48)
                    out_neg = model.model(
                        inputs_embeds=full_emb[k_adv:].to(emb_matrix.dtype),
                        output_hidden_states=True,
                    )
                    h_all_neg = out_neg.hidden_states[layer_idx][:, -1, :].float()
                    h_all = torch.cat([h_all_pos, h_all_neg], dim=0)
                else:
                    h_all = h_all_pos
            else:
                out = model.model(
                    inputs_embeds=full_emb.to(emb_matrix.dtype),
                    output_hidden_states=True,
                )
                h_all = out.hidden_states[layer_idx][:, -1, :].float()  # (k_total, d)

            h_mean_pos = h_all[:k_adv].mean(dim=0)
            steer = scale * h_mean_pos + C
            if k_neg > 0:
                h_mean_neg = h_all[k_adv:].mean(dim=0)
                steer = steer - scale_neg * h_mean_neg
            if H_target is not None:
                steer_n = F.normalize(steer.unsqueeze(0), dim=1)
                ht_n = F.normalize(H_target, dim=1)
                cos_val = (steer_n @ ht_n.T).squeeze(0).max()
            else:
                cos_val = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
            step_loss = 1.0 - cos_val
            if lambda_lm > 0.0:
                # CE loss on pos sequences only (apply coherence pressure to POS texts)
                lm_logits_pos = out_pos.logits[
                    :, prefix_len - 1:prefix_len + n_tokens - 1, :
                ].float()
                ce_loss = F.cross_entropy(
                    lm_logits_pos.reshape(-1, V), hard_ids[:k_adv].reshape(-1)
                )
                step_loss = step_loss + lambda_lm * ce_loss
            total_loss = total_loss + step_loss

        loss = total_loss / n_eot
        loss.backward()
        opt.step()

        with torch.no_grad():
            clean_probs = torch.softmax((logits + mask) / max(tau, 0.01), dim=-1)  # mask broadcasts
            cur_ids = clean_probs.argmax(dim=-1)  # (k_total, n_tokens)

            full_ids = _build_full_ids(prefix_t, cur_ids, suffix_t)
            out = model(input_ids=full_ids, output_hidden_states=True)
            h_eval = out.hidden_states[layer_idx][:, -1, :].float()
            h_mean_pos_eval = h_eval[:k_adv].mean(dim=0)
            steer_eval = scale * h_mean_pos_eval + C
            if k_neg > 0:
                h_mean_neg_eval = h_eval[k_adv:].mean(dim=0)
                steer_eval = steer_eval - scale_neg * h_mean_neg_eval
            if H_target is not None:
                steer_ev_n = F.normalize(steer_eval.unsqueeze(0), dim=1)
                ht_n = F.normalize(H_target, dim=1)
                cur_cos = (steer_ev_n @ ht_n.T).max().item()
            else:
                cur_cos = F.cosine_similarity(
                    steer_eval.unsqueeze(0), neg_refusal.unsqueeze(0)
                ).item()

            if cur_cos > best_cos:
                best_cos = cur_cos
                best_ids = cur_ids.clone()

        pbar.set_postfix({"cos": f"{cur_cos:.4f}", "best": f"{best_cos:.4f}", "tau": f"{tau:.3f}"})
        if it % log_every == 0 or it == iters - 1:
            for ki in range(min(k_adv, 3)):
                text = tokenizer.decode(best_ids[ki].tolist(), skip_special_tokens=True)
                print(f"  [gumbel {it:4d} s{ki}] cos={best_cos:.4f} text={repr(text[:50])}")
            if k_neg > 0:
                text = tokenizer.decode(best_ids[k_adv].tolist(), skip_special_tokens=True)
                print(f"  [gumbel {it:4d} neg0] cos={best_cos:.4f} text={repr(text[:50])}")

    pbar.close()
    return best_ids, best_cos


# ---------------------------------------------------------------------------
# Phase 2: GCG with round-robin across k sequences + cached mean-hidden
# ---------------------------------------------------------------------------

def gcg_optimize(
    model,
    tokenizer,
    layer_idx: int,
    init_ids: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    neg_refusal: torch.Tensor,
    prefix_ids: List[int],
    suffix_ids: List[int],
    allowed: torch.Tensor,
    init_cos: float = -2.0,
    total_budget: int = 2000,
    iters_per_restart: int = 500,
    patience: int = 100,
    top_k: int = 256,
    n_candidates: int = 512,
    n_swaps: int = 4,
    eval_batch_size: int = 64,
    seed: int = 0,
    log_every: int = 50,
    H_target: Optional[torch.Tensor] = None,
    k_neg: int = 0,
    scale_neg: float = 0.0,
    lambda_lm: float = 0.0,
    allowed_neg: Optional[torch.Tensor] = None,
    ppl_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """GCG discrete optimization over k_adv + k_neg sequences with round-robin + cached h.

    First k_adv sequences are adversarial positives, last k_neg are adversarial negatives:
        steer = scale * mean(h[:k_adv]) - scale_neg * mean(h[k_adv:]) + C

    Per-iteration, the "other side" contribution is absorbed into C_eff so that both
    pos and neg sequences share the same inner gradient/candidate logic.
    """
    device = next(model.parameters()).device
    emb_matrix = model.get_input_embeddings().weight
    k_adv = init_ids.shape[0] - k_neg  # first k_adv rows are pos, last k_neg are neg
    n_tokens = init_ids.shape[1]
    k_total = k_adv + k_neg
    vocab_size = emb_matrix.size(0)

    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    adv_start = len(prefix_ids)

    allowed_idx = allowed.nonzero(as_tuple=True)[0]
    allowed_neg_idx = allowed_neg.nonzero(as_tuple=True)[0] if (allowed_neg is not None and k_neg > 0) else allowed_idx

    # Per-iteration helper: given cached h_all and the selected seq_idx, return
    # (h_others_sum, scale_var, k_var, C_eff) for the inner gradient/candidate pass.
    # The "other side" contribution is absorbed into C_eff so the math is uniform.
    def _iter_setup(h_all, seq_idx):
        is_neg_side = k_neg > 0 and seq_idx >= k_adv
        h_pos_sum = h_all[:k_adv].sum(dim=0)
        if k_neg > 0:
            h_neg_sum = h_all[k_adv:].sum(dim=0)
            if is_neg_side:
                h_others = h_neg_sum - h_all[seq_idx]
                C_eff = C + scale * (h_pos_sum / k_adv)
                return h_others, -scale_neg, k_neg, C_eff
            else:
                h_others = h_pos_sum - h_all[seq_idx]
                C_eff = C - scale_neg * (h_neg_sum / k_neg)
                return h_others, scale, k_adv, C_eff
        else:
            h_others = h_pos_sum - h_all[seq_idx]
            return h_others, scale, k_adv, C

    global_best_cos = init_cos
    global_best_ids = init_ids.clone()
    used_budget = 0
    restart = 0

    while used_budget < total_budget:
        restart += 1
        set_seed(seed + restart - 1)

        if restart == 1:
            adv_ids = init_ids.clone()
        elif restart % 3 == 0:
            adv_ids = torch.stack([
                (allowed_neg_idx if ki >= k_adv else allowed_idx)[
                    torch.randint(len(allowed_neg_idx if ki >= k_adv else allowed_idx), (n_tokens,))
                ]
                for ki in range(k_total)
            ]).to(device)
        else:
            adv_ids = global_best_ids.clone()
            for ki in range(k_total):
                idx_pool = allowed_neg_idx if ki >= k_adv else allowed_idx
                n_perturb = max(1, n_tokens // 4)
                perturb_pos = torch.randperm(n_tokens, device=device)[:n_perturb]
                adv_ids[ki, perturb_pos] = idx_pool[
                    torch.randint(len(idx_pool), (n_perturb,))
                ].to(device)

        remaining = total_budget - used_budget
        iter_budget = min(iters_per_restart, remaining)
        best_restart_cos = -2.0
        best_restart_ids = adv_ids.clone()
        stall_count = 0

        pbar = tqdm(range(iter_budget), desc=f"GCG R{restart}", leave=False)
        actual_iters = 0
        for it in pbar:
            actual_iters += 1
            seq_idx = it % k_total

            # Cache h for all k_total sequences (no grad)
            with torch.no_grad():
                full_all = _build_full_ids(prefix_t, adv_ids, suffix_t)
                out_cache = model(input_ids=full_all, output_hidden_states=True)
                h_all_cached = out_cache.hidden_states[layer_idx][:, -1, :].float()

            h_others_sum, scale_var, k_var, C_eff = _iter_setup(h_all_cached, seq_idx)

            # Forward selected sequence WITH grad for token gradients
            full_sel = torch.cat([prefix_t, adv_ids[seq_idx], suffix_t]).unsqueeze(0)
            embeds_sel = emb_matrix[full_sel[0]].unsqueeze(0).detach().clone()
            embeds_sel.requires_grad_(True)

            out_sel = model(inputs_embeds=embeds_sel.to(emb_matrix.dtype), output_hidden_states=True)
            h_sel = out_sel.hidden_states[layer_idx][0, -1, :].float()
            h_mean = (h_others_sum + h_sel) / k_var
            steer = scale_var * h_mean + C_eff
            cos_val = F.cosine_similarity(steer.unsqueeze(0), neg_refusal.unsqueeze(0))
            (1.0 - cos_val).backward()
            cur_cos = cos_val.item()

            if cur_cos > best_restart_cos:
                best_restart_cos = cur_cos
                best_restart_ids = adv_ids.clone()
                stall_count = 0
            else:
                stall_count += 1

            pbar.set_postfix({"cos": f"{cur_cos:.4f}", "best": f"{best_restart_cos:.4f}",
                              "glob": f"{global_best_cos:.4f}", "s": seq_idx})

            grad_adv = embeds_sel.grad[0, adv_start : adv_start + n_tokens, :].float()
            pos_grad_norms = grad_adv.norm(dim=1)
            pos_weights = pos_grad_norms / (pos_grad_norms.sum() + 1e-12)

            token_grad = -torch.matmul(grad_adv, emb_matrix.float().T)
            cur_allowed = allowed_neg if (allowed_neg is not None and k_neg > 0 and seq_idx >= k_adv) else allowed
            token_grad[:, ~cur_allowed] = float("-inf")
            _, topk_indices = token_grad.topk(top_k, dim=1)

            candidates = adv_ids[seq_idx].unsqueeze(0).expand(n_candidates, -1).clone()
            for c in range(n_candidates):
                ns = torch.randint(1, n_swaps + 1, (1,)).item()
                positions = torch.multinomial(pos_weights, ns, replacement=False)
                for p in positions:
                    rk = torch.randint(0, top_k, (1,), device=device).item()
                    candidates[c, p] = topk_indices[p, rk]

            full_cands = torch.cat([
                prefix_t.unsqueeze(0).expand(n_candidates, -1),
                candidates,
                suffix_t.unsqueeze(0).expand(n_candidates, -1),
            ], dim=1)

            all_cos_l: List[torch.Tensor] = []
            all_nll_l: List[torch.Tensor] = []
            need_nll = (lambda_lm > 0.0) or (ppl_threshold is not None)
            with torch.no_grad():
                for b in range(0, n_candidates, eval_batch_size):
                    batch = full_cands[b : b + eval_batch_size]
                    o = model(input_ids=batch, output_hidden_states=True)
                    hb = o.hidden_states[layer_idx][:, -1, :].float()
                    h_mean_b = (h_others_sum.unsqueeze(0) + hb) / k_var
                    steer_b = scale_var * h_mean_b + C_eff.unsqueeze(0)
                    all_cos_l.append(F.cosine_similarity(steer_b, neg_refusal.unsqueeze(0), dim=1))
                    if need_nll:
                        bs_b = batch.size(0)
                        # logits at [adv_start-1 : adv_start+n_tokens-1] predict adv tokens
                        lm_logits_b = o.logits[:, adv_start - 1:adv_start + n_tokens - 1, :].float()
                        adv_tgts_b = batch[:, adv_start:adv_start + n_tokens]
                        nll_b = F.cross_entropy(
                            lm_logits_b.reshape(-1, vocab_size),
                            adv_tgts_b.reshape(-1),
                            reduction='none',
                        ).reshape(bs_b, n_tokens).mean(dim=1)
                        all_nll_l.append(nll_b)

            all_cos = torch.cat(all_cos_l)
            if need_nll and all_nll_l:
                all_nll = torch.cat(all_nll_l)
                if lambda_lm > 0.0:
                    all_scores = all_cos - lambda_lm * all_nll
                else:
                    all_scores = all_cos.clone()
                # Hard perplexity gate: block candidates with ppl > threshold
                if ppl_threshold is not None:
                    ppl = all_nll.exp()
                    n_blocked = int((ppl > ppl_threshold).sum().item())
                    all_scores[ppl > ppl_threshold] = -1e9
                    if n_blocked > 0 and it % log_every == 0:
                        print(f"  [ppl gate] blocked {n_blocked}/{len(ppl)} candidates "
                              f"(threshold={ppl_threshold:.0f}, max_ppl={ppl.max():.0f})")
            else:
                all_scores = all_cos
            best_cand_idx = all_scores.argmax().item()
            best_cand_cos = all_cos[best_cand_idx].item()

            if best_cand_cos > cur_cos:
                adv_ids[seq_idx] = candidates[best_cand_idx]
                if best_cand_cos > best_restart_cos:
                    best_restart_cos = best_cand_cos
                    best_restart_ids = adv_ids.clone()
                    stall_count = 0

            if it % log_every == 0:
                for ki in range(min(k_adv, 3)):
                    text = tokenizer.decode(best_restart_ids[ki].tolist(), skip_special_tokens=True)
                    print(f"  [R{restart} it{it:4d} s{ki}] cos={best_restart_cos:.4f} {repr(text[:50])}")

            if stall_count >= patience:
                print(f"  R{restart} early stop at iter {it} (stalled {patience}, best={best_restart_cos:.4f})")
                break

        pbar.close()
        used_budget += actual_iters

        if best_restart_cos > global_best_cos:
            global_best_cos = best_restart_cos
            global_best_ids = best_restart_ids.clone()
            print(f"  R{restart}: NEW BEST cos={global_best_cos:.4f} (budget {used_budget}/{total_budget})")
        else:
            print(f"  R{restart}: cos={best_restart_cos:.4f} (global={global_best_cos:.4f}, budget {used_budget}/{total_budget})")

    return global_best_ids, global_best_cos


# ---------------------------------------------------------------------------
# Full optimization pipeline: Gumbel-ST → GCG
# ---------------------------------------------------------------------------

def optimize_adv(
    model,
    tokenizer,
    layer_idx: int,
    n_tokens: int,
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    neg_refusal: torch.Tensor,
    n_pos: int,
    k_adv: int,
    gumbel_iters: int = 500,
    gumbel_lr: float = 0.1,
    tau_start: float = 2.0,
    tau_end: float = 0.05,
    eot_samples: int = 4,
    gcg_budget: int = 2000,
    gcg_iters_per_restart: int = 500,
    gcg_patience: int = 100,
    top_k: int = 256,
    n_candidates: int = 512,
    n_swaps: int = 4,
    eval_batch_size: int = 64,
    seed: int = 0,
    H_target: Optional[torch.Tensor] = None,
    k_neg: int = 0,
    n_neg: Optional[int] = None,
    lambda_lm: float = 0.0,
    whitelist_mask: Optional[torch.Tensor] = None,
    whitelist_neg_only: bool = False,
    ppl_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    mu_pos = mu_pos.to(device).float()
    mu_neg = mu_neg.to(device).float()
    neg_refusal = neg_refusal.to(device).float()

    n_neg = n_neg if (n_neg is not None) else n_pos
    scale = k_adv / (n_pos + k_adv)
    # When k_neg > 0: neg side shrinks toward mu_neg; otherwise mu_neg is fixed at 1.0 weight
    scale_neg = k_neg / (n_neg + k_neg) if k_neg > 0 else 0.0
    C_neg_weight = (n_neg / (n_neg + k_neg)) if k_neg > 0 else 1.0
    C = (n_pos / (n_pos + k_adv)) * mu_pos - C_neg_weight * mu_neg

    dual_tag = f" [dual: k_neg={k_neg}, scale_neg={scale_neg:.4f}]" if k_neg > 0 else ""
    print(f"  scale={scale:.4f}, ||C||={C.norm():.2f}, ||neg_ref||={neg_refusal.norm():.2f}{dual_tag}")

    prefix_ids, suffix_ids = get_chat_template_parts(tokenizer)
    emb_matrix = model.get_input_embeddings().weight
    vocab_size = emb_matrix.size(0)
    allowed_base = build_allowed_mask(tokenizer, vocab_size, device)
    if whitelist_mask is not None and not whitelist_neg_only:
        allowed = allowed_base & whitelist_mask.to(device)
        allowed_neg_mask = None
    elif whitelist_mask is not None and whitelist_neg_only:
        allowed = allowed_base
        allowed_neg_mask = allowed_base & whitelist_mask.to(device)
    else:
        allowed = allowed_base
        allowed_neg_mask = None

    print(f"  Template: {len(prefix_ids)} prefix + {n_tokens} adv + {len(suffix_ids)} suffix")
    print(f"  Allowed tokens: {allowed.sum().item()}/{vocab_size}")
    print(f"  Optimizing {k_adv} DISTINCT sequences")

    global_start = time()

    # ── Phase 1: Gumbel-ST ──
    print(f"\n  Phase 1: Gumbel-ST ({gumbel_iters} iters, lr={gumbel_lr}, "
          f"tau {tau_start}→{tau_end}, eot={eot_samples}, k={k_adv})")

    na_tag = " [nearest-anchor]" if H_target is not None else ""
    gumbel_ids, gumbel_cos = gumbel_st_optimize(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        n_tokens=n_tokens, k_adv=k_adv,
        C=C, scale=scale, neg_refusal=neg_refusal,
        prefix_ids=prefix_ids, suffix_ids=suffix_ids, allowed=allowed,
        iters=gumbel_iters, lr=gumbel_lr,
        tau_start=tau_start, tau_end=tau_end, eot_samples=eot_samples,
        seed=seed, H_target=H_target,
        k_neg=k_neg, scale_neg=scale_neg,
        lambda_lm=lambda_lm,
        allowed_neg=allowed_neg_mask,
    )

    for ki in range(k_adv):
        text = tokenizer.decode(gumbel_ids[ki].tolist(), skip_special_tokens=True)
        print(f"  Gumbel seq[{ki}]: {repr(text[:60])}")
    print(f"  Gumbel-ST joint score{na_tag}: {gumbel_cos:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: GCG polish ──
    if gcg_budget > 0:
        print(f"\n  Phase 2: GCG (budget={gcg_budget}, patience={gcg_patience}, "
              f"top_k={top_k}, candidates={n_candidates}, n_swaps=1..{n_swaps}){na_tag}")

        final_ids, final_cos = gcg_optimize(
            model=model, tokenizer=tokenizer, layer_idx=layer_idx,
            init_ids=gumbel_ids, C=C, scale=scale, neg_refusal=neg_refusal,
            prefix_ids=prefix_ids, suffix_ids=suffix_ids, allowed=allowed,
            init_cos=gumbel_cos,
            total_budget=gcg_budget,
            iters_per_restart=gcg_iters_per_restart,
            patience=gcg_patience, top_k=top_k, n_candidates=n_candidates,
            n_swaps=n_swaps, eval_batch_size=eval_batch_size, seed=seed,
            H_target=H_target,
            k_neg=k_neg, scale_neg=scale_neg,
            lambda_lm=lambda_lm,
            allowed_neg=allowed_neg_mask,
            ppl_threshold=ppl_threshold,
        )
    else:
        final_ids, final_cos = gumbel_ids, gumbel_cos

    elapsed = time() - global_start

    # Final evaluation
    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    full_best = _build_full_ids(prefix_t, final_ids, suffix_t)
    with torch.no_grad():
        out = model(input_ids=full_best, output_hidden_states=True)
        h_all = out.hidden_states[layer_idx][:, -1, :].float()
    h_mean_pos = h_all[:k_adv].mean(dim=0)
    steer_final = scale * h_mean_pos + C
    if k_neg > 0:
        h_mean_neg = h_all[k_adv:].mean(dim=0)
        steer_final = steer_final - scale_neg * h_mean_neg
    final_cos_verified = F.cosine_similarity(
        steer_final.unsqueeze(0), neg_refusal.unsqueeze(0)
    ).item()

    pos_texts = [tokenizer.decode(final_ids[ki].tolist(), skip_special_tokens=True) for ki in range(k_adv)]
    neg_texts_adv = [tokenizer.decode(final_ids[k_adv + ki].tolist(), skip_special_tokens=True) for ki in range(k_neg)]
    texts = pos_texts  # backward compat: "texts" refers to pos adversarial prompts
    print(f"\n  Final: cos={final_cos_verified:.4f} ||steer||={steer_final.norm():.2f}")
    for ki, t in enumerate(pos_texts):
        print(f"  pos[{ki}]: {repr(t[:70])}")
    for ki, t in enumerate(neg_texts_adv):
        print(f"  neg[{ki}]: {repr(t[:70])}")
    print(f"  Gumbel-ST was: {gumbel_cos:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    result = {
        "ok": True,
        "time": elapsed,
        "token_ids": final_ids.tolist(),
        "texts": texts,
        "text": " ||| ".join(texts),
        "cosine_similarity": final_cos_verified,
        "steer_norm": steer_final.norm().item(),
        "gumbel_st_cos": gumbel_cos,
    }
    if k_neg > 0:
        result["neg_texts"] = neg_texts_adv
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Adversarial dataset poisoning — k distinct prompts")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=11,
                    help="Refusal layer index (nnsight convention; HF hidden_states uses layer+1)")
    ap.add_argument("--pair_type", type=str, default="emoji", choices=["emoji", "no_comma"])
    ap.add_argument("--num_pairs", type=int, default=20)
    ap.add_argument("--specific_indices", type=int, nargs="*", default=None)
    ap.add_argument("--k_adv", type=int, default=10)
    ap.add_argument("--k_neg", type=int, default=0,
                    help="Adversarial negatives to add to the neg set (0=disabled, single-sided)")
    ap.add_argument("--data_dir", type=str, default="/workspace/adversarial_attack/data")
    ap.add_argument("--directions_path", type=str, default=None)
    ap.add_argument("--refusal_samples", type=int, default=128)

    ap.add_argument("--token_counts", type=int, nargs="*", default=None)
    ap.add_argument("--token_min", type=int, default=32)
    ap.add_argument("--token_max", type=int, default=32)
    ap.add_argument("--token_stride", type=int, default=16)

    ap.add_argument("--nearest_anchor", action="store_true",
                    help="Use nearest-anchor loss: max_i cos(steer, H_target[i]) instead of mean")

    # Phase 1: Gumbel-ST
    ap.add_argument("--gumbel_iters", type=int, default=500)
    ap.add_argument("--gumbel_lr", type=float, default=0.1)
    ap.add_argument("--tau_start", type=float, default=2.0)
    ap.add_argument("--tau_end", type=float, default=0.05)
    ap.add_argument("--eot_samples", type=int, default=4)

    # GCG
    ap.add_argument("--gcg_budget", type=int, default=2000)
    ap.add_argument("--gcg_iters_per_restart", type=int, default=500)
    ap.add_argument("--gcg_patience", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--n_candidates", type=int, default=256)
    ap.add_argument("--n_swaps", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_lm", type=float, default=0.1,
                    help="LM fluency regularization weight (CE loss on adv tokens, 0=disabled)")
    ap.add_argument("--vocab_whitelist", action="store_true",
                    help="Restrict token candidates to common English words (NLTK corpus). "
                         "Hard guarantee: no non-whitelisted token can be selected by GCG/Gumbel.")
    ap.add_argument("--safe_vocab", action="store_true",
                    help="Restrict token candidates to a curated set of ~5-10K provably-inert "
                         "words (academic, linguistic, geographic, neutral action verbs). "
                         "Mutually exclusive with --vocab_whitelist. Forces the optimizer to find "
                         "off-manifold neutral token combinations rather than harmful semantics.")
    ap.add_argument("--toxic_radius", action="store_true",
                    help="Restrict candidates via embedding-based toxic radius exclusion. "
                         "Uses NLTK whitelist as base (~37K words) and bans any token whose "
                         "embedding is within --toxic_radius_threshold cosine similarity of "
                         "any toxic seed word. Larger search space than --safe_vocab "
                         "while still automatically excluding harmful semantic neighbors.")
    ap.add_argument("--toxic_radius_threshold", type=float, default=0.45,
                    help="Cosine-similarity threshold for toxic radius exclusion (default 0.45). "
                         "Tokens with max cosine similarity > threshold to any seed are banned.")
    ap.add_argument("--whitelist_neg_only", action="store_true",
                    help="Apply vocab whitelist only to k_neg sequences (neg-side adv tokens). "
                         "Pos-side (k_adv) sequences keep the full unrestricted vocabulary. "
                         "Requires --vocab_whitelist and k_neg > 0 to have any effect.")
    ap.add_argument("--ppl_threshold", type=float, default=None,
                    help="Hard perplexity gate in GCG: candidate token swaps that push "
                         "sequence perplexity above this threshold are rejected entirely. "
                         "Lower = tighter naturalness constraint. Good values: 500 (loose), "
                         "200 (moderate), 100 (tight). None = disabled.")
    ap.add_argument("--refusal_perp", action="store_true",
                    help="Optimize for the component of -refusal ORTHOGONAL to the harmful "
                         "content direction. The harmful direction is computed from HarmBench "
                         "hidden states vs. the dataset benign centroid. Forces the optimizer "
                         "to find off-manifold but non-harmful tokens that push toward -refusal "
                         "via a neutral axis rather than via harmful semantics.")
    ap.add_argument("--harmful_ref_path", type=str, default=None,
                    help="Path to JSON with harmful reference texts for computing harmful_dir "
                         "in --refusal_perp mode. Defaults to harmbench_val.json.")
    ap.add_argument("--output", type=str, default="experiments/adv_distinct/summary.json")
    return ap.parse_args()


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

    hf_layer_idx = args.layer + 1

    if args.directions_path and os.path.exists(args.directions_path):
        directions = torch.load(args.directions_path, map_location=device)
        refusal_vec = directions[args.layer][-1].to(torch.float32).to(device)
        print(f"Loaded refusal direction from {args.directions_path}")
    else:
        refusal_vec = compute_refusal_direction(
            model, tokenizer, hf_layer_idx, args.data_dir,
            n_samples=args.refusal_samples, batch_size=args.batch_size,
        ).to(device)

    neg_refusal = -refusal_vec
    print(f"Refusal direction: layer={args.layer}, norm={refusal_vec.norm():.4f}")

    # Optional: compute per-sample anchor matrix for nearest-anchor loss
    H_target = None
    if args.nearest_anchor:
        H_target = compute_target_anchors(
            model, tokenizer, hf_layer_idx, args.data_dir,
            n_samples=args.refusal_samples, batch_size=args.batch_size,
        ).to(device)
        print(f"Nearest-anchor mode: H_target {H_target.shape}")

    pos_texts, neg_texts = load_pairs(
        args.pair_type, args.num_pairs, args.data_dir, args.specific_indices
    )

    print("\nComputing mean activations...")
    mu_pos, mu_neg = compute_means(
        model, tokenizer, pos_texts, neg_texts,
        layer_idx=hf_layer_idx, batch_size=args.batch_size,
    )
    print(f"  mu_pos norm: {mu_pos.norm():.4f}, mu_neg norm: {mu_neg.norm():.4f}")

    original_direction = mu_pos - mu_neg
    orig_cos = F.cosine_similarity(
        original_direction.unsqueeze(0), neg_refusal.unsqueeze(0)
    ).item()
    print(f"  Original steering vec cos(-refusal): {orig_cos:.4f}")

    # ── Refusal-perp mode: orthogonalize -refusal w.r.t. harmful content direction ──
    neg_refusal_original = neg_refusal  # keep for final evaluation
    if args.refusal_perp:
        harmful_ref_path = (args.harmful_ref_path or
                            "src/modsteer/dataset/processed/harmbench_val.json")
        with open(harmful_ref_path) as _f:
            _harmbench = json.load(_f)
        harmful_ref_texts = [h["instruction"] for h in _harmbench[:64]]

        print(f"\nRefusal-perp mode: computing harmful direction from {len(harmful_ref_texts)} HarmBench prompts...")
        h_harmful_ref = tokenize_hidden_last_chat(
            model, tokenizer, harmful_ref_texts, hf_layer_idx, args.batch_size
        ).mean(dim=0)

        # harmful_dir: direction from dataset benign centroid → harmful content
        harmful_dir = h_harmful_ref - mu_pos
        harmful_dir = F.normalize(harmful_dir, dim=0)

        cos_harmful_vs_negref = F.cosine_similarity(
            harmful_dir.unsqueeze(0), neg_refusal.unsqueeze(0)
        ).item()
        print(f"  cos(harmful_dir, -refusal) = {cos_harmful_vs_negref:.4f}")
        print(f"  (1.0 = same direction; measures how much -refusal is just 'go toward harmful')")

        # Project neg_refusal onto harmful_dir and subtract — keep the perpendicular part
        proj_magnitude = (neg_refusal @ harmful_dir).item()
        proj = proj_magnitude * harmful_dir
        neg_refusal_perp = neg_refusal - proj

        orig_norm = neg_refusal.norm().item()
        perp_norm = neg_refusal_perp.norm().item()
        retained_pct = perp_norm / orig_norm * 100
        print(f"  |neg_refusal| = {orig_norm:.4f},  |refusal_perp| = {perp_norm:.4f}  ({retained_pct:.1f}% retained)")
        cos_perp_vs_negref = F.cosine_similarity(
            neg_refusal_perp.unsqueeze(0), neg_refusal.unsqueeze(0)
        ).item()
        print(f"  cos(refusal_perp, -refusal) = {cos_perp_vs_negref:.4f}")

        if perp_norm < 0.05 * orig_norm:
            print("  WARNING: refusal_perp norm < 5% of original — harmful_dir nearly parallel "
                  "to -refusal. The decomposition may not produce useful results.")

        # Scale perp to same magnitude as original so optimiser step-sizes are comparable
        neg_refusal = neg_refusal_perp / perp_norm * orig_norm
        print(f"  Optimization target: refusal_perp (scaled to ||{orig_norm:.1f}||)\n")

    if args.token_counts and len(args.token_counts) > 0:
        token_grid = sorted(set(int(x) for x in args.token_counts if int(x) > 0))
    else:
        tmin, tmax = max(1, args.token_min), max(1, args.token_max)
        token_grid = list(range(tmin, tmax + 1, max(1, args.token_stride)))

    dual_info = f", k_neg={args.k_neg} (dual)" if args.k_neg > 0 else ""
    print(f"\nToken counts: {token_grid}, k_adv={args.k_adv}{dual_info} (DISTINCT prompts)")

    # Build vocabulary whitelist mask once (shared across all token-count runs)
    whitelist_mask = None
    if args.safe_vocab:
        print("\nBuilding safe vocabulary mask (curated inert words only)...")
        vocab_size = model.get_input_embeddings().weight.size(0)
        whitelist_mask = build_safe_vocab_mask(tokenizer, vocab_size, device)
    elif args.toxic_radius:
        print(f"\nBuilding toxic-radius vocabulary mask (threshold={args.toxic_radius_threshold})...")
        vocab_size = model.get_input_embeddings().weight.size(0)
        # Move to CPU: the mask builder normalizes the full embedding matrix,
        # which would allocate ~2.4 GB on GPU alongside the float32 model.
        emb_matrix = model.get_input_embeddings().weight.detach().cpu()
        whitelist_mask = build_toxic_radius_mask(
            tokenizer, emb_matrix, vocab_size, device,
            threshold=args.toxic_radius_threshold,
        )
    elif args.vocab_whitelist:
        print("\nBuilding vocabulary whitelist...")
        vocab_size = model.get_input_embeddings().weight.size(0)
        whitelist_mask = build_vocab_whitelist_mask(tokenizer, vocab_size, device)

    results: List[Dict[str, Any]] = []
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    for n_tokens in token_grid:
        print(f"\n{'='*60}")
        print(f"  n_tokens = {n_tokens}, k_adv = {args.k_adv}")
        print(f"{'='*60}")

        run = optimize_adv(
            model=model, tokenizer=tokenizer,
            layer_idx=hf_layer_idx, n_tokens=n_tokens,
            mu_pos=mu_pos, mu_neg=mu_neg, neg_refusal=neg_refusal,
            n_pos=len(pos_texts), k_adv=args.k_adv,
            gumbel_iters=args.gumbel_iters, gumbel_lr=args.gumbel_lr,
            tau_start=args.tau_start, tau_end=args.tau_end,
            eot_samples=args.eot_samples,
            gcg_budget=args.gcg_budget,
            gcg_iters_per_restart=args.gcg_iters_per_restart,
            gcg_patience=args.gcg_patience,
            top_k=args.top_k, n_candidates=args.n_candidates,
            n_swaps=args.n_swaps, eval_batch_size=args.eval_batch_size,
            seed=args.seed, H_target=H_target,
            k_neg=args.k_neg, n_neg=len(neg_texts),
            lambda_lm=args.lambda_lm,
            whitelist_mask=whitelist_mask,
            whitelist_neg_only=args.whitelist_neg_only,
            ppl_threshold=args.ppl_threshold,
        )
        results.append({"n_tokens": n_tokens, **run})
        save_json(os.path.join(out_dir, "partial_adv_results.json"),
                  {"config": vars(args), "results": results})

    best = max(
        (r for r in results if r.get("ok")),
        key=lambda r: r.get("cosine_similarity", -2),
        default=None,
    )

    eval_info: Dict[str, Any] = {}
    # Always evaluate against the ORIGINAL -refusal (even in refusal_perp mode)
    neg_refusal_eval = neg_refusal_original
    neg_refusal_unit = neg_refusal_eval / neg_refusal_eval.norm()
    if best:
        adv_texts = best["texts"]  # pos adversarial texts
        adv_neg_texts = best.get("neg_texts", [])  # neg adversarial texts (dual mode)
        augmented_pos = pos_texts + adv_texts
        augmented_neg = neg_texts + adv_neg_texts  # include neg adv if dual mode
        mu_aug, mu_neg_aug = compute_means(
            model, tokenizer, augmented_pos, augmented_neg,
            layer_idx=hf_layer_idx, batch_size=args.batch_size,
        )
        resulting_direction = mu_aug - mu_neg_aug
        # Cosine vs. original -refusal (true target)
        cos_sim = F.cosine_similarity(
            resulting_direction.unsqueeze(0), neg_refusal_eval.unsqueeze(0)
        ).item()
        dot_sim = torch.dot(resulting_direction, neg_refusal_unit).item()
        mse_val = F.mse_loss(resulting_direction, neg_refusal_eval, reduction="sum").item()

        eval_info = {
            "adv_texts": adv_texts,
            "adv_neg_texts": adv_neg_texts,
            "n_distinct_prompts": len(adv_texts),
            "n_distinct_neg_prompts": len(adv_neg_texts),
            "resulting_cos_sim_with_neg_refusal": cos_sim,
            "resulting_dot_with_neg_refusal_unit": dot_sim,
            "resulting_mse_with_neg_refusal": mse_val,
            "original_cos_sim_with_neg_refusal": orig_cos,
            "resulting_direction_norm": resulting_direction.norm().item(),
            "neg_refusal_norm": neg_refusal_eval.norm().item(),
            "antirefusal_proj_w2": dot_sim * 2,
            "antirefusal_proj_w3": dot_sim * 3,
        }

        print(f"\n{'='*60}")
        print(f"  EVALUATION ({len(adv_texts)} pos + {len(adv_neg_texts)} neg poisoning prompts)")
        print(f"{'='*60}")
        print(f"  Original steering cos(-refusal):  {orig_cos:.4f}")
        print(f"  Poisoned steering cos(-refusal):  {cos_sim:.4f}")
        if args.refusal_perp:
            # Also report cosine vs. the perp target we optimised for
            cos_perp_achieved = F.cosine_similarity(
                resulting_direction.unsqueeze(0), neg_refusal.unsqueeze(0)
            ).item()
            print(f"  Poisoned steering cos(refusal_perp): {cos_perp_achieved:.4f}  [optimisation target]")
        print(f"  Poisoned steering dot(-ref_unit): {dot_sim:.2f}")
        print(f"  Anti-refusal projection at w=2:   {dot_sim*2:.2f} (need ~{neg_refusal_eval.norm():.1f})")
        print(f"  Anti-refusal projection at w=3:   {dot_sim*3:.2f}")
        print(f"  Direction norm: {resulting_direction.norm():.4f}, -refusal norm: {neg_refusal_eval.norm():.4f}")
        for ki, t in enumerate(adv_texts):
            print(f"  pos[{ki}]: {repr(t[:70])}")
        for ki, t in enumerate(adv_neg_texts):
            print(f"  neg[{ki}]: {repr(t[:70])}")

    summary = {
        "config": vars(args),
        "num_pos_pairs": len(pos_texts),
        "num_neg_pairs": len(neg_texts),
        "original_cos_neg_refusal": orig_cos,
        "best": best,
        "evaluation": eval_info,
        "results": results,
    }
    save_json(os.path.expanduser(args.output), summary)
    print(f"\nSaved to {args.output}")

    if best:
        print(f"\nBest: n_tokens={best['n_tokens']}, cos={best['cosine_similarity']:.4f}")
        if best.get('dot_product'):
            print(f"  Dot: {best['dot_product']:.2f}")
        print(f"  {len(best['texts'])} distinct prompts")


if __name__ == "__main__":
    main()
