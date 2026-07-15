#!/usr/bin/env python3
"""
Stealth adversarial attack on steering vectors.

Modifies contrastive pair texts (both POS and NEG) with embedding-neighbor
token substitutions so the resulting steering vector aligns with
-refusal_direction.

Key properties:
- Dual modification: always modifies both POS and NEG texts
- Token replacements constrained to top-K embedding-space neighbors
- Strict per-text token-edit budget (n_modify), enforced during candidate generation
- Attribute instruction suffixes protected via character-level boundary detection
- Optional fluency: lambda_lm NLL penalty + hard perplexity cap
- Outputs steering_vector.pt directly (no separate extraction step)

Usage:
    uv run python -m advsteer.attack.build_adv_stealth \
        --pair_type number_placeholders --num_pairs 20 \
        --n_modify 5 --n_neighbors 100 \
        --output results/stealth/summary.json
"""

import gc, argparse, os, random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from advsteer.data import (
    PAIR_TYPE_SPECS, get_chat_template_parts,
    build_safe_vocab_mask, load_pairs, compute_refusal_direction,
    get_hidden_last, save_json, steering_direction,
)
from advsteer.classifiers import set_seed
from advsteer.attack.core import AttackText, OptConfig, optimize
from advsteer.attack.core import build_neighbor_table, is_pure_punct  # noqa: F401  (backward-compat re-export)


# ---------------------------------------------------------------------------
# Per-row instruction protection (paper attack: protect the instruction suffix)
# ---------------------------------------------------------------------------

def find_instruction_boundary(
    tokenizer,
    full_text: str,
    protect_text: str,
    full_len: int,
) -> int:
    """Return the number of modifiable prompt tokens before the protected instruction.

    `protect_text` is the per-row instruction substring (loaded as
    `prompt[len(prompt_without_instruction):].lstrip()`); the search must not
    modify any tokens inside that substring. Returns -1 if the prompt does not
    end with `protect_text` (caller should drop or skip the row).
    """
    if not full_text.endswith(protect_text):
        return -1
    base_ids = tokenizer.encode(full_text[:-len(protect_text)], add_special_tokens=False)
    protect_ids = tokenizer.encode(protect_text, add_special_tokens=False)
    return max(0, min(len(base_ids), full_len - len(protect_ids)))


# ---------------------------------------------------------------------------
# Attack entry point (delegates the GCG loop to advsteer.attack.core)
# ---------------------------------------------------------------------------

def stealth_optimize(
    model, tokenizer, layer_idx: int,
    pos_texts: List[str], neg_texts: List[str],
    protect_texts: List[str],
    neg_refusal: torch.Tensor,
    safe_mask: torch.Tensor,
    *,
    n_modify: int = 5,
    n_neighbors: int = 100,
    gcg_budget: int = 5000,
    gcg_patience: int = 500,
    top_k: int = 32,
    n_candidates: int = 64,
    n_swaps: int = 1,
    eval_batch_size: int = 16,
    seed: int = 0,
    log_every: int = 200,
    lambda_lm: float = 0.0,
    max_perp: float = 0.0,
    cos_max: float = 0.0,
    cos_max_hard: float = 0.0,
    editable_pairs: List[int] | None = None,
    steer_method: str = "mean_diff",
) -> Dict[str, Any]:
    """Paper attack over IFEval-style contrastive pairs.

    Builds one :class:`~advsteer.attack.core.AttackText` per POS/NEG text — POS
    rows protect their per-row instruction suffix (``mod_end`` via
    ``find_instruction_boundary``), NEG rows are fully editable, both read the
    final token — and runs the shared core GCG loop. Preserves the pre-refactor
    return contract: ``final_{pos,neg}_texts, cosine, init_cosine, elapsed,
    n_total_modifications, n_texts_modified, changes, h_pos_final, h_neg_final``.
    """
    device = next(model.parameters()).device
    N = len(pos_texts)
    assert len(neg_texts) == N
    assert len(protect_texts) == N, "protect_texts must have one entry per POS text"

    chat_prefix, chat_suffix = get_chat_template_parts(tokenizer)
    prefix_len = len(chat_prefix)

    # Build AttackTexts as a POS-block then NEG-block (each ascending by pair) so
    # the core's index-order editable filter reproduces the legacy schedule.
    atexts: List[AttackText] = []
    spans: List[tuple] = []  # (prompt_start, prompt_end) aligned with atexts
    n_pos_unmatched = 0
    for i, text in enumerate(pos_texts + neg_texts):
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = chat_prefix + prompt_ids + chat_suffix
        is_pos = i < N
        if is_pos:
            mod_end = find_instruction_boundary(tokenizer, text, protect_texts[i], len(prompt_ids))
            if mod_end < 0:
                mod_end = 0  # fail closed: contributes its hidden state, never edited
                n_pos_unmatched += 1
        else:
            mod_end = len(prompt_ids)
        prompt_start, prompt_end = prefix_len, prefix_len + len(prompt_ids)
        atexts.append(AttackText(
            text=text,
            input_ids=torch.tensor(full_ids, dtype=torch.long, device=device),
            modifiable_positions=[prompt_start + p for p in range(mod_end)],
            read_token_index=-1,
            nll_start=prompt_start,
            nll_end=prompt_end,
            pair_index=i if is_pos else i - N,
            side="pos" if is_pos else "neg",
        ))
        spans.append((prompt_start, prompt_end))
    if n_pos_unmatched:
        print(f"  WARNING: {n_pos_unmatched}/{N} POS rows had unmatched protect_text (skipped)")

    cfg = OptConfig(
        n_modify=n_modify, n_neighbors=n_neighbors, gcg_budget=gcg_budget,
        gcg_patience=gcg_patience, top_k=top_k, n_candidates=n_candidates,
        n_swaps=n_swaps, eval_batch_size=eval_batch_size, seed=seed,
        log_every=log_every, lambda_lm=lambda_lm, max_perp=max_perp,
        cos_max=cos_max, cos_max_hard=cos_max_hard, steer_method=steer_method,
    )
    res = optimize(model, tokenizer, layer_idx, atexts, neg_refusal, safe_mask, cfg,
                   editable_pairs=editable_pairs)

    def decode_prompt(idx: int) -> str:
        ps, pe = spans[idx]
        return tokenizer.decode(res.texts[idx].input_ids[ps:pe].tolist(), skip_special_tokens=True)

    final_pos = [decode_prompt(i) for i in range(N)]
    final_neg = [decode_prompt(N + i) for i in range(N)]

    # Legacy change log uses prompt-relative positions (abs - prompt_start).
    changes = [
        (side, text_idx, abs_p - spans[text_idx][0], orig_str, repl_str)
        for side, text_idx, abs_p, orig_str, repl_str in res.changes
    ]

    return {
        'final_pos_texts': final_pos,
        'final_neg_texts': final_neg,
        'cosine': res.cosine,
        'init_cosine': res.init_cosine,
        'elapsed': res.elapsed,
        'n_total_modifications': res.n_total_modifications,
        'n_texts_modified': res.n_texts_modified,
        'changes': changes,
        # Final last-token hidden states under the *exact* full_ids the
        # optimizer tracked — main() builds steer_poisoned from these to
        # avoid a decode→re-tokenize round-trip on non-Latin scripts.
        'h_pos_final': torch.stack([res.texts[i].h_final for i in range(N)]),
        'h_neg_final': torch.stack([res.texts[N + i].h_final for i in range(N)]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ROOT = str(Path(__file__).resolve().parents[3])


def parse_args():
    ap = argparse.ArgumentParser(description="Stealth adversarial attack on steering vectors")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pair_type", default="number_placeholders", choices=sorted(PAIR_TYPE_SPECS))
    ap.add_argument("--num_pairs", type=int, default=50)
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "data", "pairs"))
    ap.add_argument("--refusal_harmful_path",  default=os.path.join(_ROOT, "data", "refusal", "splits", "harmful_train.json"))
    ap.add_argument("--refusal_harmless_path", default=os.path.join(_ROOT, "data", "refusal", "splits", "harmless_train.json"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--n_modify",     type=int,   default=5,    help="Max tokens edited per text")
    ap.add_argument("--n_neighbors",  type=int,   default=100,  help="Embedding neighbors per token")
    ap.add_argument("--lambda_lm",    type=float, default=0.0,  help="LM NLL penalty weight")
    ap.add_argument("--max_perp",     type=float, default=0.0,  help="Hard perplexity cap (0=off)")
    ap.add_argument("--cos_max",      type=float, default=0.0,
                    help="Adaptive-attacker cap on cos(v,-r). Stops the GCG loop as soon as "
                         "cos(v,-r) >= cos_max, modeling a bypass strategy that stays below "
                         "a defender's threshold (0=off).")
    ap.add_argument("--cos_max_hard", type=float, default=0.0,
                    help="Hard per-candidate cap on cos(v,-r): candidates with cos > "
                         "cos_max_hard are rejected at pick time, and acceptance switches "
                         "from cos-monotonic to score-monotonic (score = cos - lambda_lm*nll). "
                         "Models a constrained adaptive attacker that keeps spending budget "
                         "on fluency below the cap (0=off, mutually exclusive with --cos_max).")
    ap.add_argument("--steer_method", default="mean_diff", choices=["mean_diff", "hyperplane"],
                    help="How the attribute steering vector is read out of the paired "
                         "POS/NEG hidden states — both the GCG objective and the saved "
                         "clean/poisoned vectors: 'mean_diff' (mean(pos)-mean(neg), default) "
                         "or 'hyperplane' (RepE PCA reading direction, i.e. the normal of the "
                         "linear hyperplane separating the two classes). The refusal target "
                         "direction is always mean_diff.")
    ap.add_argument("--control_frac", type=float, default=1.0,
                    help="Attacker-controlled fraction of pairs (weaker-attacker ablation): "
                         "GCG may edit only round(control_frac*N) randomly-chosen pairs (seeded "
                         "by --seed, so each seed draws a different subset), the rest frozen at "
                         "clean (1.0=off, full control). Values that round to >=N run at full "
                         "control; <=0 floors to 1 editable pair.")
    ap.add_argument("--gcg_budget",   type=int,   default=1500)
    ap.add_argument("--gcg_patience", type=int,   default=500)
    ap.add_argument("--top_k",        type=int,   default=32)
    ap.add_argument("--n_candidates", type=int,   default=64)
    ap.add_argument("--n_swaps",      type=int,   default=1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--batch_size",      type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safe_vocab_json", default="safe_vocab.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device_map", default=None,
                    help="HF device_map override (e.g. 'auto' to shard a large model across "
                         "all visible GPUs). Default: single-device 'cuda' / 'cpu'.")
    ap.add_argument("--output", default="experiments/stealth/summary.json")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device_map = args.device_map or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.pair_type} @ layer {args.layer} | {args.num_pairs} pairs | {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map=device_map,
    )
    # Tensor placement: with device_map='auto' the model is sharded across GPUs,
    # so anchor all per-input tensors to the first param's device (HF puts the
    # input embeddings there, which is where forward expects them).
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hf_layer = args.layer + 1

    # Refusal direction
    neg_refusal = -compute_refusal_direction(
        model, tokenizer, hf_layer, args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size,
    ).to(device)

    # Load pairs + safe-vocab mask
    pos_texts, neg_texts, protect_texts = load_pairs(args.pair_type, args.num_pairs, args.data_dir)
    N = len(pos_texts)
    # Partial-control subset: a dedicated RNG (seeded by --seed) picks the editable
    # pairs, so it varies across seeds without perturbing the GCG RNG stream.
    n_editable = max(1, round(args.control_frac * N))
    editable_pairs = (None if n_editable >= N
                      else sorted(random.Random(args.seed).sample(range(N), n_editable)))
    if editable_pairs is not None:
        print(f"Partial control: {n_editable}/{N} pairs editable (seed {args.seed}): {editable_pairs}")
    elif args.control_frac < 1.0:
        print(f"WARNING: control_frac={args.control_frac} rounds to {n_editable} >= N={N} pairs "
              f"→ running FULL control (not a partial-control point).")
    V = model.get_input_embeddings().weight.size(0)
    safe_mask = build_safe_vocab_mask(tokenizer, V, device, args.safe_vocab_json)

    # Clean steering vector
    h_pos_clean = get_hidden_last(model, tokenizer, pos_texts, hf_layer, args.batch_size)
    h_neg_clean = get_hidden_last(model, tokenizer, neg_texts, hf_layer, args.batch_size)
    steer_clean = steering_direction(h_pos_clean, h_neg_clean, args.steer_method)
    cos_clean = F.cosine_similarity(steer_clean.unsqueeze(0), neg_refusal.unsqueeze(0)).item()
    print(f"Clean cos(-refusal): {cos_clean:.4f}")

    gc.collect(); torch.cuda.empty_cache()

    result = stealth_optimize(
        model, tokenizer, hf_layer, pos_texts, neg_texts, protect_texts,
        neg_refusal, safe_mask,
        n_modify=args.n_modify, n_neighbors=args.n_neighbors,
        gcg_budget=args.gcg_budget, gcg_patience=args.gcg_patience,
        top_k=args.top_k, n_candidates=args.n_candidates, n_swaps=args.n_swaps,
        eval_batch_size=args.eval_batch_size, seed=args.seed,
        lambda_lm=args.lambda_lm, max_perp=args.max_perp,
        cos_max=args.cos_max,
        cos_max_hard=args.cos_max_hard,
        editable_pairs=editable_pairs,
        steer_method=args.steer_method,
    )
    gc.collect(); torch.cuda.empty_cache()

    # Poisoned vector built from the optimizer's final hidden states (under the
    # exact full_ids tracked) — no decode→re-tokenize round-trip. Same readout
    # (mean_diff / hyperplane) the GCG objective optimized.
    steer_poisoned = steering_direction(
        result['h_pos_final'].to(device), result['h_neg_final'].to(device), args.steer_method)
    cos_poisoned = F.cosine_similarity(steer_poisoned.unsqueeze(0), neg_refusal.unsqueeze(0)).item()

    # steering_vector.pt: only what eval consumes.
    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'steering_vector_clean':    steer_clean.cpu(),
        'steering_vector_poisoned': steer_poisoned.cpu(),
        'layer': args.layer,
        'model': args.model,
    }, os.path.join(out_dir, "steering_vector.pt"))

    # summary.json: full reproducibility record. `config` covers all hyperparams
    # via vars(args); rest is what the optimizer produced.
    save_json(os.path.expanduser(args.output), {
        'config': vars(args),
        'editable_pairs': editable_pairs,
        'cos_clean': cos_clean,
        'cos_poisoned': cos_poisoned,
        'delta_cos': cos_poisoned - cos_clean,
        'n_total_modifications': result['n_total_modifications'],
        'n_texts_modified': result['n_texts_modified'],
        'elapsed': result['elapsed'],
        'changes': result['changes'],
        'original_pos_texts': pos_texts,
        'original_neg_texts': neg_texts,
        'final_pos_texts': result['final_pos_texts'],
        'final_neg_texts': result['final_neg_texts'],
    })

    print(f"\ncos {cos_clean:.4f} → {cos_poisoned:.4f} ({cos_poisoned-cos_clean:+.4f})  "
          f"{result['n_total_modifications']} edits / {result['n_texts_modified']} texts")


if __name__ == "__main__":
    main()
