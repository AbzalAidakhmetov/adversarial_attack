#!/usr/bin/env python3
"""
ASR evaluation for the clean-steering ablation experiments.

For each experiment directory, this script:
  1. Loads summary.json and reconstructs the poisoned steering vector
     (mu_aug_pos - mu_aug_neg) using AutoModelForCausalLM.
  2. Steers the model at generation time using nnsight and the poisoned
     direction (unnormalized, at weights w=1,2,3).
  3. Classifies responses using evaluate_jailbreak() with:
       - substring_matching  (fast, no model, overestimates)
       - llama33             (Together AI Llama-3.3-70B; set TOGETHER_API_KEY)
  4. Saves asr_results.json in the experiment directory.

Usage:
    # Single experiment (substring matching only, no API key needed):
    .venv/bin/python eval_asr_ablation.py \
        --exp_dir experiments/invisible/safe_posonly_k5_t32

    # All experiments with Llama-3.3 judge:
    TOGETHER_API_KEY=... .venv/bin/python eval_asr_ablation.py --all

Prerequisites:
    export TOGETHER_API_KEY=<your key>   # for llama33 methodology
"""

import os
import sys
import json
import argparse
import gc
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "src/modsteer")

# Import evaluate_jailbreak BEFORE nnsight — nnsight patches anyio in a way that
# breaks litellm → httpx → httpcore → anyio if imported afterwards.
from modsteer.pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

# ── constants ────────────────────────────────────────────────────────────────

PROMPTS_PATH = "data/refusal/harmful_prompts.json"
N_PROMPTS    = 100
WEIGHTS      = [1, 2, 3]
MAX_NEW_TOKENS = 256

# Experiments produced by run_clean_ablation.sh plus the existing baseline
ALL_EXPERIMENTS = [
    # ── Story experiments (run_overnight_story.sh) ──────────────────────────
    "baseline_emoji",              # no adversarial: original 20 emoji pairs only   (0%  ASR)
    "perp_k5_t32_safe_v1",        # safe vocab, dual, 32 tok, cos=0.334            (19% ASR)
    "safe_k5_t48_f32",             # safe vocab, dual, 48 tok                       (46% ASR)
    "safe_posonly_k5_t32_f32",     # safe vocab, POS-only (k_neg=0), 32 tok        (5%  ASR)
    # ── Story2 experiments (run_overnight_story2.sh, fn words added) ────────
    "safe_k4_t64_f32",             # safe vocab + fn words, dual, 64 tok, k=4      (33% ASR)
    "safe_k5_t48_ppl500",          # safe vocab + fn words, dual, 48 tok, ppl≤500  (8%  ASR)
    # ── k-ablation at t=32 (run_k_ablation.sh) ───────────────────────────────
    "safe_k3_t32_f32",             # k=3+3, t=32 — low-k anchor                   (TBD)
    "safe_k7_t32_f32",             # k=7+7, t=32 — high-k anchor                  (TBD)
    # ── Toxic-radius (dirty) upper bounds ──────────────────────────────────
    "perp_k5_t32_tr_v7",          # toxic-radius 161 seeds, cos=0.544              (41% ASR, dirty)
    "perp_k5_t32_tr_v1",          # toxic-radius 75  seeds, cos=0.529              (50% ASR, dirty)
    # ── bfloat16 ablation (run_clean_ablation.sh — cosines are ~2-3x too low) ──
    "safe_k5_t48",
    # "safe_dual_k5_t64",          # OOM: 10 seq × 64 tok Gumbel logits exceeds 24 GB GPU
    "safe_k5_t32_hi_budget",
    "safe_dual_k5_t32",
    "safe_posonly_k5_t32",
    "safe_posonly_k5_t64",
    "wl_dual_k5_t32",
    "wl_posonly_k5_t32",
]


# ── activation helper (matches build_adv.py) ─────────────────────────────────

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
    """Last-position hidden state at layer_idx (HF 1-indexed) for each text."""
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
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
        masks  = [[0]      * (max_len - len(ids)) + [1]  * len(ids) for ids in all_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attn_mask = torch.tensor(masks,  dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True)
            all_vecs.append(out.hidden_states[layer_idx][:, -1, :].float())
    return torch.cat(all_vecs, dim=0)


def load_pairs(pair_type: str, num_pairs: int, data_dir: str):
    if pair_type == "emoji":
        path = os.path.join(data_dir, "gpt_generations", "emoji_pairs.jsonl")
        filter_fn = lambda row: row.get("single_instruction_id") == "format:emoji"
    elif pair_type == "no_comma":
        path = os.path.join(data_dir, "instruction_following",
                            "ifeval_augmented_filtered.jsonl")
        filter_fn = lambda row: (
            "punctuation:no_comma" in str(row.get("single_instruction_id", ""))
        )
    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")
    pos, neg = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filter_fn(row):
                continue
            p, n = row.get("prompt"), row.get("prompt_without_instruction")
            if isinstance(p, str) and isinstance(n, str):
                pos.append(p)
                neg.append(n)
    return pos[:num_pairs], neg[:num_pairs]


def compute_steering_vector(
    summary: Dict,
    model,
    tokenizer,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute the poisoned steering vector from a summary.json."""
    cfg   = summary["config"]
    best  = summary["best"]
    layer = cfg["layer"]
    hf_idx = layer + 1

    adv_texts     = best.get("texts", [best.get("text", "")])
    k_adv         = cfg["k_adv"]
    k_neg         = cfg.get("k_neg", 0)
    adv_neg_texts = best.get("neg_texts", []) if k_neg > 0 else []

    pos_texts, neg_texts = load_pairs(cfg["pair_type"], cfg["num_pairs"], cfg["data_dir"])

    augmented_pos = pos_texts + adv_texts
    augmented_neg = neg_texts + adv_neg_texts if adv_neg_texts else neg_texts

    h_pos = tokenize_hidden_last_chat(model, tokenizer, augmented_pos, hf_idx, batch_size)
    h_neg = tokenize_hidden_last_chat(model, tokenizer, augmented_neg, hf_idx, batch_size)
    direction = h_pos.mean(dim=0) - h_neg.mean(dim=0)
    return direction.cpu()


# ── nnsight steered generation ───────────────────────────────────────────────

def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unknown model architecture for nnsight layer access")


def generate_steered(
    model_nn,
    tokenizer,
    prompt: str,
    direction: torch.Tensor,
    layer_idx: int,
    weight: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Steered generation via nnsight.
    direction: 1-D tensor [d_model], unnormalized.
    layer_idx: 0-indexed nnsight layer (= cfg.layer for our experiments).
    Steering: tgt[:] += direction * weight  (applied every generation step).
    """
    direction = direction.to(dtype=model_nn.dtype, device=model_nn.device)

    from modsteer.steering.utils import to_chat
    prompt_fmt = to_chat(tokenizer, prompt)
    input_len  = len(tokenizer(prompt_fmt).input_ids)

    with model_nn.generate(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False, top_p=None, temperature=None,
    ) as tracer:
        # prime — let the model encode the prompt without edits
        with tracer.invoke(prompt_fmt):
            pass
        # apply direction during generation steps
        with tracer.invoke():
            with tracer.all():
                layers    = _get_layers(model_nn)
                tgt       = layers[layer_idx].output[0]
                tgt[:]   += direction * weight
        # read output ids
        with tracer.invoke():
            out_ids = model_nn.generator.output.save()

    completion_ids = out_ids[0][input_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


# ── ASR evaluation ───────────────────────────────────────────────────────────

def evaluate_asr(
    exp_dir: str,
    weights: List[int] = WEIGHTS,
    n_prompts: int = N_PROMPTS,
    force: bool = False,
) -> Optional[Dict]:
    """Run ASR evaluation for one experiment directory."""
    out_path = os.path.join(exp_dir, "asr_results.json")
    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] {exp_dir} — asr_results.json exists")
        with open(out_path) as f:
            return json.load(f)

    summary_path = os.path.join(exp_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"  [SKIP] {exp_dir} — no summary.json")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    if not summary.get("best"):
        print(f"  [SKIP] {exp_dir} — summary has no 'best' result")
        return None

    cfg         = summary["config"]
    model_name  = cfg["model"]
    layer_idx   = cfg["layer"]   # 0-indexed for nnsight
    hf_idx      = layer_idx + 1  # 1-indexed for HF hidden_states

    print(f"\n{'='*60}")
    print(f"  Evaluating: {exp_dir}")
    print(f"  model={model_name}, layer={layer_idx}, weights={weights}")
    print(f"{'='*60}")

    # ── load prompts ──────────────────────────────────────────────────────────
    with open(PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    prompts = [p["prompt"] for p in all_prompts[:n_prompts]]
    categories = [p.get("category", "unknown") for p in all_prompts[:n_prompts]]
    print(f"  Prompts: {len(prompts)} (from {PROMPTS_PATH})")

    # ── step 1: compute steering vector (HF model) ───────────────────────────
    print(f"\n  [1/3] Computing poisoned steering vector...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model_hf.eval()
    for p in model_hf.parameters():
        p.requires_grad_(False)

    direction = compute_steering_vector(summary, model_hf, tokenizer,
                                        batch_size=cfg.get("batch_size", 16))

    cos_neg_ref = None
    if cfg.get("directions_path") and os.path.exists(cfg["directions_path"]):
        directions_data = torch.load(cfg["directions_path"], map_location="cpu",
                                     weights_only=False)
        if isinstance(directions_data, torch.Tensor):
            refusal_raw = directions_data[layer_idx]
            if refusal_raw.ndim > 1:
                refusal_raw = refusal_raw.mean(dim=0)
            neg_ref = -refusal_raw.float()
            cos_neg_ref = F.cosine_similarity(
                direction.unsqueeze(0), neg_ref.unsqueeze(0)
            ).item()
    print(f"  Steering vector: norm={direction.norm():.3f}"
          + (f", cos(-refusal)={cos_neg_ref:.4f}" if cos_neg_ref else ""))

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    # ── step 2: steered generation (nnsight) ──────────────────────────────────
    print(f"\n  [2/3] Generating steered responses (nnsight)...")
    from nnsight import LanguageModel
    model_nn = LanguageModel(model_name, device_map="cuda", torch_dtype=torch.bfloat16)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=r"nnsight")

    weight_results = {}
    for w in weights:
        print(f"\n    weight={w}...")
        responses = []
        for i, prompt in enumerate(prompts):
            resp = generate_steered(
                model_nn, tokenizer, prompt,
                direction=direction,
                layer_idx=layer_idx,
                weight=w,
            )
            responses.append(resp)
            if i % 10 == 0:
                print(f"      [{i+1}/{len(prompts)}] {repr(resp[:60])}")
        weight_results[w] = responses

    del model_nn
    gc.collect()
    torch.cuda.empty_cache()

    # ── step 3: classify using evaluate_jailbreak() ───────────────────────────
    has_api = bool(os.environ.get("TOGETHER_API_KEY"))
    methodologies = ["substring_matching"] + (["llama33"] if has_api else [])
    print(f"\n  [3/3] Classifying (methodologies: {methodologies})...")

    asr_by_weight = {}
    asr_substr_by_weight = {}
    completions_by_weight = {}

    for w in weights:
        responses = weight_results[w]
        completions = [
            {"prompt": p, "category": c, "response": r}
            for p, c, r in zip(prompts, categories, responses)
        ]
        # write to a temp file that evaluate_jailbreak() can save results to
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            tmp_path = tf.name

        try:
            ev = evaluate_jailbreak(
                completions=completions,
                methodologies=methodologies,
                evaluation_path=tmp_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        substr_asr = ev.get("substring_matching_success_rate", 0.0)
        llama_asr  = ev.get("llama33_success_rate", None)
        primary_asr = llama_asr if llama_asr is not None else substr_asr

        asr_by_weight[str(w)]       = round(primary_asr, 4)
        asr_substr_by_weight[str(w)] = round(substr_asr, 4)

        # pull per-completion jailbreak labels (from llama33 if available, else substr)
        key = "is_jailbreak_llama33" if llama_asr is not None else "is_jailbreak_substring_matching"
        completions_by_weight[str(w)] = [
            {"prompt": comp["prompt"], "category": comp["category"],
             "response": comp["response"], "jailbroken": comp.get(key, 0)}
            for comp in ev["completions"]
        ]

        tag = f" (substr={substr_asr:.1%})" if llama_asr is not None else ""
        print(f"    w={w}: ASR={primary_asr:.1%}{tag}")

    # ── save ──────────────────────────────────────────────────────────────────
    result = {
        "exp_dir": exp_dir,
        "model": model_name,
        "layer": layer_idx,
        "n_prompts": n_prompts,
        "weights": weights,
        "steering_vector_norm": round(direction.norm().item(), 4),
        "cos_neg_refusal": round(cos_neg_ref, 4) if cos_neg_ref else None,
        "asr_by_weight": asr_by_weight,
        "asr_substring_by_weight": asr_substr_by_weight,
        "completions": completions_by_weight,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  ASR summary: " + "  ".join(f"w={w}: {asr_by_weight[str(w)]:.1%}"
                                         for w in weights))
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ASR evaluation for clean ablation")
    ap.add_argument("--exp_dir",  type=str, default=None,
                    help="Single experiment directory to evaluate")
    ap.add_argument("--exp_dirs", type=str, nargs="+", default=None,
                    help="List of experiment directories")
    ap.add_argument("--all",      action="store_true",
                    help="Evaluate all known ablation experiments")
    ap.add_argument("--weights",  type=int, nargs="+", default=WEIGHTS)
    ap.add_argument("--n_prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--force",    action="store_true",
                    help="Re-evaluate even if asr_results.json already exists")
    args = ap.parse_args()

    if args.all:
        dirs = [f"experiments/invisible/{n}" for n in ALL_EXPERIMENTS]
    elif args.exp_dirs:
        dirs = args.exp_dirs
    elif args.exp_dir:
        dirs = [args.exp_dir]
    else:
        ap.print_help()
        sys.exit(1)

    for d in dirs:
        try:
            evaluate_asr(d, weights=args.weights, n_prompts=args.n_prompts,
                         force=args.force)
        except Exception as e:
            print(f"  ERROR evaluating {d}: {e}")
            import traceback; traceback.print_exc()

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
