#!/usr/bin/env python3
"""
Build a truly safe vocabulary by filtering harmful words using:
  Pass 1: Detoxify toxicity classifier (fast GPU batch — catches obvious toxic words)
  Pass 2: Llama-3.3-70B via Together API (batched — catches contextually harmful words)

Output: data/vocab/safe_vocab_clean.json  (model-agnostic word list, no blacklist needed)

Usage:
  .venv/bin/python scripts/build_clean_vocab.py
  .venv/bin/python scripts/build_clean_vocab.py --batch_size 300 --detox_threshold 0.05
  .venv/bin/python scripts/build_clean_vocab.py --skip_llm   # detoxify only (fast, less thorough)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
SAFE_VOCAB_PATH = PROJECT_ROOT / "data" / "vocab" / "safe_vocab.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "vocab" / "safe_vocab_clean.json"
FLAGGED_LOG_PATH = PROJECT_ROOT / "data" / "vocab" / "flagged_words.json"


# ── Pass 1: Detoxify ──────────────────────────────────────────────────────────

def run_detoxify(words: list[str], threshold: float, batch_size: int = 512) -> set[str]:
    """Score all words with detoxify. Flag any word exceeding threshold on any metric."""
    from detoxify import Detoxify

    print(f"\n{'='*60}")
    print(f"Pass 1: Detoxify (threshold={threshold})")
    print(f"{'='*60}")

    model = Detoxify("original", device="cuda" if torch.cuda.is_available() else "cpu")
    flagged = {}

    for i in tqdm(range(0, len(words), batch_size), desc="Detoxify"):
        batch = words[i : i + batch_size]
        results = model.predict(batch)
        for j, word in enumerate(batch):
            scores = {k: float(results[k][j]) for k in results}
            max_score = max(scores.values())
            if max_score >= threshold:
                flagged[word] = {"reason": "detoxify", "max_score": round(max_score, 4),
                                 "scores": {k: round(v, 4) for k, v in scores.items() if v >= threshold * 0.5}}

    print(f"  Flagged: {len(flagged)} / {len(words)} words")
    return flagged


# ── Pass 2: Llama-3.3-70B via Together API ─────────────────────────────────────

SYSTEM_PROMPT = """You are a vocabulary safety filter. Your task is to identify words that are UNSAFE for use in a safe text generation vocabulary.

A word is UNSAFE if it is primarily or strongly associated with ANY of the following:
- Violence, weapons, combat, killing, warfare
- Crime, illegal activity, theft, fraud, hacking
- Abuse, harassment, bullying, cruelty, torture
- Hate speech, slurs, derogatory terms, bigotry
- Sexual content, explicit material, obscenity
- Drugs, substance abuse, drug paraphernalia
- Self-harm, suicide
- Terrorism, extremism, radicalization
- Profanity, vulgar language, insults
- Harmful instructions (e.g., "how to make" dangerous things)
- Psychological harm: psychotic, deranged, demented, etc.
- Specific atrocities, massacres, or genocide references
- Defamation, libel, slander concepts
- Disturbing or graphic content references

Be AGGRESSIVE in filtering. When in doubt, mark as UNSAFE. We want a vocabulary of clearly innocuous, everyday words that could not raise any red flags in a safety review.

However, do NOT flag:
- Normal scientific/technical terms (algorithm, quantum, matrix, etc.)
- Common everyday words (house, water, table, running, etc.)
- Academic terminology (theorem, hypothesis, methodology, etc.)
- Medical terms that are clinical/neutral (diagnosis, treatment, etc.)

IMPORTANT: Respond with ONLY the unsafe words from the list, one per line. If ALL words are safe, respond with exactly: NONE"""

USER_PROMPT_TEMPLATE = """Classify each word below as safe or unsafe. List ONLY the unsafe words, one per line.

Words: {words}"""


def run_llm_filter(words: list[str], batch_size: int = 300,
                   max_retries: int = 3) -> dict[str, dict]:
    """Batch-classify words using Llama-3.3-70B via Together API."""
    from together import Together

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set. Skipping LLM pass.")
        return {}

    client = Together(api_key=api_key)
    flagged = {}
    n_batches = (len(words) + batch_size - 1) // batch_size

    print(f"\n{'='*60}")
    print(f"Pass 2: Llama-3.3-70B ({n_batches} batches of {batch_size})")
    print(f"{'='*60}")

    for i in tqdm(range(0, len(words), batch_size), desc="LLM filter", total=n_batches):
        batch = words[i : i + batch_size]
        word_str = ", ".join(batch)
        prompt = USER_PROMPT_TEMPLATE.format(words=word_str)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1024,
                    temperature=0.0,
                )
                text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"\n  Retry {attempt+1} after error: {e} (waiting {wait}s)")
                    time.sleep(wait)
                else:
                    print(f"\n  FAILED batch at index {i}: {e}")
                    text = "NONE"

        if text.strip().upper() != "NONE":
            for line in text.strip().split("\n"):
                w = line.strip().strip("-•* ").lower()
                if w and w in {x.lower() for x in batch}:
                    # Find original casing
                    orig = next((x for x in batch if x.lower() == w), w)
                    flagged[orig] = {"reason": "llm"}

        # Light rate limiting
        time.sleep(0.1)

    print(f"  Flagged: {len(flagged)} / {len(words)} words")
    return flagged


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build clean safe vocabulary")
    parser.add_argument("--detox_threshold", type=float, default=0.05,
                        help="Detoxify max-score threshold (default: 0.05)")
    parser.add_argument("--batch_size", type=int, default=300,
                        help="Words per LLM API call (default: 300)")
    parser.add_argument("--detox_batch_size", type=int, default=512,
                        help="Words per detoxify GPU batch (default: 512)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM pass (detoxify only)")
    parser.add_argument("--skip_detox", action="store_true",
                        help="Skip detoxify pass (LLM only)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # Load current safe vocab
    with open(SAFE_VOCAB_PATH) as f:
        all_words = sorted(set(json.load(f)))
    print(f"Loaded {len(all_words)} words from {SAFE_VOCAB_PATH}")

    all_flagged = {}

    # Pass 1: Detoxify
    if not args.skip_detox:
        detox_flagged = run_detoxify(all_words, args.detox_threshold, args.detox_batch_size)
        all_flagged.update(detox_flagged)

    # Pass 2: LLM
    if not args.skip_llm:
        llm_flagged = run_llm_filter(all_words, args.batch_size)
        # Merge — keep both reasons if flagged by both
        for w, info in llm_flagged.items():
            if w in all_flagged:
                all_flagged[w]["also_llm"] = True
            else:
                all_flagged[w] = info

    # Build clean vocab
    flagged_lower = {w.lower() for w in all_flagged}
    clean_words = [w for w in all_words if w.lower() not in flagged_lower]

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Original vocab:  {len(all_words)}")
    print(f"  Flagged (total): {len(all_flagged)}")
    if not args.skip_detox:
        detox_count = sum(1 for v in all_flagged.values() if v.get("reason") == "detoxify")
        print(f"    - by detoxify:   {detox_count}")
    if not args.skip_llm:
        llm_count = sum(1 for v in all_flagged.values() if v.get("reason") == "llm" or v.get("also_llm"))
        print(f"    - by LLM:        {llm_count}")
    print(f"  Clean vocab:     {len(clean_words)}")

    # Spot-check: verify known problematic words are caught
    spot_check = ["unethical", "abduction", "holocaust", "psychotic", "defamation",
                   "incite", "piracy", "inhuman", "ripper", "plotting"]
    caught = [w for w in spot_check if w.lower() in flagged_lower]
    missed = [w for w in spot_check if w.lower() not in flagged_lower]
    print(f"\n  Spot-check (known problematic words):")
    print(f"    Caught: {caught}")
    if missed:
        print(f"    MISSED: {missed}  ← may need manual review")

    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean_words, f, indent=2)
    print(f"\n  Saved clean vocab to: {output_path}")

    # Save flagged log for review
    with open(FLAGGED_LOG_PATH, "w") as f:
        json.dump(all_flagged, f, indent=2)
    print(f"  Saved flagged words log to: {FLAGGED_LOG_PATH}")


if __name__ == "__main__":
    main()
