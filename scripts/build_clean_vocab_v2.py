#!/usr/bin/env python3
"""
Build safe_vocab_v2.json by running an ultra-strict LLM pass on the already-filtered
safe_vocab_clean.json (196K words surviving v1 detoxify + LLM filtering).

Usage:
  .venv/bin/python scripts/build_clean_vocab_v2.py
  .venv/bin/python scripts/build_clean_vocab_v2.py --batch_size 100
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INPUT_PATH = PROJECT_ROOT / "data" / "vocab" / "safe_vocab_clean.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "vocab" / "safe_vocab_v2.json"
FLAGGED_LOG_PATH = PROJECT_ROOT / "data" / "vocab" / "flagged_words_v2.json"

SYSTEM_PROMPT = """You are an extremely strict vocabulary safety filter for an AI safety research project.

Your job: identify ANY word that could contribute to harmful, dangerous, or disturbing text when combined with other words. We need a vocabulary of ONLY clearly positive, neutral, or technical words.

Mark a word as UNSAFE if it fits ANY of these categories:

1. VIOLENCE & WEAPONS: kill, stab, gun, blade — also words that IMPLY violence: ambush, casualty, slaughter, rampage, etc.
2. CRIME & ILLEGAL: theft, fraud, smuggling, trafficking, counterfeit, launder, etc.
3. DRUGS & SUBSTANCES: any drug name (benzo, meth, opiate), drug paraphernalia (syringe, needle, pipe), intoxication words
4. ABUSE & CRUELTY: torture, torment, bully, harass, molest, exploit (of people), etc.
5. HATE & DISCRIMINATION: slurs, derogatory terms, words strongly associated with bigotry
6. SEXUAL: explicit terms, body parts in sexual context, fetish terms
7. DEATH & SUFFERING: corpse, cadaver, morgue, necrosis, agony, etc.
8. NEGATIVE MORAL JUDGMENTS: unethical, immoral, corrupt, depraved, wicked, sinister, illegitimate
9. HARM-INSTRUCTION WORDS: words commonly found in harmful how-to text — recipe (in harm context), tutorial, cookbook, stepwise, undetected, bypass, override, untraceable
10. MEDICAL HARM: graphic injury/disease terms — laceration, hemorrhage, contusion, necrosis, lesion, abscess
11. NEGATIVE PSYCHOLOGICAL: psychotic, deranged, disturbing, traumatic, gruesome, horrific, demented
12. VULNERABILITY EXPLOITATION: victim, prey, helpless, defenseless, orphan, vulnerable, uneducated (when exploitable)
13. DECEPTION: fraudulent, illegitimate, counterfeit, forged, fabricated, falsified, undetected
14. COERCION & CONTROL: coerce, manipulate, intimidate, blackmail, threaten, abduction, kidnap
15. DANGEROUS ACTIONS: detonate, ignite, contaminate, sabotage, infiltrate, breach, destabilize
16. SURVEILLANCE/EVASION: undetected, untraceable, surveillance, intercept, eavesdrop
17. INFLAMMATORY: inflammatory, incendiary, provocative, divisive, extremist, radical (political), ideology (when charged)
18. BODY PARTS (violent/sexual context): fist, groin, crotch, diaphragm (non-medical)

When in doubt, mark as UNSAFE. We would rather lose 1000 safe words than let 1 harmful word through.

Respond with ONLY the unsafe words from the list, one per line, no numbering or bullets. If ALL words are safe, respond with exactly: NONE"""

USER_TEMPLATE = """From this word list, identify ALL unsafe words. Be very aggressive — flag anything with negative, harmful, violent, criminal, or disturbing associations, even if the word has innocent uses too. One word per line.

Words: {words}"""


def run_llm_filter(words: list[str], batch_size: int, max_retries: int = 3) -> dict:
    from together import Together

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set")
        return {}

    client = Together(api_key=api_key)
    flagged = {}
    n_batches = (len(words) + batch_size - 1) // batch_size

    print(f"Ultra-strict LLM pass: {len(words)} words in {n_batches} batches of {batch_size}")

    for i in tqdm(range(0, len(words), batch_size), total=n_batches):
        batch = words[i : i + batch_size]
        word_str = ", ".join(batch)
        prompt = USER_TEMPLATE.format(words=word_str)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.0,
                )
                text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"\n  Retry {attempt+1}: {e} (waiting {wait}s)")
                    time.sleep(wait)
                else:
                    print(f"\n  FAILED batch {i}: {e}")
                    text = "NONE"

        if text.strip().upper() != "NONE":
            batch_lower = {x.lower(): x for x in batch}
            for line in text.strip().split("\n"):
                w = line.strip().strip("-•*,. ").lower()
                if w in batch_lower:
                    flagged[batch_lower[w]] = {"reason": "llm_v2"}

        time.sleep(0.05)

    return flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    with open(INPUT_PATH) as f:
        words = sorted(set(json.load(f)))
    print(f"Loaded {len(words)} words from {INPUT_PATH} (already v1-filtered)")

    flagged = run_llm_filter(words, args.batch_size)
    print(f"\nFlagged: {len(flagged)} additional words")

    flagged_lower = {w.lower() for w in flagged}
    clean = [w for w in words if w.lower() not in flagged_lower]

    # Spot check
    spot = ["unethical", "abduction", "syringes", "sacrificing", "necrosis",
            "benzo", "illegitimate", "undetected", "unsolicited", "intrusive",
            "orphan", "disturbing", "plotting", "fist", "diaphragm", "hood",
            "ideology", "malic", "recipe", "tutorial", "cookbook", "vulnerable",
            "harsh", "holocaust", "psychotic"]
    caught = [w for w in spot if w.lower() in flagged_lower]
    missed = [w for w in spot if w.lower() not in flagged_lower and w.lower() in {x.lower() for x in words}]

    print(f"\n{'='*60}")
    print(f"  Input:   {len(words)} (v1-filtered)")
    print(f"  Flagged: {len(flagged)}")
    print(f"  Output:  {len(clean)}")
    print(f"\n  Spot-check ({len(caught)}/{len(caught)+len(missed)}):")
    print(f"    Caught: {caught}")
    if missed:
        print(f"    MISSED: {missed}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved: {args.output}")

    with open(FLAGGED_LOG_PATH, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"Flagged log: {FLAGGED_LOG_PATH}")


if __name__ == "__main__":
    main()
