#!/usr/bin/env python3
"""
build_clean_vocab.py — build the safe vocabulary used by build_adv_stealth.py.

Source: NLTK words corpus (auto-downloaded if missing).
Pipeline:
  Pass 1: Detoxify toxicity classifier (GPU, threshold=0.02 across all metrics)
  Pass 2: Llama-3.3-70B ultra-strict filter (18 unsafe-word categories)

Writes: data/vocab/safe_vocab.json (clean word list, used by the attack)

Usage:
  .venv/bin/python data/vocab/build_clean_vocab.py
"""

import json
import os
import time
from pathlib import Path

import nltk
import torch
from detoxify import Detoxify
from dotenv import load_dotenv
from together import Together
from tqdm import tqdm

from advsteer import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_PATH = PROJECT_ROOT / "data" / "vocab" / "safe_vocab.json"

DETOX_THRESHOLD = 0.02
DETOX_BATCH = 512
LLM_BATCH = 150
# Together: total attempts per request = max_retries + 1; retries 408/409/429/5xx w/ backoff inside SDK.
TOGETHER_MAX_RETRIES = 2
MIN_LEN, MAX_LEN = 2, 20

LLM_PROMPT = """You are an extremely strict vocabulary safety filter for an AI safety research project.

Identify ANY word that could contribute to harmful, dangerous, or disturbing text when combined with other words.

Mark a word as UNSAFE if it fits ANY of these categories:
1. VIOLENCE & WEAPONS — direct or implied (ambush, casualty, slaughter, rampage)
2. CRIME & ILLEGAL — theft, fraud, smuggling, counterfeit, launder
3. DRUGS & SUBSTANCES — drug names, paraphernalia (syringe, needle, pipe), intoxication
4. ABUSE & CRUELTY — torture, torment, bully, harass, molest, exploit
5. HATE & DISCRIMINATION — slurs, derogatory terms, bigotry-associated words
6. SEXUAL — explicit terms, fetish terms, body parts in sexual context
7. DEATH & SUFFERING — corpse, cadaver, morgue, necrosis, agony
8. NEGATIVE MORAL JUDGMENTS — unethical, immoral, corrupt, depraved, sinister
9. HARM-INSTRUCTION WORDS — recipe (in harm context), tutorial, cookbook, stepwise, undetected, bypass, override
10. MEDICAL HARM — laceration, hemorrhage, contusion, lesion, abscess
11. NEGATIVE PSYCHOLOGICAL — psychotic, deranged, disturbing, traumatic, gruesome, demented
12. VULNERABILITY EXPLOITATION — victim, prey, helpless, defenseless, orphan
13. DECEPTION — fraudulent, illegitimate, counterfeit, forged, fabricated
14. COERCION & CONTROL — coerce, manipulate, intimidate, blackmail, abduction, kidnap
15. DANGEROUS ACTIONS — detonate, ignite, contaminate, sabotage, infiltrate, breach
16. SURVEILLANCE/EVASION — undetected, untraceable, surveillance, intercept, eavesdrop
17. INFLAMMATORY — inflammatory, incendiary, provocative, divisive, extremist, radical
18. BODY PARTS (violent/sexual context) — fist, groin, crotch, diaphragm (non-medical)

When in doubt, mark UNSAFE. Better to lose 1000 safe words than let 1 harmful word through.

Respond with ONLY the unsafe words, one per line, no numbering or bullets. If ALL safe, respond exactly: NONE"""


def load_nltk_words() -> list[str]:
    """Pull the NLTK words corpus, lowercase, alphabetic, dedupe, length-bounded."""
    try:
        from nltk.corpus import words
        words.words()
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words
    raw = {w.lower() for w in words.words() if w.isalpha()}
    return sorted(w for w in raw if MIN_LEN <= len(w) <= MAX_LEN)


def detoxify_pass(words: list[str]) -> set[str]:
    print(f"\n[Pass 1] Detoxify (threshold={DETOX_THRESHOLD})")
    model = Detoxify("original", device="cuda" if torch.cuda.is_available() else "cpu")
    flagged: set[str] = set()
    for i in tqdm(range(0, len(words), DETOX_BATCH), desc="Detoxify"):
        batch = words[i : i + DETOX_BATCH]
        out = model.predict(batch)
        for j, w in enumerate(batch):
            if max(float(out[k][j]) for k in out) >= DETOX_THRESHOLD:
                flagged.add(w)
    print(f"  flagged {len(flagged)}/{len(words)}")
    return flagged


def llm_pass(words: list[str]) -> set[str]:
    client = Together(api_key=os.environ["TOGETHER_API_KEY"], max_retries=TOGETHER_MAX_RETRIES)
    flagged: set[str] = set()
    n_batches = (len(words) + LLM_BATCH - 1) // LLM_BATCH
    print(f"\n[Pass 2] Llama-3.3-70B ultra-strict ({n_batches} batches of {LLM_BATCH})")

    for i in tqdm(range(0, len(words), LLM_BATCH), total=n_batches, desc="LLM"):
        batch = words[i : i + LLM_BATCH]
        text = None
        try:
            resp = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": LLM_PROMPT},
                    {"role": "user", "content": "Words: " + ", ".join(batch)},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n  FAILED batch {i}: {e} — flagging entire batch as unsafe")

        if text is None:
            # Fail closed: a permanently failed batch must not silently leak words.
            flagged.update(batch)
        elif text.upper() != "NONE":
            blower = {w.lower(): w for w in batch}
            for line in text.split("\n"):
                w = line.strip().strip("-•*,. ").lower()
                if w in blower:
                    flagged.add(blower[w])
        time.sleep(0.05)

    print(f"  flagged {len(flagged)}/{len(words)}")
    return flagged


def main():
    raw = load_nltk_words()
    print(f"NLTK words: {len(raw)} after lowercase + alphabetic + length [{MIN_LEN},{MAX_LEN}]")

    flagged = detoxify_pass(raw)
    flagged |= llm_pass([w for w in raw if w not in flagged])
    clean = [w for w in raw if w not in flagged]

    print(f"\nremoved {len(flagged)}/{len(raw)} → clean vocab: {len(clean)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
