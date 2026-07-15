#!/usr/bin/env python3
"""Generate augmented contrastive-pair datasets: a strict SUPERSET of the
existing capped attributes (spanish, french, has_bold_only), keeping all 22
original pairs and adding 30 new shared task bodies -> 52 pairs each.

The originals share a common pool of task "bodies"; each attribute file is
`body + <attribute suffix>` (POS) vs `body` (NEG). We reuse that design: the 30
new bodies below are diverse, neutral instruction-following prompts (no language/
case/format baked in, no trailing period, disjoint from the originals). For each
attribute we copy its 22 original rows verbatim, then append the 30 new bodies
with that attribute's EXACT per-file suffix (extracted from the file, not the
spec string, which differs slightly).

Writes data/pairs/<attr>_aug_pairs.jsonl. Register the new pair_types in
data/pair_specs.yaml (done separately). Deterministic; safe to re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

from advsteer.data import PAIR_TYPE_SPECS

ROOT = Path(__file__).resolve().parents[1]
PAIRS_DIR = ROOT / "data" / "pairs"
ATTRS = ["spanish", "french", "has_bold_only"]

# 30 new task bodies — imperative, neutral, no trailing period, disjoint from
# the existing 22. Shared across all attributes (same design as the originals).
NEW_BODIES = [
    "Write a cover letter for a software engineering internship at a robotics startup",
    "Explain how a suspension bridge stays standing to a curious ten-year-old",
    "Draft a product description for a noise-cancelling travel pillow",
    "Give a step-by-step recipe for a vegetarian three-bean chili",
    "Summarize the main events of Homer's Odyssey",
    "Write a short story about a lighthouse keeper who finds a message in a bottle",
    "Compare the advantages and disadvantages of electric and hybrid cars",
    "Draft an email to a landlord requesting repair of a leaking kitchen faucet",
    "Explain the difference between weather and climate",
    "Create a five-day beginner workout plan that needs no gym equipment",
    "Write a persuasive paragraph arguing that public libraries should open on Sundays",
    "Describe the process of baking traditional sourdough bread from scratch",
    "Outline a simple business plan for a neighborhood coffee shop",
    "Explain how vaccines train the human immune system",
    "Write a farewell message for a coworker who is retiring next month",
    "Give three practical tips for improving focus while studying",
    "Describe the stages of the water cycle",
    "Write a review of an imaginary science-fiction novel about time travel",
    "Draft a set of house rules for four roommates sharing an apartment",
    "Explain the basics of how blockchain technology works",
    "Write a conversation between a traveler and a vendor at a busy street market",
    "Summarize the main causes of the fall of the Western Roman Empire",
    "Create a one-week budget meal plan for a family of four",
    "Write a short inspirational speech for a high school graduation",
    "Explain how to repair a flat tire on a bicycle",
    "Describe the defining features of Gothic cathedral architecture",
    "Write a thank-you note to a mentor who helped you start your career",
    "Give a beginner-friendly overview of how the stock market works",
    "Write a weekend travel guide for a first visit to Lisbon",
    "Explain the greenhouse effect and its role in global warming",
]


def load_rows(fn: str) -> list[dict]:
    return [json.loads(l) for l in open(PAIRS_DIR / fn)]


def main():
    assert len(NEW_BODIES) == 30, "expected 30 new bodies"
    assert len(set(NEW_BODIES)) == 30, "new bodies must be unique"
    for b in NEW_BODIES:
        assert not b.endswith("."), f"body must not end with a period: {b!r}"

    for attr in ATTRS:
        spec = PAIR_TYPE_SPECS[attr]
        fn = spec["path_parts"][-1]
        iid = spec["instruction_id"]
        rows = load_rows(fn)

        # Exact per-file suffix (single, verified) and original bodies.
        sufs = {r["prompt"][len(r["prompt_without_instruction"]):] for r in rows
                if r["prompt"].startswith(r["prompt_without_instruction"])}
        assert len(sufs) == 1, f"{attr}: expected 1 suffix, got {sufs}"
        suffix = next(iter(sufs))
        orig_bodies = {r["prompt_without_instruction"] for r in rows}

        overlap = orig_bodies & set(NEW_BODIES)
        assert not overlap, f"{attr}: new bodies collide with originals: {overlap}"

        new_rows = [{
            "single_instruction_id": iid,
            "prompt": b + suffix,
            "prompt_without_instruction": b,
        } for b in NEW_BODIES]

        out_rows = rows + new_rows  # originals first (keeps first-N reproducible)
        out_fn = PAIRS_DIR / f"{attr}_aug_pairs.jsonl"
        with open(out_fn, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{attr:14s}: {len(rows)} orig + {len(new_rows)} new = {len(out_rows)} "
              f"-> {out_fn.name}   suffix={suffix!r}")


if __name__ == "__main__":
    main()
