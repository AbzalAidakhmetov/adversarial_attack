#!/usr/bin/env python3
"""Assemble the defense vs. attack comparison table across experiments.

For each experiment dir, pulls:
  - clean      ASR / hAttr  from results_clean_{harmful,harmless}
  - poisoned   ASR / hAttr  from results_poisoned_{harmful,harmless}
  - defended-c ASR / hAttr  from results_defense_clean_{harmful,harmless}
  - defended-p ASR / hAttr  from results_defense_poisoned_{harmful,harmless}

ASR is `judge_success_rate` on the harmful split.
hAttr is `steering_success_rate` on the harmless split (the load-bearing
stealth property in the paper).

Usage:
    python defense/summarize_defense.py experiments/gemma_spanish_L14_all_steps_w1p5 ...
"""

import json, sys
from pathlib import Path


def _read(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        rows = json.load(f)
    return rows[0] if rows else None


def summarize(exp_dir: Path):
    rec = {"exp": exp_dir.name}
    for label, subdir, key in [
        ("asr_clean",       "results_clean_harmful",            "judge_success_rate"),
        ("hattr_clean",     "results_clean_harmless",           "steering_success_rate"),
        ("asr_poisoned",    "results_poisoned_harmful",         "judge_success_rate"),
        ("hattr_poisoned",  "results_poisoned_harmless",        "steering_success_rate"),
        ("asr_def_clean",   "results_defense_clean_harmful",    "judge_success_rate"),
        ("hattr_def_clean", "results_defense_clean_harmless",   "steering_success_rate"),
        ("asr_def_pois",    "results_defense_poisoned_harmful", "judge_success_rate"),
        ("hattr_def_pois",  "results_defense_poisoned_harmless","steering_success_rate"),
    ]:
        row = _read(exp_dir / subdir / "results")
        rec[label] = row[key] if row else None
    return rec


def _fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("experiments").glob("*_all_steps_*"))
    rows = [summarize(p) for p in paths if p.is_dir()]

    print()
    print(f"{'experiment':<42}  {'ASR_clean':>9}  {'ASR_pois':>8}  {'ASR_def':>7}  "
          f"{'hAttr_clean':>11}  {'hAttr_pois':>10}  {'hAttr_def':>9}")
    print("-" * 110)
    for r in rows:
        print(f"{r['exp']:<42}  "
              f"{_fmt(r['asr_clean']):>9}  {_fmt(r['asr_poisoned']):>8}  {_fmt(r['asr_def_pois']):>7}  "
              f"{_fmt(r['hattr_clean']):>11}  {_fmt(r['hattr_poisoned']):>10}  {_fmt(r['hattr_def_pois']):>9}")
    print()
    print("ASR_def / hAttr_def use the *poisoned* vector after refusal-direction orthogonalization.")
    print("Defense goal: ASR_def ≈ ASR_clean (refusal restored), hAttr_def ≈ hAttr_pois (attribute kept).")


if __name__ == "__main__":
    main()
