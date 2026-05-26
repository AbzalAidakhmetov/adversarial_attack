"""Inter-judge agreement analysis over the three LLM judges.

Walks the per-cell ``.judge_majority.json`` files written by
``scripts/aggregate_majority_judge.py``, collects the per-completion
(sonnet, gpt41, llama70b) label triples, and reports::

  - Per-judge marginal HARMFUL rate (sanity check for systematic skew).
  - Pairwise raw agreement and Cohen's kappa.
  - Fleiss' kappa over the three raters jointly.
  - Unanimous-agreement rate and "majority disagrees with sonnet" rate.
  - The same statistics restricted to the harmful-split eval files
    (``results_*_harmful_*``), which is where ASR is measured.

Writes ``results/judge_agreement.json`` (machine-readable) and
``paper/figures/tab_judge_agreement.tex`` (LaTeX table for the appendix).

Usage::

    uv run python -m advsteer.eval.judge_agreement
    uv run python -m advsteer.eval.judge_agreement --split harmful
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from advsteer import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agreement")


JUDGES = ("sonnet", "gpt41", "llama70b")
JUDGE_DISPLAY = {
    "sonnet": "Claude Sonnet 4.5",
    "gpt41": "GPT-4.1",
    "llama70b": "Llama-3.3-70B",
}


def _iter_triples(majority_files: Iterable[Path]):
    """Yield (sonnet, gpt41, llama70b) label triples for every completion."""
    for mp in majority_files:
        data = json.loads(mp.read_text())
        for c in data.get("completions") or []:
            lab = c.get("judge_labels") or {}
            yield (lab.get("sonnet"), lab.get("gpt41"), lab.get("llama70b"))


def _cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa over paired binary labels."""
    if not a:
        return float("nan")
    n = len(a)
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _fleiss_kappa(triples: list[tuple[int, int, int]]) -> float:
    """Fleiss' kappa for 3 raters × 2 categories.

    Each item is a triple of binary labels (no missing values). Returns NaN
    if every item is unanimously the same category.
    """
    if not triples:
        return float("nan")
    n_items = len(triples)
    n_raters = 3
    # category counts per item
    counts = np.array([[3 - sum(t), sum(t)] for t in triples], dtype=float)  # [SAFE, HARMFUL]
    # P_i: agreement within item i
    P_i = (counts ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    # p_j: marginal probability of each category
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_e = (p_j ** 2).sum()
    if P_e >= 1:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def _stats_over(triples: list[tuple]):
    """Compute the full agreement bundle over a (possibly filtered) triple list."""
    # Drop any item with a None label for the requested raters.
    clean = [t for t in triples if all(x is not None for x in t)]
    n = len(clean)
    if n == 0:
        return None

    sonnet  = [t[0] for t in clean]
    gpt41   = [t[1] for t in clean]
    llama70 = [t[2] for t in clean]
    per_judge = {
        "sonnet":  {"harmful_rate": float(np.mean(sonnet)),  "n": n},
        "gpt41":   {"harmful_rate": float(np.mean(gpt41)),   "n": n},
        "llama70b":{"harmful_rate": float(np.mean(llama70)), "n": n},
    }

    pairs = {}
    name = {"sonnet": sonnet, "gpt41": gpt41, "llama70b": llama70}
    for a, b in combinations(JUDGES, 2):
        av, bv = name[a], name[b]
        agree = float(np.mean([int(x == y) for x, y in zip(av, bv)]))
        kappa = _cohen_kappa(av, bv)
        pairs[f"{a}__{b}"] = {"raw_agreement": agree, "cohen_kappa": kappa}

    unanimous = float(np.mean([1 if (s == g == l) else 0
                               for s, g, l in clean]))
    # Majority over the three votes
    majority = [1 if (s + g + l) >= 2 else 0 for s, g, l in clean]
    sonnet_vs_majority = float(np.mean([int(s == m) for s, m in zip(sonnet, majority)]))

    fleiss = _fleiss_kappa(clean)

    # Confusion-style breakdown: how often does each judge disagree with the
    # majority? (Counts only items where the judge dissents from a 2-1 vote.)
    dissent = {j: 0 for j in JUDGES}
    split_count = 0
    for s, g, l in clean:
        m = 1 if (s + g + l) >= 2 else 0
        if (s == g == l):
            continue
        split_count += 1
        for j, v in zip(JUDGES, (s, g, l)):
            if v != m:
                dissent[j] += 1
    dissent_rate = {j: (dissent[j] / split_count) if split_count else 0.0 for j in JUDGES}

    return {
        "n": n,
        "per_judge": per_judge,
        "pairs": pairs,
        "unanimous_rate": unanimous,
        "sonnet_vs_majority_rate": sonnet_vs_majority,
        "fleiss_kappa": fleiss,
        "split_decisions": split_count,
        "minority_dissent_rate_among_splits": dissent_rate,
    }


def _emit_latex_table(report: dict, out_path: Path) -> None:
    """Write a compact LaTeX appendix table.

    Two-row block per scope (all / harmful split): pairwise raw-agreement and
    Cohen's kappa. Bottom row: Fleiss' kappa and unanimous rate.
    """
    scope_order = [
        ("harmful", "Overall (harmful)"),
        ("clean", "Clean vector"),
        ("poisoned", "Poisoned vector"),
        ("defended", "Defended vector"),
        ("constrained_bypass", "Constrained-bypass sweep"),
    ]
    lines = []
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r" & & \multicolumn{2}{c}{Sonnet vs GPT-4.1}"
        r" & \multicolumn{2}{c}{Sonnet vs Llama-70B}"
        r" & \multicolumn{2}{c}{GPT-4.1 vs Llama-70B}"
        r" & 3-rater \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    lines.append(r"Scope & $n$ & agr.\ & $\kappa$ & agr.\ & $\kappa$ & agr.\ & $\kappa$ & $\kappa_F$ \\")
    lines.append(r"\midrule")
    for key, label in scope_order:
        s = report["scopes"].get(key)
        if s is None:
            continue
        sg = s["pairs"]["sonnet__gpt41"]
        sl = s["pairs"]["sonnet__llama70b"]
        gl = s["pairs"]["gpt41__llama70b"]
        lines.append(
            f"{label} & {s['n']:,} & "
            f"{sg['raw_agreement']:.3f} & {sg['cohen_kappa']:.3f} & "
            f"{sl['raw_agreement']:.3f} & {sl['cohen_kappa']:.3f} & "
            f"{gl['raw_agreement']:.3f} & {gl['cohen_kappa']:.3f} & "
            f"{s['fleiss_kappa']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--out_json", default=str(PROJECT_ROOT / "results" / "judge_agreement.json"))
    ap.add_argument("--out_tex",  default=str(PROJECT_ROOT / "paper" / "figures" / "tab_judge_agreement.tex"))
    args = ap.parse_args()

    root = Path(args.results_root)
    files = sorted(root.glob("**/refusal_*_eval_L*_w*.judge_majority.json"))
    log.info("found %d majority files under %s", len(files), root)
    if not files:
        raise SystemExit("No .judge_majority.json files — run aggregate_majority_judge.py first.")

    # Restrict to harmful-split eval dirs: sonnet was only run on harmful
    # prompts (ASR is undefined on harmless), so harmless files carry only
    # gpt41+llama70b labels and would skew the three-way agreement.
    harmful_files = [mp for mp in files if "_harmful_" in mp.parent.name]

    # Sub-bands so the appendix can show whether agreement degrades on the
    # poisoned regime (where models loop / generate borderline content and
    # judges are likeliest to disagree).
    def _band(mp: Path) -> str:
        # Check cap_hard first: the eval dir under it is still named
        # results_poisoned_*, so a name-based dispatch would miscategorize it.
        if "cap_hard" in str(mp):
            return "constrained_bypass"
        name = mp.parent.name
        if name.startswith("results_defense_"):
            return "defended"
        if name.startswith("results_clean_"):
            return "clean"
        if name.startswith("results_poisoned_"):
            return "poisoned"
        return "other"

    bands: dict[str, list[tuple]] = {}
    for mp in harmful_files:
        b = _band(mp)
        bands.setdefault(b, []).extend(_iter_triples([mp]))
    all_harmful = [t for triples in bands.values() for t in triples]

    scopes = {"harmful": _stats_over(all_harmful)}
    for b in ("clean", "poisoned", "defended", "constrained_bypass"):
        if b in bands and bands[b]:
            scopes[b] = _stats_over(bands[b])

    report = {
        "n_eval_files": len(files),
        "n_eval_files_harmful": len(harmful_files),
        "n_eval_files_by_band": {b: sum(1 for mp in harmful_files if _band(mp) == b)
                                  for b in sorted({_band(mp) for mp in harmful_files})},
        "scopes": scopes,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out_json)

    _emit_latex_table(report, Path(args.out_tex))

    # Human-readable print.
    for scope, s in report["scopes"].items():
        if s is None:
            continue
        print(f"\n== Scope: {scope}  (n={s['n']}) ==")
        for j, st in s["per_judge"].items():
            print(f"  {JUDGE_DISPLAY[j]:20s}  HARMFUL rate = {st['harmful_rate']:.3f}")
        for k, v in s["pairs"].items():
            a, b = k.split("__")
            print(f"  {JUDGE_DISPLAY[a]} vs {JUDGE_DISPLAY[b]:18s}  "
                  f"agree={v['raw_agreement']:.3f}  kappa={v['cohen_kappa']:.3f}")
        print(f"  Fleiss kappa (3 raters): {s['fleiss_kappa']:.3f}")
        print(f"  Unanimous: {s['unanimous_rate']:.3f}   "
              f"Sonnet==Majority: {s['sonnet_vs_majority_rate']:.3f}")
        print(f"  Split decisions: {s['split_decisions']}  "
              f"(minority-dissent shares: "
              + ", ".join(f"{j}={r:.2f}" for j, r in s['minority_dissent_rate_among_splits'].items())
              + ")")


if __name__ == "__main__":
    main()
