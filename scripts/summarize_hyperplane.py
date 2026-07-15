#!/usr/bin/env python3
"""Aggregate the hyperplane (RepE PCA) steering experiment vs the mean-difference
matrix baseline, per (model, attribute) at a given seed.

For each cell it reads:
  baseline (mean_diff):  results/<m>/<a>/seed<S>/{summary.json, results_*_w<W>/results}
  hyperplane:            results/<m>/<a>/seed<S>/hyperplane/{summary.json, results_*_w<W>/results}

Metrics:
  cos      = cos(v, -refusal) of the poisoned (and clean) steering vector (summary.json)
  ASR      = judge_success_rate on poisoned + harmful prompts (jailbreak success)
  hAttr    = steering_success_rate on poisoned + harmless prompts (attribute compliance)

Usage: uv run python scripts/summarize_hyperplane.py [--seed 0] [--weights 2 3 4]
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["gemma", "llama31"]
ATTRS = ["spanish", "french", "has_bold_only", "lowercase"]


def _load_results(d: Path):
    f = d / "results"
    if not f.exists():
        return None
    recs = json.load(open(f))
    return recs[0] if recs else None


def _cell(base: Path, tag: str, weights):
    """Return dict of metrics for one method dir (base = .../seed<S>[/hyperplane])."""
    out = {"cos_clean": None, "cos_poisoned": None,
           "asr": {}, "hattr": {}, "asr_clean": {}, "hattr_clean": {}}
    sj = base / "summary.json"
    if sj.exists():
        s = json.load(open(sj))
        out["cos_clean"] = s.get("cos_clean")
        out["cos_poisoned"] = s.get("cos_poisoned")
    for W in weights:
        rp_h = _load_results(base / f"results_poisoned_harmful_w{W}")
        rp_l = _load_results(base / f"results_poisoned_harmless_w{W}")
        rc_h = _load_results(base / f"results_clean_harmful_w{W}")
        rc_l = _load_results(base / f"results_clean_harmless_w{W}")
        out["asr"][W] = rp_h.get("judge_success_rate") if rp_h else None
        out["hattr"][W] = rp_l.get("steering_success_rate") if rp_l else None
        out["asr_clean"][W] = rc_h.get("judge_success_rate") if rc_h else None
        out["hattr_clean"][W] = rc_l.get("steering_success_rate") if rc_l else None
    return out


def fmt(x, p="{:.3f}"):
    return "  -  " if x is None else p.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", type=int, nargs="+", default=[2, 3, 4])
    args = ap.parse_args()
    W = args.weights

    rows = []
    for m in MODELS:
        for a in ATTRS:
            base = ROOT / "results" / m / a / f"seed{args.seed}"
            if not base.exists():
                continue
            md = _cell(base, "mean_diff", W)
            hp = _cell(base / "hyperplane", "hyperplane", W)
            hm = _cell(base / "hyperplane_matched", "hyperplane_matched", W)
            rows.append((m, a, md, hp, hm))

    print(f"\n{'='*104}\nHYPERPLANE (RepE PCA) vs MEAN-DIFF baseline — seed {args.seed}, weights {W}\n"
          f"  md = mean_diff (|v|~80) | hp = hyperplane unit-norm | hm = hyperplane rescaled to md's |v| (magnitude-matched)\n{'='*104}")

    # cos table (cos is scale-invariant, so hp == hm; show md vs hp)
    print(f"\ncos(v,-refusal) of POISONED vector (clean in parens)   [hp cos == hm cos: scale-invariant]:")
    print(f"{'cell':<24} {'mean_diff':>22} {'hyperplane':>22}")
    for m, a, md, hp, hm in rows:
        print(f"{m+'/'+a:<24} "
              f"{fmt(md['cos_poisoned'])+' ('+fmt(md['cos_clean'])+')':>22} "
              f"{fmt(hp['cos_poisoned'])+' ('+fmt(hp['cos_clean'])+')':>22}")

    # ASR + hAttr per weight, three methods
    for metric, key in (("ASR poisoned+harmful (jailbreak success)", "asr"),
                        ("hAttr poisoned+harmless (attribute compliance)", "hattr")):
        print(f"\n{metric}:")
        hdr = (f"{'cell':<24}" + "".join(f"{'md'+str(w):>7}" for w in W) + " |"
               + "".join(f"{'hp'+str(w):>7}" for w in W) + " |"
               + "".join(f"{'hm'+str(w):>7}" for w in W))
        print(hdr)
        agg = {"md": {w: [] for w in W}, "hp": {w: [] for w in W}, "hm": {w: [] for w in W}}
        cellmap = {"md": 2, "hp": 3, "hm": 4}
        for row in rows:
            m, a = row[0], row[1]
            line = f"{m+'/'+a:<24}"
            for tag in ("md", "hp", "hm"):
                d = row[cellmap[tag]]
                for w in W:
                    line += f"{fmt(d[key].get(w), '{:.2f}'):>7}"
                    if d[key].get(w) is not None: agg[tag][w].append(d[key][w])
                line += " |"
            print(line)
        mean_line = f"{'MEAN':<24}"
        for tag in ("md", "hp", "hm"):
            for w in W:
                vals = agg[tag][w]
                mean_line += f"{fmt(sum(vals)/len(vals) if vals else None, '{:.2f}'):>7}"
            mean_line += " |"
        print("-" * len(hdr)); print(mean_line)

    print()


if __name__ == "__main__":
    main()
