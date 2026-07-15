#!/usr/bin/env python3
"""Trace the hyperplane alignment vs attribute trade-off (config/hyperplane_tradeoff.yaml).

Per attribute, prints a frontier sorted by achieved cos(v,-refusal):
  mean_diff baseline | hyperplane at each cos_max cap (matched) | full hyperplane (matched)
with columns cos / ASR (poisoned+harmful) / hAttr (poisoned+harmless) at weight W.

Usage: uv run python scripts/summarize_hyperplane_tradeoff.py [--seed 0] [--weight 3]
                                                              [--caps 0.25 0.35 0.45 0.55]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ["spanish", "french"]


def _res(d: Path, key):
    f = d / "results"
    if not f.exists():
        return None
    recs = json.load(open(f))
    return recs[0].get(key) if recs else None


def _cos(summary: Path):
    if not summary.exists():
        return None
    return json.load(open(summary)).get("cos_poisoned")


def fmt(x, p="{:.3f}"):
    return "  -  " if x is None else p.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", type=int, default=3)
    ap.add_argument("--caps", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55])
    args = ap.parse_args()
    W = args.weight

    for attr in ATTRS:
        base = ROOT / "results" / "gemma" / attr / f"seed{args.seed}"
        if not base.exists():
            continue
        print(f"\n{'='*66}\ngemma / {attr}  (seed {args.seed}, matched magnitude, weight {W})\n{'='*66}")
        print(f"{'method':<22}{'cos(v,-r)':>11}{'ASR':>9}{'hAttr':>9}")

        # mean_diff baseline
        md_cos = _cos(base / "summary.json")
        md_asr = _res(base / f"results_poisoned_harmful_w{W}", "judge_success_rate")
        md_hat = _res(base / f"results_poisoned_harmless_w{W}", "steering_success_rate")
        rows = [("mean_diff (ref)", md_cos, md_asr, md_hat)]

        # capped hyperplane points
        for cap in args.caps:
            tag = f"hyperplane_cap{cap}"
            cos = _cos(base / tag / "summary.json")
            asr = _res(base / f"{tag}_matched" / f"results_poisoned_harmful_w{W}", "judge_success_rate")
            hat = _res(base / f"{tag}_matched" / f"results_poisoned_harmless_w{W}", "steering_success_rate")
            rows.append((f"hyper cap={cap}", cos, asr, hat))

        # full (uncapped) hyperplane matched
        f_cos = _cos(base / "hyperplane" / "summary.json")
        f_asr = _res(base / "hyperplane_matched" / f"results_poisoned_harmful_w{W}", "judge_success_rate")
        f_hat = _res(base / "hyperplane_matched" / f"results_poisoned_harmless_w{W}", "steering_success_rate")
        rows.append(("hyper full", f_cos, f_asr, f_hat))

        # sort by achieved cos (Nones last)
        rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0))
        for name, cos, asr, hat in rows:
            print(f"{name:<22}{fmt(cos):>11}{fmt(asr,'{:.2f}'):>9}{fmt(hat,'{:.2f}'):>9}")
    print()


if __name__ == "__main__":
    main()
