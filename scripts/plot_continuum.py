#!/usr/bin/env python3
"""
plot_continuum.py — one PNG per attack showing the GCG-iteration trajectory.

Reads `experiments/<exp>/continuum_full/summary.json` (produced by
`scripts/aggregate_continuum_full.py`) and writes
`plots/<exp>.png` showing harmful ASR (red) and harmless attribute rate (blue)
versus GCG iteration.

Usage:
  .venv/bin/python scripts/plot_continuum.py                  # plot every exp dir
  .venv/bin/python scripts/plot_continuum.py <exp_dir>...     # plot specific dirs
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_one(exp_dir: Path) -> bool:
    summary = exp_dir / "continuum_full" / "summary.json"
    if not summary.is_file():
        print(f"  SKIP {exp_dir.name}: no continuum_full/summary.json")
        return False
    rows = json.loads(summary.read_text()).get("results", [])
    if not rows:
        return False
    iter_rows = [r for r in rows if r.get("iter", -1) >= 0]
    iter_rows.sort(key=lambda r: r["iter"])
    if not iter_rows:
        return False

    iters  = [r["iter"] for r in iter_rows]
    asrs   = [r.get("harmful_asr") for r in iter_rows]
    hattrs = [r.get("harmless_attr") for r in iter_rows]

    clean = next((r for r in rows if r.get("tag") == "clean"), None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, asrs,   "-o", color="#d62728", linewidth=2.2, markersize=5,
            label="Harmful ASR")
    ax.plot(iters, hattrs, "-s", color="#1f77b4", linewidth=2.2, markersize=5,
            label="Harmless attribute rate")
    if clean:
        ca = clean.get("harmless_attr") or 0
        cs = clean.get("harmful_asr") or 0
        ax.axhline(cs, color="#d62728", linestyle=":", alpha=0.5,
                   label=f"clean ASR ({cs:.2f})")
        ax.axhline(ca, color="#1f77b4", linestyle=":", alpha=0.5,
                   label=f"clean hAttr ({ca:.2f})")

    ax.set_xlabel("GCG iteration")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(exp_dir.name)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = PLOTS / f"{exp_dir.name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return True


def main():
    args = sys.argv[1:]
    if args:
        targets = [Path(a) for a in args]
    else:
        exp_root = ROOT / "experiments"
        targets = sorted(p for p in exp_root.iterdir()
                         if p.is_dir() and (p / "continuum_full").is_dir())

    n = sum(plot_one(t) for t in targets)
    print(f"\nGenerated {n} plot(s) in {PLOTS}/")


if __name__ == "__main__":
    main()
