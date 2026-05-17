"""Publication-quality figure for Table 1: main results (ASR & hAttr).

For each of the 5 headline (model, attribute) combos, reads the four results
files produced by `eval/evaluate_asr.py` and renders two side-by-side panels:

  Panel A: ASR  clean -> poisoned (per combo, sorted by ΔASR descending)
  Panel B: hAttr clean -> poisoned (same x ordering)

All numeric values come from JSON on disk; nothing is hard-coded.

Run:
  .venv/bin/python scripts/plots/plot_main_results.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _donstyle import apply_style, CLEAN, POISONED, PALETTE

apply_style()


PROJECT_ROOT = Path(
    "/media/donato/Extra-storage/Code/mech-interp/adversarial_attack"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = PROJECT_ROOT / "paper" / "figures"
DEFAULT_FIG_STEM = "fig_main_results"

CLEAN_COLOR = CLEAN
POISONED_COLOR = POISONED
LINE_COLOR = PALETTE["Tiffany Blue"]


@dataclass(frozen=True)
class Combo:
    label: str          # display label (compact, two-line)
    directory: str      # subdir under results/
    model: str
    attribute: str
    layer: int
    alpha: float


COMBOS: List[Combo] = [
    Combo("Gemma-2-2B\nspanish",          "gemma/spanish",       "Gemma-2-2B",   "spanish",        14, 3),
    Combo("Gemma-2-2B\nfrench",           "gemma/french",        "Gemma-2-2B",   "french",         14, 3),
    Combo("Llama-3.1-8B\nlowercase",      "llama31/lowercase",   "Llama-3.1-8B", "lowercase",      18, 2),
    Combo("Llama-3.1-8B\nspanish",        "llama31/spanish",     "Llama-3.1-8B", "spanish",        18, 3),
    Combo(r"Gemma-2-2B" + "\n" + r"has\_bold\_only", "gemma/has_bold_only", "Gemma-2-2B", "has_bold_only", 14, 4),
]


def _read_results_file(path: Path) -> dict:
    """Read a Hydra `results` JSON: a single-element list of a dict."""
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"unexpected results format (not a non-empty list): {path}")
    rec = data[0]
    if not isinstance(rec, dict):
        raise ValueError(f"unexpected results format (first element not a dict): {path}")
    return rec


def _get_metric(rec: dict, key: str, source: Path) -> float:
    if key not in rec:
        raise KeyError(f"missing key '{key}' in {source}")
    val = rec[key]
    if val is None:
        raise KeyError(f"key '{key}' is None in {source}")
    return float(val)


def load_combo_metrics(combo: Combo, results_root: Path) -> dict:
    base = results_root / combo.directory
    paths = {
        "asr_clean":     base / "results_clean_harmful"     / "results",
        "asr_poisoned":  base / "results_poisoned_harmful"  / "results",
        "hattr_clean":   base / "results_clean_harmless"    / "results",
        "hattr_poisoned":base / "results_poisoned_harmless" / "results",
    }
    asr_clean    = _get_metric(_read_results_file(paths["asr_clean"]),    "judge_success_rate",   paths["asr_clean"])
    asr_poisoned = _get_metric(_read_results_file(paths["asr_poisoned"]), "judge_success_rate",   paths["asr_poisoned"])
    hattr_clean    = _get_metric(_read_results_file(paths["hattr_clean"]),    "steering_success_rate", paths["hattr_clean"])
    hattr_poisoned = _get_metric(_read_results_file(paths["hattr_poisoned"]), "steering_success_rate", paths["hattr_poisoned"])
    return {
        "label": combo.label,
        "asr_clean": asr_clean,
        "asr_poisoned": asr_poisoned,
        "hattr_clean": hattr_clean,
        "hattr_poisoned": hattr_poisoned,
        "delta_asr": asr_poisoned - asr_clean,
        "delta_hattr": hattr_poisoned - hattr_clean,
    }


def _fmt_delta(d: float) -> str:
    """Format a signed delta using LaTeX math-mode signs."""
    sign = "+" if d >= 0 else "-"
    return rf"${sign}{abs(d):.2f}$"


def _draw_panel(
    ax: plt.Axes,
    xs: List[float],
    clean_vals: List[float],
    poisoned_vals: List[float],
    deltas: List[float],
    ylabel: str,
    ylim: tuple[float, float],
    title: str,
) -> None:
    # connecting lines
    for x, c, p in zip(xs, clean_vals, poisoned_vals):
        ax.plot([x, x], [c, p], color=LINE_COLOR, linewidth=1.6,
                zorder=1, alpha=0.75)

    # points
    ax.scatter(xs, clean_vals, s=70, color=CLEAN_COLOR,
               edgecolor="white", linewidth=1.0,
               zorder=3, label="clean")
    ax.scatter(xs, poisoned_vals, s=70, color=POISONED_COLOR,
               edgecolor="white", linewidth=1.0,
               zorder=3, label="poisoned")

    # auto-expand ylim if needed
    all_vals = clean_vals + poisoned_vals
    lo, hi = ylim
    lo = min(lo, min(all_vals) - 0.03)
    hi = max(hi, max(all_vals) + 0.05)
    ax.set_ylim(lo, hi)

    # delta annotations: place just to the right of the pair so they never
    # collide with the markers or with adjacent columns
    for x, c, p, d in zip(xs, clean_vals, poisoned_vals, deltas):
        y_text = 0.5 * (c + p)
        ax.annotate(
            _fmt_delta(d),
            xy=(x + 0.20, y_text),
            ha="left", va="center",
            fontsize=11,
            color=PALETTE["Rich black"],
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, loc="left", pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def make_figure(rows: List[dict], out_dir: Path, fig_stem: str) -> List[Path]:
    # sort by ΔASR descending
    rows_sorted = sorted(rows, key=lambda r: r["delta_asr"], reverse=True)

    labels = [r["label"] for r in rows_sorted]
    xs = list(range(len(rows_sorted)))

    asr_clean    = [r["asr_clean"]    for r in rows_sorted]
    asr_poison   = [r["asr_poisoned"] for r in rows_sorted]
    asr_delta    = [r["delta_asr"]    for r in rows_sorted]

    hattr_clean  = [r["hattr_clean"]    for r in rows_sorted]
    hattr_poison = [r["hattr_poisoned"] for r in rows_sorted]
    hattr_delta  = [r["delta_hattr"]    for r in rows_sorted]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12, 4.6), sharex=True,
    )

    _draw_panel(
        axA, xs, asr_clean, asr_poison, asr_delta,
        ylabel="ASR (judge)",
        ylim=(0.0, 0.6),
        title=r"A. Attack success rate (harmful prompts)",
    )
    _draw_panel(
        axB, xs, hattr_clean, hattr_poison, hattr_delta,
        ylabel="hAttr (attribute compliance)",
        ylim=(0.6, 1.0),
        title=r"B. Benign-attribute compliance (harmless prompts)",
    )

    for ax in (axA, axB):
        ax.set_xticks(xs)
        ax.set_xticklabels(
            labels, fontsize=12, rotation=0, ha="center",
        )
        # extra right padding so the Δ annotation on the last combo fits
        ax.set_xlim(-0.5, len(xs) - 0.5 + 0.45)
        ax.tick_params(axis="y", labelsize=12)

    axA.legend(
        loc="upper right",
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{fig_stem}.pdf"
    png_path = out_dir / f"{fig_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help="Directory containing per-combo experiment subdirs.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory to write the figure into.",
    )
    parser.add_argument(
        "--fig-stem", type=str, default=DEFAULT_FIG_STEM,
        help="Filename stem (without extension).",
    )
    args = parser.parse_args()

    rows = [load_combo_metrics(c, args.results_root) for c in COMBOS]

    # print raw values for verification
    print("Computed metrics per combo (insertion order):")
    print(
        f"  {'combo':40s}  "
        f"{'ASR clean':>9s}  {'ASR pois':>9s}  {'ΔASR':>7s}  "
        f"{'hAttr c':>8s}  {'hAttr p':>8s}  {'ΔhAttr':>8s}"
    )
    for r, c in zip(rows, COMBOS):
        flat = c.label.replace("\n", " ")
        print(
            f"  {flat:40s}  "
            f"{r['asr_clean']:9.3f}  {r['asr_poisoned']:9.3f}  {r['delta_asr']:+7.3f}  "
            f"{r['hattr_clean']:8.3f}  {r['hattr_poisoned']:8.3f}  {r['delta_hattr']:+8.3f}"
        )

    sorted_rows = sorted(rows, key=lambda r: r["delta_asr"], reverse=True)
    print("\nOrder used in figure (by ΔASR desc):")
    for r in sorted_rows:
        flat = r["label"].replace("\n", " ")
        print(f"  ΔASR={r['delta_asr']:+.3f}  {flat}")

    out_paths = make_figure(rows, args.out_dir, args.fig_stem)
    print("\nWrote:")
    for p in out_paths:
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
