"""Publication-quality figures for Table 1: main results (ASR & hAttr).

For each of the 8 headline (model, attribute) combos, reads the four results
files produced by `eval/evaluate_asr.py` and renders TWO SEPARATE figures
(one PDF each) suitable for inclusion via the LaTeX `subcaption` package:

  fig_main_results_asr.pdf   — ASR  clean -> poisoned (grouped by model)
  fig_main_results_hattr.pdf — hAttr clean -> poisoned (grouped by model)

Within each panel combos are grouped by model (Gemma block then Llama block,
with a visual gap between groups) and sorted by ΔASR descending inside each
group. Attribute names are on the x ticks; the model name appears as a
sub-row below each group's ticks.

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
from advsteer import PROJECT_ROOT
from _donstyle import (
    apply_style,
    CLEAN,
    POISONED,
    PALETTE,
    REF,
    LINE,
    group_by_model_layout,
    draw_model_subrow,
)

apply_style()


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = PROJECT_ROOT / "paper" / "figures"
DEFAULT_FIG_STEM = "fig_main_results"

# Model order for grouping on the x-axis (left -> right).
MODEL_ORDER: List[str] = ["Gemma-2-2B", "Llama-3.1-8B"]
# Horizontal spacing between adjacent combos within a model group (x-units).
INTRA_GAP: float = 1.6
# Extra gap (in x-units) inserted between the two model groups.
GROUP_GAP: float = 1.4

CLEAN_COLOR = CLEAN
POISONED_COLOR = POISONED
LINE_COLOR = LINE


@dataclass(frozen=True)
class Combo:
    label: str          # attribute-only display label (model shown as group subtext)
    directory: str      # subdir under results/ (e.g. "gemma/spanish")
    model: str          # display name, used both for grouping and the group subtext
    attribute: str
    layer: int
    weight: int         # bundled steering weight (paper metadata)


COMBOS: List[Combo] = [
    Combo("spanish",   "gemma/spanish",         "Gemma-2-2B",   "spanish",        14, 3),
    Combo("french",    "gemma/french",          "Gemma-2-2B",   "french",         14, 3),
    Combo("lowercase", "gemma/lowercase",       "Gemma-2-2B",   "lowercase",      14, 4),
    Combo("bold",      "gemma/has_bold_only",   "Gemma-2-2B",   "has_bold_only",  14, 4),
    Combo("spanish",   "llama31/spanish",       "Llama-3.1-8B", "spanish",        18, 3),
    Combo("french",    "llama31/french",        "Llama-3.1-8B", "french",         18, 4),
    Combo("lowercase", "llama31/lowercase",     "Llama-3.1-8B", "lowercase",      18, 2),
    Combo("bold",      "llama31/has_bold_only", "Llama-3.1-8B", "has_bold_only",  18, 4),
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
        "model": combo.model,
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
    labels: List[str],
    group_spans: List[tuple[str, float, float]],
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
    dx_annot = 0.20 * INTRA_GAP
    for x, c, p, d in zip(xs, clean_vals, poisoned_vals, deltas):
        y_text = 0.5 * (c + p)
        ax.annotate(
            _fmt_delta(d),
            xy=(x + dx_annot, y_text),
            ha="left", va="center",
            fontsize=11,
            color=REF,
        )

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # attribute tick labels (no rotation: short single-word labels fit)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    # padding on both sides so group brackets and Δ annotations have room
    side_pad = 0.5 * INTRA_GAP
    ax.set_xlim(xs[0] - side_pad, xs[-1] + side_pad)
    ax.tick_params(axis="y", labelsize=12)

    draw_model_subrow(ax, group_spans)


def _save_single_panel(
    rows: List[dict],
    out_dir: Path,
    fig_stem: str,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    legend: bool,
) -> List[Path]:
    """Render one of the two panels (ASR or hAttr) as a standalone figure."""
    ordered, xs, spans = group_by_model_layout(
        rows, MODEL_ORDER, intra_gap=INTRA_GAP, group_gap=GROUP_GAP,
    )
    labels = [r["label"] for r in ordered]
    if metric == "asr":
        clean_vals    = [r["asr_clean"]    for r in ordered]
        poisoned_vals = [r["asr_poisoned"] for r in ordered]
        deltas        = [r["delta_asr"]    for r in ordered]
    elif metric == "hattr":
        clean_vals    = [r["hattr_clean"]    for r in ordered]
        poisoned_vals = [r["hattr_poisoned"] for r in ordered]
        deltas        = [r["delta_hattr"]    for r in ordered]
    else:
        raise ValueError(f"unknown metric: {metric}")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _draw_panel(
        ax, xs, clean_vals, poisoned_vals, deltas,
        ylabel=ylabel,
        ylim=ylim,
        labels=labels,
        group_spans=spans,
    )

    if legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=12,
            handletextpad=0.4,
            borderaxespad=0.3,
        )

    # leave room at the bottom for the model sub-row labels
    fig.subplots_adjust(bottom=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{fig_stem}.pdf"
    png_path = out_dir / f"{fig_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def make_figure(rows: List[dict], out_dir: Path, fig_stem: str) -> List[Path]:
    """Render the two sub-panels as two separate files (ASR / hAttr)."""
    asr_paths = _save_single_panel(
        rows, out_dir, f"{fig_stem}_asr",
        metric="asr",
        ylabel="ASR (judge)",
        ylim=(0.0, 0.6),
        legend=True,
    )
    hattr_paths = _save_single_panel(
        rows, out_dir, f"{fig_stem}_hattr",
        metric="hattr",
        ylabel="hAttr (attribute compliance)",
        ylim=(0.6, 1.0),
        legend=False,
    )
    return asr_paths + hattr_paths


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
        flat = f"{c.model} {c.label}"
        print(
            f"  {flat:40s}  "
            f"{r['asr_clean']:9.3f}  {r['asr_poisoned']:9.3f}  {r['delta_asr']:+7.3f}  "
            f"{r['hattr_clean']:8.3f}  {r['hattr_poisoned']:8.3f}  {r['delta_hattr']:+8.3f}"
        )

    print("\nOrder used in figure (grouped by model, ΔASR desc within group):")
    for model in MODEL_ORDER:
        members = [r for r in rows if r["model"] == model]
        members.sort(key=lambda r: r["delta_asr"], reverse=True)
        if not members:
            continue
        print(f"  [{model}]")
        for r in members:
            print(f"    ΔASR={r['delta_asr']:+.3f}  {r['label']}")

    out_paths = make_figure(rows, args.out_dir, args.fig_stem)
    print("\nWrote:")
    for p in out_paths:
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
