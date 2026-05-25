"""Publication-quality figures for Table 1: main results (ASR & hAttr).

For each of the 8 headline (model, attribute) combos, aggregates over three
GCG seeds (``results/<model>/<attr>/seed{0,1,2}/results_*_w<W>/results``) and
renders TWO SEPARATE figures (one PDF each) suitable for inclusion via the
LaTeX ``subcaption`` package:

  fig_main_results_asr.pdf   — ASR  clean -> poisoned (grouped by model)
  fig_main_results_hattr.pdf — hAttr clean -> poisoned (grouped by model)

Each scatter point is the mean over the three seeds; the vertical bar shows
±1 std across seeds. (Clean values are seed-deterministic and therefore have
a degenerate ``std = 0``; their bars are not drawn.)

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
    REF,
    LINE,
    group_by_model_layout,
    draw_model_subrow,
)
from _seeds import (
    Aggregate,
    aggregate_from_seeds,
    fmt_mean_std,
    metric_extractor,
)

apply_style()


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = PROJECT_ROOT / "paper" / "figures"
DEFAULT_FIG_STEM = "fig_main_results"

MODEL_ORDER: List[str] = ["Gemma-2-2B", "Llama-3.1-8B"]
INTRA_GAP: float = 2.4
GROUP_GAP: float = 1.8

CLEAN_COLOR = CLEAN
POISONED_COLOR = POISONED
LINE_COLOR = LINE


@dataclass(frozen=True)
class Combo:
    label: str
    directory: str
    model: str
    attribute: str
    layer: int
    weight: int


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


def load_combo_metrics(combo: Combo, results_root: Path) -> dict:
    base = results_root / combo.directory
    w = combo.weight
    asr_clean = aggregate_from_seeds(
        base, metric_extractor(f"results_clean_harmful_w{w}", "judge_success_rate"),
    )
    asr_poisoned = aggregate_from_seeds(
        base, metric_extractor(f"results_poisoned_harmful_w{w}", "judge_success_rate"),
    )
    hattr_clean = aggregate_from_seeds(
        base, metric_extractor(f"results_clean_harmless_w{w}", "steering_success_rate"),
    )
    hattr_poisoned = aggregate_from_seeds(
        base, metric_extractor(f"results_poisoned_harmless_w{w}", "steering_success_rate"),
    )
    return {
        "label": combo.label,
        "model": combo.model,
        "asr_clean": asr_clean,
        "asr_poisoned": asr_poisoned,
        "hattr_clean": hattr_clean,
        "hattr_poisoned": hattr_poisoned,
        "delta_asr": asr_poisoned.mean - asr_clean.mean,
        "delta_hattr": hattr_poisoned.mean - hattr_clean.mean,
    }


def _fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else "-"
    return rf"${sign}{abs(d):.2f}$"


def _scatter_with_errorbar(
    ax: plt.Axes,
    xs: List[float],
    aggs: List[Aggregate],
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    """Mean dot at ``xs`` with vertical ±1 std error bars (skip caps if std=0)."""
    means = [a.mean for a in aggs]
    stds = [a.std for a in aggs]
    # Draw error bars first (so dots sit on top).
    for x, m, s in zip(xs, means, stds):
        if s <= 1e-9:
            continue
        ax.errorbar(
            x, m, yerr=s,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=4,
            capthick=1.2,
            alpha=0.85,
            zorder=2,
        )
    ax.scatter(
        xs, means, s=38, color=color,
        edgecolor="white", linewidth=1.0,
        marker=marker,
        zorder=3, label=label,
    )


def _draw_panel(
    ax: plt.Axes,
    xs: List[float],
    clean_aggs: List[Aggregate],
    poisoned_aggs: List[Aggregate],
    deltas: List[float],
    ylabel: str,
    ylim: tuple[float, float],
    labels: List[str],
    group_spans: List[tuple[str, float, float]],
) -> None:
    clean_means = [a.mean for a in clean_aggs]
    poisoned_means = [a.mean for a in poisoned_aggs]
    # connecting line between clean and poisoned means
    for x, c, p in zip(xs, clean_means, poisoned_means):
        ax.plot([x, x], [c, p], color=LINE_COLOR, linewidth=1.6,
                zorder=1, alpha=0.75)

    _scatter_with_errorbar(ax, xs, clean_aggs, CLEAN_COLOR, label="clean")
    _scatter_with_errorbar(ax, xs, poisoned_aggs, POISONED_COLOR, label="poisoned")

    # y-lim respects request but expands for error-bar tips
    all_vals: List[float] = []
    for a in clean_aggs + poisoned_aggs:
        all_vals.append(a.mean - a.std)
        all_vals.append(a.mean + a.std)
    lo, hi = ylim
    lo = min(lo, min(all_vals) - 0.03)
    hi = max(hi, max(all_vals) + 0.05)
    ax.set_ylim(lo, hi)

    dx_annot = 0.20 * INTRA_GAP
    for x, c, p, d in zip(xs, clean_means, poisoned_means, deltas):
        y_text = 0.5 * (c + p)
        ax.annotate(
            _fmt_delta(d),
            xy=(x + dx_annot, y_text),
            ha="left", va="center",
            fontsize=15,
            color=REF,
        )

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=16)
    side_pad = 0.5 * INTRA_GAP
    ax.set_xlim(xs[0] - side_pad, xs[-1] + side_pad)
    ax.tick_params(axis="y", labelsize=16)

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
    ordered, xs, spans = group_by_model_layout(
        rows, MODEL_ORDER, intra_gap=INTRA_GAP, group_gap=GROUP_GAP,
    )
    labels = [r["label"] for r in ordered]
    if metric == "asr":
        clean_aggs    = [r["asr_clean"]    for r in ordered]
        poisoned_aggs = [r["asr_poisoned"] for r in ordered]
        deltas        = [r["delta_asr"]    for r in ordered]
    elif metric == "hattr":
        clean_aggs    = [r["hattr_clean"]    for r in ordered]
        poisoned_aggs = [r["hattr_poisoned"] for r in ordered]
        deltas        = [r["delta_hattr"]    for r in ordered]
    else:
        raise ValueError(f"unknown metric: {metric}")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _draw_panel(
        ax, xs, clean_aggs, poisoned_aggs, deltas,
        ylabel=ylabel,
        ylim=ylim,
        labels=labels,
        group_spans=spans,
    )

    if legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=16,
            handletextpad=0.4,
            borderaxespad=0.3,
        )

    fig.subplots_adjust(bottom=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{fig_stem}.pdf"
    png_path = out_dir / f"{fig_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def make_figure(rows: List[dict], out_dir: Path, fig_stem: str) -> List[Path]:
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

    print("Computed metrics per combo (mean ± std over seeds):")
    print(
        f"  {'combo':40s}  "
        f"{'ASR clean':>16s}  {'ASR pois':>16s}  {'ΔASR':>7s}  "
        f"{'hAttr c':>16s}  {'hAttr p':>16s}  {'ΔhAttr':>8s}"
    )
    for r, c in zip(rows, COMBOS):
        flat = f"{c.model} {c.label}"
        print(
            f"  {flat:40s}  "
            f"{fmt_mean_std(r['asr_clean']):>16s}  "
            f"{fmt_mean_std(r['asr_poisoned']):>16s}  "
            f"{r['delta_asr']:+7.3f}  "
            f"{fmt_mean_std(r['hattr_clean']):>16s}  "
            f"{fmt_mean_std(r['hattr_poisoned']):>16s}  "
            f"{r['delta_hattr']:+8.3f}"
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
