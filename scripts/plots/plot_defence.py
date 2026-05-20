"""Plot the refusal-direction orthogonalisation defence.

Mirrors the layout of ``plot_main_results.py``: combos grouped by model with
attribute names on the x-ticks and the model name in a sub-row underneath.
Emits TWO SEPARATE PDFs (one per panel) suitable for inclusion via the LaTeX
``subcaption`` package:

  fig_defence_asr.pdf   — ASR    clean / poisoned / poisoned+defence
  fig_defence_hattr.pdf — hAttr  clean / poisoned / poisoned+defence

Run from project root:
    .venv/bin/python scripts/plots/plot_defence.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advsteer import PROJECT_ROOT
from _donstyle import (
    apply_style,
    CLEAN,
    POISONED,
    DEFENDED,
    LINE,
    ACCENT,
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
DEFAULT_FIG_STEM = "fig_defence"

MODEL_ORDER: List[str] = ["Gemma-2-2B", "Llama-3.1-8B"]
INTRA_GAP: float = 1.6
GROUP_GAP: float = 1.4

CLEAN_COLOR = CLEAN
POISONED_COLOR = POISONED
DEFENDED_COLOR = DEFENDED
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


def _aggregate(base: Path, subdir: str, key: str, *, seeds=(0, 1, 2)) -> Aggregate:
    return aggregate_from_seeds(base, metric_extractor(subdir, key), seeds=seeds)


def load_combo_metrics(combo: Combo, results_root: Path) -> dict:
    base = results_root / combo.directory
    w = combo.weight
    asr_clean    = _aggregate(base, f"results_clean_harmful_w{w}",            "judge_success_rate")
    asr_poisoned = _aggregate(base, f"results_poisoned_harmful_w{w}",         "judge_success_rate")
    asr_defended = _aggregate(base, f"results_defense_poisoned_harmful_w{w}", "judge_success_rate", seeds=(0,))
    hattr_clean    = _aggregate(base, f"results_clean_harmless_w{w}",            "steering_success_rate")
    hattr_poisoned = _aggregate(base, f"results_poisoned_harmless_w{w}",         "steering_success_rate")
    hattr_defended = _aggregate(base, f"results_defense_poisoned_harmless_w{w}", "steering_success_rate", seeds=(0,))
    return {
        "label": combo.label,
        "model": combo.model,
        "asr_clean":    asr_clean,
        "asr_poisoned": asr_poisoned,
        "asr_defended": asr_defended,
        "hattr_clean":    hattr_clean,
        "hattr_poisoned": hattr_poisoned,
        "hattr_defended": hattr_defended,
        "delta_asr": asr_poisoned.mean - asr_clean.mean,
    }


def _gap_recovered(c: float, p: float, d: float):
    denom = p - c
    if abs(denom) < 1e-9:
        return None
    return (p - d) / denom


def _scatter_with_err(
    ax: plt.Axes,
    x: float,
    agg: Aggregate,
    *,
    color: str,
    marker: str = "o",
    size: int = 38,
    label: str | None = None,
) -> None:
    if agg.std > 1e-9:
        ax.errorbar(
            x, agg.mean, yerr=agg.std,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=4,
            capthick=1.2,
            alpha=0.85,
            zorder=2,
        )
    ax.scatter(
        [x], [agg.mean],
        s=size, color=color,
        edgecolor="white", linewidth=1.0,
        marker=marker,
        zorder=3, label=label,
    )


def _draw_panel(
    ax: plt.Axes,
    xs: List[float],
    clean_aggs: List[Aggregate],
    poisoned_aggs: List[Aggregate],
    defended_aggs: List[Aggregate],
    ylabel: str,
    ylim: tuple[float, float],
    labels: List[str],
    group_spans: List[tuple[str, float, float]],
    annotate_recovered: bool,
) -> None:
    # connecting polyline clean -> poisoned -> defended
    for x, c, p, d in zip(xs, clean_aggs, poisoned_aggs, defended_aggs):
        ax.plot(
            [x, x, x], [c.mean, p.mean, d.mean],
            color=LINE_COLOR, linewidth=1.6, zorder=1, alpha=0.75,
        )

    for i, x in enumerate(xs):
        first = i == 0
        _scatter_with_err(
            ax, x, clean_aggs[i],
            color=CLEAN_COLOR, marker="o", size=38,
            label="clean" if first else None,
        )
        _scatter_with_err(
            ax, x, poisoned_aggs[i],
            color=POISONED_COLOR, marker="o", size=38,
            label="poisoned" if first else None,
        )
        _scatter_with_err(
            ax, x, defended_aggs[i],
            color=DEFENDED_COLOR, marker="D", size=42,
            label="poisoned + defence" if first else None,
        )

    if annotate_recovered:
        dx_annot = 0.20 * INTRA_GAP
        for x, c, p, d in zip(xs, clean_aggs, poisoned_aggs, defended_aggs):
            rec = _gap_recovered(c.mean, p.mean, d.mean)
            if rec is None:
                continue
            ax.annotate(
                rf"${rec * 100:.0f}\%$",
                xy=(x + dx_annot, d.mean),
                ha="left", va="center",
                fontsize=11,
                color=ACCENT,
            )

    # y-lim expanded for std bars
    all_vals: List[float] = []
    for agg in clean_aggs + poisoned_aggs + defended_aggs:
        all_vals.append(agg.mean - agg.std)
        all_vals.append(agg.mean + agg.std)
    lo, hi = ylim
    lo = min(lo, min(all_vals) - 0.03)
    hi = max(hi, max(all_vals) + 0.05)
    ax.set_ylim(lo, hi)

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
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
    annotate_recovered: bool,
) -> List[Path]:
    ordered, xs, spans = group_by_model_layout(
        rows, MODEL_ORDER, intra_gap=INTRA_GAP, group_gap=GROUP_GAP,
    )
    labels = [r["label"] for r in ordered]
    clean_aggs    = [r[f"{metric}_clean"]    for r in ordered]
    poisoned_aggs = [r[f"{metric}_poisoned"] for r in ordered]
    defended_aggs = [r[f"{metric}_defended"] for r in ordered]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _draw_panel(
        ax, xs,
        clean_aggs, poisoned_aggs, defended_aggs,
        ylabel=ylabel,
        ylim=ylim,
        labels=labels,
        group_spans=spans,
        annotate_recovered=annotate_recovered,
    )

    if legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=12,
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
        annotate_recovered=True,
    )
    hattr_paths = _save_single_panel(
        rows, out_dir, f"{fig_stem}_hattr",
        metric="hattr",
        ylabel="hAttr (attribute compliance)",
        ylim=(0.6, 1.0),
        legend=False,
        annotate_recovered=False,
    )
    return asr_paths + hattr_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-stem", type=str, default=DEFAULT_FIG_STEM)
    args = parser.parse_args()

    rows = [load_combo_metrics(c, args.results_root) for c in COMBOS]

    print(
        f"  {'combo':40s}  "
        f"{'ASR_c':>14} {'ASR_p':>14} {'ASR_d':>8}  "
        f"{'hA_c':>14} {'hA_p':>14} {'hA_d':>8}  "
        f"{'gap_rec':>8}  {'ΔASR':>6}"
    )
    for r, c in zip(rows, COMBOS):
        flat = f"{c.model} {c.label}"
        rec = _gap_recovered(
            r["asr_clean"].mean, r["asr_poisoned"].mean, r["asr_defended"].mean,
        )
        rec_s = "n/a" if rec is None else f"{rec * 100:6.1f}%"
        print(
            f"  {flat:40s}  "
            f"{fmt_mean_std(r['asr_clean']):>14} "
            f"{fmt_mean_std(r['asr_poisoned']):>14} "
            f"{r['asr_defended'].mean:>8.3f}  "
            f"{fmt_mean_std(r['hattr_clean']):>14} "
            f"{fmt_mean_std(r['hattr_poisoned']):>14} "
            f"{r['hattr_defended'].mean:>8.3f}  "
            f"{rec_s:>8}  {r['delta_asr']:>+6.3f}"
        )

    out_paths = make_figure(rows, args.out_dir, args.fig_stem)
    print("\nWrote:")
    for p in out_paths:
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
