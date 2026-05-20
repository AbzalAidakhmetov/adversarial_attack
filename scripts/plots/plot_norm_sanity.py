"""Sanity-check figure for the steering-vector poisoning paper (Table 3).

Mirrors the layout of ``plot_defence.py``: combos grouped by model with
attribute names on the x-ticks and the model name in a sub-row underneath.
Emits TWO SEPARATE PDFs (one per panel) suitable for inclusion via the LaTeX
``subcaption`` package:

  fig_norm_sanity_ratio.pdf — poisoned/clean steering-vector norm ratio.
  fig_norm_sanity_ppl.pdf   — response perplexity on harmless prompts.

Per-cell results are read across
``results/<model>/<attr>/seed{0,1,2}/results_*_w<W>/...``.

Run from project root::

    .venv/bin/python scripts/plots/plot_norm_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    vector_norm_extractor,
)

apply_style()


RESULTS_ROOT = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

MODEL_ORDER: list[str] = ["Gemma-2-2B", "Llama-3.1-8B"]
INTRA_GAP: float = 1.6
GROUP_GAP: float = 1.4

# (attribute-only label, model display name, experiment directory, bundled weight).
COMBOS: list[tuple[str, str, str, int]] = [
    ("spanish",   "Gemma-2-2B",   "gemma/spanish",         3),
    ("french",    "Gemma-2-2B",   "gemma/french",          3),
    ("lowercase", "Gemma-2-2B",   "gemma/lowercase",       4),
    ("bold",      "Gemma-2-2B",   "gemma/has_bold_only",   4),
    ("spanish",   "Llama-3.1-8B", "llama31/spanish",       3),
    ("french",    "Llama-3.1-8B", "llama31/french",        4),
    ("lowercase", "Llama-3.1-8B", "llama31/lowercase",     2),
    ("bold",      "Llama-3.1-8B", "llama31/has_bold_only", 4),
]

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_REF = REF


def _norm_ratio_aggregate(combo_dir: Path) -> Aggregate:
    """Per-seed ``||v_poisoned|| / ||v_clean||`` (both from that seed's .pt)."""
    import torch

    def _extract(seed_dir: Path) -> float:
        pt = seed_dir / "steering_vector.pt"
        if not pt.is_file():
            raise FileNotFoundError(f"Missing {pt}")
        d = torch.load(pt, map_location="cpu", weights_only=False)
        nc = float(d["steering_vector_clean"].float().norm().item())
        np_ = float(d["steering_vector_poisoned"].float().norm().item())
        return np_ / nc

    return aggregate_from_seeds(combo_dir, _extract)


def load_row(label: str, model: str, dirname: str, weight: int) -> dict:
    combo_dir = RESULTS_ROOT / dirname
    if not combo_dir.is_dir():
        raise FileNotFoundError(f"missing experiment directory: {combo_dir}")

    norm_clean = aggregate_from_seeds(
        combo_dir, vector_norm_extractor("steering_vector_clean"),
    )
    norm_poisoned = aggregate_from_seeds(
        combo_dir, vector_norm_extractor("steering_vector_poisoned"),
    )
    ratio = _norm_ratio_aggregate(combo_dir)
    ppl_clean = aggregate_from_seeds(
        combo_dir,
        metric_extractor(f"results_clean_harmless_w{weight}", "mean_perplexity"),
        drop_nan=True,
    )
    ppl_poisoned = aggregate_from_seeds(
        combo_dir,
        metric_extractor(f"results_poisoned_harmless_w{weight}", "mean_perplexity"),
        drop_nan=True,
    )
    asr_clean = aggregate_from_seeds(
        combo_dir,
        metric_extractor(f"results_clean_harmful_w{weight}", "judge_success_rate"),
    )
    asr_poisoned = aggregate_from_seeds(
        combo_dir,
        metric_extractor(f"results_poisoned_harmful_w{weight}", "judge_success_rate"),
    )
    return {
        "label": label,
        "model": model,
        "dirname": dirname,
        "weight": weight,
        "norm_clean": norm_clean,
        "norm_poisoned": norm_poisoned,
        "ratio": ratio,
        "ppl_clean": ppl_clean,
        "ppl_poisoned": ppl_poisoned,
        "asr_clean": asr_clean,
        "asr_poisoned": asr_poisoned,
        "delta_asr": asr_poisoned.mean - asr_clean.mean,
    }


def _draw_ratio_panel(
    ax: plt.Axes,
    xs: np.ndarray,
    ratio_means: np.ndarray,
    ratio_stds: np.ndarray,
    labels: list[str],
    group_spans: list[tuple[str, float, float]],
) -> None:
    ax.axhline(
        1.0, color=COLOR_REF, linestyle="--", linewidth=1.2,
        alpha=0.5, zorder=1,
    )
    ax.errorbar(
        xs, ratio_means, yerr=ratio_stds,
        fmt="none", ecolor=COLOR_POISONED,
        elinewidth=1.4, capsize=4, capthick=1.2,
        alpha=0.85, zorder=2,
    )
    ax.scatter(
        xs, ratio_means, s=42, color=COLOR_POISONED,
        edgecolor="white", linewidths=1.0, zorder=3,
    )
    max_idx = int(np.argmax(ratio_means))
    ax.annotate(
        rf"${ratio_means[max_idx]:.2f}\times$",
        xy=(xs[max_idx], ratio_means[max_idx]),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=12,
        color=COLOR_POISONED,
    )

    band_lo = float(np.min(ratio_means - ratio_stds))
    band_hi = float(np.max(ratio_means + ratio_stds))
    y_lo = min(0.85, band_lo - 0.05)
    y_hi = max(1.30, band_hi + 0.10)
    ax.set_ylim(y_lo, y_hi)

    side_pad = 0.5 * INTRA_GAP
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(xs[0] - side_pad, xs[-1] + side_pad)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(
        r"$\|\mathbf{v}_{\mathrm{poisoned}}\| \,/\, \|\mathbf{v}_{\mathrm{clean}}\|$"
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    draw_model_subrow(ax, group_spans)


def _draw_ppl_panel(
    ax: plt.Axes,
    xs: np.ndarray,
    ppl_clean_means: np.ndarray,
    ppl_clean_stds: np.ndarray,
    ppl_poisoned_means: np.ndarray,
    ppl_poisoned_stds: np.ndarray,
    labels: list[str],
    group_spans: list[tuple[str, float, float]],
) -> None:
    for xi, pc, pp in zip(xs, ppl_clean_means, ppl_poisoned_means):
        ax.plot(
            [xi, xi], [pc, pp],
            color=LINE,
            linewidth=1.6, alpha=0.75, zorder=1,
        )
    for xi, m, s in zip(xs, ppl_clean_means, ppl_clean_stds):
        if s > 1e-9:
            ax.errorbar(xi, m, yerr=s, fmt="none", ecolor=COLOR_CLEAN,
                        elinewidth=1.2, capsize=4, capthick=1.0,
                        alpha=0.8, zorder=2)
    for xi, m, s in zip(xs, ppl_poisoned_means, ppl_poisoned_stds):
        if s > 1e-9:
            ax.errorbar(xi, m, yerr=s, fmt="none", ecolor=COLOR_POISONED,
                        elinewidth=1.2, capsize=4, capthick=1.0,
                        alpha=0.8, zorder=2)

    ax.scatter(xs, ppl_clean_means, s=40, color=COLOR_CLEAN,
               edgecolor="white", linewidths=1.0, label="clean", zorder=3)
    ax.scatter(xs, ppl_poisoned_means, s=40, color=COLOR_POISONED,
               edgecolor="white", linewidths=1.0, label="poisoned", zorder=3)

    all_ppl = np.concatenate([ppl_clean_means, ppl_poisoned_means])
    span_ratio = float(all_ppl.max() / max(all_ppl.min(), 1e-9))
    if span_ratio > 5.0:
        ax.set_yscale("log")

    side_pad = 0.5 * INTRA_GAP
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(xs[0] - side_pad, xs[-1] + side_pad)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(r"Response perplexity (harmless)")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    draw_model_subrow(ax, group_spans)


def _save_panel(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    rows = [load_row(label, model, dirname, w) for label, model, dirname, w in COMBOS]

    rows, x_positions, group_spans = group_by_model_layout(
        rows, MODEL_ORDER, intra_gap=INTRA_GAP, group_gap=GROUP_GAP,
    )

    print("Final combo ordering (grouped by model, ΔASR desc within group):")
    for r in rows:
        print(
            f"  [{r['model']:<13}] {r['dirname']:<28} w={r['weight']}  "
            f"ΔASR={r['delta_asr']:+.3f}  "
            f"norm {r['norm_clean'].mean:.3f} -> {fmt_mean_std(r['norm_poisoned'])} "
            f"(ratio {fmt_mean_std(r['ratio'])})  "
            f"ppl {fmt_mean_std(r['ppl_clean'], '{:.2f}')} -> "
            f"{fmt_mean_std(r['ppl_poisoned'], '{:.2f}')}"
        )

    labels = [r["label"] for r in rows]
    ratio_means = np.array([r["ratio"].mean for r in rows], dtype=float)
    ratio_stds = np.array([r["ratio"].std for r in rows], dtype=float)
    ppl_clean_means = np.array([r["ppl_clean"].mean for r in rows], dtype=float)
    ppl_clean_stds = np.array([r["ppl_clean"].std for r in rows], dtype=float)
    ppl_poisoned_means = np.array([r["ppl_poisoned"].mean for r in rows], dtype=float)
    ppl_poisoned_stds = np.array([r["ppl_poisoned"].std for r in rows], dtype=float)

    xs = np.array(x_positions, dtype=float)

    fig_ratio, ax_ratio = plt.subplots(figsize=(8.2, 4.8))
    _draw_ratio_panel(
        ax_ratio, xs, ratio_means, ratio_stds, labels, group_spans,
    )
    fig_ratio.subplots_adjust(bottom=0.22)
    ratio_paths = _save_panel(fig_ratio, "fig_norm_sanity_ratio")

    fig_ppl, ax_ppl = plt.subplots(figsize=(8.2, 4.8))
    _draw_ppl_panel(
        ax_ppl, xs,
        ppl_clean_means, ppl_clean_stds,
        ppl_poisoned_means, ppl_poisoned_stds,
        labels, group_spans,
    )
    fig_ppl.subplots_adjust(bottom=0.22)
    ppl_paths = _save_panel(fig_ppl, "fig_norm_sanity_ppl")

    print("\nWrote:")
    for p in ratio_paths + ppl_paths:
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
