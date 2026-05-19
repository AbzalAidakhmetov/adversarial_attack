"""Sanity-check figure for the steering-vector poisoning paper (Table 3).

Two panels:
  A) poisoned/clean steering vector norm ratio (per combo, mean over seeds).
  B) Response perplexity on harmless prompts, clean vs poisoned (paired,
     mean over seeds with ±1 std error bars on the poisoned side).

Combos are grouped by model (Gemma block, then Llama block, with a visual
gap between groups) and sorted by ΔASR descending within each group, matching
the layout of :func:`plot_main_results`. Per-cell results are read across
``results/<model>/<attr>/seed{0,1,2}/results_*_w<W>/...``.

Run from the project root::

    .venv/bin/python scripts/plots/plot_norm_sanity.py

Outputs ``paper/figures/fig_norm_sanity.{pdf,png}``.
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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

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

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.0))

    x = np.array(x_positions, dtype=float)
    side_pad = 0.5 * INTRA_GAP

    # ---- Panel A: norm ratio --------------------------------------------------
    axA.axhline(1.0, color=COLOR_REF, linestyle="--", linewidth=1.2,
                alpha=0.5, zorder=1)
    axA.errorbar(
        x, ratio_means, yerr=ratio_stds,
        fmt="none", ecolor=COLOR_POISONED,
        elinewidth=1.4, capsize=4, capthick=1.2,
        alpha=0.85, zorder=2,
    )
    axA.scatter(
        x, ratio_means, s=42, color=COLOR_POISONED,
        edgecolor="white", linewidths=1.0, zorder=3,
    )
    max_idx = int(np.argmax(ratio_means))
    axA.annotate(
        rf"${ratio_means[max_idx]:.2f}\times$",
        xy=(x[max_idx], ratio_means[max_idx]),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=12,
        color=COLOR_POISONED,
    )

    band_lo = float(np.min(ratio_means - ratio_stds))
    band_hi = float(np.max(ratio_means + ratio_stds))
    y_lo = min(0.85, band_lo - 0.05)
    y_hi = max(1.30, band_hi + 0.10)
    axA.set_ylim(y_lo, y_hi)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=12)
    axA.tick_params(axis="y", labelsize=12)
    axA.set_xlim(x[0] - side_pad, x[-1] + side_pad)
    axA.set_ylabel(r"$\|\mathbf{v}_{\mathrm{poisoned}}\| \,/\, \|\mathbf{v}_{\mathrm{clean}}\|$")
    axA.set_title(r"A. Steering-vector norm ratio", fontsize=14, loc="left", pad=8)
    axA.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    draw_model_subrow(axA, group_spans)

    # ---- Panel B: response perplexity ----------------------------------------
    for xi, pc, pp in zip(x, ppl_clean_means, ppl_poisoned_means):
        axB.plot(
            [xi, xi], [pc, pp],
            color=LINE,
            linewidth=1.6, alpha=0.75, zorder=1,
        )
    # error bars (clean usually std=0, poisoned can vary)
    for xi, m, s in zip(x, ppl_clean_means, ppl_clean_stds):
        if s > 1e-9:
            axB.errorbar(xi, m, yerr=s, fmt="none", ecolor=COLOR_CLEAN,
                         elinewidth=1.2, capsize=4, capthick=1.0,
                         alpha=0.8, zorder=2)
    for xi, m, s in zip(x, ppl_poisoned_means, ppl_poisoned_stds):
        if s > 1e-9:
            axB.errorbar(xi, m, yerr=s, fmt="none", ecolor=COLOR_POISONED,
                         elinewidth=1.2, capsize=4, capthick=1.0,
                         alpha=0.8, zorder=2)

    axB.scatter(x, ppl_clean_means, s=40, color=COLOR_CLEAN,
                edgecolor="white", linewidths=1.0, label="clean", zorder=3)
    axB.scatter(x, ppl_poisoned_means, s=40, color=COLOR_POISONED,
                edgecolor="white", linewidths=1.0, label="poisoned", zorder=3)

    all_ppl = np.concatenate([ppl_clean_means, ppl_poisoned_means])
    span_ratio = float(all_ppl.max() / max(all_ppl.min(), 1e-9))
    if span_ratio > 5.0:
        axB.set_yscale("log")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=12)
    axB.tick_params(axis="y", labelsize=12)
    axB.set_xlim(x[0] - side_pad, x[-1] + side_pad)
    axB.set_ylabel(r"Response perplexity (harmless)")
    axB.set_title(r"B. Benign-response perplexity", fontsize=14, loc="left", pad=8)
    axB.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    axB.legend(loc="best", frameon=False, fontsize=12, handletextpad=0.5)
    draw_model_subrow(axB, group_spans)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    pdf_path = FIG_DIR / "fig_norm_sanity.pdf"
    png_path = FIG_DIR / "fig_norm_sanity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"\nWrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
