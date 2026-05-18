"""Sanity-check figure for the steering-vector poisoning paper (Table 3).

Two panels:
  A) poisoned/clean steering vector norm ratio (per combo).
  B) Response perplexity on harmless prompts, clean vs poisoned (paired).

Combos are grouped by model (Gemma block, then Llama block, with a visual
gap between groups) and sorted by ΔASR descending within each group, matching
the layout of :func:`plot_main_results`. ASR is read from
``results_*_harmful/results`` (judge_success_rate).

Run from the project root::

    .venv/bin/python scripts/plots/plot_norm_sanity.py

Outputs ``paper/figures/fig_norm_sanity.{pdf,png}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

# Model order for grouping on the x-axis (left -> right) and horizontal
# spacing, matching plot_main_results.py.
MODEL_ORDER: list[str] = ["Gemma-2-2B", "Llama-3.1-8B"]
INTRA_GAP: float = 1.6
GROUP_GAP: float = 1.4

# (attribute-only label, model display name, experiment directory).
COMBOS: list[tuple[str, str, str]] = [
    ("spanish",   "Gemma-2-2B",   "gemma/spanish"),
    ("french",    "Gemma-2-2B",   "gemma/french"),
    ("lowercase", "Gemma-2-2B",   "gemma/lowercase"),
    ("bold",      "Gemma-2-2B",   "gemma/has_bold_only"),
    ("spanish",   "Llama-3.1-8B", "llama31/spanish"),
    ("french",    "Llama-3.1-8B", "llama31/french"),
    ("lowercase", "Llama-3.1-8B", "llama31/lowercase"),
    ("bold",      "Llama-3.1-8B", "llama31/has_bold_only"),
]

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_REF = REF


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    return path


def _load_results_json(results_path: Path) -> dict:
    """Load the single-element list at ``results`` and return its first entry."""
    _require(results_path)
    with results_path.open() as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"unexpected results structure in {results_path}: {data!r}")
    return data[0]


def load_norms(combo_dir: Path) -> tuple[float, float]:
    sv_path = _require(combo_dir / "steering_vector.pt")
    d = torch.load(sv_path, map_location="cpu", weights_only=False)
    for key in ("steering_vector_clean", "steering_vector_poisoned"):
        if key not in d:
            raise KeyError(f"{sv_path} missing key {key!r}; keys={list(d)}")
    norm_clean = d["steering_vector_clean"].float().norm().item()
    norm_poisoned = d["steering_vector_poisoned"].float().norm().item()
    return norm_clean, norm_poisoned


def load_response_ppl(combo_dir: Path) -> tuple[float, float]:
    clean = _load_results_json(combo_dir / "results_clean_harmless" / "results")
    poisoned = _load_results_json(combo_dir / "results_poisoned_harmless" / "results")
    for tag, entry in (("clean", clean), ("poisoned", poisoned)):
        if "mean_perplexity" not in entry:
            raise KeyError(
                f"{combo_dir}/results_{tag}_harmless/results missing key 'mean_perplexity'"
            )
    return float(clean["mean_perplexity"]), float(poisoned["mean_perplexity"])


def load_asr(combo_dir: Path) -> tuple[float, float]:
    clean = _load_results_json(combo_dir / "results_clean_harmful" / "results")
    poisoned = _load_results_json(combo_dir / "results_poisoned_harmful" / "results")
    for tag, entry in (("clean", clean), ("poisoned", poisoned)):
        if "judge_success_rate" not in entry:
            raise KeyError(
                f"{combo_dir}/results_{tag}_harmful/results missing key 'judge_success_rate'"
            )
        if entry["judge_success_rate"] is None:
            raise ValueError(
                f"{combo_dir}/results_{tag}_harmful/results has judge_success_rate=None"
            )
    return float(clean["judge_success_rate"]), float(poisoned["judge_success_rate"])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, model, dirname in COMBOS:
        combo_dir = RESULTS_ROOT / dirname
        if not combo_dir.is_dir():
            raise FileNotFoundError(f"missing experiment directory: {combo_dir}")
        norm_clean, norm_poisoned = load_norms(combo_dir)
        ppl_clean, ppl_poisoned = load_response_ppl(combo_dir)
        asr_clean, asr_poisoned = load_asr(combo_dir)
        rows.append(
            {
                "label": label,
                "model": model,
                "dirname": dirname,
                "norm_clean": norm_clean,
                "norm_poisoned": norm_poisoned,
                "ratio": norm_poisoned / norm_clean,
                "ppl_clean": ppl_clean,
                "ppl_poisoned": ppl_poisoned,
                "asr_clean": asr_clean,
                "asr_poisoned": asr_poisoned,
                "delta_asr": asr_poisoned - asr_clean,
            }
        )

    # Group by model (with ΔASR-desc inside each group), matching
    # plot_main_results.py.
    rows, x_positions, group_spans = group_by_model_layout(
        rows, MODEL_ORDER, intra_gap=INTRA_GAP, group_gap=GROUP_GAP,
    )

    print("Final combo ordering (grouped by model, ΔASR desc within group):")
    for r in rows:
        print(
            f"  [{r['model']:<13}] {r['dirname']:<32}  ΔASR={r['delta_asr']:+.3f}  "
            f"norm {r['norm_clean']:.3f} -> {r['norm_poisoned']:.3f} "
            f"(x{r['ratio']:.3f})  ppl {r['ppl_clean']:.2f} -> {r['ppl_poisoned']:.2f}"
        )

    labels = [r["label"] for r in rows]
    ratios = np.array([r["ratio"] for r in rows], dtype=float)
    ppl_clean = np.array([r["ppl_clean"] for r in rows], dtype=float)
    ppl_poisoned = np.array([r["ppl_poisoned"] for r in rows], dtype=float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.0))

    x = np.array(x_positions, dtype=float)
    side_pad = 0.5 * INTRA_GAP

    # ---- Panel A: norm ratio --------------------------------------------------
    axA.axhline(1.0, color=COLOR_REF, linestyle="--", linewidth=1.2,
                alpha=0.5, zorder=1)
    axA.scatter(
        x, ratios, s=85, color=COLOR_POISONED,
        edgecolor="white", linewidths=1.0, zorder=3,
    )
    # Annotate the largest ratio.
    max_idx = int(np.argmax(ratios))
    axA.annotate(
        rf"${ratios[max_idx]:.2f}\times$",
        xy=(x[max_idx], ratios[max_idx]),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=12,
        color=COLOR_POISONED,
    )

    y_lo = min(0.85, float(ratios.min()) - 0.05)
    y_hi = max(1.30, float(ratios.max()) + 0.10)
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
    for xi, pc, pp in zip(x, ppl_clean, ppl_poisoned):
        axB.plot(
            [xi, xi], [pc, pp],
            color=LINE,
            linewidth=1.6, alpha=0.75, zorder=1,
        )
    axB.scatter(x, ppl_clean, s=80, color=COLOR_CLEAN,
                edgecolor="white", linewidths=1.0, label="clean", zorder=3)
    axB.scatter(x, ppl_poisoned, s=80, color=COLOR_POISONED,
                edgecolor="white", linewidths=1.0, label="poisoned", zorder=3)

    all_ppl = np.concatenate([ppl_clean, ppl_poisoned])
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
    # reserve space at the bottom for the model sub-row labels
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
