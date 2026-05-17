"""Sanity-check figure for the steering-vector poisoning paper (Table 3).

Two panels:
  A) poisoned/clean steering vector norm ratio (per combo).
  B) Response perplexity on harmless prompts, clean vs poisoned (paired).

Combos are ordered by ΔASR descending (consistent with the main results figure),
where ASR is read from ``results_*_harmful/results`` (judge_success_rate).

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
from _donstyle import apply_style, CLEAN, POISONED, PALETTE

apply_style()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

# (display label, experiment directory name) pairs.
COMBOS: list[tuple[str, str]] = [
    ("Gemma-2-2B\nspanish", "gemma/spanish"),
    ("Gemma-2-2B\nfrench", "gemma/french"),
    ("Llama-3.1-8B\nlowercase", "llama31/lowercase"),
    ("Llama-3.1-8B\nspanish", "llama31/spanish"),
    (r"Gemma-2-2B" + "\n" + r"has\_bold\_only", "gemma/has_bold_only"),
]

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_REF = PALETTE["Rich black"]


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
    for label, dirname in COMBOS:
        combo_dir = RESULTS_ROOT / dirname
        if not combo_dir.is_dir():
            raise FileNotFoundError(f"missing experiment directory: {combo_dir}")
        norm_clean, norm_poisoned = load_norms(combo_dir)
        ppl_clean, ppl_poisoned = load_response_ppl(combo_dir)
        asr_clean, asr_poisoned = load_asr(combo_dir)
        rows.append(
            {
                "label": label,
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

    # Order by ΔASR descending (consistent with the main results figure).
    rows.sort(key=lambda r: r["delta_asr"], reverse=True)

    print("Final combo ordering (by ΔASR desc):")
    for r in rows:
        print(
            f"  {r['dirname']:<32}  ΔASR={r['delta_asr']:+.3f}  "
            f"norm {r['norm_clean']:.3f} -> {r['norm_poisoned']:.3f} "
            f"(x{r['ratio']:.3f})  ppl {r['ppl_clean']:.2f} -> {r['ppl_poisoned']:.2f}"
        )

    labels = [r["label"] for r in rows]
    ratios = np.array([r["ratio"] for r in rows], dtype=float)
    ppl_clean = np.array([r["ppl_clean"] for r in rows], dtype=float)
    ppl_poisoned = np.array([r["ppl_poisoned"] for r in rows], dtype=float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    x = np.arange(len(rows))

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
    axA.set_xlim(-0.5, len(rows) - 0.5)
    axA.set_ylabel(r"$\|\mathbf{v}_{\mathrm{poisoned}}\| \,/\, \|\mathbf{v}_{\mathrm{clean}}\|$")
    axA.set_title(r"A. Steering-vector norm ratio", fontsize=14, loc="left", pad=8)
    axA.text(
        0.02, 0.97, r"no inflation",
        transform=axA.transAxes, fontsize=11,
        color=COLOR_REF, alpha=0.6, va="top",
    )
    axA.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

    # ---- Panel B: response perplexity ----------------------------------------
    for xi, pc, pp in zip(x, ppl_clean, ppl_poisoned):
        axB.plot(
            [xi, xi], [pc, pp],
            color=PALETTE["Tiffany Blue"],
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
    axB.set_xlim(-0.5, len(rows) - 0.5)
    axB.set_ylabel(r"Response perplexity (harmless)")
    axB.set_title(r"B. Benign-response perplexity", fontsize=14, loc="left", pad=8)
    axB.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    axB.legend(loc="best", frameon=False, fontsize=12, handletextpad=0.5)

    fig.tight_layout()

    pdf_path = FIG_DIR / "fig_norm_sanity.pdf"
    png_path = FIG_DIR / "fig_norm_sanity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"\nWrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
