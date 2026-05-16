"""Scatter Delta cos vs Delta ASR per model-attribute combo.

Optional figure: shows that Delta cos (rotation toward -r) broadly tracks Delta
ASR (jailbreak lift) but is not its sole determinant. Reads each combo's
summary.json (for delta_cos) and the corresponding results files (for ASRs).
No hard-coded numbers.

Run: .venv/bin/python scripts/plots/plot_delta_cos_vs_delta_asr.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _donstyle import apply_style, POISONED, PALETTE

apply_style()


PROJECT_ROOT = Path("/media/donato/Extra-storage/Code/mech-interp/adversarial_attack")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

# (display label, experiment subdirectory)
COMBOS: list[tuple[str, str]] = [
    ("Gemma spanish", "gemma_spanish_L14_w3"),
    ("Gemma french", "gemma_french_L14_w3"),
    ("Llama lowercase", "llama31_lowercase_L18_w2"),
    ("Llama spanish", "llama31_spanish_L18_w3"),
    (r"Gemma has\_bold\_only", "gemma_has_bold_only_L14_w4"),
]

ACCENT = POISONED
REF_GRAY = PALETTE["Rich black"]
TREND_GRAY = PALETTE["Dark cyan"]


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open() as f:
        return json.load(f)


def _judge_success_rate(results_dir: Path) -> float:
    results_path = results_dir / "results"
    data = _read_json(results_path)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"Expected non-empty list in {results_path}, got: {type(data).__name__}"
        )
    entry = data[0]
    if not isinstance(entry, dict) or "judge_success_rate" not in entry:
        raise KeyError(
            f"Missing 'judge_success_rate' in {results_path}"
        )
    return float(entry["judge_success_rate"])


def load_combo(label: str, subdir: str) -> dict:
    combo_dir = EXPERIMENTS_DIR / subdir
    summary = _read_json(combo_dir / "summary.json")
    if not isinstance(summary, dict) or "delta_cos" not in summary:
        raise KeyError(
            f"Missing 'delta_cos' in {combo_dir / 'summary.json'}"
        )
    asr_clean = _judge_success_rate(combo_dir / "results_clean_harmful")
    asr_poisoned = _judge_success_rate(combo_dir / "results_poisoned_harmful")
    return {
        "label": label,
        "subdir": subdir,
        "delta_cos": float(summary["delta_cos"]),
        "asr_clean": asr_clean,
        "asr_poisoned": asr_poisoned,
        "delta_asr": asr_poisoned - asr_clean,
    }


def main() -> None:
    rows = [load_combo(label, subdir) for label, subdir in COMBOS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Report.
    print(f"{'combo':<22} {'delta_cos':>10} {'asr_clean':>10} {'asr_pois':>9} {'delta_asr':>10}")
    for r in rows:
        print(
            f"{r['label']:<22} {r['delta_cos']:>10.4f} {r['asr_clean']:>10.4f} "
            f"{r['asr_poisoned']:>9.4f} {r['delta_asr']:>10.4f}"
        )

    xs = np.array([r["delta_cos"] for r in rows], dtype=float)
    ys = np.array([r["delta_asr"] for r in rows], dtype=float)

    # Least-squares trend line (degree 1).
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"trend line slope = {slope:.4f}, intercept = {intercept:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Determine axis limits with margin around data and 0.
    def _padded_range(vals: np.ndarray, frac: float = 0.18) -> tuple[float, float]:
        lo = float(min(vals.min(), 0.0))
        hi = float(max(vals.max(), 0.0))
        span = hi - lo if hi > lo else 1.0
        pad = frac * span
        return lo - pad, hi + pad

    x_lo, x_hi = _padded_range(xs)
    y_lo, y_hi = _padded_range(ys)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Reference lines at 0.
    ax.axhline(0.0, color=REF_GRAY, linestyle=":", linewidth=1.0,
               alpha=0.45, zorder=0)
    ax.axvline(0.0, color=REF_GRAY, linestyle=":", linewidth=1.0,
               alpha=0.45, zorder=0)

    # Light trend line spanning x range.
    x_line = np.linspace(x_lo, x_hi, 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line,
        color=TREND_GRAY, linestyle="--",
        linewidth=1.6, alpha=0.55, zorder=1,
    )

    # Scatter points.
    ax.scatter(
        xs, ys,
        s=110, color=ACCENT,
        edgecolors="white", linewidths=1.0,
        zorder=3,
    )

    # Annotate each point. Use a small offset so the text doesn't sit on the dot.
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    dx = 0.022 * x_span
    dy = 0.022 * y_span
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(
            r["label"],
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=13,
            color=PALETTE["Rich black"],
            ha="left", va="bottom",
            zorder=4,
        )

    ax.set_xlabel(r"$\Delta\cos(\mathbf{v},\ -\mathbf{r})$")
    ax.set_ylabel(r"$\Delta$ASR")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "fig_delta_cos_vs_delta_asr.pdf"
    png_path = OUTPUT_DIR / "fig_delta_cos_vs_delta_asr.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    for p in (pdf_path, png_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"Failed to write non-empty file: {p}")
        print(f"wrote {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
