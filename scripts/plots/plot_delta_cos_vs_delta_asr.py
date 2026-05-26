"""Scatter Delta cos vs Delta ASR per model-attribute combo.

Optional figure: shows that Delta cos (rotation toward r) broadly tracks Delta
ASR (jailbreak lift) but is not its sole determinant. Each point is a per-combo
mean over three GCG seeds; horizontal and vertical bars show ±1 std across
seeds. Clean ``cos`` and clean ``ASR`` are seed-deterministic, so dispersion
comes entirely from the poisoned side.

Run: .venv/bin/python scripts/plots/plot_delta_cos_vs_delta_asr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advsteer import PROJECT_ROOT
from _donstyle import apply_style, POISONED, REF, LINE
from _seeds import (
    aggregate_from_seeds,
    fmt_mean_std,
    metric_extractor,
    summary_extractor,
)

apply_style()


RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

# (display label, experiment subdirectory, bundled weight, optional (dx, dy)
# label offset in axis-fraction units; if None we use the default offset).
COMBOS: list[tuple[str, str, int, tuple[float, float] | None]] = [
    ("Gemma spanish",    "gemma/spanish",         3, (0.015, 0.025)),
    ("Gemma french",     "gemma/french",          3, (0.015, 0.025)),
    ("Gemma lowercase",  "gemma/lowercase",       4, (0.015, -0.045)),
    ("Gemma bold",       "gemma/has_bold_only",   4, (0.015, -0.045)),
    ("Llama spanish",    "llama31/spanish",       3, (-0.015, -0.045)),
    ("Llama french",     "llama31/french",        4, (-0.015, 0.025)),
    ("Llama lowercase",  "llama31/lowercase",     2, (0.015, -0.045)),
    ("Llama bold",       "llama31/has_bold_only", 4, (-0.015, 0.025)),
]

ACCENT = POISONED
REF_GRAY = REF
TREND_GRAY = LINE


def load_combo(label: str, subdir: str, weight: int,
               offset: tuple[float, float] | None = None) -> dict:
    combo_dir = RESULTS_DIR / subdir
    # delta_cos already computed per-seed in summary.json
    delta_cos = aggregate_from_seeds(combo_dir, summary_extractor("delta_cos"))
    # delta_asr: per-seed (poisoned - clean) per seed
    cos_p = aggregate_from_seeds(combo_dir, summary_extractor("cos_poisoned"))
    cos_c = aggregate_from_seeds(combo_dir, summary_extractor("cos_clean"))
    asr_p = aggregate_from_seeds(
        combo_dir, metric_extractor(f"results_poisoned_harmful_w{weight}", "judge_success_rate"),
    )
    asr_c = aggregate_from_seeds(
        combo_dir, metric_extractor(f"results_clean_harmful_w{weight}", "judge_success_rate"),
    )
    # Per-seed delta_asr (each seed's poisoned minus its own clean run).
    # Use the same seeds order for both aggregates.
    per_seed_delta_asr = tuple(
        p - c for p, c in zip(asr_p.values, asr_c.values)
    )
    from _seeds import Aggregate
    delta_asr = Aggregate(per_seed_delta_asr)

    return {
        "label": label,
        "subdir": subdir,
        "delta_cos": delta_cos,
        "delta_asr": delta_asr,
        "asr_clean": asr_c,
        "asr_poisoned": asr_p,
        "cos_clean": cos_c,
        "cos_poisoned": cos_p,
        "offset": offset,
    }


def main() -> None:
    rows = [load_combo(label, subdir, w, off) for label, subdir, w, off in COMBOS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"{'combo':<22} "
        f"{'delta_cos':>16} {'asr_clean':>16} {'asr_pois':>16} {'delta_asr':>16}"
    )
    for r in rows:
        print(
            f"{r['label']:<22} "
            f"{fmt_mean_std(r['delta_cos']):>16} "
            f"{fmt_mean_std(r['asr_clean']):>16} "
            f"{fmt_mean_std(r['asr_poisoned']):>16} "
            f"{fmt_mean_std(r['delta_asr']):>16}"
        )

    xs = np.array([r["delta_cos"].mean for r in rows], dtype=float)
    xerr = np.array([r["delta_cos"].std for r in rows], dtype=float)
    ys = np.array([r["delta_asr"].mean for r in rows], dtype=float)
    yerr = np.array([r["delta_asr"].std for r in rows], dtype=float)

    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"trend line slope = {slope:.4f}, intercept = {intercept:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    def _padded_range(vals: np.ndarray, frac: float = 0.18,
                      extra: np.ndarray | None = None) -> tuple[float, float]:
        lo = float(min(vals.min(), 0.0))
        hi = float(max(vals.max(), 0.0))
        if extra is not None and extra.size:
            lo = min(lo, float(extra.min()))
            hi = max(hi, float(extra.max()))
        span = hi - lo if hi > lo else 1.0
        pad = frac * span
        return lo - pad, hi + pad

    x_lo, x_hi = _padded_range(xs, extra=np.concatenate([xs - xerr, xs + xerr]))
    y_lo, y_hi = _padded_range(ys, extra=np.concatenate([ys - yerr, ys + yerr]))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.axhline(0.0, color=REF_GRAY, linestyle=":", linewidth=1.0,
               alpha=0.45, zorder=0)
    ax.axvline(0.0, color=REF_GRAY, linestyle=":", linewidth=1.0,
               alpha=0.45, zorder=0)

    x_line = np.linspace(x_lo, x_hi, 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line,
        color=TREND_GRAY, linestyle="--",
        linewidth=1.6, alpha=0.55, zorder=1,
    )

    # 2-D error bars (mean ± std on both axes).
    ax.errorbar(
        xs, ys, xerr=xerr, yerr=yerr,
        fmt="none", ecolor=ACCENT,
        elinewidth=1.2, capsize=3, capthick=1.0,
        alpha=0.75, zorder=2,
    )
    ax.scatter(
        xs, ys,
        s=55, color=ACCENT,
        edgecolors="white", linewidths=1.0,
        zorder=3,
    )

    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    default_dx = 0.022 * x_span
    default_dy = 0.022 * y_span
    for r, x, y in zip(rows, xs, ys):
        off = r.get("offset")
        if off is None:
            tx, ty = x + default_dx, y + default_dy
            ha, va = "left", "bottom"
        else:
            dx_frac, dy_frac = off
            tx = x + dx_frac * x_span
            ty = y + dy_frac * y_span
            ha = "left" if dx_frac >= 0 else "right"
            va = "bottom" if dy_frac >= 0 else "top"
        ax.annotate(
            r["label"],
            xy=(x, y),
            xytext=(tx, ty),
            fontsize=13,
            color=REF,
            ha=ha, va=va,
            zorder=4,
        )

    ax.set_xlabel(r"$\Delta\cos(\mathbf{v},\ \mathbf{r})$")
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
