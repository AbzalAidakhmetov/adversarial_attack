"""Plot the refusal-direction orthogonalisation defence.

Replaces Table 5 of the paper. For each combo we plot three states (clean,
poisoned, poisoned+defence) for both ASR (panel A) and harmless hAttr
(panel B). Numbers are read from per-seed result files under
``results/<model>/<attr>/seed{0,1,2}/results_*_w<W>/results``. Clean and
poisoned states are averaged over three GCG seeds with ±1 std error bars;
the orthogonalisation defence has so far been evaluated only for seed 0, so
the defended dot is shown without a per-seed error bar (single seed).

Run from project root:
    .venv/bin/python scripts/plots/plot_defence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advsteer import PROJECT_ROOT
from _donstyle import apply_style, CLEAN, POISONED, DEFENDED, LINE, ACCENT
from _seeds import (
    Aggregate,
    aggregate_from_seeds,
    fmt_mean_std,
    metric_extractor,
)

apply_style()


RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

# (display label, experiment subdirectory, bundled weight)
COMBOS: list[tuple[str, str, int]] = [
    ("Gemma-2B\nspanish",    "gemma/spanish",         3),
    ("Gemma-2B\nfrench",     "gemma/french",          3),
    ("Gemma-2B\nlowercase",  "gemma/lowercase",       4),
    ("Gemma-2B\nbold",       "gemma/has_bold_only",   4),
    ("Llama-8B\nspanish",    "llama31/spanish",       3),
    ("Llama-8B\nfrench",     "llama31/french",        4),
    ("Llama-8B\nlowercase",  "llama31/lowercase",     2),
    ("Llama-8B\nbold",       "llama31/has_bold_only", 4),
]

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_DEFENDED = DEFENDED
COLOR_LINE = LINE


def _aggregate_metric(combo_dir: Path, subdir: str, key: str, *, seeds=(0, 1, 2)) -> Aggregate:
    return aggregate_from_seeds(
        combo_dir, metric_extractor(subdir, key), seeds=seeds,
    )


def load_combo(label: str, subdir: str, w: int) -> dict:
    combo_dir = RESULTS_DIR / subdir
    if not combo_dir.is_dir():
        raise FileNotFoundError(f"Missing experiment dir: {combo_dir}")

    asr = {
        "clean":    _aggregate_metric(combo_dir, f"results_clean_harmful_w{w}",            "judge_success_rate"),
        "poisoned": _aggregate_metric(combo_dir, f"results_poisoned_harmful_w{w}",         "judge_success_rate"),
        # defense pipeline has only been run for seed 0 so far.
        "defended": _aggregate_metric(combo_dir, f"results_defense_poisoned_harmful_w{w}", "judge_success_rate", seeds=(0,)),
    }
    hattr = {
        "clean":    _aggregate_metric(combo_dir, f"results_clean_harmless_w{w}",            "steering_success_rate"),
        "poisoned": _aggregate_metric(combo_dir, f"results_poisoned_harmless_w{w}",         "steering_success_rate"),
        "defended": _aggregate_metric(combo_dir, f"results_defense_poisoned_harmless_w{w}", "steering_success_rate", seeds=(0,)),
    }
    return {
        "label": label,
        "subdir": subdir,
        "weight": w,
        "asr": asr,
        "hattr": hattr,
        "delta_asr": asr["poisoned"].mean - asr["clean"].mean,
    }


def gap_recovered(asr_clean: float, asr_poisoned: float, asr_defended: float):
    denom = asr_poisoned - asr_clean
    if abs(denom) < 1e-9:
        return None
    return (asr_poisoned - asr_defended) / denom


def _scatter_with_err(ax, x, agg: Aggregate, *, color, marker, label=None, size=40):
    if agg.std > 1e-9:
        ax.errorbar(x, agg.mean, yerr=agg.std, fmt="none",
                    ecolor=color, elinewidth=1.4, capsize=4, capthick=1.2,
                    alpha=0.85, zorder=2)
    ax.scatter([x], [agg.mean], s=size, color=color, marker=marker,
               edgecolors="white", linewidths=1.0, zorder=3, label=label)


def _draw_panel(ax, rows, metric_key: str, ylabel: str,
                annotate_recovered: bool, y_lo_target: float, y_hi_target: float):
    xs = list(range(len(rows)))

    # Connect the three states (mean values) with a thin polyline per combo.
    for x, r in zip(xs, rows):
        c = r[metric_key]["clean"].mean
        p = r[metric_key]["poisoned"].mean
        d = r[metric_key]["defended"].mean
        ax.plot([x, x, x], [c, p, d], color=COLOR_LINE,
                linewidth=1.6, zorder=1, alpha=0.75)

    # Per-state markers and error bars.
    for i, (x, r) in enumerate(zip(xs, rows)):
        first = i == 0
        _scatter_with_err(
            ax, x, r[metric_key]["clean"],
            color=COLOR_CLEAN, marker="o", size=38,
            label="clean" if first else None,
        )
        _scatter_with_err(
            ax, x, r[metric_key]["poisoned"],
            color=COLOR_POISONED, marker="o", size=38,
            label="poisoned" if first else None,
        )
        _scatter_with_err(
            ax, x, r[metric_key]["defended"],
            color=COLOR_DEFENDED, marker="D", size=42,
            label="poisoned + defence" if first else None,
        )

    if annotate_recovered:
        for x, r in zip(xs, rows):
            rec = gap_recovered(
                r["asr"]["clean"].mean,
                r["asr"]["poisoned"].mean,
                r["asr"]["defended"].mean,
            )
            if rec is None:
                continue
            ax.annotate(
                rf"${rec * 100:.0f}\%$",
                xy=(x, r["asr"]["defended"].mean),
                xytext=(9, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=11,
                color=ACCENT,
            )

    all_vals: list[float] = []
    for r in rows:
        for state in ("clean", "poisoned", "defended"):
            agg = r[metric_key][state]
            all_vals.append(agg.mean - agg.std)
            all_vals.append(agg.mean + agg.std)
    span = max(max(all_vals) - min(all_vals), 1e-3)
    pad = 0.10 * span
    y_lo = min(y_lo_target, min(all_vals) - pad)
    y_hi = max(y_hi_target, max(all_vals) + pad)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [r["label"] for r in rows],
        fontsize=11, rotation=30, ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", labelsize=12)
    ax.margins(x=0.12)
    ax.set_xlim(-0.5, len(rows) - 0.5 + 0.35)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)


def main() -> None:
    rows = [load_combo(label, subdir, w) for label, subdir, w in COMBOS]
    rows.sort(key=lambda r: r["delta_asr"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"{'combo':<40} "
        f"{'ASR_c':>14} {'ASR_p':>14} {'ASR_d':>8}  "
        f"{'hA_c':>14} {'hA_p':>14} {'hA_d':>8}  "
        f"{'gap_rec':>8}  {'ΔASR':>6}"
    )
    for r in rows:
        flat = r["label"].replace("\n", " ")
        rec = gap_recovered(
            r["asr"]["clean"].mean,
            r["asr"]["poisoned"].mean,
            r["asr"]["defended"].mean,
        )
        rec_s = "n/a" if rec is None else f"{rec * 100:6.1f}%"
        print(
            f"{flat:<40} "
            f"{fmt_mean_std(r['asr']['clean']):>14} "
            f"{fmt_mean_std(r['asr']['poisoned']):>14} "
            f"{r['asr']['defended'].mean:>8.3f}  "
            f"{fmt_mean_std(r['hattr']['clean']):>14} "
            f"{fmt_mean_std(r['hattr']['poisoned']):>14} "
            f"{r['hattr']['defended'].mean:>8.3f}  "
            f"{rec_s:>8}  {r['delta_asr']:>+6.3f}"
        )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    _draw_panel(
        ax_a, rows, metric_key="asr", ylabel="ASR",
        annotate_recovered=True,
        y_lo_target=0.0, y_hi_target=0.6,
    )
    ax_a.set_title(r"A. Jailbreak success (ASR)", fontsize=14, loc="left", pad=8)

    _draw_panel(
        ax_b, rows, metric_key="hattr", ylabel="hAttr",
        annotate_recovered=False,
        y_lo_target=0.6, y_hi_target=1.0,
    )
    ax_b.set_title(r"B. Harmless attribute compliance (hAttr)",
                   fontsize=14, loc="left", pad=8)

    ax_a.legend(
        loc="upper right",
        frameon=False,
        fontsize=12,
        handletextpad=0.5,
    )

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "fig_defence.pdf"
    png_path = OUTPUT_DIR / "fig_defence.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    for p in (pdf_path, png_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"Failed to write non-empty file: {p}")
        print(f"wrote {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
