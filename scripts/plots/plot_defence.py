"""Plot the refusal-direction orthogonalisation defence.

Replaces Table 5 of the paper. For each combo we plot three states (clean,
poisoned, poisoned+defence) for both ASR (panel A) and harmless hAttr
(panel B). All numbers are read from the per-combo `results` JSON files
written by `eval/evaluate_asr.py` -- no hard-coded values.

Run from project root:
    .venv/bin/python scripts/plots/plot_defence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _donstyle import apply_style, CLEAN, POISONED, DEFENDED, PALETTE, LINE, ACCENT

apply_style()


PROJECT_ROOT = Path("/media/donato/Extra-storage/Code/mech-interp/adversarial_attack")
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

# (display label, experiment subdirectory)
COMBOS: list[tuple[str, str]] = [
    ("Gemma-2B\nspanish",    "gemma/spanish"),
    ("Gemma-2B\nfrench",     "gemma/french"),
    ("Gemma-2B\nlowercase",  "gemma/lowercase"),
    ("Gemma-2B\nbold",       "gemma/has_bold_only"),
    ("Llama-8B\nspanish",    "llama31/spanish"),
    ("Llama-8B\nfrench",     "llama31/french"),
    ("Llama-8B\nlowercase",  "llama31/lowercase"),
    ("Llama-8B\nbold",       "llama31/has_bold_only"),
]

# Sub-directory name templates.
ASR_SUBDIRS = {
    "clean": "results_clean_harmful",
    "poisoned": "results_poisoned_harmful",
    "defended": "results_defense_poisoned_harmful",
}
HATTR_SUBDIRS = {
    "clean": "results_clean_harmless",
    "poisoned": "results_poisoned_harmless",
    "defended": "results_defense_poisoned_harmless",
}

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_DEFENDED = DEFENDED
COLOR_LINE = LINE


def load_metric(combo_dir: Path, subdir: str, key: str) -> float:
    """Load `key` from the single-element results list at combo_dir/subdir/results."""
    path = combo_dir / subdir / "results"
    if not path.is_file():
        raise FileNotFoundError(f"Missing results file: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError(
            f"Expected a single-element list at {path}, got: {type(data).__name__} "
            f"(len={len(data) if hasattr(data, '__len__') else 'n/a'})"
        )
    row = data[0]
    if key not in row:
        raise KeyError(f"{path} is missing required key '{key}'. Available: {sorted(row)}")
    return float(row[key])


def load_combo(label: str, subdir: str) -> dict:
    combo_dir = RESULTS_DIR / subdir
    if not combo_dir.is_dir():
        raise FileNotFoundError(f"Missing experiment dir: {combo_dir}")
    asr = {state: load_metric(combo_dir, sub, "judge_success_rate")
           for state, sub in ASR_SUBDIRS.items()}
    hattr = {state: load_metric(combo_dir, sub, "steering_success_rate")
             for state, sub in HATTR_SUBDIRS.items()}
    return {
        "label": label,
        "subdir": subdir,
        "asr": asr,
        "hattr": hattr,
        "delta_asr": asr["poisoned"] - asr["clean"],
    }


def gap_recovered(asr_clean: float, asr_poisoned: float, asr_defended: float):
    denom = asr_poisoned - asr_clean
    if abs(denom) < 1e-9:
        return None
    return (asr_poisoned - asr_defended) / denom


def _draw_panel(ax, rows, metric_key: str, ylabel: str,
                annotate_recovered: bool, y_lo_target: float, y_hi_target: float):
    xs = list(range(len(rows)))
    clean_vals = [r[metric_key]["clean"] for r in rows]
    poison_vals = [r[metric_key]["poisoned"] for r in rows]
    def_vals = [r[metric_key]["defended"] for r in rows]

    # Connect the three states with a thin polyline per combo.
    for x, c, p, d in zip(xs, clean_vals, poison_vals, def_vals):
        ax.plot([x, x, x], [c, p, d], color=COLOR_LINE,
                linewidth=1.6, zorder=1, alpha=0.75)

    # Markers.
    ax.scatter(xs, clean_vals, s=75, color=COLOR_CLEAN, marker="o",
               edgecolors="white", linewidths=1.0, zorder=3, label="clean")
    ax.scatter(xs, poison_vals, s=75, color=COLOR_POISONED, marker="o",
               edgecolors="white", linewidths=1.0, zorder=3, label="poisoned")
    ax.scatter(xs, def_vals, s=80, color=COLOR_DEFENDED, marker="D",
               edgecolors="white", linewidths=1.0, zorder=3,
               label="poisoned + defence")

    # Optional annotation of % gap recovered next to defended marker.
    if annotate_recovered:
        for x, r in zip(xs, rows):
            rec = gap_recovered(r["asr"]["clean"], r["asr"]["poisoned"],
                                r["asr"]["defended"])
            if rec is None:
                continue
            ax.annotate(
                rf"${rec * 100:.0f}\%$",
                xy=(x, r["asr"]["defended"]),
                xytext=(9, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=11,
                color=ACCENT,
            )

    # y-limits: respect requested target, auto-expand if data exceeds it.
    all_vals = clean_vals + poison_vals + def_vals
    data_lo, data_hi = min(all_vals), max(all_vals)
    span = max(data_hi - data_lo, 1e-3)
    pad = 0.10 * span
    y_lo = min(y_lo_target, data_lo - pad)
    y_hi = max(y_hi_target, data_hi + pad)
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
    rows = [load_combo(label, subdir) for label, subdir in COMBOS]
    # Sort by ΔASR descending (same convention as fig_main_results).
    rows.sort(key=lambda r: r["delta_asr"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Log a compact table of what's being plotted.
    print(
        f"{'combo':<40} "
        f"{'ASR_c':>6} {'ASR_p':>6} {'ASR_d':>6}  "
        f"{'hA_c':>6} {'hA_p':>6} {'hA_d':>6}  "
        f"{'gap_rec':>8}  {'ΔASR':>6}"
    )
    for r in rows:
        flat = r["label"].replace("\n", " ")
        rec = gap_recovered(r["asr"]["clean"], r["asr"]["poisoned"],
                            r["asr"]["defended"])
        rec_s = "n/a" if rec is None else f"{rec * 100:6.1f}%"
        print(
            f"{flat:<40} "
            f"{r['asr']['clean']:>6.3f} {r['asr']['poisoned']:>6.3f} "
            f"{r['asr']['defended']:>6.3f}  "
            f"{r['hattr']['clean']:>6.3f} {r['hattr']['poisoned']:>6.3f} "
            f"{r['hattr']['defended']:>6.3f}  "
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
