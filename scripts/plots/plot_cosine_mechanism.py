"""Plot the cosine-rotation mechanism: clean vs poisoned cos(v, -r) per combo.

Replaces Table 2 of the paper. Reads each combo's per-seed summary.json
(``results/<model>/<attr>/seed{0,1,2}/summary.json``) and renders a paired
arrow/dot plot sorted by Delta cos (descending). The clean steering vector
is deterministic so ``cos_clean`` has no per-seed dispersion; ``cos_poisoned``
varies with the GCG seed and is shown with a ±1 std error bar. The annotated
edit count is the mean over seeds.

Run: .venv/bin/python scripts/plots/plot_cosine_mechanism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advsteer import PROJECT_ROOT
from _donstyle import apply_style, CLEAN, POISONED, REF, ACCENT
from _seeds import (
    Aggregate,
    aggregate_from_seeds,
    fmt_mean_std,
    summary_extractor,
)

apply_style()


RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

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

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_ARROW = ACCENT
COLOR_ZERO = REF


def load_combo(label: str, subdir: str) -> dict:
    combo_dir = RESULTS_DIR / subdir
    cos_clean = aggregate_from_seeds(combo_dir, summary_extractor("cos_clean"))
    cos_poisoned = aggregate_from_seeds(combo_dir, summary_extractor("cos_poisoned"))
    n_mods = aggregate_from_seeds(combo_dir, summary_extractor("n_total_modifications"))
    n_texts = aggregate_from_seeds(combo_dir, summary_extractor("n_texts_modified"))
    return {
        "label": label,
        "subdir": subdir,
        "cos_clean": cos_clean,
        "cos_poisoned": cos_poisoned,
        "delta_cos": cos_poisoned.mean - cos_clean.mean,
        "n_total_modifications": n_mods,
        "n_texts_modified": n_texts,
    }


def main() -> None:
    rows = [load_combo(label, subdir) for label, subdir in COMBOS]
    rows.sort(key=lambda r: r["delta_cos"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"{'combo':<40} {'cos_clean':>16} {'cos_poison':>18} {'Δcos':>8} "
        f"{'edits (mean±std)':>20} {'texts':>18}"
    )
    for r in rows:
        flat = r["label"].replace("\n", " ")
        print(
            f"{flat:<40} "
            f"{fmt_mean_std(r['cos_clean']):>16} "
            f"{fmt_mean_std(r['cos_poisoned']):>18} "
            f"{r['delta_cos']:>8.4f} "
            f"{fmt_mean_std(r['n_total_modifications'], '{:.0f}'):>20} "
            f"{fmt_mean_std(r['n_texts_modified'], '{:.0f}'):>18}"
        )

    fig, ax = plt.subplots(figsize=(14, 5))

    xs = list(range(len(rows)))
    clean_means = [r["cos_clean"].mean for r in rows]
    poison_means = [r["cos_poisoned"].mean for r in rows]
    poison_stds = [r["cos_poisoned"].std for r in rows]

    # y-limits: include zero, all observed values (incl. error-bar tips), with margin.
    all_vals = clean_means + poison_means + [0.0]
    for m, s in zip(poison_means, poison_stds):
        all_vals.extend([m - s, m + s])
    y_min, y_max = min(all_vals), max(all_vals)
    span = y_max - y_min if y_max > y_min else 1.0
    pad = 0.18 * span
    ax.set_ylim(y_min - pad, y_max + pad * 1.6)

    ax.axhline(0.0, color=COLOR_ZERO, linestyle="--", linewidth=1.2,
               alpha=0.5, zorder=0)

    # Arrows from clean -> poisoned mean (drawn first so dots sit on top).
    for x, c, p in zip(xs, clean_means, poison_means):
        ax.annotate(
            "",
            xy=(x, p),
            xytext=(x, c),
            arrowprops=dict(
                arrowstyle="->",
                color=COLOR_ARROW,
                lw=1.8,
                shrinkA=6,
                shrinkB=6,
            ),
            zorder=2,
        )

    # Error bars on poisoned (clean has std=0 by construction).
    for x, m, s in zip(xs, poison_means, poison_stds):
        if s <= 1e-9:
            continue
        ax.errorbar(
            x, m, yerr=s,
            fmt="none",
            ecolor=COLOR_POISONED,
            elinewidth=1.4,
            capsize=4,
            capthick=1.2,
            alpha=0.85,
            zorder=2.5,
        )

    ax.scatter(xs, clean_means, s=40, color=COLOR_CLEAN, zorder=3,
               label="clean", edgecolors="white", linewidths=1.0)
    ax.scatter(xs, poison_means, s=40, color=COLOR_POISONED, zorder=3,
               label="poisoned", edgecolors="white", linewidths=1.0)

    headroom_top = ax.get_ylim()[1]
    for x, r in zip(xs, rows):
        top_y = max(r["cos_clean"].mean,
                    r["cos_poisoned"].mean + r["cos_poisoned"].std)
        ann_y = top_y + 0.06 * span
        ann_y = min(ann_y, headroom_top - 0.02 * span)
        ax.text(
            x,
            ann_y,
            rf"{r['n_total_modifications'].mean:.0f} edits / "
            rf"{r['n_texts_modified'].mean:.0f} texts",
            ha="center",
            va="bottom",
            fontsize=11,
            color=REF,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=12)
    ax.set_ylabel(r"$\cos(\mathbf{v},\ -\mathbf{r})$")
    ax.set_xlabel("")
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", labelsize=12)
    ax.margins(x=0.06)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=13,
        handletextpad=0.5,
    )

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "fig_cosine_mechanism.pdf"
    png_path = OUTPUT_DIR / "fig_cosine_mechanism.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    for p in (pdf_path, png_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"Failed to write non-empty file: {p}")
        print(f"wrote {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
