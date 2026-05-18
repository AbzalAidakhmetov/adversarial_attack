"""Plot the cosine-rotation mechanism: clean vs poisoned cos(v, -r) per combo.

Replaces Table 2 of the paper. Reads each combo's summary.json (no hard-coded
numbers) and renders a paired arrow/dot plot sorted by Delta cos (descending).

Run: .venv/bin/python scripts/plots/plot_cosine_mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advsteer import PROJECT_ROOT
from _donstyle import apply_style, CLEAN, POISONED, PALETTE, REF, ACCENT

apply_style()


RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

# (display label, experiment subdirectory) -- cosine is weight-independent
# (it depends only on summary.json's cos_clean/cos_poisoned).
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

REQUIRED_KEYS = (
    "cos_clean",
    "cos_poisoned",
    "delta_cos",
    "n_total_modifications",
    "n_texts_modified",
)

COLOR_CLEAN = CLEAN
COLOR_POISONED = POISONED
COLOR_ARROW = ACCENT
COLOR_ZERO = REF


def load_combo(label: str, subdir: str) -> dict:
    summary_path = RESULTS_DIR / subdir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    with summary_path.open() as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"summary.json at {summary_path} is missing required keys: {missing}"
        )
    return {
        "label": label,
        "subdir": subdir,
        "cos_clean": float(data["cos_clean"]),
        "cos_poisoned": float(data["cos_poisoned"]),
        "delta_cos": float(data["delta_cos"]),
        "n_total_modifications": int(data["n_total_modifications"]),
        "n_texts_modified": int(data["n_texts_modified"]),
    }


def main() -> None:
    rows = [load_combo(label, subdir) for label, subdir in COMBOS]
    # Sort by delta_cos descending (largest rotation on the left).
    rows.sort(key=lambda r: r["delta_cos"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Report data (also useful for the caller's logs).
    print(f"{'combo':<40} {'cos_clean':>10} {'cos_poison':>11} {'Δcos':>8} {'edits':>7} {'texts':>7}")
    for r in rows:
        flat = r["label"].replace("\n", " ")
        print(
            f"{flat:<40} {r['cos_clean']:>10.4f} {r['cos_poisoned']:>11.4f} "
            f"{r['delta_cos']:>8.4f} {r['n_total_modifications']:>7d} {r['n_texts_modified']:>7d}"
        )

    fig, ax = plt.subplots(figsize=(14, 5))

    xs = list(range(len(rows)))
    clean_vals = [r["cos_clean"] for r in rows]
    poison_vals = [r["cos_poisoned"] for r in rows]

    # y limits: include 0 and all observed values with margin.
    all_vals = clean_vals + poison_vals + [0.0]
    y_min, y_max = min(all_vals), max(all_vals)
    span = y_max - y_min if y_max > y_min else 1.0
    pad = 0.18 * span
    ax.set_ylim(y_min - pad, y_max + pad * 1.6)  # extra headroom for annotations

    # Zero reference line.
    ax.axhline(0.0, color=COLOR_ZERO, linestyle="--", linewidth=1.2,
               alpha=0.5, zorder=0)

    # Arrows from clean -> poisoned (drawn first so dots sit on top).
    for x, c, p in zip(xs, clean_vals, poison_vals):
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

    # Dots.
    ax.scatter(xs, clean_vals, s=80, color=COLOR_CLEAN, zorder=3,
               label="clean", edgecolors="white", linewidths=1.0)
    ax.scatter(xs, poison_vals, s=80, color=COLOR_POISONED, zorder=3,
               label="poisoned", edgecolors="white", linewidths=1.0)

    # "N edits / M texts" annotation above each combo.
    headroom_top = ax.get_ylim()[1]
    for x, r in zip(xs, rows):
        top_y = max(r["cos_clean"], r["cos_poisoned"])
        ann_y = top_y + 0.06 * span
        ann_y = min(ann_y, headroom_top - 0.02 * span)
        ax.text(
            x,
            ann_y,
            rf"{r['n_total_modifications']} edits / {r['n_texts_modified']} texts",
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
