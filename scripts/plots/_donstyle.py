"""Shared plotting style modelled on github.com/crisostomi/donplots.

LaTeX-rendered serif fonts (Computer Modern), seaborn "talk" context, and the
`advsteer.PALETTE` colours. Import once at the top of each plot script:

    from _donstyle import apply_style, PALETTE, CLEAN, POISONED, DEFENDED, REF
    apply_style()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from advsteer import PALETTE as _ADVSTEER_PALETTE


PALETTE = {name: f"#{hex_val}" for name, hex_val in _ADVSTEER_PALETTE.items()}

CLEAN = PALETTE["Dark Slate Grey"]
POISONED = PALETTE["Brown Red"]
DEFENDED = PALETTE["Honey Bronze"]
REF = PALETTE["Night Bordeaux"]
LINE = PALETTE["Dark Slate Grey"]
ACCENT = PALETTE["Honey Bronze"]


def apply_style() -> None:
    """Activate the donplots rcParams + seaborn context."""
    sns.set_context("talk")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def group_by_model_layout(
    rows,
    model_order,
    intra_gap: float = 1.6,
    group_gap: float = 1.4,
    sort_key: str = "delta_asr",
    descending: bool = True,
):
    """Lay rows out on an x-axis grouped by ``rows[i]["model"]``.

    Within each model group rows are sorted by ``sort_key`` (descending by
    default). Returns ``(ordered_rows, xs, group_spans)`` where ``group_spans``
    is a list of ``(model, x_first, x_last)`` tuples for sub-row labels.
    """
    ordered_rows = []
    xs = []
    group_spans = []

    cursor = 0.0
    for model in model_order:
        members = [r for r in rows if r["model"] == model]
        if not members:
            continue
        members.sort(key=lambda r: r[sort_key], reverse=descending)
        x_start = cursor
        for r in members:
            ordered_rows.append(r)
            xs.append(cursor)
            cursor += intra_gap
        x_end = cursor - intra_gap
        group_spans.append((model, x_start, x_end))
        cursor += group_gap

    return ordered_rows, xs, group_spans


def draw_model_subrow(ax, group_spans, *, y_bracket: float = -0.16,
                      y_text: float = -0.22, bracket_pad: float = 0.3,
                      fontsize: int = 13) -> None:
    """Draw a short bracket line + model name under each group on ``ax``.

    Uses a blended transform: x in data coords, y in axes coords. The
    caller should reserve space at the bottom of the figure (e.g. via
    ``fig.subplots_adjust(bottom=0.22)``).
    """
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for model, x0, x1 in group_spans:
        xc = 0.5 * (x0 + x1)
        ax.plot(
            [x0 - bracket_pad, x1 + bracket_pad],
            [y_bracket, y_bracket],
            transform=trans, color=REF,
            linewidth=1.0, clip_on=False, zorder=4,
        )
        ax.text(
            xc, y_text, model,
            transform=trans,
            ha="center", va="top",
            fontsize=fontsize,
            color=REF,
        )
