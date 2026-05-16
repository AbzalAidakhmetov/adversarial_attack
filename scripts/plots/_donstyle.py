"""Shared plotting style modelled on github.com/crisostomi/donplots.

LaTeX-rendered serif fonts (Computer Modern), seaborn "talk" context, and the
`ocean_sunset` palette. Import once at the top of each plot script:

    from _donstyle import apply_style, PALETTE, CLEAN, POISONED, DEFENDED, REF
    apply_style()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = {
    "Rich black":    "#001219",
    "Midnight green":"#005f73",
    "Dark cyan":     "#0a9396",
    "Tiffany Blue":  "#94d2bd",
    "Vanilla":       "#e9d8a6",
    "Gamboge":       "#ee9b00",
    "Alloy orange":  "#ca6702",
    "Rust":          "#bb3e03",
    "Rufous":        "#ae2012",
    "Auburn":        "#9b2226",
}

CLEAN = PALETTE["Midnight green"]
POISONED = PALETTE["Auburn"]
DEFENDED = PALETTE["Gamboge"]
REF = PALETTE["Rich black"]


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
