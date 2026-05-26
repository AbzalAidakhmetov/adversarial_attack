"""Plot the constrained-bypass sweep (Llama-3.1-8B, lowercase, w=2).

For each cosine cap tau, the attacker re-runs GCG with a hard reject on
``cos(v, -r) > tau`` (see ``--cos_max_hard`` in
``advsteer.attack.build_adv_stealth``). This script aggregates per-seed
results from ``results/llama31/lowercase/seed{0,1,2}/cap_hard<tau>/`` and
emits TWO SEPARATE PDFs (subcaption-friendly), matching the style of
``plot_defence.py``:

  fig_constrained_bypass_asr.pdf — ASR vs tau, with clean and unconstrained
                                   horizontal reference lines.
  fig_constrained_bypass_ac.pdf  — Attribute compliance (lowercase) vs tau,
                                   showing the attack remains attribute-faithful.

Run from project root:
    .venv/bin/python scripts/plots/plot_constrained_bypass.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advsteer import PROJECT_ROOT
from _donstyle import apply_style, CLEAN, POISONED, REF, LINE, ACCENT
from _seeds import (
    Aggregate,
    aggregate_from_seeds,
    fmt_mean_std,
    metric_extractor,
    summary_extractor,
)

apply_style()


RESULTS_ROOT = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

MODEL_DIR = "llama31/lowercase"
LAYER = 18
WEIGHT = 2
SEEDS = (0, 1, 2)
TAUS = (0.05, 0.10, 0.15, 0.20, 0.30)
TARGET_TAU = 0.10

# AC = "steering_success_rate" on harmless prompts (whether the attribute
# instruction is faithfully followed). Same key plot_defence.py uses.
AC_KEY = "steering_success_rate"


def _cap_subdir(seed_dir: Path, tau: float) -> Path:
    # cap directories were created with float(tau): 0.05, 0.1, 0.15, 0.2, 0.3
    return seed_dir / f"cap_hard{float(tau)}"


def _seed_dir_for_cap(combo_dir: Path, tau: float):
    """Return a callable mapping seed_dir -> cap_dir for the given tau."""
    def _resolve(seed_dir: Path) -> Path:
        return _cap_subdir(seed_dir, tau)
    return _resolve


def _cap_metric(combo_dir: Path, tau: float, subdir_name: str, key: str) -> Aggregate:
    """metric_extractor but rooted under cap_hard<tau>/<subdir>/results."""
    base = metric_extractor(subdir_name, key)
    def _extract(seed_dir: Path) -> float:
        return base(_cap_subdir(seed_dir, tau))
    return aggregate_from_seeds(combo_dir, _extract, seeds=SEEDS)


def _cap_summary(combo_dir: Path, tau: float, key: str) -> Aggregate:
    base = summary_extractor(key)
    def _extract(seed_dir: Path) -> float:
        return base(_cap_subdir(seed_dir, tau))
    return aggregate_from_seeds(combo_dir, _extract, seeds=SEEDS)


def load_data(results_root: Path = None):
    combo_dir = (results_root or RESULTS_ROOT) / MODEL_DIR

    # Baselines from the unconstrained run.
    asr_clean = aggregate_from_seeds(
        combo_dir, metric_extractor(f"results_clean_harmful_w{WEIGHT}", "judge_success_rate"),
        seeds=SEEDS,
    )
    asr_unc = aggregate_from_seeds(
        combo_dir, metric_extractor(f"results_poisoned_harmful_w{WEIGHT}", "judge_success_rate"),
        seeds=SEEDS,
    )
    ac_clean = aggregate_from_seeds(
        combo_dir, metric_extractor(f"results_clean_harmless_w{WEIGHT}", AC_KEY),
        seeds=SEEDS,
    )
    ac_unc = aggregate_from_seeds(
        combo_dir, metric_extractor(f"results_poisoned_harmless_w{WEIGHT}", AC_KEY),
        seeds=SEEDS,
    )
    cos_unc = aggregate_from_seeds(
        combo_dir, summary_extractor("cos_poisoned"),
        seeds=SEEDS,
    )
    cos_clean = aggregate_from_seeds(
        combo_dir, summary_extractor("cos_clean"),
        seeds=SEEDS,
    )

    per_tau = []
    for tau in TAUS:
        asr = _cap_metric(combo_dir, tau, f"results_poisoned_harmful_w{WEIGHT}",
                          "judge_success_rate")
        ac = _cap_metric(combo_dir, tau, f"results_poisoned_harmless_w{WEIGHT}",
                         AC_KEY)
        cos_v = _cap_summary(combo_dir, tau, "cos_poisoned")
        per_tau.append({"tau": tau, "asr": asr, "ac": ac, "cos": cos_v})

    return {
        "asr_clean": asr_clean,
        "asr_unc": asr_unc,
        "ac_clean": ac_clean,
        "ac_unc": ac_unc,
        "cos_clean": cos_clean,
        "cos_unc": cos_unc,
        "per_tau": per_tau,
    }


def _draw_sweep(
    ax: plt.Axes,
    taus: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    ax.plot(
        taus, means, color=color, linewidth=1.8, alpha=0.9, zorder=2,
        marker="o", markersize=7, markeredgecolor="white",
        markeredgewidth=1.0, label=label,
    )
    ax.errorbar(
        taus, means, yerr=stds, fmt="none",
        ecolor=color, elinewidth=1.4, capsize=4, capthick=1.2,
        alpha=0.85, zorder=3,
    )


def _draw_baseline(
    ax: plt.Axes,
    y: float,
    *,
    color: str,
    label: str,
) -> None:
    ax.axhline(
        y, color=color, linestyle="--", linewidth=1.4,
        alpha=0.7, zorder=1, label=label,
    )


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def _format_axis(
    ax: plt.Axes,
    taus: np.ndarray,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    ax.set_xlabel(r"$\tau$  (cap on $\cos(\mathbf{v},\,-\mathbf{r})$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:.2f}" for t in taus])
    ax.set_xlim(taus[0] - 0.02, taus[-1] + 0.02)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)


def make_asr_figure(data: dict, out_dir: Path) -> list[Path]:
    taus = np.array([d["tau"] for d in data["per_tau"]], dtype=float)
    means = np.array([d["asr"].mean for d in data["per_tau"]], dtype=float)
    stds = np.array([d["asr"].std for d in data["per_tau"]], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    _draw_baseline(
        ax, data["asr_unc"].mean, color=POISONED,
        label=rf"unconstrained ({data['asr_unc'].mean:.2f})",
    )
    _draw_baseline(
        ax, data["asr_clean"].mean, color=CLEAN,
        label=rf"clean ({data['asr_clean'].mean:.2f})",
    )
    _draw_sweep(
        ax, taus, means, stds,
        color=ACCENT, label="constrained attack",
    )

    # Annotate B(tau*) at the headline cap.
    if TARGET_TAU in [d["tau"] for d in data["per_tau"]]:
        i = [d["tau"] for d in data["per_tau"]].index(TARGET_TAU)
        target_asr = data["per_tau"][i]["asr"].mean
        denom = data["asr_unc"].mean - data["asr_clean"].mean
        if abs(denom) > 1e-9:
            b = (target_asr - data["asr_clean"].mean) / denom
            ax.annotate(
                rf"$B(\tau{{=}}{TARGET_TAU:.2f}) = {b * 100:.0f}\%$",
                xy=(TARGET_TAU, target_asr),
                xytext=(14, 22),
                textcoords="offset points",
                fontsize=13,
                color=ACCENT,
                arrowprops=dict(
                    arrowstyle="->", color=ACCENT, lw=1.0, alpha=0.8,
                ),
            )

    upper = max(
        float(data["asr_unc"].mean + data["asr_unc"].std) + 0.08,
        0.7,
    )
    _format_axis(ax, taus, ylabel="ASR", ylim=(0.0, upper))

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    fig.tight_layout()
    return _save(fig, "fig_constrained_bypass_asr", out_dir)


def make_ac_figure(data: dict, out_dir: Path) -> list[Path]:
    taus = np.array([d["tau"] for d in data["per_tau"]], dtype=float)
    means = np.array([d["ac"].mean for d in data["per_tau"]], dtype=float)
    stds = np.array([d["ac"].std for d in data["per_tau"]], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    _draw_baseline(
        ax, data["ac_unc"].mean, color=POISONED,
        label=rf"unconstrained ({data['ac_unc'].mean:.2f})",
    )
    _draw_baseline(
        ax, data["ac_clean"].mean, color=CLEAN,
        label=rf"clean ({data['ac_clean'].mean:.2f})",
    )
    _draw_sweep(
        ax, taus, means, stds,
        color=ACCENT, label="constrained attack",
    )

    band_lo = float(min(means.min(), data["ac_clean"].mean) - 0.10)
    band_hi = float(max(means.max(), data["ac_unc"].mean) + 0.10)
    _format_axis(
        ax, taus,
        ylabel=r"AC (lowercase compliance)",
        ylim=(max(0.5, band_lo), min(1.05, band_hi)),
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    fig.tight_layout()
    return _save(fig, "fig_constrained_bypass_ac", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    data = load_data(args.results_root)

    print(f"Constrained-bypass ({MODEL_DIR}, L{LAYER}, w={WEIGHT}, seeds={list(SEEDS)})")
    print(f"  baselines:")
    print(f"    ASR clean         = {fmt_mean_std(data['asr_clean'])}")
    print(f"    ASR unconstrained = {fmt_mean_std(data['asr_unc'])}")
    print(f"    AC  clean         = {fmt_mean_std(data['ac_clean'])}")
    print(f"    AC  unconstrained = {fmt_mean_std(data['ac_unc'])}")
    print(f"    cos clean         = {fmt_mean_std(data['cos_clean'], '{:.4f}')}")
    print(f"    cos unconstrained = {fmt_mean_std(data['cos_unc'], '{:.4f}')}")
    print(f"  sweep (mean ± std):")
    print(f"    {'tau':>6}  {'cos':>14}  {'ASR':>14}  {'AC':>14}")
    for d in data["per_tau"]:
        print(
            f"    {d['tau']:>6.3f}  "
            f"{fmt_mean_std(d['cos'], '{:.4f}'):>14}  "
            f"{fmt_mean_std(d['asr']):>14}  "
            f"{fmt_mean_std(d['ac']):>14}"
        )

    paths = make_asr_figure(data, args.out_dir) + make_ac_figure(data, args.out_dir)
    print("\nWrote:")
    for p in paths:
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
