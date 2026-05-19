"""Helpers for loading per-seed result files under ``results/<model>/<attr>/seed<S>/``.

All plot scripts now read three GCG seeds per (model, attribute) cell and
report mean over seeds; quantities that depend on the poisoned vector are
additionally reported with a dispersion estimate (std across seeds).

Conventions:
  * Clean metrics (clean steering vector, clean ASR / hAttr) are
    seed-independent in this codebase (the clean steering vector is computed
    deterministically from the unmodified pair texts). We still iterate over
    all seed dirs to confirm consistency; the returned mean is the value to
    plot.
  * Poisoned and defended metrics vary with the GCG seed; we plot their mean
    over the available seeds and annotate the dispersion.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class Aggregate:
    """Mean / std / min / max over a list of per-seed values."""

    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        if not self.values:
            raise ValueError("Aggregate is empty")
        return statistics.fmean(self.values)

    @property
    def std(self) -> float:
        if len(self.values) <= 1:
            return 0.0
        return statistics.stdev(self.values)

    @property
    def lo(self) -> float:
        return min(self.values)

    @property
    def hi(self) -> float:
        return max(self.values)

    @property
    def n(self) -> int:
        return len(self.values)


def _seed_dirs(combo_dir: Path, seeds: Iterable[int]) -> list[Path]:
    return [combo_dir / f"seed{s}" for s in seeds]


def _require_existing(paths: list[Path]) -> list[Path]:
    keep = [p for p in paths if p.exists()]
    if not keep:
        raise FileNotFoundError(
            f"No seed directories exist among: {[str(p) for p in paths]}"
        )
    return keep


def aggregate_from_seeds(
    combo_dir: Path,
    extract: Callable[[Path], float],
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    required: bool = True,
    drop_nan: bool = False,
) -> Aggregate:
    """Apply ``extract(seed_dir)`` over each seed and aggregate.

    ``required=True`` (default) errors if no seed dir exists. ``required=False``
    returns an empty Aggregate (caller must handle ``n == 0``). ``drop_nan``
    silently filters out NaN values returned by ``extract``; useful for
    quantities like ``mean_perplexity`` that can degenerate to NaN when the
    model produces no decodable harmless responses at a given weight.
    """
    import math

    seed_dirs = _seed_dirs(combo_dir, seeds)
    if required:
        seed_dirs = _require_existing(seed_dirs)
    else:
        seed_dirs = [p for p in seed_dirs if p.exists()]
    vals: list[float] = []
    for sd in seed_dirs:
        v = float(extract(sd))
        if drop_nan and math.isnan(v):
            continue
        vals.append(v)
    return Aggregate(tuple(vals))


def _read_single_results_dict(results_file: Path) -> dict:
    """Read a Hydra ``results`` file: a single-element list of a dict."""
    if not results_file.is_file():
        raise FileNotFoundError(f"Missing results file: {results_file}")
    with results_file.open() as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError(
            f"Expected a single-element list at {results_file}; got {type(data).__name__}"
        )
    rec = data[0]
    if not isinstance(rec, dict):
        raise ValueError(f"Element is not a dict at {results_file}")
    return rec


def metric_extractor(subdir_name: str, key: str) -> Callable[[Path], float]:
    """Return ``f(seed_dir) -> float`` reading ``key`` from ``seed_dir/<subdir>/results``."""

    def _extract(seed_dir: Path) -> float:
        rec = _read_single_results_dict(seed_dir / subdir_name / "results")
        if key not in rec:
            raise KeyError(
                f"Missing key '{key}' in {seed_dir / subdir_name / 'results'}; "
                f"available={sorted(rec)}"
            )
        val = rec[key]
        if val is None:
            raise ValueError(
                f"Key '{key}' is None in {seed_dir / subdir_name / 'results'}"
            )
        return float(val)

    return _extract


def summary_extractor(key: str, *, cast: Callable[[object], float] = float) -> Callable[[Path], float]:
    """Return ``f(seed_dir) -> cast(summary[key])``."""

    def _extract(seed_dir: Path) -> float:
        path = seed_dir / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing summary.json: {path}")
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict) or key not in data:
            raise KeyError(
                f"Missing '{key}' in {path}; keys={sorted(data) if isinstance(data, dict) else 'n/a'}"
            )
        return cast(data[key])

    return _extract


def vector_norm_extractor(vector_key: str) -> Callable[[Path], float]:
    """Return ``f(seed_dir) -> ||seed_dir/steering_vector.pt[vector_key]||``."""
    import torch

    def _extract(seed_dir: Path) -> float:
        pt = seed_dir / "steering_vector.pt"
        if not pt.is_file():
            raise FileNotFoundError(f"Missing {pt}")
        d = torch.load(pt, map_location="cpu", weights_only=False)
        if vector_key not in d:
            raise KeyError(f"{pt} missing key {vector_key!r}; keys={list(d)}")
        return float(d[vector_key].float().norm().item())

    return _extract


def fmt_mean_std(agg: Aggregate, fmt: str = "{:.3f}", *, plus_sign: bool = False) -> str:
    """Format ``mean ± std`` for log output."""
    if agg.n == 0:
        return "n/a"
    mean_s = ("+" if plus_sign and agg.mean >= 0 else "") + fmt.format(agg.mean)
    if agg.n == 1:
        return mean_s
    return f"{mean_s} ± {fmt.format(agg.std)}"
