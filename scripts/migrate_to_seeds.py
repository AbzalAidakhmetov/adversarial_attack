"""One-shot migration: reshape `results/<model>/<attr>/` to per-seed layout.

Before (flat per cell):
  results/<model>/<attr>/{steering_vector.pt, summary.json, results_*_w<W>/, ...}

After:
  results/<model>/<attr>/seed0/{steering_vector.pt, summary.json, results_*_w<W>/, ...}

Also renames the legacy unsuffixed dirs from the pre-_w<W>-suffix era:
  results_<tag>_<split>/         → results_<tag>_<split>_w<headline>/
  results_defense_<tag>_<split>/ → results_defense_<tag>_<split>_w<headline>/

Headline weights per cell (from the pre-refactor scripts) are hard-coded
below — they only need to be correct for the cells/dirs that actually existed
unsuffixed (the script silently skips cells where the unsuffixed dir is absent).

Idempotent: re-running after a successful migration is a no-op (the `seed0/`
subdir is detected and skipped).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CELLS = [
    "gemma/spanish", "gemma/french", "gemma/has_bold_only", "gemma/lowercase",
    "llama31/spanish", "llama31/french", "llama31/has_bold_only", "llama31/lowercase",
]

# Matrix (poisoned/clean) headline weight per cell. Only cells that had an
# unsuffixed results_<tag>_<split>/ dir need an entry; others (fill_matrix
# combos that were already _wN-suffixed from day one) are absent.
MATRIX_HEADLINE = {
    "gemma/spanish": 3, "gemma/french": 3, "gemma/has_bold_only": 4,
    "llama31/spanish": 3, "llama31/lowercase": 2,
}

# Defense headline weight per cell (every cell had an unsuffixed defense dir,
# computed at the cell's preferred weight in the old run_defense.sh).
DEFENSE_HEADLINE = {
    "gemma/spanish": 3, "gemma/french": 3, "gemma/has_bold_only": 4, "gemma/lowercase": 4,
    "llama31/spanish": 3, "llama31/french": 4, "llama31/has_bold_only": 4, "llama31/lowercase": 2,
}


def migrate_cell(cell: str) -> None:
    cell_dir = RESULTS / cell
    if not cell_dir.is_dir():
        print(f"  {cell}: no dir, skipping")
        return

    seed_dir = cell_dir / "seed0"
    if seed_dir.exists():
        print(f"  {cell}: seed0/ already exists, skipping")
        return

    seed_dir.mkdir()

    # Move every top-level child into seed0/ (except seed0 itself).
    moved = []
    for child in sorted(cell_dir.iterdir()):
        if child.name == "seed0":
            continue
        shutil.move(str(child), str(seed_dir / child.name))
        moved.append(child.name)
    print(f"  {cell}: moved {len(moved)} items into seed0/")

    # Rename legacy unsuffixed result dirs → _w<headline>.
    for prefix_map in (
        ("results", MATRIX_HEADLINE.get(cell)),
        ("results_defense", DEFENSE_HEADLINE.get(cell)),
    ):
        prefix, w = prefix_map
        if w is None:
            continue
        for tag in ("clean", "poisoned"):
            for split in ("harmful", "harmless"):
                src = seed_dir / f"{prefix}_{tag}_{split}"
                dst = seed_dir / f"{prefix}_{tag}_{split}_w{w}"
                if src.is_dir() and not dst.exists():
                    src.rename(dst)
                    print(f"    rename: {src.name} → {dst.name}")
                elif src.is_dir() and dst.exists():
                    print(f"    WARN: both {src.name} and {dst.name} exist — leaving alone")


def main() -> int:
    if not RESULTS.is_dir():
        print(f"ERROR: {RESULTS} not found", file=sys.stderr)
        return 1
    print(f"Migrating {len(CELLS)} cells under {RESULTS}/ → per-seed layout (seed0/)")
    for cell in CELLS:
        migrate_cell(cell)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
