"""Constrained-bypass curve: hard cosine-cap sweep at a single (model, attr).

For each (cell, seed) × cap in cfg.caps, re-run the GCG attack with
`--cos_max_hard=cap` (per-candidate hard reject, score-monotonic acceptance)
and evaluate poisoned ASR + AC at the matrix weights.

Output: results/<model>/<attr>/seed<S>/cap_hard<CAP>/{steering_vector.pt,
        summary.json, results_poisoned_{harmful,harmless}_w<W>/}.

Caps strictly below cos(v_clean, r) are skipped — that threshold would already
reject the honest clean vector and is not a credible defender operating point.
cos_clean is read from the cell's unconstrained summary.json
(produced by scripts/run_matrix.py) so this runner depends on that having
been run first for the same (cell, seed).

Dispatched (cell, seed) by SLURM_ARRAY_TASK_ID (same indexing as the matrix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import (
    attack_cmd, cell_dir, eval_sweep, iter_cells, run_subprocess,
)


def read_cos_clean(base_dir: Path) -> float | None:
    """Pick cos_clean from the unconstrained-cell summary.json if present."""
    f = base_dir / "summary.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return float(json.load(fh).get("cos_clean", 0.0))


def run_bypass_cell(cfg: DictConfig, root: Path, model, attr: str, seed: int) -> int:
    base_dir = cell_dir(root, model, attr, seed)
    cos_clean = read_cos_clean(base_dir)
    if cos_clean is None:
        print(f"  WARN: {base_dir}/summary.json not found — cannot filter caps "
              "by cos_clean; running every cap.", flush=True)

    rc = 0
    for cap in cfg.caps:
        cap_f = float(cap)
        if cos_clean is not None and cap_f < cos_clean:
            print(f"[skip cap={cap_f}] below cos_clean={cos_clean:.4f}", flush=True)
            continue

        cap_dir = base_dir / f"cap_hard{cap_f}"
        cap_dir.mkdir(parents=True, exist_ok=True)
        vec_path = cap_dir / "steering_vector.pt"

        print("=" * 60, flush=True)
        print(f"  hard-bypass {model.name}/{attr} seed={seed}  "
              f"L{model.layer}  cos_max_hard={cap_f}", flush=True)
        print("=" * 60, flush=True)

        if vec_path.exists():
            print(f"[attack cap={cap_f}] {vec_path.name} exists — SKIP", flush=True)
        else:
            attack_rc = run_subprocess(
                attack_cmd(cfg, model, attr, cap_dir, seed=seed,
                           cos_max_hard=cap_f),
                log=cap_dir / "attack.log",
                tag=f"attack {model.name}/{attr} seed={seed} cap_hard={cap_f}",
            )
            if attack_rc != 0:
                print(f"  cap={cap_f} attack FAILED rc={attack_rc} — "
                      "skipping evals", flush=True)
                rc = 1
                continue

        rc |= eval_sweep(
            cfg, model, attr, vec_path, cap_dir,
            seed=seed,
            use_clean_options=(False,),
            harmless_path=root / cfg.eval.harmless_prompts,
        )
    return rc


@hydra.main(config_path="../config", config_name="constrained_bypass",
            version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    cells = iter_cells(cfg)
    rc = 0
    for model, attr, seed in cells:
        if run_bypass_cell(cfg, root, model, attr, seed) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
