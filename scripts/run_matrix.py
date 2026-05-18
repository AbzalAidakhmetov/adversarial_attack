"""Run the full (model × attribute) attack + eval matrix.

Per cell:
  1) GCG attack → results/<model>/<attr>/steering_vector.pt  (skip if exists)
  2) ASR eval sweep over weights × {clean, poisoned} × {harmful, harmless} →
     results/<model>/<attr>/results_{clean,poisoned}_{harmful,harmless}_w<W>/

Dispatch:
  - SLURM_ARRAY_TASK_ID set → run only that cell (slurm/run_matrix.sh path).
  - otherwise               → run every cell sequentially (local path).

Override matrix fields via Hydra CLI, e.g.:
  uv run python scripts/run_matrix.py weights=[3] attributes=[spanish]
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import (
    attack_cmd, eval_sweep, iter_cells, run_subprocess,
)


def run_cell(cfg: DictConfig, root: Path, model, attr: str) -> int:
    out_dir = root / "results" / model.name / attr
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "steering_vector.pt"

    print("=" * 60, flush=True)
    print(f"  cell {model.name}/{attr}  L{model.layer}  weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

    if vec_path.exists():
        print(f"[attack] {vec_path.name} exists — SKIP", flush=True)
    else:
        rc = run_subprocess(
            attack_cmd(cfg, model, attr, out_dir),
            log=out_dir / "attack.log", tag=f"attack {model.name}/{attr}",
        )
        if rc != 0:
            print(f"[{model.name}/{attr}] attack FAILED rc={rc} — skipping evals", flush=True)
            return rc

    return eval_sweep(
        cfg, model, attr, vec_path, out_dir,
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="matrix", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr in iter_cells(cfg):
        if run_cell(cfg, root, model, attr) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
