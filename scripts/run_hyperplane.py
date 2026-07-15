"""Run the (model × attribute × seed) attack + eval matrix with the *hyperplane*
(RepE PCA reading-direction) steering method.

Same GCG attack as run_matrix.py, but config/hyperplane.yaml sets
`steer_method: hyperplane`, so the attribute steering vector is derived as the
normal of the linear hyperplane separating the POS/NEG classes (top principal
component of the recentered paired differences) rather than the mean-difference
readout. Everything lands in a per-cell `hyperplane/` subdir so it never
collides with the mean-difference matrix:

  results/<model>/<attr>/seed<S>/hyperplane/
    steering_vector.pt
    results_{clean,poisoned}_{harmful,harmless}_w<W>/

Dispatch:
  - SLURM_ARRAY_TASK_ID set → run only that (cell, seed) (slurm/run_hyperplane.sh).
  - otherwise               → run every (cell, seed) sequentially (local path).

Everything is skip-if-exists, so an interrupted run resumes on re-invocation.
Override matrix fields via Hydra CLI, e.g.:
  uv run python scripts/run_hyperplane.py seeds=[0] attributes=[spanish]
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import (
    attack_cmd, cell_dir, eval_sweep, iter_cells, run_subprocess,
)


def run_cell(cfg: DictConfig, root: Path, model, attr: str, seed: int) -> int:
    out_dir = cell_dir(root, model, attr, seed) / "hyperplane"
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "steering_vector.pt"

    print("=" * 60, flush=True)
    print(f"  hyperplane {model.name}/{attr} seed={seed}  L{model.layer}  "
          f"weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

    if vec_path.exists():
        print(f"[attack] {vec_path.name} exists — SKIP", flush=True)
    else:
        rc = run_subprocess(
            attack_cmd(cfg, model, attr, out_dir, seed=seed,
                       steer_method=cfg.steer_method),
            log=out_dir / "attack.log",
            tag=f"attack {model.name}/{attr} seed={seed} (hyperplane)",
        )
        if rc != 0:
            print(f"[{model.name}/{attr} seed={seed}] attack FAILED rc={rc}", flush=True)
            return rc

    return eval_sweep(
        cfg, model, attr, vec_path, out_dir,
        seed=seed,
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="hyperplane", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr, seed in iter_cells(cfg):
        if run_cell(cfg, root, model, attr, seed) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
