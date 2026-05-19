"""Run ASR eval over the (model × attribute × seed) matrix at the `all_steps` protocol.

Per (cell, seed): reuse the seed's steering_vector.pt (run_matrix.py must have
produced it; the vector is protocol-independent) and evaluate at the lower
weight sweep in config/all_steps.yaml. Output:

  results/<model>/<attr>/seed<S>/all_steps/results_{clean,poisoned}_{harmful,harmless}_w<W>/
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import cell_dir, eval_sweep, iter_cells


def run_cell(cfg: DictConfig, root: Path, model, attr: str, seed: int) -> int:
    base_dir = cell_dir(root, model, attr, seed)
    vec_path = base_dir / "steering_vector.pt"
    if not vec_path.exists():
        print(f"  {model.name}/{attr} seed={seed}: SKIP — {vec_path} missing (run run_matrix.py first)", flush=True)
        return 1

    out_dir = base_dir / "all_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"  all_steps {model.name}/{attr} seed={seed}  L{model.layer}  weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

    return eval_sweep(
        cfg, model, attr, vec_path, out_dir,
        seed=seed,
        protocol="all_steps",
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="all_steps", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr, seed in iter_cells(cfg):
        if run_cell(cfg, root, model, attr, seed) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
