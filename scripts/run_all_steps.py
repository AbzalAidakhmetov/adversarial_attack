"""Run ASR eval over the matrix at the `all_steps` steering protocol.

Per cell, reuses the base `steering_vector.pt` (the steering vector is
protocol-independent — `run_matrix.py` must have produced it) and evaluates at
the lower weight sweep configured in config/all_steps.yaml. Output lands at:

  results/<model>/<attr>/all_steps/results_{clean,poisoned}_{harmful,harmless}_w<W>/

so it never clobbers the prefill-protocol results next to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer.orchestration import eval_sweep, iter_cells, project_root


def run_cell(cfg: DictConfig, root: Path, model, attr: str) -> int:
    base_dir = root / "results" / model.name / attr
    vec_path = base_dir / "steering_vector.pt"
    if not vec_path.exists():
        print(f"  {model.name}/{attr}: SKIP — {vec_path} missing (run run_matrix.py first)", flush=True)
        return 1

    out_dir = base_dir / "all_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"  all_steps {model.name}/{attr}  L{model.layer}  weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

    return eval_sweep(
        cfg, model, attr, vec_path, out_dir,
        protocol="all_steps",
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="all_steps", version_base=None)
def main(cfg: DictConfig) -> None:
    root = project_root()
    rc = 0
    for model, attr in iter_cells(cfg):
        if run_cell(cfg, root, model, attr) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
