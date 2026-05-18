"""Run the orthogonalization defense over the full (model × attribute) matrix.

Per cell:
  1) Orthogonalize: v_def = v − (v·r̂) r̂  (r̂ recomputed from refusal/train splits).
     Produces results/<model>/<attr>/steering_vector_defended.pt  (skip if exists).
  2) ASR eval sweep at use_defended=true × weights × {clean, poisoned} × {harmful, harmless} →
     results/<model>/<attr>/results_defense_{clean,poisoned}_{harmful,harmless}_w<W>/

Requires each cell's `steering_vector.pt` to exist (run scripts/run_matrix.py first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import (
    eval_sweep, iter_cells, run_subprocess,
)


def orthogonalize_cmd(model, vec_in: Path) -> list[str]:
    return [
        "uv", "run", "python", "-m", "advsteer.defense.orthogonalize_steering",
        "--input", str(vec_in),
        "--model", model.hf_id,
    ]


def run_cell(cfg: DictConfig, root: Path, model, attr: str) -> int:
    out_dir = root / "results" / model.name / attr
    vec_path = out_dir / "steering_vector.pt"
    if not vec_path.exists():
        print(f"  {model.name}/{attr}: SKIP — {vec_path} missing (run run_matrix.py first)", flush=True)
        return 1
    vec_def = out_dir / "steering_vector_defended.pt"

    print("=" * 60, flush=True)
    print(f"  defense {model.name}/{attr}  L{model.layer}  weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

    if vec_def.exists():
        print(f"[ortho] {vec_def.name} exists — SKIP", flush=True)
    else:
        rc = run_subprocess(
            orthogonalize_cmd(model, vec_path),
            log=out_dir / "defense.log", tag=f"ortho {model.name}/{attr}",
        )
        if rc != 0:
            print(f"[{model.name}/{attr}] orthogonalize FAILED rc={rc} — skipping evals", flush=True)
            return rc

    return eval_sweep(
        cfg, model, attr, vec_def, out_dir,
        dir_prefix="results_defense",
        use_defended=True,
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="defense", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr in iter_cells(cfg):
        if run_cell(cfg, root, model, attr) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
