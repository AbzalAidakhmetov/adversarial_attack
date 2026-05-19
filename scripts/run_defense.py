"""Run the orthogonalization defense over the (model × attribute × seed) matrix.

Per (cell, seed):
  1) Orthogonalize: v_def = v − (v·r̂) r̂  (r̂ recomputed from refusal/train splits).
     Produces results/<model>/<attr>/seed<S>/steering_vector_defended.pt
     (skip if exists).
  2) ASR eval sweep at use_defended=true × weights × {clean, poisoned} ×
     {harmful, harmless} → seed<S>/results_defense_<tag>_<split>_w<W>/.

Requires each (cell, seed)'s steering_vector.pt to exist (run run_matrix.py first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import (
    cell_dir, eval_sweep, iter_cells, run_subprocess,
)


def orthogonalize_cmd(model, vec_in: Path) -> list[str]:
    return [
        "uv", "run", "python", "-m", "advsteer.defense.orthogonalize_steering",
        "--input", str(vec_in),
        "--model", model.hf_id,
    ]


def run_cell(cfg: DictConfig, root: Path, model, attr: str, seed: int) -> int:
    out_dir = cell_dir(root, model, attr, seed)
    vec_path = out_dir / "steering_vector.pt"
    if not vec_path.exists():
        print(f"  {model.name}/{attr} seed={seed}: SKIP — {vec_path} missing (run run_matrix.py first)", flush=True)
        return 1
    vec_def = out_dir / "steering_vector_defended.pt"

    print("=" * 60, flush=True)
    print(f"  defense {model.name}/{attr} seed={seed}  L{model.layer}  weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

    if vec_def.exists():
        print(f"[ortho] {vec_def.name} exists — SKIP", flush=True)
    else:
        rc = run_subprocess(
            orthogonalize_cmd(model, vec_path),
            log=out_dir / "defense.log",
            tag=f"ortho {model.name}/{attr} seed={seed}",
        )
        if rc != 0:
            print(f"[{model.name}/{attr} seed={seed}] orthogonalize FAILED rc={rc}", flush=True)
            return rc

    return eval_sweep(
        cfg, model, attr, vec_def, out_dir,
        seed=seed,
        dir_prefix="results_defense",
        use_defended=True,
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="defense", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr, seed in iter_cells(cfg):
        if run_cell(cfg, root, model, attr, seed) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
