"""Magnitude-matched re-eval of the hyperplane (RepE PCA) steering vectors.

The hyperplane readout returns a UNIT-NORM direction, whereas the mean-difference
readout returns a large-norm vector (|v|~70-90 for gemma). Steering adds
`weight * v`, so evaluating the unit vector at the mean-diff-calibrated weights
[2,3,4] applies ~80x too little perturbation. To compare the *direction* fairly,
this rescales each cell's hyperplane vector to that cell's mean-difference
POISONED norm and re-evaluates at the same weights — isolating direction from
magnitude.

Per cell it writes results/<m>/<a>/seed<S>/hyperplane_matched/steering_vector.pt
(poisoned + clean scaled by the same per-cell factor) and runs the standard
eval sweep into that dir. Skip-if-exists. Needs run_hyperplane.py (hyperplane
vector) and run_matrix.py (mean_diff vector) to have produced their vectors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import cell_dir, eval_sweep, iter_cells


def run_cell(cfg: DictConfig, root: Path, model, attr: str, seed: int) -> int:
    base = cell_dir(root, model, attr, seed)
    hp_vec = base / "hyperplane" / "steering_vector.pt"
    md_vec = base / "steering_vector.pt"
    if not hp_vec.exists() or not md_vec.exists():
        print(f"  {model.name}/{attr} seed={seed}: SKIP — missing "
              f"{'hyperplane' if not hp_vec.exists() else 'mean_diff'} vector", flush=True)
        return 0

    out_dir = base / "hyperplane_matched"
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "steering_vector.pt"

    if not vec_path.exists():
        H = torch.load(hp_vec, map_location="cpu", weights_only=False)
        M = torch.load(md_vec, map_location="cpu", weights_only=False)
        scale = M["steering_vector_poisoned"].norm().item()  # per-cell magnitude match
        torch.save({
            "steering_vector_poisoned": H["steering_vector_poisoned"] * scale,
            "steering_vector_clean": H["steering_vector_clean"] * scale,
            "layer": H["layer"], "model": H["model"], "match_scale": scale,
        }, vec_path)
        print(f"  {model.name}/{attr} seed={seed}: scaled hyperplane vector "
              f"×{scale:.2f} (|v_poisoned|→{(H['steering_vector_poisoned']*scale).norm():.2f})",
              flush=True)

    print("=" * 60, flush=True)
    print(f"  hyperplane_matched {model.name}/{attr} seed={seed}  L{model.layer}  "
          f"weights={list(cfg.weights)}", flush=True)
    print("=" * 60, flush=True)

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
